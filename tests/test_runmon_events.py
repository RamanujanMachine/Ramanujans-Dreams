"""Tests for :mod:`runmon.events` — the structured event log.

  - Event shape (schema keys, level normalisation, per-process sequence).
  - ``EventWriter``: append + flush per write, JSON-unsafe values stringified,
    close is idempotent.
  - ``FileTailer``: incremental reads, a torn last line is held back until it
    is completed, invalid lines are skipped.
  - ``JsonlHandler``: ``logging`` records become events, with ``extra``
    ``event``/``fields`` and exception text.
"""

import json
import logging

import pytest

from runmon import events as ev


def test_event_shape():
    e = ev.make_event("pool_created", "8 workers", "WARNING", workers=8)
    assert set(e) == {"v", "ts", "seq", "pid", "role", "level", "kind", "msg", "ctx", "f"}
    assert e["v"] == ev.SCHEMA_VERSION and e["kind"] == "pool_created"
    assert e["level"] == "WARNING" and e["f"] == {"workers": 8}
    assert ev.make_event("x")["seq"] > e["seq"]


@pytest.mark.parametrize("given, name", [
    (logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"), (25, "INFO"), (26, "INFO"),
    (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR"), (logging.CRITICAL, "CRITICAL"),
    ("warning", "WARNING"), ("nonsense", "INFO"), (None, "INFO"),
])
def test_level_names(given, name):
    assert ev.level_name(given) == name


def test_writer_round_trip(tmp_path):
    path = tmp_path / "sub" / "events.jsonl"
    w = ev.EventWriter(str(path))
    w.write(ev.make_event("a", obj=object()))           # not JSON-serialisable -> str
    w.write_many([ev.make_event("b"), ev.make_event("c")])
    # Flushed per write: readable before close.
    kinds = [e["kind"] for e in ev.iter_events(str(path))]
    assert kinds == ["a", "b", "c"]
    w.close()
    w.close()                                             # idempotent
    w.write(ev.make_event("after-close"))                 # ignored, no crash
    assert len(list(ev.iter_events(str(path)))) == 3


def test_tailer_holds_back_torn_line(tmp_path):
    path = tmp_path / "w.jsonl"
    t = ev.FileTailer(str(path))
    assert t.read_new() == []                             # missing file
    line1 = json.dumps({"kind": "one"})
    line2 = json.dumps({"kind": "two"})
    path.write_text(line1 + "\n" + line2[:7])             # second line torn mid-write
    assert [e["kind"] for e in t.read_new()] == ["one"]
    with open(path, "a") as f:
        f.write(line2[7:] + "\nnot json\n" + json.dumps({"kind": "three"}) + "\n")
    assert [e["kind"] for e in t.read_new()] == ["two", "three"]
    assert t.read_new() == []


def test_iter_events_skips_garbage(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"kind":"a"}\n\n{broken\n{"kind":"b"}\n')
    assert [e["kind"] for e in ev.iter_events(str(p))] == ["a", "b"]


def test_jsonl_handler():
    got = []
    log = logging.getLogger("runmon-test-handler")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    h = ev.JsonlHandler(got.append)
    log.addHandler(h)
    try:
        log.info("plain line")
        log.warning("worker died", extra={"event": "worker_died", "fields": {"pool": "eval"}})
        try:
            raise ValueError("boom")
        except ValueError:
            log.error("failed", exc_info=True)
    finally:
        log.removeHandler(h)
    assert [e["kind"] for e in got] == ["log", "worker_died", "log"]
    assert got[1]["level"] == "WARNING" and got[1]["f"]["pool"] == "eval"
    assert "test_runmon_events.py" in got[0]["f"]["src"]
    assert "ValueError: boom" in got[2]["f"]["exc"]


def test_handler_never_raises():
    def bad_sink(_):
        raise RuntimeError("sink down")
    h = ev.JsonlHandler(bad_sink)
    h.handleError = lambda record: None                  # silence stderr
    h.emit(logging.makeLogRecord({"msg": "x", "levelno": 20}))   # must not raise
