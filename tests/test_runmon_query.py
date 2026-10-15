"""Tests for ``runmon query`` (:mod:`runmon.query`) and the CLI wiring.

  - Time parsing: relative (30m / 2h / 3d), epoch, ISO; bad input is an error.
  - Filters: kinds (OR), minimum level, regex over message + fields, time
    window, pid / role, task path substring; heartbeats hidden unless asked.
  - Output: readable lines or raw JSON, ``--limit`` keeps the last N; exit
    code 1 when nothing matched.
  - ``watch host:port`` without a token is refused with a clear message.
"""

import io
import json
import time

import pytest

from runmon import query
from runmon.__main__ import main

NOW = 1_760_000_000.0


def _ev(kind, level="INFO", ts=NOW, msg="", pid=1, role="main", task=None, **f):
    return {"v": 1, "ts": ts, "seq": 0, "pid": pid, "role": role, "level": level, "kind": kind,
            "msg": msg, "ctx": {"task": task} if task else {}, "f": f}


EVENTS = [
    _ev("run_start", ts=NOW - 7200, msg="run started"),
    _ev("heartbeat", "DEBUG", ts=NOW - 3600),
    _ev("pool_created", ts=NOW - 3000, msg="[eval pool] 8 workers", task="search/annealing", workers=8),
    _ev("worker_died", "WARNING", ts=NOW - 600, msg="[eval pool] a worker died", pid=7, task="search/annealing"),
    _ev("task_failed", "WARNING", ts=NOW - 300, msg="task t-17 failed", pid=99, role="worker", key="t-17"),
    _ev("log", "ERROR", ts=NOW - 60, msg="LIReC timed out", task="analysis"),
]


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EVENTS) + "garbage\n")
    return str(tmp_path)


def _q(run_dir, **kw):
    out = io.StringIO()
    n = query.run(run_dir, out=out, **kw)
    return n, out.getvalue().splitlines()


def test_parse_when():
    assert query.parse_when("30m", now=NOW) == NOW - 1800
    assert query.parse_when("2h", now=NOW) == NOW - 7200
    assert query.parse_when("1760000000") == NOW
    assert query.parse_when(None) is None
    with pytest.raises(ValueError):
        query.parse_when("yesterday-ish")


def test_filters(run_dir, monkeypatch):
    monkeypatch.setattr(query.time, "time", lambda: NOW)
    assert _q(run_dir)[0] == 5                                   # heartbeat hidden
    assert _q(run_dir, kinds=["heartbeat"])[0] == 1
    assert _q(run_dir, level="WARNING")[0] == 3
    assert _q(run_dir, kinds=["worker_died", "pool_created"])[0] == 2
    assert _q(run_dir, grep="t-17")[0] == 1                      # matches fields too
    assert _q(run_dir, grep="LIREC")[0] == 1                     # case-insensitive
    assert _q(run_dir, since="15m")[0] == 3
    assert _q(run_dir, role="worker")[0] == 1
    assert _q(run_dir, pid=7)[0] == 1
    assert _q(run_dir, task="search/annealing")[0] == 2


def test_output_formats(run_dir):
    n, lines = _q(run_dir, level="WARNING", limit=1)
    assert n == 3 and len(lines) == 1 and "LIReC timed out" in lines[0] and "[analysis]" in lines[0]
    n, lines = _q(run_dir, kinds=["pool_created"], as_json=True)
    assert json.loads(lines[0])["f"]["workers"] == 8


def test_cli_exit_codes(run_dir, capsys):
    assert main(["query", run_dir, "--kind", "worker_died"]) == 0
    assert "a worker died" in capsys.readouterr().out
    assert main(["query", run_dir, "--kind", "nothing_like_this"]) == 1


def test_watch_host_port_needs_token(monkeypatch):
    monkeypatch.delenv("RUNMON_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="needs --token"):
        main(["watch", "localhost:4000"])
