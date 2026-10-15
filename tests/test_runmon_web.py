"""Tests for the browser view (``runmon web``, :mod:`runmon.web.server`).

  - The page is served (self-contained: no external scripts or styles).
  - ``/api/snapshot`` carries the source label/state and the run snapshot.
  - Requests whose ``Host`` is not the loopback address are refused (DNS rebinding).
  - ``/api/history`` rebuilds CPU / memory totals from the run's heartbeats.
  - ``/api/query`` applies the ``runmon query`` filters; bad input is an error, not a crash.
  - ``/api/stream`` sends a ``snapshot`` Server-Sent Event right away.
"""

import http.client
import json
import threading
import time

import pytest

from runmon import query as q
from runmon.client.sources import FileSource
from runmon.hub import Hub, HubConfig
from runmon.web.server import make_server


@pytest.fixture
def web(tmp_path):
    h = Hub(str(tmp_path / "run"), "web-test", HubConfig(sample_s=0.05, drain_s=0.02, status_s=0.05,
                                                          heartbeat_s=0.1, walltime_check_s=0.05)).start()
    h.event("worker_died", "a worker died", "WARNING", pool="eval")
    h.event("pool_created", "eval pool", workers=4)
    deadline = time.time() + 10
    while time.time() < deadline and not any(
            e.get("kind") == "heartbeat" for e in q.iter_matching(q._lines_local(h.run_dir), kinds=["heartbeat"])):
        time.sleep(0.05)
    httpd = make_server(FileSource(h.run_dir), lines=lambda: q._lines_local(h.run_dir))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield httpd.server_address[1], h
    httpd.shutdown()
    httpd.server_close()
    h.stop()


def _get(port, path, host="127.0.0.1"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path, headers={"Host": f"{host}:{port}"})
    r = c.getresponse()
    return r.status, r.getheader("Content-Type"), r


def test_page_is_self_contained(web):
    port, _ = web
    status, ctype, r = _get(port, "/")
    body = r.read().decode()
    assert status == 200 and ctype.startswith("text/html")
    assert "<title>Run Monitor</title>" in body
    assert "src=\"http" not in body and "href=\"http" not in body      # works offline


def test_snapshot(web):
    port, h = web
    status, _, r = _get(port, "/api/snapshot")
    data = json.loads(r.read())
    assert status == 200 and data["source"].startswith("file ")
    assert data["snapshot"]["run"]["id"] == "web-test"


def test_foreign_host_refused(web):
    port, _ = web
    status, _, r = _get(port, "/api/snapshot", host="evil.example")
    assert status == 403
    assert _get(port, "/api/snapshot", host="localhost")[0] == 200


def test_history_from_heartbeats(web):
    port, _ = web
    hist = json.loads(_get(port, "/api/history")[2].read())["history"]
    assert hist and {"ts", "cpu_percent", "rss_mb"} <= set(hist[0])


def test_query(web):
    port, _ = web
    r = json.loads(_get(port, "/api/query?level=WARNING")[2].read())
    assert [e["kind"] for e in r["events"]] == ["worker_died"] and r["total"] == 1
    r = json.loads(_get(port, "/api/query?kind=pool_created,worker_died&limit=1")[2].read())
    assert len(r["events"]) == 1 and r["total"] == 2
    assert "error" in json.loads(_get(port, "/api/query?grep=%5B")[2].read())       # bad regex "["
    assert "error" in json.loads(_get(port, "/api/query?since=someday")[2].read())


def test_stream_sends_snapshot(web):
    port, _ = web
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/api/stream", headers={"Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    assert r.getheader("Content-Type") == "text/event-stream"
    assert r.fp.readline().strip() == b"event: snapshot"
    payload = r.fp.readline().decode()
    assert payload.startswith("data: ") and json.loads(payload[6:])["snapshot"]["run"]["id"] == "web-test"
    c.close()
