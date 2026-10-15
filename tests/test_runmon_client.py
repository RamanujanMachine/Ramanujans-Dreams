"""Tests for the terminal view and the simulated run (``python -m runmon demo``).

  - ``render`` draws every panel of a snapshot (stages, estimated progress with
    metrics, processes against the allocation, pools, alerts, events) and the
    waiting screen when there is no snapshot yet.
  - The demo runs end to end (fast): the expected event kinds appear, the
    final status says the run ended, and the frame renders from it.
  - The sampler names multiprocessing helper processes.
"""

import collections
import json
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("rich")
from rich.console import Console

from runmon import events as ev
from runmon.client.tui import read_status, render

SNAPSHOT = {
    "v": 1, "type": "snapshot", "ts": time.time(),
    "run": {"id": "demo-1", "host": "n017", "pid": 4242, "uptime": 3725, "stopped": False},
    "sample_age": 0.8, "stage": "search", "current": "search/annealing/shard 0",
    "tasks": [
        {"name": "loading", "status": "done", "fraction": 1.0, "children": []},
        {"name": "search", "status": "running", "kind": "stage", "done": 1, "total": 3,
         "fraction": 0.5, "estimated": False, "eta": 600, "metrics": {}, "children": [
             {"name": "shard 0", "status": "running", "fraction": 0.42, "estimated": True,
              "done": 0, "total": None, "eta": 120, "metrics": {"T": 0.123, "best_delta": 0.0168},
              "children": []}]},
    ],
    "resources": {
        "processes": [{"pid": 4242, "role": "main", "cpu": 3.0, "rss_mb": 400, "status": "running"},
                      {"pid": 4300, "role": "eval worker", "cpu": 99.0, "rss_mb": 900, "status": "running"}],
        "totals": {"n": 2, "cpu_percent": 102, "rss_mb": 1300, "cpu_util": 0.255, "mem_util": 0.16},
        "allocation": {"cores": 4, "memory_mb": 8192, "cores_source": "allocation"},
        "machine": {"cores": 80, "mem_total_mb": 386000, "mem_available_mb": 344000},
    },
    "pools": [{"name": "eval", "workers": 2, "requested": 4, "in_flight": 2, "completed": 140,
               "deaths": 1, "shrinks": 1, "grows": 0, "peak_mb": 880}],
    "walltime": {"end": time.time() + 600, "remaining": 600, "source": "pbs", "limit": 86400},
    "alerts": {"stall": "no progress for 31m00s"},
    "events": [{"ts": time.time(), "level": "WARNING", "kind": "worker_died",
                "msg": "[eval] a worker process died"}],
}


def _text(renderable, width=120):
    c = Console(width=width, record=True, force_terminal=False)
    c.print(renderable)
    return c.export_text()


def test_render_snapshot():
    out = _text(render(SNAPSHOT, "test"))
    for needle in ("demo-1", "n017", "▶ search", "✓ loading", "~ 42.0%", "T=0.123",
                   "best_delta=0.0168", "eval worker", "of 4 cores (allocation)", "of 8192 MB", "0.99",
                   "80 cores", "2/4", "worker_died", "walltime left 10m00s", "of 1d00h",
                   "no progress for 31m00s"):
        assert needle in out, needle


def test_render_waiting_and_narrow():
    assert "waiting for a run status" in _text(render(None, "x"))
    _text(render(SNAPSHOT, "t"), width=60)            # narrow terminal must not crash


def test_demo_end_to_end(tmp_path):
    run_dir = str(tmp_path / "demo")
    out = subprocess.run([sys.executable, "-m", "runmon", "demo", "--no-watch", "--dir", run_dir,
                          "--speed", "12", "--workers", "2"],
                         capture_output=True, text=True, timeout=240,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    kinds = collections.Counter(e["kind"] for e in ev.iter_events(os.path.join(run_dir, "events.jsonl")))
    for k in ("run_start", "pool_created", "watchdog", "worker_died", "walltime", "pool_closed", "run_end"):
        assert kinds[k] >= 1, (k, kinds)
    snap = read_status(run_dir)
    assert snap["run"]["stopped"] is True
    assert all(t["status"] == "done" for t in snap["tasks"])
    assert "postprocess" in _text(render(snap, "file"))
