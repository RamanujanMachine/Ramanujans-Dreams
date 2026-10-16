"""Tests for the run timeline (:mod:`runmon.timeline`) and the hub alerts that ride on it.

  - The aggregator reduces 1 s samples to mean / max per window, per role, and
    counts pool tasks (also across a replaced pool).
  - The hub writes ``timeline.jsonl`` while running and a final partial window
    on stop; reading survives a torn line.
  - ``/api/history`` plots the timeline; runs without one fall back to heartbeats.
  - ``memory_pressure`` / ``memory_recovered`` and ``disk_low`` fire once, with
    hysteresis; a SIG* stop leaves a ``signal_received`` event.
"""

import json
import os
import threading
import time

import pytest

import runmon
from runmon import events as ev
from runmon import timeline as tl
from runmon.client.sources import FileSource
from runmon.hub import Hub, HubConfig
from runmon.web.server import make_server

FAST = dict(sample_s=0.03, drain_s=0.02, worker_tail_s=0.02, status_s=0.05,
            heartbeat_s=0.1, walltime_check_s=0.02)


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    return list(ev.iter_events(path)) if os.path.exists(path) else []


def _sample(ts, cpu, rss, workers=4, roles=None, avail=1000):
    procs = roles or [{"role": "main", "cpu": cpu / 2, "rss_mb": rss / 2},
                      {"role": "pool worker", "cpu": cpu / 2, "rss_mb": rss / 2}]
    return {"ts": ts, "processes": procs,
            "totals": {"cpu_percent": cpu, "rss_mb": rss, "n": len(procs)},
            "pool_workers": {"workers": workers},
            "machine": {"mem_available_mb": avail},
            "allocation": {"cores": 8, "memory_mb": 4000}}


def test_aggregator_mean_max_and_roles():
    agg = tl.TimelineAggregator()
    agg.add(_sample(100.0, 100, 1000))
    agg.add(_sample(101.0, 300, 3000, workers=6, avail=400))
    rec = agg.flush(102.0, stage="search", disk_free_mb=12345.6)
    assert rec["t0"] == 100.0 and rec["ts"] == 102.0 and rec["n"] == 2
    assert (rec["cpu"], rec["cpu_max"]) == (200.0, 300.0)
    assert (rec["rss_mb"], rec["rss_max"]) == (2000.0, 3000.0)
    assert (rec["workers"], rec["workers_max"]) == (5.0, 6)
    assert rec["mem_avail_min"] == 400 and rec["cores"] == 8 and rec["memory_mb"] == 4000
    assert rec["roles"]["pool worker"] == {"cpu": 100.0, "rss_mb": 1000.0, "rss_max": 1500.0, "n": 1}
    assert rec["stage"] == "search" and rec["disk_free_mb"] == 12346
    assert agg.flush(103.0) is None            # window was reset; nothing new
    json.dumps(rec)


def test_aggregator_counts_tasks_across_replaced_pool():
    agg = tl.TimelineAggregator()
    agg.add_pools([{"name": "eval", "completed": 10}])
    agg.add_pools([{"name": "eval", "completed": 25}])
    agg.add_pools([{"name": "eval", "completed": 4}])      # a new pool with the same name
    agg.add(_sample(1.0, 1, 1))
    assert agg.flush(2.0)["tasks"] == 10 + 15 + 4


def test_parse_lines_skips_torn_and_foreign_lines(tmp_path):
    path = tmp_path / tl.TIMELINE_FILE
    path.write_bytes(b'{"ts": 1, "cpu": 2}\n\nnot json\n{"no_ts": 1}\n{"ts": 3, "cp')
    assert [r["ts"] for r in tl.parse_lines(tl.lines_local(str(tmp_path)))] == [1]
    assert list(tl.lines_local(str(tmp_path / "missing"))) == []


def test_hub_writes_timeline_and_final_window(tmp_path):
    h = Hub(str(tmp_path / "run"), config=HubConfig(timeline_s=0.2, **FAST)).start()
    with h.tree.task("search", kind="stage"):
        assert _wait(lambda: len(list(tl.parse_lines(tl.lines_local(h.run_dir)))) >= 2)
    h.stop()
    recs = list(tl.parse_lines(tl.lines_local(h.run_dir)))
    assert all(r["cpu"] is not None and r["rss_mb"] > 0 and r["n"] >= 1 for r in recs)
    assert [r["ts"] for r in recs] == sorted(r["ts"] for r in recs)
    assert recs[0]["stage"] == "search" and recs[0]["disk_free_mb"] > 0


def test_web_history_uses_timeline_with_heartbeat_fallback(tmp_path):
    run = tmp_path / "run"
    h = Hub(str(run), config=HubConfig(timeline_s=0.1, **FAST)).start()
    assert _wait(lambda: len(list(tl.parse_lines(tl.lines_local(h.run_dir)))) >= 2)
    h.stop()

    def history(timeline_lines):
        from runmon import query as q
        httpd = make_server(FileSource(h.run_dir), lines=lambda: q._lines_local(h.run_dir),
                            timeline_lines=timeline_lines)
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            import http.client
            port = httpd.server_address[1]
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", "/api/history", headers={"Host": f"127.0.0.1:{port}"})
            return json.loads(c.getresponse().read())["history"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    from_timeline = history(lambda: tl.lines_local(h.run_dir))
    assert from_timeline and "cpu_max" in from_timeline[0] and "rss_max" in from_timeline[0]
    from_beats = history(None)
    assert from_beats and "cpu_max" not in from_beats[0]


def test_memory_pressure_fires_once_and_recovers(tmp_path):
    avail = {"mb": 10.0}
    h = Hub(str(tmp_path / "r"), config=HubConfig(memory_probe=lambda: avail["mb"], **FAST)).start()
    try:
        assert _wait(lambda: "memory_pressure" in h.alerts)
        time.sleep(0.2)
        avail["mb"] = 1e9
        assert _wait(lambda: "memory_pressure" not in h.alerts)
    finally:
        h.stop()
    kinds = [e["kind"] for e in _events(h.run_dir)]
    assert kinds.count("memory_pressure") == 1 and kinds.count("memory_recovered") == 1
    pressure = next(e for e in _events(h.run_dir) if e["kind"] == "memory_pressure")
    assert pressure["f"]["biggest"] and pressure["level"] == "WARNING"


def test_disk_low_event(tmp_path):
    h = Hub(str(tmp_path / "r"), config=HubConfig(timeline_s=0.1, disk_low_mb=1e12, **FAST)).start()
    try:
        assert _wait(lambda: "disk_low" in h.alerts)
    finally:
        h.stop()
    assert [e["kind"] for e in _events(h.run_dir)].count("disk_low") == 1


def test_signal_stop_records_signal_received(tmp_path):
    runmon.start(str(tmp_path / "r"), handle_signals=False, **FAST)
    run_dir = runmon.hub().run_dir
    runmon.stop("SIGTERM")
    events = _events(run_dir)
    sig = [e for e in events if e["kind"] == "signal_received"]
    assert len(sig) == 1 and sig[0]["f"]["signal"] == "SIGTERM"
    assert events[-1]["kind"] == "run_end" and events[-1]["f"]["reason"] == "SIGTERM"


def test_aggregator_counts_spawned_processes_and_task_time():
    agg = tl.TimelineAggregator()
    two = [{"pid": 1, "role": "main", "cpu": 1, "rss_mb": 1}, {"pid": 2, "role": "w", "cpu": 1, "rss_mb": 1}]
    three = two + [{"pid": 3, "role": "w", "cpu": 1, "rss_mb": 1}]
    agg.add(_sample(1.0, 1, 1, roles=two))             # baseline: nothing counts as spawned
    agg.add(_sample(2.0, 1, 1, roles=three))           # pid 3 appeared
    agg.add(_sample(3.0, 1, 1, roles=[three[0], three[2]]))   # pid 2 left
    agg.add(_sample(4.0, 1, 1, roles=three))           # a recycled worker came back as new pid 2
    agg.add_pools([{"name": "eval", "completed": 5, "task_seconds": 20.0}])
    agg.add_pools([{"name": "eval", "completed": 9, "task_seconds": 50.0}])
    rec = agg.flush(5.0)
    assert rec["spawned"] == 2 and rec["tasks"] == 9 and rec["task_s"] == 50.0
    agg.add(_sample(6.0, 1, 1, roles=three))
    assert agg.flush(7.0)["spawned"] == 0              # per window, baseline kept across windows


def test_jitter_separates_flat_drifting_and_spiky_signals():
    def jitter(values):
        agg = tl.TimelineAggregator()
        for i, v in enumerate(values):
            agg.add(_sample(float(i), v, 1000))
        rec = agg.flush(99.0)
        return rec["cpu_jitter"], rec["cpu_std"]

    flat = jitter([400] * 10)
    drift = jitter([100 + 10 * i for i in range(10)])            # slow ramp: large spread, small steps
    spiky = jitter([100, 700] * 5)                                # same mean as 400, huge steps
    assert flat == (0.0, 0.0)
    assert drift[0] < 0.1 < spiky[0]
    assert spiky[0] == pytest.approx(600 / 400)
    assert spiky[1] == pytest.approx(300)
    assert tl.TimelineAggregator().flush(1.0) is None


def test_utilization_peaks_and_run_summary():
    agg, summary = tl.TimelineAggregator(), tl.RunSummary()
    for cpu_util, mem_util in ((0.25, 0.5), (0.75, 0.25), (0.5, 0.25)):
        s = _sample(1.0, 100, 100)
        s["totals"].update(cpu_util=cpu_util, mem_util=mem_util)
        agg.add(s)
        summary.add(s)
    rec = agg.flush(2.0)
    assert (rec["cpu_util_max"], rec["mem_util_max"]) == (0.75, 0.5)
    out = summary.as_dict()
    assert out["samples"] == 3 and out["cpu_util_max"] == 0.75 and out["cpu_util_mean"] == 0.5
    assert out["mem_util_mean"] == pytest.approx(0.333, abs=1e-3)
    assert tl.RunSummary().as_dict()["cpu_util_max"] is None
    json.dumps(out)


def test_hub_writes_resource_summary_and_snapshot(tmp_path):
    h = Hub(str(tmp_path / "run"), config=HubConfig(**FAST)).start()
    assert _wait(lambda: h.summary.samples >= 3)
    assert h.snapshot()["summary"]["samples"] >= 3
    h.stop()
    ev_ = [e for e in _events(h.run_dir) if e["kind"] == "resource_summary"]
    assert len(ev_) == 1 and "peak CPU" in ev_[0]["msg"] and "jitter" in ev_[0]["msg"]
    assert ev_[0]["f"]["cpu_util_max"] is not None
    final = json.load(open(os.path.join(h.run_dir, "status.json")))
    assert final["summary"]["samples"] >= 3
