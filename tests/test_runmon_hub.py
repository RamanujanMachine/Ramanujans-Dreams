"""Tests for :mod:`runmon.hub` and the :mod:`runmon` facade.

A real hub runs in a temp directory with fast timings.

  - Events reach ``events.jsonl`` with the active task as context; ERROR+ is
    written immediately (before the hub's next drain).
  - ``status.json`` holds a full, consistent snapshot (tasks, resources,
    pools, alerts) and is replaced atomically.
  - Worker processes' own event files are merged into the run's log.
  - Sampling sees child processes, labels pool workers, and flags a run that
    uses more memory than its allocation.
  - Alerts: stall (no progress), walltime warnings and the grace period
    (callbacks run, everything flushed), from ``RUNMON_WALLTIME_END``.
  - Heartbeats are compact; ``stop`` is idempotent and writes ``run_end``.
  - Facade: every call is a no-op when monitoring is off; ``init_worker``
    writes the worker's own file; the watchdog fires as an event.
  - Overhead: the hub's tick takes milliseconds.
"""

import json
import multiprocessing as mp
import os
import time

import pytest

import runmon
from runmon import events as ev
from runmon.hub import Hub, HubConfig

FAST = dict(sample_s=0.05, drain_s=0.02, worker_tail_s=0.02, status_s=0.05,
            heartbeat_s=0.1, walltime_check_s=0.02)


def _events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    return list(ev.iter_events(path)) if os.path.exists(path) else []


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def hub(tmp_path):
    h = Hub(str(tmp_path / "run"), "test-run", HubConfig(**FAST)).start()
    yield h
    h.stop("test-teardown")


def test_events_and_context(hub):
    with hub.tree.task("search", total=1, kind="stage"):
        with hub.tree.task("shard 7", kind="shard"):
            hub.event("pool_created", "eval pool", workers=4)
    assert _wait(lambda: any(e["kind"] == "pool_created" for e in _events(hub.run_dir)))
    e = next(e for e in _events(hub.run_dir) if e["kind"] == "pool_created")
    assert e["ctx"]["task"] == "search/shard 7" and e["f"]["workers"] == 4
    assert _events(hub.run_dir)[0]["kind"] == "run_start"


def test_errors_are_written_immediately(tmp_path):
    h = Hub(str(tmp_path / "r"), config=HubConfig(drain_s=3600, **{k: v for k, v in FAST.items() if k != "drain_s"}))
    # Hub thread not started: only the synchronous path can write.
    h.event("info_only", "queued")
    h.event("crash", "boom", "ERROR")
    kinds = [e["kind"] for e in _events(h.run_dir)]
    assert "crash" in kinds and "info_only" in kinds     # the error flushes the queue too
    h.stop()


def test_status_snapshot(hub):
    with hub.tree.task("analysis", total=4, kind="stage") as st:
        st.set(done=1)
        assert _wait(lambda: os.path.exists(os.path.join(hub.run_dir, "status.json")))
        assert _wait(lambda: json.load(open(os.path.join(hub.run_dir, "status.json")))["stage"] == "analysis")
        snap = json.load(open(os.path.join(hub.run_dir, "status.json")))
    assert snap["run"]["id"] == "test-run" and snap["type"] == "snapshot"
    assert snap["tasks"][0]["fraction"] == pytest.approx(0.25)
    assert snap["resources"]["totals"]["n"] >= 1
    assert snap["resources"]["processes"][0]["role"] == "main"
    assert not any(f.endswith(".tmp" + str(os.getpid())) for f in os.listdir(hub.run_dir))


def test_worker_files_are_merged(hub):
    # pid 99999 is not running: once merged, its file is removed (content kept in events.jsonl).
    w = ev.EventWriter(runmon.worker_file(hub.run_dir, pid=99999))
    e = ev.make_event("task_failed", "bad trajectory", "WARNING", key="t-17")
    e["pid"], e["role"] = 99999, "worker"
    w.write(e)
    assert _wait(lambda: any(x["kind"] == "task_failed" for x in _events(hub.run_dir)))
    assert any(x["kind"] == "task_failed" for x in hub.ring)
    w.close()
    assert _wait(lambda: not os.path.exists(runmon.worker_file(hub.run_dir, pid=99999)))
    assert sum(x["kind"] == "task_failed" for x in _events(hub.run_dir)) == 1


def test_live_worker_file_is_kept_and_silent_workers_leave_none(hub):
    path = runmon.worker_file(hub.run_dir)                    # this (live) process as "worker"
    w = ev.EventWriter(path, lazy=True)
    assert not os.path.exists(path)                           # nothing written → no file
    w.write(ev.make_event("still_working"))
    assert _wait(lambda: any(x["kind"] == "still_working" for x in _events(hub.run_dir)))
    time.sleep(0.1)
    assert os.path.exists(path)                               # writer alive: kept
    w.close()
    os.remove(path)


def _sleep_child(seconds):
    time.sleep(seconds)


class _FakePool:
    def __init__(self, pids):
        self.pids = pids

    def monitor_info(self):
        return {"name": "eval", "workers": len(self.pids), "pids": self.pids, "deaths": 0}


def test_sampling_sees_and_labels_children(hub):
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_sleep_child, args=(5,))
    child.start()
    pool = _FakePool([child.pid])
    handle = hub.register_pool(pool)
    try:
        assert _wait(lambda: hub.last_sample and any(
            p["pid"] == child.pid and p["role"] == "eval worker"
            for p in hub.last_sample["processes"]))
        assert hub.snapshot()["pools"][0]["name"] == "eval"
        main = next(p for p in hub.last_sample["processes"] if p["role"] == "main")
        assert main["cores"] == pytest.approx(main["cpu"] / 100, abs=0.01) and main["threads"] >= 1
    finally:
        child.terminate()
        child.join()
        hub.unregister_pool(handle)
    del pool
    assert hub.pools_info() == []


def test_over_allocation_alert(tmp_path):
    h = Hub(str(tmp_path / "r"), config=HubConfig(memory_mb=1.0, **FAST)).start()
    try:
        assert _wait(lambda: "over_allocation" in h.alerts)
        assert _wait(lambda: any(e["kind"] == "over_allocation" for e in _events(h.run_dir)))
    finally:
        h.stop()


def test_stall_alert(tmp_path):
    h = Hub(str(tmp_path / "r"), config=HubConfig(stall_s=0.05, **FAST))
    node = h.tree.open("stuck", kind="stage")
    h.tree.last_progress = time.time() - 10
    h._check_stall()
    assert "stall" in h.alerts
    node.advance()
    h._check_stall()
    assert "stall" not in h.alerts
    kinds = [e["kind"] for e in h.ring]
    assert "stall" in kinds and "stall_cleared" in kinds
    h.stop()


def test_walltime_grace(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNMON_WALLTIME_END", str(time.time() + 120))
    called = []
    h = Hub(str(tmp_path / "r"), config=HubConfig(walltime_grace_s=300, **FAST))
    h.on_grace(lambda: called.append(True))
    h.start()
    try:
        assert _wait(lambda: called == [True])
        # The callbacks run first; the status file follows within the same tick.
        assert _wait(lambda: os.path.exists(os.path.join(h.run_dir, "status.json")))
        kinds = [e["kind"] for e in _events(h.run_dir)]
        assert "walltime" in kinds and "walltime_grace" in kinds
        assert kinds.count("walltime_warning") == 3        # 60, 15 and 5 min marks
        snap = json.load(open(os.path.join(h.run_dir, "status.json")))
        assert 0 < snap["walltime"]["remaining"] <= 120 and snap["walltime"]["source"] == "env"
        time.sleep(0.1)
        assert called == [True]                             # only once
    finally:
        h.stop()


def test_heartbeat_and_stop(tmp_path):
    h = Hub(str(tmp_path / "r"), config=HubConfig(**FAST)).start()
    assert _wait(lambda: any(e["kind"] == "heartbeat" for e in _events(h.run_dir)))
    hb = next(e for e in _events(h.run_dir) if e["kind"] == "heartbeat")
    assert "processes" not in json.dumps(hb)                # compact: totals, not the process list
    open_node = h.tree.open("never-closed")
    h.stop("SIGTERM")
    h.stop("again")                                         # idempotent
    ends = [e for e in _events(h.run_dir) if e["kind"] == "run_end"]
    assert len(ends) == 1 and ends[0]["f"]["reason"] == "SIGTERM" and ends[0]["level"] == "WARNING"
    assert open_node.status == "stopped"
    assert json.load(open(os.path.join(h.run_dir, "status.json")))["run"]["stopped"] is True


def test_tick_is_cheap(hub):
    assert _wait(lambda: hub.ticks > 20)
    assert hub.tick_cost_ms < 100


class TestFacade:
    def test_noop_when_off(self):
        runmon.stop()
        assert not runmon.active()
        runmon.event("x", "y")
        with runmon.task("t", total=3) as t:
            t.advance()
            t.estimate(0.5, T=1)
        with runmon.watch("w", 0.01):
            pass
        assert runmon.register_pool(object()) is None
        assert runmon.current_task().fraction() is None

    def test_start_task_event_stop(self, tmp_path):
        h = runmon.start(str(tmp_path / "r"), handle_signals=False, **FAST)
        try:
            with runmon.task("stage", total=2, kind="stage") as st:
                assert runmon.current_task() is st
                runmon.event("custom", "hello", answer=42)
            with runmon.watch("slow block", 0.05, detail="shard 9"):
                time.sleep(0.2)
        finally:
            runmon.stop()
        kinds = [e["kind"] for e in _events(h.run_dir)]
        assert {"run_start", "custom", "watchdog", "run_end"} <= set(kinds)
        wd = next(e for e in _events(h.run_dir) if e["kind"] == "watchdog")
        assert "slow block" in wd["msg"] and "shard 9" in wd["msg"]
        assert not runmon.active()

    def test_init_worker_writes_own_file(self, tmp_path):
        runmon.stop()
        run_dir = str(tmp_path / "r")
        runmon.init_worker(run_dir)
        try:
            runmon.event("task_failed", "x", "WARNING")
            with runmon.watch("walk", 0.05):
                time.sleep(0.15)
            evs = list(ev.iter_events(runmon.worker_file(run_dir)))
            assert [e["kind"] for e in evs][:1] == ["task_failed"]
            assert any(e["kind"] == "watchdog" for e in evs)
            assert evs[0]["role"] == "worker"
        finally:
            runmon._worker_writer.close()
            runmon._worker_writer = None
            runmon._worker_watcher = None
            ev.set_role("main")


def test_stage_nodes_emit_start_and_end_events(hub):
    with hub.tree.task("analysis", kind="stage"):
        with hub.tree.task("not a stage", kind="shard"):
            pass
    with pytest.raises(RuntimeError):
        with hub.tree.task("search", kind="stage"):
            raise RuntimeError("boom")
    assert _wait(lambda: sum(e["kind"].startswith("stage_") for e in _events(hub.run_dir)) == 4)
    st = [(e["kind"], e["f"]["stage"], e["f"].get("status")) for e in _events(hub.run_dir)
          if e["kind"].startswith("stage_")]
    assert st == [("stage_start", "analysis", None), ("stage_end", "analysis", "done"),
                  ("stage_start", "search", None), ("stage_end", "search", "failed")]
    end = next(e for e in _events(hub.run_dir) if e["kind"] == "stage_end" and e["f"]["stage"] == "search")
    assert end["level"] == "WARNING" and end["f"]["elapsed"] >= 0
    start = next(e for e in _events(hub.run_dir) if e["kind"] == "stage_start")
    assert start["ts"] == pytest.approx(hub.snapshot()["tasks"][0]["started"])
