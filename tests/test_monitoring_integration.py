"""Ramanujan's Dreams ↔ runmon wiring (:mod:`dreamer.utils.monitoring`).

  - ``monitor.ENABLED = False``: no run directory, no thread, nothing changes
    (the watchdog falls back to its own thread).
  - A monitored ``System.run``: one run directory (+ ``latest``), ``run_start`` /
    ``run_end``, a stage node per pipeline stage, the run's allocation as the
    utilization baseline, log lines from ``EVENT_LOG_LEVEL`` up, structured
    ``Logger.event`` fields, and the reason a run ended (error → ``error: …``).
  - Pools: ``pool_created`` / ``pool_closed`` events, ``monitor_info`` in the
    snapshot, and **worker** warnings / watchdogs / task failures merged from
    the workers' own event files.
  - Progress: ``SmartTQDM`` bars nest under stage → shard nodes (global count +
    per-shard level); simulated annealing reports an estimate with its
    temperature (real 2F1 log(2) shard).
  - Monitor alerts (walltime) are echoed to the project log exactly once.
"""

import json
import logging
import os
import time
from typing import cast

import pytest

import runmon
from dreamer.configs import config
from dreamer.system import system as system_mod
from dreamer.system.system import System
from dreamer.utils import monitoring
from dreamer.utils.logger import Logger, sys_logger
from dreamer.utils.resilient_pool import ResilientPool
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.ui.tqdm_config import SmartTQDM
from runmon import events as ev
from runmon.progress import TaskNode


def _events(run_dir):
    return list(ev.iter_events(os.path.join(run_dir, "events.jsonl")))


def _wait(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


# --- worker task functions (module level: picklable) --------------------------

def _warn_in_worker(x):
    Logger.event("custom_worker_event", f"worker saw {x}", Logger.Levels.warning, x=x)
    return x * 2


def _slow_in_worker(x):
    with Logger.watchdog("slow worker step", 0.2, detail=f"item {x}"):
        time.sleep(0.6)
    return x


class TestDisabled:
    def test_nothing_starts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config.monitor, "ENABLED", False)
        assert monitoring.start_run("r1") is None
        assert not monitoring.active() and not runmon.active()
        assert not os.path.exists(config.monitor.RUN_DIR)
        with monitoring.stage("search") as node:
            node.advance()
        assert monitoring.worker_run_dir() is None

    def test_watchdog_falls_back_to_thread(self, monkeypatch):
        monkeypatch.setattr(config.monitor, "ENABLED", False)
        records = []
        h = logging.Handler()
        h.emit = records.append
        sys_logger.addHandler(h)
        try:
            with Logger.watchdog("legacy path", 0.1, repeat=False):
                time.sleep(0.3)
        finally:
            sys_logger.removeHandler(h)
        assert any("legacy path" in r.getMessage() for r in records)


class TestRunLifecycle:
    def test_start_events_levels_and_end(self, monkeypatch):
        monkeypatch.setattr(config.monitor, "EVENT_LOG_LEVEL", "INFO")
        d = monitoring.start_run("abc123", {"constants": ["log(2)"]})
        assert d is not None and os.path.basename(d).endswith("-abc123")
        latest = os.path.join(config.monitor.RUN_DIR, "latest")
        assert os.path.realpath(latest) == os.path.realpath(d)
        Logger("an info line", Logger.Levels.info).log()
        sys_logger.debug("a debug line")
        Logger.event("pool_shrink", "shrinking", Logger.Levels.warning, workers=2)
        monitoring.end_run("exit")
        evs = _events(d)
        kinds = [e["kind"] for e in evs]
        assert kinds[0] == "run_start" and kinds[-1] == "run_end"
        assert any(e["kind"] == "log" and e["msg"] == "an info line" for e in evs)
        assert not any(e["msg"] == "a debug line" for e in evs)
        shrink = next(e for e in evs if e["kind"] == "pool_shrink")
        assert shrink["f"]["workers"] == 2 and shrink["level"] == "WARNING"
        assert evs[0]["f"]["meta"]["constants"] == ["log(2)"]
        assert "cores=" in evs[0]["f"]["meta"]["allocation"]

    def test_allocation_is_the_baseline(self, monkeypatch):
        monkeypatch.setattr(config.system, "TOTAL_CORES", 3)
        monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", 2048)
        d = monitoring.start_run("alloc")
        try:
            assert _wait(lambda: runmon.hub().last_sample is not None)
            alloc = runmon.hub().last_sample["allocation"]
            assert alloc["cores"] == 3 and alloc["memory_mb"] == 2048
        finally:
            monitoring.end_run()

    def test_monitor_alerts_echo_once(self, monkeypatch):
        monkeypatch.setenv("RUNMON_WALLTIME_END", str(time.time() + 100))
        records = []
        h = logging.Handler()
        h.emit = records.append
        sys_logger.addHandler(h)
        try:
            d = monitoring.start_run("wall")
            assert _wait(lambda: any(e["kind"] == "walltime_grace" for e in _events(d)))
            monitoring.end_run()
        finally:
            sys_logger.removeHandler(h)
        echoed = [r.getMessage() for r in records if r.getMessage().startswith("[monitor]")]
        assert any("grace period" in m for m in echoed)
        assert sum("grace period" in e["msg"] for e in _events(d)) == 1   # not re-published


class _PipelineSearcher:
    """Stand-in search module exercising pools, bars and shard nodes."""

    def __init__(self, priorities, *args, **kwargs):
        self.priorities = priorities

    def execute(self):
        for shard_no in SmartTQDM(range(2), desc="Searching in shards: "):
            with monitoring.shard(f"s{shard_no}", "log(2)"):
                with ResilientPool(2, name="eval pool") as pool:
                    assert list(pool.imap(_warn_in_worker, range(4))) == [0, 2, 4, 6]
                    assert list(pool.imap(_slow_in_worker, [shard_no])) == [shard_no]
                for _ in SmartTQDM(range(3), desc="Ascending ... "):
                    pass
        return None


@pytest.mark.timeout(300)
def test_monitored_system_run(monkeypatch, tmp_path):
    from dreamer.configs.logging import logging_config
    monkeypatch.setattr(logging_config, "GENERATE_LOGS", False)
    searchables_dir = tmp_path / "searchables"
    searchables_dir.mkdir()
    monkeypatch.setattr(system_mod.sys_config, "PATH_TO_SEARCHABLES", str(searchables_dir))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", None)
    system = System(function_sources=[], extractor=None, analyzers=[],
                    searcher=cast(type[SearcherModScheme], _PipelineSearcher))
    monkeypatch.setattr(system, "_System__validate_constants", lambda _c: [])
    monkeypatch.setattr(system, "_System__loading_stage", lambda _c: ({}, {}))
    monkeypatch.setattr(system, "_System__analysis_stage", lambda _d: {"log(2)": ["shard"]})
    monkeypatch.setattr(system, "_System__search_stage",
                        lambda pr: _PipelineSearcher(pr).execute())

    system.run(constants=[])

    run_dir = os.path.realpath(os.path.join(config.monitor.RUN_DIR, "latest"))
    evs = _events(run_dir)
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert evs[-1]["f"]["reason"] == "exit"
    assert evs[0]["f"]["meta"]["modules"]["searcher"] == "_PipelineSearcher"
    # Pools and worker-side events (merged from the workers' own files).
    assert kinds.count("pool_created") == 2 and kinds.count("pool_closed") == 2
    worker_evs = [e for e in evs if e["kind"] == "custom_worker_event"]
    assert len(worker_evs) == 8 and all(e["role"] == "worker" for e in worker_evs)
    assert {e["f"]["x"] for e in worker_evs} == {0, 1, 2, 3}
    wd = [e for e in evs if e["kind"] == "watchdog"]
    assert wd and all(e["role"] == "worker" for e in wd) and "slow worker step" in wd[0]["msg"]
    assert any(f.startswith("events.w") for f in os.listdir(run_dir))
    # Final progress tree: stages → bar → shard → pool-free inner bar.
    snap = json.load(open(os.path.join(run_dir, "status.json")))
    names = [t["name"] for t in snap["tasks"]]
    assert names == ["loading", "analysis", "search"]
    assert all(t["status"] == "done" for t in snap["tasks"])
    bar = snap["tasks"][-1]["children"][0]
    assert bar["name"] == "Searching in shards" and bar["done"] == 2 and bar["total"] == 2
    shard_node = bar["children"][-1]
    assert shard_node["name"] == "shard s1 / log(2)" and shard_node["status"] == "done"
    assert shard_node["children"][-1]["name"] == "Ascending ..."
    assert not monitoring.active()


def test_run_end_reason_on_error(monkeypatch):
    with pytest.raises(ValueError):
        with monitoring.monitored_run("boom") as d:
            raise ValueError("bad shard")
    end = _events(d)[-1]
    assert end["kind"] == "run_end" and end["f"]["reason"].startswith("error: ValueError: bad shard")
    assert end["level"] == "WARNING"


def test_pool_monitor_info_in_snapshot():
    monitoring.start_run("pools")
    try:
        with ResilientPool(2, name="probe pool") as pool:
            list(pool.imap(_warn_in_worker, range(2)))
            info = next(p for p in runmon.hub().snapshot()["pools"] if p["name"] == "probe pool")
            assert info["completed"] == 2 and info["workers"] >= 1 and info["pids"]
        assert not any(p["name"] == "probe pool" for p in runmon.hub().snapshot()["pools"])
    finally:
        monitoring.end_run()


@pytest.mark.timeout(600)
def test_simulated_annealing_reports_estimate(monkeypatch):
    """Real SA on the 2F1 log(2) shard: the shard node carries an estimate + T."""
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq
    from dreamer.search.methods.annealing.annealing_scan import SimulatedAnnealingSearch
    from dreamer.search.methods.flatland.geometry import FlatlandGeometry
    from dreamer import log
    from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids

    LOG2 = log(2)
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    monkeypatch.setattr(config.extraction, "STRATEGY", "heuristic")
    monkeypatch.setattr(config.extraction, "IGNORE_DUPLICATE_SEARCHABLES", True)
    monkeypatch.setattr(config.extraction, "LOAD_SHARD_CACHE", False)
    monkeypatch.setattr(config.system, "EXPORT_CMFS", None)
    monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 6)
    monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 12)
    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    shards = ShardExtractor([LOG2], pFq(LOG2, 2, 1, -1).to_cmf()).extract()
    shard = next(s for s in shards
                 if tuple(int(v) for v in s.get_interior_point().values()) == (1, 1, 3))
    cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
    seen_estimates = []

    monitoring.start_run("sa")
    try:
        with monitoring.stage("search"):
            with monitoring.shard(shard_id, LOG2) as node:
                real_estimate = TaskNode.estimate

                def spy(self, fraction, **metrics):
                    if self is node:
                        seen_estimates.append((fraction, metrics))
                    real_estimate(self, fraction, **metrics)

                monkeypatch.setattr(TaskNode, "estimate", spy)
                SimulatedAnnealingSearch(shard, LOG2, use_LIReC=True).run(
                    constant=LOG2, cmf_id=cmf_id, shard_id=shard_id, shard_encoding_str=enc,
                    sink=lambda item: None, seen_trajectories={}, handler_cache={},
                    geom=FlatlandGeometry(shard), start=shard.get_interior_point(), pool=None,
                )
    finally:
        monitoring.end_run()
    assert seen_estimates, "SA never reported progress"
    fractions = [f for f, _ in seen_estimates]
    assert all(0.0 <= f <= 1.0 for f in fractions) and fractions == sorted(fractions)
    last = seen_estimates[-1][1]
    assert {"T", "best", "accepted", "steps", "clock"} <= set(last)
    assert last["clock"] in ("accepted", "steps")
