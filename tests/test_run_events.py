"""The structured events a run leaves for later analysis (``events.jsonl``).

  - ``environment``: code version, packages, machine, scheduler job and the
    settings, collected once at the start (:mod:`dreamer.utils.run_environment`).
  - ``shard_start`` / ``shard_end`` around every shard × constant, with the
    elapsed time, status (``failed`` when the block raised) and the node's metrics.
  - ``new_best`` (throttled) and ``eval_stats`` (reuse counts, best score,
    look-ahead hit rate) from :class:`EvalStats` / :func:`log_eval_stats`.
  - ``slow_task`` and per-pool task timing from :class:`ResilientPool`.
  - ``flush`` when the walltime grace period flushes the logs.
"""

import json
import os
import time

import pytest

import runmon
from dreamer.configs import config
from dreamer.search.methods.flatland.evaluator import EvalStats, log_eval_stats
from dreamer.utils import monitoring, run_environment
from dreamer.utils.resilient_pool import ResilientPool
from runmon import events as ev


def _events(run_dir, kind=None):
    hub = runmon.hub()
    if hub is not None:
        hub._drain()                              # queued events reach the file every 0.25 s
    evs = list(ev.iter_events(os.path.join(run_dir, "events.jsonl")))
    return [e for e in evs if kind is None or e["kind"] == kind]


@pytest.fixture
def run():
    directory = monitoring.start_run("events-test")
    yield directory
    monitoring.end_run()


def _sleepy(x):
    time.sleep(x)
    return x


class TestEnvironment:
    def test_collect_is_json_ready_and_has_the_essentials(self):
        env = run_environment.collect({"system": {"A": 1, "B": object()}})
        json.dumps(env)
        assert env["machine"]["cpu_count"] and env["machine"]["python"]
        assert isinstance(env["packages"], dict)
        assert env["config"]["system"]["A"] == 1 and isinstance(env["config"]["system"]["B"], str)
        assert "commit" in env["code"]            # the repo under test is a git checkout

    def test_collect_never_raises(self, monkeypatch):
        monkeypatch.setattr(run_environment, "code_version", lambda *a: 1 / 0)
        assert "code" not in run_environment.collect()

    def test_scheduler_variables(self, monkeypatch):
        monkeypatch.setenv("PBS_JOBID", "4797602.zeus")
        monkeypatch.setenv("SLURM_JOB_ID", "77")
        assert run_environment.scheduler() == {"PBS_JOBID": "4797602.zeus", "SLURM_JOB_ID": "77"}

    def test_event_written_at_run_start(self, run):
        env = _events(run, "environment")
        assert len(env) == 1
        f = env[0]["f"]
        assert f["machine"]["host"] and f["config"]["monitor"]["TIMELINE_SECONDS"] == 60.0


class TestShardEvents:
    def test_start_end_with_metrics(self, run):
        with monitoring.shard("s1", "log(2)", label="micro-climb") as node:
            node.metric(best=0.5)
        start, = _events(run, "shard_start")
        end, = _events(run, "shard_end")
        assert start["f"]["shard"] == "s1" and start["f"]["label"] == "micro-climb"
        assert end["f"]["status"] == "done" and end["f"]["elapsed"] >= 0
        assert end["f"]["metrics"] == {"best": 0.5} and end["level"] == "INFO"

    def test_failure_is_recorded_and_reraised(self, run):
        with pytest.raises(ValueError):
            with monitoring.shard("s2"):
                raise ValueError("boom")
        end, = _events(run, "shard_end")
        assert end["f"]["status"] == "failed" and end["level"] == "WARNING"
        assert end["f"]["constant"] is None

    def test_no_op_when_monitoring_is_off(self):
        with monitoring.shard("s3", "log(2)") as node:
            node.metric(x=1)                       # a null node: accepts everything


class TestEvalStatsEvents:
    def test_new_best_is_throttled_but_best_is_exact(self, run):
        stats = EvalStats({}, "log(2)")
        for score in (0.1, 0.1004, 0.1008, float("-inf"), float("nan"), 0.05, 0.2):
            stats.note_score(score)
        assert stats.best == 0.2
        scores = [e["f"]["score"] for e in _events(run, "new_best")]
        assert scores == [0.1, 0.2]               # tiny gains within 10 s are not announced
        assert _events(run, "new_best")[0]["f"]["constant"] == "log(2)"

    def test_eval_stats_event_with_lookahead_hit_rate(self, run):
        stats = EvalStats({}, "log(2)")
        stats.walked("t1", cached_walk=False, ok=True)
        stats.note_score(0.3)
        log_eval_stats("Simulated Annealing", "shard-x", "log(2)", stats,
                       lookahead={"submitted": 10, "taken": 4, "dropped": 6})
        f = _events(run, "eval_stats")[0]["f"]
        assert f["new"] == 1 and f["best"] == 0.3 and f["evaluations"] == 1
        assert f["lookahead"]["hit_rate"] == pytest.approx(0.4)

    def test_eval_stats_without_lookahead_or_score(self, run):
        log_eval_stats("GA", "shard-y", "pi", EvalStats({}, "pi"))
        f = _events(run, "eval_stats")[0]["f"]
        assert f["best"] is None and "lookahead" not in f


class TestPoolTiming:
    def test_slow_task_event_and_timing_in_monitor_info(self, run, monkeypatch):
        monkeypatch.setattr(config.monitor, "SLOW_TASK_SECONDS", 0.2)
        with ResilientPool(2, name="timing pool") as pool:
            list(pool.imap(_sleepy, [0.3, 0.3, 0.3]))
            info = pool.monitor_info()
        assert info["task_seconds"] >= 0.8 and info["longest_s"] >= 0.3
        slow = _events(run, "slow_task")
        assert len(slow) == 1                      # rate limited to one a minute per pool
        assert slow[0]["f"]["seconds"] >= 0.3 and slow[0]["f"]["fn"] == "_sleepy"
        closed = _events(run, "pool_closed")[-1]["f"]
        assert closed["slow_tasks"] == 3 and closed["task_seconds"] >= 0.8

    def test_no_slow_events_when_off(self, run, monkeypatch):
        monkeypatch.setattr(config.monitor, "SLOW_TASK_SECONDS", 0)
        with ResilientPool(1, name="quiet pool") as pool:
            list(pool.imap(_sleepy, [0.05]))
        assert not _events(run, "slow_task")


def test_grace_flush_leaves_an_event(run):
    monitoring._flush_logs()
    assert _events(run, "flush")[0]["level"] == "WARNING"
