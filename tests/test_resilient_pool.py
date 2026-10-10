"""Tests for the dead-worker-safe process pool.

Workers are killed with ``SIGKILL`` from inside a task — exactly what the
kernel's out-of-memory killer does — which made ``multiprocessing.Pool`` wait
forever (reproduced 2026-10-10).  These tests pin that the resilient pool:

  - recovers from a one-off death (task re-run, all results correct, order kept)
    and halves its worker count;
  - isolates a task that always kills its worker and returns the ``on_death``
    result for it while every other task completes — no hang;
  - raises ``WorkerDeathError`` for such a task when no ``on_death`` is given;
  - keeps ``imap`` in-flight work bounded and ``apply_async`` / ``ready`` working;
  - plugs into ``StagePool`` (poisoned trajectory → ``TaskError``) and the eval
    pool (dead walk → ``WalkError``).

Each test runs under a hard timeout: a regression shows up as a failure, not a
hung test session.
"""

import os
import signal
import tempfile

import pytest

from dreamer.utils.resilient_pool import ResilientPool, WorkerDeathError, WorkerInitError

pytestmark = pytest.mark.timeout(240)


# ---------------------------------------------------------------------------
# Module-level task functions (picklable for spawned workers)
# ---------------------------------------------------------------------------

def _square(x):
    return x * x


def _die_once(args):
    """Kill the worker the first time it runs for ``x == victim``; succeed after."""
    x, victim, marker = args
    if x == victim and not os.path.exists(marker):
        open(marker, "w").close()
        os.kill(os.getpid(), signal.SIGKILL)
    return x * 10


def _poison(x):
    """Always kill the worker for ``x == 3``."""
    if x == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return x + 100


def _poison_tuple(args):
    return _poison(args[0])


# ---------------------------------------------------------------------------
# Unit behaviour
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_one_off_death_is_recovered_in_order(self, tmp_path):
        marker = str(tmp_path / "died")
        with ResilientPool(4, name="t") as pool:
            out = list(pool.imap(_die_once, [(x, 5, marker) for x in range(12)]))
        assert out == [x * 10 for x in range(12)]
        assert os.path.exists(marker)                 # the death really happened
        assert pool.deaths == 1 and pool._processes == 2   # halved

    def test_poison_task_is_isolated_and_reported(self):
        with ResilientPool(3, on_death=lambda fn, args: ("dead", args[0]), name="t") as pool:
            out = list(pool.imap(_poison, range(8)))
        assert out[3] == ("dead", 3)
        assert [o for i, o in enumerate(out) if i != 3] == [x + 100 for x in range(8) if x != 3]
        assert pool.deaths >= 2

    def test_poison_without_on_death_raises(self):
        with pytest.raises(WorkerDeathError):
            with ResilientPool(2, name="t") as pool:
                list(pool.imap(_poison, range(5)))

    def test_unordered_and_apply_async(self):
        with ResilientPool(3, on_death=lambda fn, args: -1, name="t") as pool:
            assert sorted(pool.imap_unordered(_square, range(10))) == [x * x for x in range(10)]
            handles = [pool.apply_async(_poison, (x,)) for x in range(6)]
            results = [h.get() for h in handles]
            assert results[3] == -1 and results[0] == 100
            assert all(h.ready() for h in handles)

    def test_imap_window_is_bounded(self):
        with ResilientPool(2, window=2, name="t") as pool:
            consumed = []

            def gen():
                for i in range(50):
                    consumed.append(i)
                    yield i

            it = pool.imap(_square, gen())
            assert next(it) == 0
            assert len(consumed) <= 2 * 2 + 1          # not the whole input
            assert list(it) == [x * x for x in range(1, 50)]

    def test_empty_and_single_worker(self):
        with ResilientPool(1, name="t") as pool:
            assert list(pool.imap(_square, [])) == []
            assert list(pool.imap(_square, [3])) == [9]


# ---------------------------------------------------------------------------
# Integration with the pipeline pools
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_stage_pool_reports_dead_trajectory_as_task_error(self, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.utils.stage_pool import StagePool, TaskError, Tier1Task

        monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", 2)
        monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
        tasks = [Tier1Task(key=f"k{i}", shard_idx=0, trajectory=i, start=None, constants=(),
                           cmf_id="", shard_id="", shard_encoding_str="") for i in range(5)]
        with StagePool([]) as pool:
            out = list(pool.imap(_stage_poison, tasks))
        assert isinstance(out[3], TaskError) and out[3].key == "k3"
        assert "out of memory" in out[3].message
        assert [o for i, o in enumerate(out) if i != 3] == [0, 1, 2, 4]

    def test_eval_pool_death_result_is_a_failed_walk(self):
        from dreamer.search.methods.flatland.parallel_eval import WalkError, _walk_death_result

        idx, res = _walk_death_result(None, ((7, "direction", "const", "c", "s", "e"),))
        assert idx == 7 and isinstance(res, WalkError)

    def test_previous_hang_reproduction_now_finishes(self, monkeypatch):
        """The exact scenario that hung ``multiprocessing.Pool`` (a SIGKILLed
        worker under StagePool.imap) completes."""
        from dreamer.configs.system import sys_config
        from dreamer.utils.stage_pool import StagePool, TaskError

        monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", 2)
        monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
        with StagePool([]) as pool:
            out = list(pool.imap(_poison_plain, range(6)))
        assert isinstance(out[3], TaskError)


def _stage_poison(task):
    if task.trajectory == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return task.trajectory


def _poison_plain(x):
    if x == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


# ---------------------------------------------------------------------------
# Start method, initializer failures, peak memory, slow start
# ---------------------------------------------------------------------------

def _bad_init():
    raise RuntimeError("broken initializer")


def _allocate(mb):
    """Touch ``mb`` MB, release it, return the size (peak RSS rises, current falls)."""
    buf = bytearray(mb * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    del buf
    return mb


class TestStartMethod:
    def test_auto_is_forkserver_on_linux_spawn_on_macos(self, monkeypatch):
        import sys
        from dreamer.configs.system import sys_config
        from dreamer.utils.mp_context import start_method

        monkeypatch.setattr(sys_config, "POOL_START_METHOD", "auto")
        expected = "forkserver" if sys.platform.startswith("linux") else "spawn"
        assert start_method() == expected

    def test_explicit_override_and_invalid(self, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.utils.mp_context import start_method

        monkeypatch.setattr(sys_config, "POOL_START_METHOD", "spawn")
        assert start_method() == "spawn"
        monkeypatch.setattr(sys_config, "POOL_START_METHOD", "nonsense")
        with pytest.raises(ValueError):
            start_method()

    @pytest.mark.parametrize("method", ["spawn", "forkserver"])
    def test_pool_runs_under_each_dynamic_method(self, monkeypatch, method):
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "POOL_START_METHOD", method)
        with ResilientPool(2, name="t") as pool:
            assert pool._executor._mp_context.get_start_method() == method
            assert list(pool.imap(_square, range(6))) == [x * x for x in range(6)]


class TestInitializerFailure:
    def test_initializer_failure_is_not_reported_as_oom(self):
        with pytest.raises(WorkerInitError, match="broken initializer"):
            with ResilientPool(2, initializer=_bad_init, name="t") as pool:
                list(pool.imap(_square, range(4)))
        assert pool.deaths == 0                      # no OOM-style retries


class TestPeakMemory:
    def test_peak_rss_is_reported_by_workers(self):
        with ResilientPool(1, name="t") as pool:
            assert list(pool.imap(_allocate, [200])) == [200]
        assert pool.peak_worker_mb >= 200            # the released 200 MB still counts

    def test_slow_start_caps_initial_workers(self, monkeypatch):
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "MEMORY_AWARE_WORKERS", True)
        monkeypatch.setattr(sys_config, "POOL_SLOW_START", 2)
        monkeypatch.setattr(sys_config, "WORKER_MEMORY_MB", 1)   # memory is no limit
        with ResilientPool(6, name="t") as pool:
            assert pool._processes == 2 and pool._requested == 6
            assert list(pool.imap(_square, range(4))) == [0, 1, 4, 9]

    def test_growth_counts_peak_not_current_rss(self, monkeypatch):
        """Workers that peaked at ~1 GB and fell back must not let the pool add
        workers sized by their (small) current RSS."""
        from dreamer.configs.system import sys_config
        from dreamer.utils import memory_budget

        monkeypatch.setattr(sys_config, "MEMORY_AWARE_WORKERS", True)
        monkeypatch.setattr(sys_config, "POOL_SLOW_START", 2)
        monkeypatch.setattr(sys_config, "WORKER_MEMORY_MB", 1)
        monkeypatch.setattr(sys_config, "MEMORY_RESERVE_MB", 0)
        with ResilientPool(8, name="t") as pool:
            pool._GROW_INTERVAL = float("inf")               # no automatic growth while warming
            list(pool.imap(_square, range(8)))       # warm both workers
            assert pool._processes == 2
            pool._GROW_INTERVAL = 0.0
            pool.peak_worker_mb = 1000.0             # as if a task peaked at 1 GB
            pool._completed_since_resize = 100
            # 2 live workers need ~2.5 GB to climb back; 3 GB leaves room for 0 more.
            monkeypatch.setattr(memory_budget, "available_memory_mb", lambda: 3000.0)
            pool._check_memory()
            assert pool._processes == 2 and pool.grows == 0
            monkeypatch.setattr(memory_budget, "available_memory_mb", lambda: 6000.0)
            pool._check_memory()
            assert pool._processes > 2 and pool.grows == 1
