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

from dreamer.utils.resilient_pool import ResilientPool, WorkerDeathError

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
