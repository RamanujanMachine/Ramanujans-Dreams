"""Tests for releasing memory that outlives a trajectory task.

sympy's global cache kept the huge rationals of long walks alive after their
handler was dropped (~20 MB per 4F3 L2-128 trajectory, growing until the LRU
filled).  ``release_task_memory`` clears it after heavy tasks only.
"""

import sympy as sp
import sympy.core.cache as sc

from dreamer.configs import config
from dreamer.utils.task_memory import release_task_memory, releases_task_memory


def _big_rational_cached():
    """Create a large Rational through sympy's cached constructor."""
    sp.Rational(3 ** 20000 + 1, 7 ** 15000)
    return _rational_cache_size() > 0


def _rational_cache_size():
    return sum(f.cache_info().currsize for f in sc.CACHE if "Rational" in getattr(f, "__qualname__", ""))


def test_heavy_task_clears_caches(monkeypatch):
    monkeypatch.setattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", 1.0)
    assert _big_rational_cached()
    assert release_task_memory(2.0)
    assert _rational_cache_size() == 0


def test_light_task_keeps_caches(monkeypatch):
    monkeypatch.setattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", 1.0)
    _big_rational_cached()
    assert not release_task_memory(0.01)
    assert _rational_cache_size() > 0


def test_disabled_and_always(monkeypatch):
    monkeypatch.setattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", None)
    assert not release_task_memory(100.0)
    monkeypatch.setattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", 0)
    assert release_task_memory(0.0)


def test_decorator_runs_release_even_on_error(monkeypatch):
    calls = []
    monkeypatch.setattr("dreamer.utils.task_memory.release_task_memory", lambda s: calls.append(s))

    @releases_task_memory
    def task(x):
        if x < 0:
            raise ValueError
        return x * 2

    assert task(3) == 6
    try:
        task(-1)
    except ValueError:
        pass
    assert len(calls) == 2


def test_worker_tasks_are_wrapped():
    from dreamer.post_process.tier3_post_process_mod import _tier3_task
    from dreamer.search.methods.flatland.parallel_eval_worker import _pool_walk
    from dreamer.utils.stage_pool import run_tier1_task

    for fn in (run_tier1_task, _pool_walk, _tier3_task):
        assert hasattr(fn, "__wrapped__"), fn.__name__
