"""Tests for releasing memory that outlives a trajectory task.

sympy's global cache kept the huge rationals of long walks alive after their
handler was dropped (~20 MB per 4F3 L2-128 trajectory, growing until the LRU
filled).  ``release_task_memory`` clears it after heavy tasks only.
"""

import pytest
import sympy as sp
import sympy.core.cache as sc
from ramanujantools import Matrix, Position
from ramanujantools.cmf import pFq
from sympy.abc import n

from dreamer.configs import config
from dreamer.utils import task_memory
from dreamer.utils.task_memory import (
    TRAJECTORY_CACHE_KEEP,
    clear_trajectory_caches,
    release_task_memory,
    releases_task_memory,
)


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


# ---------------------------------------------------------------------------
# Per-trajectory ramanujantools caches (2026-10-16)
#
# Matrix._companionization is an lru_cache(128) on an instance method; each Tier-3
# `relation` task left ~3.7 MB in it, so a worker grew 121 -> 607 MB over 150 tasks
# although no other trajectory can ever hit those entries.  The release must
# (a) bound exactly the caches that cannot be hit across trajectories,
# (b) leave caches that ARE shared between trajectories alone, and
# (c) never change a computed value.
# ---------------------------------------------------------------------------

def _companionize(k: int):
    """Companionize a distinct 2x2 matrix (one new ``_companionization`` entry)."""
    return Matrix([[0, 1], [k * n + 1, n + k]])._companionization(n)


@pytest.fixture
def empty_caches():
    Matrix._companionization.cache_clear()
    yield
    Matrix._companionization.cache_clear()


def test_caches_within_keep_are_left_alone(empty_caches):
    """A trajectory repeated soon on the same worker must still hit."""
    for k in range(1, TRAJECTORY_CACHE_KEEP + 1):
        _companionize(k)
    assert clear_trajectory_caches() == 0
    assert Matrix._companionization.cache_info().currsize == TRAJECTORY_CACHE_KEEP
    hits = Matrix._companionization.cache_info().hits
    _companionize(1)
    assert Matrix._companionization.cache_info().hits == hits + 1


def test_cache_over_keep_is_cleared(empty_caches):
    for k in range(1, TRAJECTORY_CACHE_KEEP + 2):
        _companionize(k)
    assert Matrix._companionization.cache_info().currsize == TRAJECTORY_CACHE_KEEP + 1
    assert clear_trajectory_caches() >= 1
    assert Matrix._companionization.cache_info().currsize == 0


def test_keep_zero_clears_everything(empty_caches):
    _companionize(1)
    assert clear_trajectory_caches(keep=0) >= 1
    assert Matrix._companionization.cache_info().currsize == 0


def test_light_task_still_bounds_trajectory_caches(empty_caches, monkeypatch):
    """The bound applies to every task — it must not wait for the heavy-task
    threshold (Tier-3 tasks take ~0.3 s, none reached the 1 s threshold)."""
    monkeypatch.setattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", 1.0)
    for k in range(1, 2 * TRAJECTORY_CACHE_KEEP):
        _companionize(k)
    assert not release_task_memory(0.01)                 # sympy cache rule unchanged
    assert Matrix._companionization.cache_info().currsize == 0


def test_results_are_unchanged_by_clearing(empty_caches):
    """Dropping a cache entry only costs a recomputation, never a different value."""
    first = _companionize(3)
    clear_trajectory_caches(keep=0)
    again = _companionize(3)
    assert again is not first                            # really recomputed
    assert again.recurrence == first.recurrence
    assert again.companion == first.companion


def test_caches_shared_between_trajectories_are_not_touched():
    """``CMF._calculate_diagonal_matrix`` entries are reused by other trajectories
    (measured 50 % hits over 100 walk tasks), so the release must leave them."""
    from ramanujantools.cmf import CMF

    cmf = pFq(2, 1, -1)
    syms = list(cmf.matrices.keys())
    ones = Position({s: 1 for s in syms})        # a simple diagonal (all steps <= 1)
    cmf.trajectory_matrix(ones, ones)
    before = CMF._calculate_diagonal_matrix.cache_info().currsize
    assert before > 0, "test is vacuous: no diagonal matrix was cached"
    clear_trajectory_caches(keep=0)
    release_task_memory(0.01)
    assert CMF._calculate_diagonal_matrix.cache_info().currsize == before


def test_policy_list_excludes_shared_caches():
    """Pins the rule: nothing shared between trajectories may enter the list."""
    listed = {(c, m) for _mod, c, methods in task_memory._PER_TRAJECTORY_CACHES for m in methods}
    assert ("Matrix", "_companionization") in listed
    assert ("CMF", "_calculate_diagonal_matrix") not in listed
    assert ("Matrix", "_walk_inner") not in listed        # heavy-task rule handles it


def test_missing_library_caches_are_skipped(monkeypatch):
    """Other ramanujantools versions may lack a class or method: skip, never fail."""
    monkeypatch.setattr(task_memory, "_PER_TRAJECTORY_CACHES", (
        ("no_such_module_xyz", "Anything", ("m",)),
        ("ramanujantools.matrix", "NoSuchClass", ("m",)),
        ("ramanujantools.matrix", "Matrix", ("no_such_method", "free_symbols")),
    ))
    assert clear_trajectory_caches(keep=0) == 0


def test_decorated_task_bounds_trajectory_caches_on_exit(empty_caches):
    """The worker tasks (``_pool_walk``, ``_tier3_task``, ``run_tier1_task``) are
    wrapped by ``releases_task_memory`` (see ``test_worker_tasks_are_wrapped``), so
    the real release — now including the bound — runs when they return, light or not."""
    @releases_task_memory
    def task():
        for k in range(1, 2 * TRAJECTORY_CACHE_KEEP):
            _companionize(k)
        return Matrix._companionization.cache_info().currsize

    assert task() == 2 * TRAJECTORY_CACHE_KEEP - 1          # all held during the task
    assert Matrix._companionization.cache_info().currsize == 0
