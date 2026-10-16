"""
Release memory that outlives a trajectory computation.

A walk of a long trajectory produces rationals with 10⁵–10⁶-digit numerators.
When the handler is dropped they should be freed, but sympy's global cache
(``sympy.core.cache``: ``Rational.__new__``, ``Integer.__new__``, … — LRU, up
to 1000 entries per function) keeps the most recent ones alive.  In a worker
that walks long trajectories one after another this grew RSS by ~20 MB per
trajectory (4F3, L2 128) until the caches filled — potentially GBs per worker.

:func:`release_task_memory` clears sympy's cache (and ramanujantools'
``Matrix.walk`` cache) after a task that took at least
``system.SYMPY_CACHE_CLEAR_AFTER_SECONDS``.  Light tasks skip it: clearing costs
~5 ms of cache re-warming per task, and light tasks leave no large numbers.
:func:`releases_task_memory` wraps a worker task function with it.

Second kind of cache, bounded after **every** task (:func:`clear_trajectory_caches`):
``ramanujantools`` methods decorated with ``lru_cache`` whose key is one
trajectory's own matrix (``Matrix._companionization`` & co.).  Each keeps up to
128 results alive for the worker's life; for a Tier-3 ``relation`` task one result
is ~3.7 MB (degree-200 polynomials with hundreds-of-digit coefficients), so a
worker grew 121 -> 607 MB over 150 tasks while a cleared one stayed at ~145 MB
(measured 2026-10-16; clearing sympy's cache or ``Matrix.inverse`` changed
nothing).  What is cleared follows one rule — **only caches no other task can
hit** — and was measured, not assumed: running real tasks with and without
clearing, ``_companionization`` computed 100 results and reused 0 over 100
trajectories, whereas ``CMF._calculate_diagonal_matrix`` (building blocks shared by
trajectories; 50 % hits) and ``Matrix._walk_inner`` are left alone.  Sympy's
cache is deliberately *not* cleared more often: for light tasks it holds nothing
sizeable (clearing after every task changed a worker by 1 MB), and long walks are
already covered by the time rule above.
"""

from __future__ import annotations

import functools
import time


#: ramanujantools ``lru_cache`` methods keyed by one trajectory's own matrix or
#: recurrence — nothing a different trajectory computes can hit them.
#: ``(module, class, method names)``; names missing in the installed version are
#: skipped.  Do **not** add caches whose entries are shared between trajectories
#: (``CMF._calculate_diagonal_matrix``) or that hold per-walk state the next task
#: of the same worker may continue from (``Matrix._walk_inner`` — released by the
#: heavy-task rule below).
_PER_TRAJECTORY_CACHES = (
    ("ramanujantools.matrix", "Matrix",
     ("_companionization", "companion_coboundary_matrix", "simplify", "inverse")),
    ("ramanujantools.linear_recurrence", "LinearRecurrence",
     ("_get_reducer", "_get_reducer_at_precision")),
)

#: Results each per-trajectory cache may keep after a task.  The same trajectory can
#: come back to a worker soon (another constant's row): with a few recent entries
#: kept it still hits (measured: 90 ms of 330 ms saved per such repeat), while the
#: memory stays at a few MB instead of 128 x ~3.7 MB.
TRAJECTORY_CACHE_KEEP = 4


def clear_trajectory_caches(keep: int = TRAJECTORY_CACHE_KEEP) -> int:
    """Bound the ramanujantools caches that only serve the trajectory just finished.

    A cache holding more than *keep* results is cleared entirely; one holding at
    most *keep* is left alone, so a trajectory repeated soon on the same worker
    still hits.  Within one task these caches matter (``relation``, ``companion``
    and ``order`` all companionize the same matrix), so this runs between tasks only.

    :param keep: Results a cache may keep (``0`` = clear every cache that holds any).
    :return: Number of caches cleared.
    """
    import importlib

    cleared = 0
    for module_name, class_name, methods in _PER_TRAJECTORY_CACHES:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        for name in methods:
            cache = getattr(cls, name, None)
            info = getattr(cache, "cache_info", None)
            if info is not None and info().currsize > keep:
                cache.cache_clear()
                cleared += 1
    return cleared


def release_task_memory(elapsed_seconds: float) -> bool:
    """Release the memory a finished trajectory task leaves behind.

    Always bounds the per-trajectory ramanujantools caches
    (:func:`clear_trajectory_caches`); clears sympy's global cache and
    ``Matrix._walk_inner`` only if the task was heavy.

    :param elapsed_seconds: Wall time of the task just finished.
    :return: Whether sympy's cache (the heavy-task caches) was cleared.
    """
    from dreamer.configs import config

    clear_trajectory_caches()        # every task, however light (see module docstring)
    threshold = getattr(config.system, "SYMPY_CACHE_CLEAR_AFTER_SECONDS", 1.0)
    if threshold is None or threshold < 0 or elapsed_seconds < threshold:
        return False
    import sympy.core.cache

    sympy.core.cache.clear_cache()
    try:
        from ramanujantools import Matrix

        Matrix._walk_inner.cache_clear()
    except (ImportError, AttributeError):
        pass
    return True


def releases_task_memory(fn):
    """Decorator: run *fn*, then :func:`release_task_memory` with its duration."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            release_task_memory(time.perf_counter() - start)

    return wrapper
