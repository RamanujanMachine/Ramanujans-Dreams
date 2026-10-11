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
"""

from __future__ import annotations

import functools
import time


def release_task_memory(elapsed_seconds: float) -> bool:
    """Clear the global caches if the finished task was heavy.

    :param elapsed_seconds: Wall time of the task just finished.
    :return: Whether the caches were cleared.
    """
    from dreamer.configs import config

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
