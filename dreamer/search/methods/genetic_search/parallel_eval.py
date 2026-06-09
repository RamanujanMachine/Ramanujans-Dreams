"""
Back-compat shim: the process-pool δ-evaluation helpers now live in the shared
:mod:`dreamer.search.methods.flatland.parallel_eval` so Genetic / Simulated
Annealing / Gradient Ascent can all reuse one implementation.  This module
re-exports them for any existing import site.
"""

from dreamer.search.methods.flatland.parallel_eval import (  # noqa: F401
    WalkError,
    _pool_init,
    _pool_walk,
    evaluate_batch,
    make_eval_pool,
    make_shared_eval_pool,
    resolve_eval_workers,
)

__all__ = [
    "WalkError",
    "_pool_init",
    "_pool_walk",
    "evaluate_batch",
    "make_eval_pool",
    "make_shared_eval_pool",
    "resolve_eval_workers",
]
