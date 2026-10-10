"""
Worker-side entry points of the optimisers' eval pools (kept light on purpose).

A spawned eval worker unpickles references to these functions, so this module —
not :mod:`parallel_eval`, which imports the flatland geometry and through it the
samplers (numba, scipy.optimize) — is what a worker imports: it only needs the
trajectory-attributes machinery to walk.
"""

from collections import namedtuple

from dreamer.configs import config

#: Returned by :func:`_pool_walk` when the walk raises, so a single bad
#: trajectory degrades to δ = −∞ (matching the serial evaluator's try/except)
#: instead of propagating through ``pool.map`` and aborting the whole batch.
WalkError = namedtuple("WalkError", ["message"])


# Per-worker context, populated once by the pool initializer.
_POOL_STATE: dict = {}


def _pool_init(config_overrides: dict, shard, start) -> None:
    """Pool initializer: re-apply config and stash the per-shard walk context.

    :param config_overrides: ``config.export_configurations()`` from the parent.
    :param shard: The shard being searched (constant-independent).
    :param start: Interior start :class:`Position` (constant-independent).
    """
    config.configure(**config_overrides)
    # Rebind this worker's per-shard (p, q) cache onto the shared append-log (if
    # one was attached by the deep-search module before the pool was built), so
    # relations identified by any worker are reused by all of them instead of
    # each worker re-deriving them via LIReC.  ``shared_pq_log`` is a Manager
    # list proxy that survives pickling into the worker and reconnects to the
    # parent's manager process; absent it, the cache stays process-local.
    from dreamer.utils.storage.frequency_list import FrequencyList
    shard.cache = FrequencyList(
        shard.cache.max_size,
        shared_log=getattr(shard, "shared_pq_log", None),
    )
    _POOL_STATE["shard"] = shard
    _POOL_STATE["start"] = start


def _pool_walk(args):
    """Worker task: perform one Case-C walk and return the sink tuple.

    :param args: ``(idx, direction, constant, cmf_id, shard_id, shard_encoding_str)``
        where ``idx`` is the caller's group index (echoed back so the parent can
        reassemble results dispatched out of order under ``imap_unordered``).
    :return: ``(idx, (trajectory_matrix, value_sympy, dto))`` on success — the inner
        tuple is identical to what the serial Case-C path passes to the sink — or
        ``(idx, WalkError)`` if the walk raised (so the main process can map it to
        δ = −∞ without the exception aborting the whole batch).
    """
    from dreamer.utils.storage.trajectory_attributes import (
        TrajectoryAttributesHandler,
        build_trajectory_dtos,
    )

    idx, direction, constant, cmf_id, shard_id, shard_encoding_str = args
    shard = _POOL_STATE["shard"]
    start = _POOL_STATE["start"]
    try:
        handler = TrajectoryAttributesHandler.from_cmf(
            shard.cmf, direction, start, constant=None, searchable=shard
        )
        dtos = build_trajectory_dtos(
            handler,
            cmf_id=cmf_id,
            shard_id=shard_id,
            cmf_name=shard.cmf_name,
            shard_encoding_str=shard_encoding_str,
            start=start,
            direction=direction,
            constants=[constant],
            # Tier-2 in the same scope as the walk (no second walk / LIReC); the
            # symbolic matrix is no longer shipped back (nothing consumes it).
            extra_metrics=tuple(config.search.TIER2_ATTRIBUTES),
        )
        if not dtos:
            return idx, WalkError("no DTO produced")
        return idx, (None, constant.value_sympy, dtos[0])
    except Exception as exc:  # noqa: BLE001 — mirror serial evaluator resilience
        return idx, WalkError(str(exc))




def _walk_death_result(fn, args):
    """Result of a walk whose worker died even when run alone: a failed walk
    (δ = −∞), exactly like a walk that raises."""
    payload = args[0] if args else (None,)
    return payload[0], WalkError("worker process died during the walk (likely out of memory)")
