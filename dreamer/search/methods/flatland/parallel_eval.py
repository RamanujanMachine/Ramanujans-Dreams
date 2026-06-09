"""
Shared process-based batch δ-evaluation for the search methods.

The Genetic / Simulated-Annealing / Gradient-Ascent methods all repeatedly need
to evaluate a *batch* of flatland directions (GA population children, SA
neighbours, GD forward-difference probes).  The per-direction cost is the
symbolic walk (``TrajectoryAttributesHandler.from_cmf`` + δ / LIReC) — heavy,
GIL-holding work that a thread pool barely speeds up.  A feasibility benchmark
(``examples/benchmark_ga_parallel_eval.py``) showed the walk is ~380 ms while
pickling the result back is ~0.3% of that, so a **persistent per-shard process
pool** is a large, clean win shared by all three methods.

Design
------
* The pool is created **once per shard** (the shard + start are constant-
  independent, so they are handed to the workers a single time via the
  ``initializer`` — copy-on-write / one pickle, not per task).
* Each task sends only the small per-walk payload and returns the exact
  ``(trajectory_matrix, value_sympy, dto)`` tuple the serial Case-C path feeds
  to the ``sink``.  The main process keeps sole ownership of the
  ``seen_trajectories`` / ``handler_cache`` dicts and the file sink, so the
  cross-run dedup semantics are unchanged.

:func:`evaluate_batch` is the single orchestration entry point: it resolves the
cheap cases in the main process (invalid → −∞; Case A — δ already cached; Case B
— handler cached this run → serial recompute, no walk) and dispatches only
genuinely-new Case-C walks, de-duplicating directions that share a primitive
ray.  ``pool=None`` evaluates serially in-process.
"""

import multiprocessing as mp
from collections import namedtuple
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

from dreamer.configs import config
from dreamer.search.methods.flatland.evaluator import (
    evaluate_in_flatland,
    flatland_trajectory_key,
)
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import search_worker_budget
from dreamer.utils.storage.attribute_registry import attribute_name

search_config = config.search


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

    :param args: ``(direction, constant, cmf_id, shard_id, shard_encoding_str)``.
    :return: ``(trajectory_matrix, value_sympy, dto)`` on success — identical to
        what the serial Case-C path passes to the sink — or a :class:`WalkError`
        if the walk raised (so the main process can map it to δ = −∞ without the
        exception aborting the whole batch).
    """
    from dreamer.utils.storage.trajectory_attributes import (
        TrajectoryAttributesHandler,
        build_trajectory_dto,
    )

    direction, constant, cmf_id, shard_id, shard_encoding_str = args
    shard = _POOL_STATE["shard"]
    start = _POOL_STATE["start"]
    try:
        handler = TrajectoryAttributesHandler.from_cmf(
            shard.cmf, direction, start, constant=None, searchable=shard
        )
        dto = build_trajectory_dto(
            handler,
            cmf_id=cmf_id,
            shard_id=shard_id,
            cmf_name=shard.cmf_name,
            shard_encoding_str=shard_encoding_str,
            start=start,
            direction=direction,
            constants=[constant],
        )
        return (handler.trajectory_matrix(), constant.value_sympy, dto)
    except Exception as exc:  # noqa: BLE001 — mirror serial evaluator resilience
        return WalkError(str(exc))


def resolve_eval_workers(cap: Optional[int]) -> int:
    """Resolve a ``*_NUM_EVAL_WORKERS`` knob against the global core budget.

    The per-method knobs are interpreted as **optional caps**: ``None`` or
    ``<= 0`` means "use the full search budget"
    (:func:`dreamer.utils.multi_processing.search_worker_budget`), while a
    positive value caps the worker count at ``min(cap, budget)``.  This is what
    lets the eval pools scale to a large machine without per-method tuning while
    still honouring the cores reserved for the Tier-2 queue + sink.

    :param cap: The configured per-method worker cap (knob value).
    :return: Effective worker count (>= 1).
    """
    budget = search_worker_budget()
    if cap is None or cap <= 0:
        return budget
    return max(1, min(cap, budget))


def make_eval_pool(shard, start, workers: int) -> Optional[Pool]:
    """Create a persistent per-shard evaluation pool, or ``None`` if disabled.

    :param shard: The shard being searched (handed to workers once).
    :param start: Interior start :class:`Position`.
    :param workers: Per-method worker cap (see :func:`resolve_eval_workers`);
        the effective count is capped by the global core budget.  A resolved
        count ``<= 1`` returns ``None`` (serial in-process evaluation).
    :return: A ready :class:`multiprocessing.Pool`, or ``None``.
    """
    effective = resolve_eval_workers(workers)
    if effective <= 1:
        return None
    return Pool(
        processes=effective,
        initializer=_pool_init,
        initargs=(config.export_configurations(), shard, start),
    )


def make_shared_eval_pool(shard, start, workers: int):
    """Per-shard eval pool with a shared ``(p, q)`` cache log attached.

    Wraps :func:`make_eval_pool`.  When a real (multi-worker) pool will be
    created, a :class:`multiprocessing.Manager` list is attached to
    ``shard.shared_pq_log`` **before** the pool is built, so every worker's
    :class:`~dreamer.utils.storage.frequency_list.FrequencyList` (rebound in
    :func:`_pool_init`) publishes/consumes identified relations through it —
    eliminating redundant LIReC re-identification across the shard's workers.

    The manager owns a server process and must be shut down by the caller
    (alongside ``pool.close()``/``join()``).  When evaluation is serial
    (resolved workers ``<= 1``) no manager is created and ``(None, None)`` is
    returned, leaving the cache process-local.

    :param shard: The shard being searched.
    :param start: Interior start :class:`Position`.
    :param workers: Per-method worker cap (see :func:`resolve_eval_workers`).
    :return: ``(pool_or_None, manager_or_None)``.
    """
    if resolve_eval_workers(workers) <= 1:
        return None, None
    manager = mp.Manager()
    shard.shared_pq_log = manager.list()
    pool = make_eval_pool(shard, start, workers)
    if pool is None:  # Defensive: resolution disagreed (shouldn't happen).
        manager.shutdown()
        shard.shared_pq_log = None
        return None, None
    return pool, manager


def evaluate_batch(
    z_list: List,
    *,
    eval_ctx: dict,
    pool=None,
    valid_fn: Optional[Callable] = None,
) -> List[Tuple[float, bool]]:
    """Evaluate a batch of flatland directions, optionally via a process *pool*.

    :param z_list: Flatland integer direction vectors to evaluate.
    :param eval_ctx: The evaluation context dict (the same kwargs
        :func:`evaluate_in_flatland` takes: ``geom, shard, start, constant,
        cmf_id, shard_id, shard_encoding_str, sink, seen_trajectories,
        handler_cache``).
    :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`.
        ``None`` (or a single-element batch) evaluates serially in-process.
    :param valid_fn: Optional ``z -> bool`` predicate; directions for which it
        returns ``False`` get ``(-inf, False)`` without a walk.  ``None`` treats
        every direction as valid.
    :return: List of ``(delta, identified)`` aligned with *z_list*.
    """
    n = len(z_list)
    if n == 0:
        return []

    # --- Serial path ---------------------------------------------------
    if pool is None or n <= 1:
        out: List[Tuple[float, bool]] = []
        for z in z_list:
            if valid_fn is not None and not valid_fn(z):
                out.append((float("-inf"), False))
            else:
                out.append(evaluate_in_flatland(z, **eval_ctx))
        return out

    # --- Parallel path -------------------------------------------------
    geom = eval_ctx["geom"]
    shard = eval_ctx["shard"]
    start = eval_ctx["start"]
    constant = eval_ctx["constant"]
    sink: Callable = eval_ctx["sink"]
    seen: dict = eval_ctx["seen_trajectories"]
    handler_cache: dict = eval_ctx["handler_cache"]
    cmf_id: str = eval_ctx["cmf_id"]
    shard_id: str = eval_ctx["shard_id"]
    shard_encoding_str: str = eval_ctx["shard_encoding_str"]
    desired = {attribute_name(s) for s in search_config.TIER2_ATTRIBUTES}

    results: List[Optional[Tuple[float, bool]]] = [None] * n
    # trajectory_id -> (direction, fingerprint, [batch indices]) for Case C.
    groups: Dict[str, tuple] = {}

    for i, z in enumerate(z_list):
        if valid_fn is not None and not valid_fn(z):
            results[i] = (float("-inf"), False)
            continue
        direction, tid, fp = flatland_trajectory_key(
            z, geom=geom, shard=shard, start=start,
            shard_id=shard_id, shard_encoding_str=shard_encoding_str,
        )
        rec = seen.get(tid)
        if rec is not None and rec.get("config_fingerprint") == fp:
            dmap = rec.get("delta_estimate") or {}
            if constant.name in dmap:  # Case A
                imap = rec.get("identified") or {}
                results[i] = (float(dmap[constant.name]),
                              bool(imap.get(constant.name, False)))
                continue
        if tid in handler_cache:  # Case B — cheap recompute, no new walk
            results[i] = evaluate_in_flatland(z, **eval_ctx)
            continue
        # Case C — dedupe directions that share a primitive ray.
        grp = groups.get(tid)
        if grp is None:
            groups[tid] = (direction, fp, [i])
        else:
            grp[2].append(i)

    if groups:
        args = [
            (direction, constant, cmf_id, shard_id, shard_encoding_str)
            for (direction, fp, idxs) in groups.values()
        ]
        walked = pool.map(_pool_walk, args)
        for (tid, (direction, fp, idxs)), res in zip(groups.items(), walked):
            if isinstance(res, WalkError):
                Logger(
                    f"Parallel walk failed — shard {shard_id}, "
                    f"trajectory {tid[:8]}…: {res.message}",
                    Logger.Levels.warning,
                ).log()
                for i in idxs:
                    results[i] = (float("-inf"), False)
                continue
            matrix, vsym, dto = res
            sink((matrix, vsym, dto))
            seen[tid] = {
                "extended_metrics": dict.fromkeys(desired),
                "delta_estimate": dict(dto.delta_estimate),
                "identified": dict(dto.identified),
                "config_fingerprint": fp,
            }
            d = float(dto.delta_estimate.get(constant.name, float("-inf")))
            ided = bool(dto.identified.get(constant.name, False))
            for i in idxs:
                results[i] = (d, ided)

    return [r if r is not None else (float("-inf"), False) for r in results]
