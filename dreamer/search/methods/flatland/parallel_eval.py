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

import json
import multiprocessing as mp
from collections import namedtuple
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from dreamer.configs import config
from dreamer.search.methods.flatland.evaluator import (
    active_objective,
    evaluate_in_flatland,
    flatland_trajectory_key,
)
from dreamer.utils.logger import Logger
from dreamer.utils.resilient_pool import ResilientPool
from dreamer.utils.storage.attribute_fingerprints import tier1_record_fresh
from dreamer.utils.multi_processing import search_worker_budget
from dreamer.utils.storage.optimization_objectives import score_record

search_config = config.search


# Worker-side entry points live in the light ``parallel_eval_worker`` module so a
# spawned eval worker does not import the flatland geometry / samplers (numba,
# scipy.optimize) just to walk trajectories.  Re-exported here for callers/tests.
from dreamer.search.methods.flatland.parallel_eval_worker import (  # noqa: E402
    _POOL_STATE,
    WalkError,
    _pool_init,
    _pool_walk,
    _walk_death_result,
)


def resolve_eval_workers(cap: Optional[int]) -> int:
    """Resolve a ``*_NUM_EVAL_WORKERS`` knob against the global core budget.

    The per-method knobs are interpreted as **optional caps**: ``None`` or
    ``<= 0`` means "use the full search budget"
    (:func:`dreamer.utils.multi_processing.search_worker_budget`), while a
    positive value caps the worker count at ``min(cap, budget)``.  This is what
    lets the eval pools scale to a large machine without per-method tuning while
    still letting a method cap its own pool.

    :param cap: The configured per-method worker cap (knob value).
    :return: Effective worker count (>= 1).
    """
    budget = search_worker_budget()
    if cap is None or cap <= 0:
        return budget
    return max(1, min(cap, budget))


def make_eval_pool(shard, start, workers: int) -> Optional[Pool]:
    """Create a shard's **eval pool**, or ``None`` if disabled.

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
    return ResilientPool(
        effective,
        initializer=_pool_init,
        initargs=(config.export_configurations(), shard, start),
        on_death=_walk_death_result,
        max_tasks_per_child=getattr(config.system, "POOL_MAX_TASKS_PER_CHILD", None),
        name="eval pool",
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


class WalkPrefetcher:
    """Asynchronous speculative walks on an eval pool, using idle workers only.

    A sequential optimiser (simulated annealing) needs the walks of step *k*
    before it can choose step *k + 1*.  A step has at most ``2·d_flat`` walks and
    waits for the slowest one, so workers sit idle twice: those the step does not
    use, and those whose walk finished early (walk times on one shard spread
    ~7x, so a batch takes ~2x its mean walk).  The prefetcher fills both with
    walks a later step is likely to need (the neighbourhoods of the possible
    next centers).  When a later :func:`evaluate_batch` asks for one of them it
    is taken from here — already finished, or still running and awaited —
    instead of being dispatched again.

    Speculative results are parked in memory and **never** sunk / recorded here:
    :func:`evaluate_batch` commits a parked walk exactly like a fresh one, at the
    moment (and in the order) it is requested, so the optimiser's decisions, the
    JSONL rows and the ``seen`` cache are the same as without speculation.
    Speculation that is never requested is simply discarded.

    Capacity: walks are submitted only into **idle** workers — the pool's
    current worker limit minus *every* task in flight (the step's own walks and
    earlier speculation), read live from the pool (``idle_slots()``) — so the
    pool's FIFO queue never holds speculation ahead of a step's own walks.
    Groups are taken in the order given (most likely first) and may be started
    **partially**: every finished walk of the real next neighbourhood shortens
    that step.

    :param pool: The per-shard eval pool (its workers hold the shard / start).
    :param eval_ctx: The evaluation context of the run (same dict passed to
        :func:`evaluate_batch`).
    :param workers: Fixed worker count; only used when the pool cannot report
        its idle slots (test doubles).  Default: ``pool._processes``, read live.
    :param max_age: Finished, unclaimed results are dropped after this many
        steps (:meth:`next_step` calls) — bounds memory.
    """

    def __init__(self, pool, eval_ctx: dict, *, workers: Optional[int] = None, max_age: int = 2):
        self.pool = pool
        self.ctx = eval_ctx
        self._fixed_workers = int(workers) if workers else None
        self.max_age = max_age
        # tid -> [AsyncResult, generation]
        self._entries: Dict[str, list] = {}
        self._gen = 0
        self.stats = {"submitted": 0, "taken": 0, "dropped": 0}

    @property
    def workers(self) -> int:
        """Current worker count of the pool (it can grow or shrink mid-run)."""
        if self._fixed_workers:
            return self._fixed_workers
        return int(getattr(self.pool, "_processes", 1) or 1)

    def outstanding(self) -> int:
        """Number of speculative walks submitted and not yet finished."""
        return sum(1 for res, _gen in self._entries.values() if not res.ready())

    def free_slots(self) -> int:
        """Idle workers available for speculation right now."""
        idle = getattr(self.pool, "idle_slots", None)
        if callable(idle) and not self._fixed_workers:
            return int(idle())
        return self.workers - self.outstanding()

    def pending_handles(self) -> list:
        """Handles of speculative walks still running (for waiting on any of them)."""
        return [res for res, _gen in self._entries.values() if not res.ready()]

    def has(self, tid: str) -> bool:
        """``True`` if a walk for trajectory *tid* was started (finished or not)."""
        return tid in self._entries

    def claim(self, tid: str):
        """Remove the entry for *tid* and return its handle without waiting.

        Claim first, wait later: a claimed walk can no longer be evicted by a
        :meth:`next_step` made before its result is read.

        :return: The ``AsyncResult`` of the walk (``.get()[1]`` is the payload).
        """
        res, _gen = self._entries.pop(tid)
        self.stats["taken"] += 1
        return res

    def take(self, tid: str):
        """Remove and return the walk result for *tid* (blocks until it is done).

        :return: The ``_pool_walk`` payload — ``(matrix, value_sympy, dto)`` or
            :class:`WalkError`.
        """
        return self.claim(tid).get()[1]

    def next_step(self) -> None:
        """Mark the start of a new optimiser step and evict stale results.

        Called once per :func:`evaluate_batch` (after it claimed what it needs),
        so a result survives ``max_age`` steps however often :meth:`speculate`
        refills idle workers within a step.
        """
        self._gen += 1
        self._evict()

    def speculate(self, groups: Sequence[Sequence]) -> int:
        """Start walks of *groups* (in order, most likely first) in idle workers.

        A group's walks are started in order until the idle workers run out; a
        group may be started partially.  Directions already scored (fresh
        ``seen`` row), walk-cached this run, or already started are skipped.

        :param groups: Sequence of direction groups (flatland vectors).
        :return: Number of walks submitted by this call.
        """
        ctx = self.ctx
        constant = ctx["constant"]
        seen = ctx["seen_trajectories"]
        free = self.free_slots()
        submitted = 0
        for group in groups:
            if free <= 0:
                break
            for z in group:
                if free <= 0:
                    break
                direction, tid, fp = flatland_trajectory_key(
                    z, geom=ctx["geom"], shard=ctx["shard"], start=ctx["start"],
                    shard_id=ctx["shard_id"], shard_encoding_str=ctx["shard_encoding_str"],
                )
                rec = seen.get(tid, {}).get(constant.name)
                if tier1_record_fresh(rec, fp):
                    continue
                if tid in ctx["handler_cache"] or tid in self._entries:
                    continue
                args = (0, direction, constant, ctx["cmf_id"], ctx["shard_id"], ctx["shard_encoding_str"])
                self._entries[tid] = [self.pool.apply_async(_pool_walk, (args,)), self._gen]
                submitted += 1
                free -= 1
        self.stats["submitted"] += submitted
        return submitted

    def _evict(self) -> None:
        """Drop finished, unclaimed results older than ``max_age`` steps."""
        stale = [
            tid for tid, (res, gen) in self._entries.items()
            if gen < self._gen - self.max_age and res.ready()
        ]
        for tid in stale:
            del self._entries[tid]
        self.stats["dropped"] += len(stale)


def evaluate_batch(
    z_list: List,
    *,
    eval_ctx: dict,
    pool=None,
    valid_fn: Optional[Callable] = None,
    prefetcher: Optional["WalkPrefetcher"] = None,
    speculate=None,
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
    :param prefetcher: Optional :class:`WalkPrefetcher` bound to *pool* and this
        *eval_ctx*.  Walks it already started (speculatively, in an earlier call)
        are taken from it instead of being dispatched again; new walks are
        dispatched asynchronously, and while they finish the idle workers are
        refilled with *speculate*.  Results are committed exactly as
        freshly-walked ones, in the same order, after the whole batch is in.
    :param speculate: Directions a *later* call is likely to request (e.g. the
        next annealing step's candidate neighbourhoods), most likely first —
        either a fixed sequence of groups, or a callable
        ``f(partial) -> groups`` that ranks them from this batch's results so
        far (``partial[i]`` is ``(delta, identified)`` for directions already
        known, else ``None``).  Handed to ``prefetcher.speculate`` after this
        batch's own walks are queued and again after each of them finishes; only
        idle workers are used and nothing is committed here.
    :return: List of ``(delta, identified)`` aligned with *z_list*.
    """
    n = len(z_list)
    if n == 0:
        return []

    # --- Serial path ---------------------------------------------------
    if pool is None or (n <= 1 and prefetcher is None):
        out: List[Tuple[float, bool]] = []
        for z in z_list:
            if valid_fn is not None and not valid_fn(z):
                out.append((float("-inf"), False))
                if eval_ctx.get("stats") is not None:
                    eval_ctx["stats"].add("outside")
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
    stats = eval_ctx.get("stats")
    objective_name = active_objective()

    results: List[Optional[Tuple[float, bool]]] = [None] * n
    # trajectory_id -> (direction, fingerprint, [batch indices]) for the walk group.
    groups: Dict[str, tuple] = {}

    for i, z in enumerate(z_list):
        if valid_fn is not None and not valid_fn(z):
            results[i] = (float("-inf"), False)
            if stats is not None:
                stats.add("outside")
            continue
        direction, tid, fp = flatland_trajectory_key(
            z, geom=geom, shard=shard, start=start,
            shard_id=shard_id, shard_encoding_str=shard_encoding_str,
        )
        rec = seen.get(tid, {}).get(constant.name)
        if tier1_record_fresh(rec, fp):
            cached = score_record(rec, objective_name)
            if cached is not None:  # already scored for this constant
                results[i] = cached
                if stats is not None:
                    stats.scored(tid)
                    stats.note_score(cached[0])
                continue
        if tid in handler_cache:  # walk cached — cheap recompute, no new walk
            results[i] = evaluate_in_flatland(z, **eval_ctx)
            continue
        # Needs a walk — dedupe directions that share a primitive ray.
        grp = groups.get(tid)
        if grp is None:
            groups[tid] = (direction, fp, [i])
        else:
            grp[2].append(i)

    if groups:
        group_items = list(groups.items())          # deterministic (insertion) order
        args = [
            (gi, direction, constant, cmf_id, shard_id, shard_encoding_str)
            for gi, (tid, (direction, fp, idxs)) in enumerate(group_items)
        ]
        # Dynamic load balancing: ``imap_unordered`` with ``chunksize=1`` lets a
        # worker pull the next trajectory the instant it is free, so a few long
        # (deep-walk) trajectories can't stall a statically-assigned ``map`` chunk
        # while other cores idle.  Results are reassembled by group index and then
        # processed in deterministic group order, so the sink / ``seen`` updates and
        # the returned batch values are identical to the old ``pool.map`` path.
        walked: List = [None] * len(group_items)
        if prefetcher is None:
            for gi, res in pool.imap_unordered(_pool_walk, args, chunksize=1):
                walked[gi] = res
        else:
            # Take walks the prefetcher already started; queue the rest first so
            # they are not stuck behind speculation.  Claim the prefetched handles
            # *before* advancing the step (which may evict).
            handles = {}
            for gi, (tid, _grp) in enumerate(group_items):
                if prefetcher.has(tid):
                    handles[gi] = prefetcher.claim(tid)
                else:
                    handles[gi] = pool.apply_async(_pool_walk, (args[gi],))
            prefetcher.next_step()
            partial = list(results)            # ranking view only — never committed
            _speculate(prefetcher, speculate, partial)
            # Collect as walks finish; after each, refill the idle workers with
            # the (re-ranked) speculation.  Commit order is unaffected (below).
            pending = set(range(len(group_items)))
            while pending:
                done = [gi for gi in sorted(pending) if handles[gi].ready()]
                if not done:
                    if _wait_any([handles[gi] for gi in pending] + prefetcher.pending_handles()):
                        _speculate(prefetcher, speculate, partial)
                        continue
                    done = [min(pending)]      # no futures to wait on: block on the oldest
                for gi in done:
                    pending.discard(gi)
                    walked[gi] = handles[gi].get()[1]
                    preview = _preview_score(walked[gi], objective_name)
                    for i in group_items[gi][1][2]:
                        partial[i] = preview
                _speculate(prefetcher, speculate, partial)
        for (tid, (direction, fp, idxs)), res in zip(group_items, walked):
            if isinstance(res, WalkError):
                Logger(
                    f"Parallel walk failed — shard {shard_id}, "
                    f"trajectory {tid[:8]}…: {res.message}",
                    Logger.Levels.warning,
                ).log()
                for i in idxs:
                    results[i] = (float("-inf"), False)
                if stats is not None:
                    stats.add("failed", len(idxs))
                continue
            matrix, vsym, dto = res
            if stats is not None:
                stats.walked(tid, cached_walk=False, ok=True)
                stats.add("revisited", len(idxs) - 1)   # same ray in this batch
            sink((matrix, vsym, dto))
            record = json.loads(dto.to_json_line())
            seen.setdefault(tid, {})[constant.name] = record
            scored = score_record(record, objective_name)
            if scored is None:
                scored = (float("-inf"), bool(dto.identified))
            elif stats is not None:
                stats.note_score(scored[0])
            for i in idxs:
                results[i] = scored

    elif prefetcher is not None and speculate:
        # Nothing to walk for this batch (all cached) — still keep the
        # speculation for the next step running.
        prefetcher.next_step()
        _speculate(prefetcher, speculate, list(results))

    return [r if r is not None else (float("-inf"), False) for r in results]


def _speculate(prefetcher: "WalkPrefetcher", speculate, partial: list) -> None:
    """Hand *speculate* (groups, or ``f(partial) -> groups``) to the prefetcher."""
    if not speculate or prefetcher.free_slots() <= 0:
        return
    groups = speculate(partial) if callable(speculate) else speculate
    if groups:
        prefetcher.speculate(groups)


def _preview_score(res, objective_name: str) -> Optional[Tuple[float, bool]]:
    """Score of a finished walk payload for *ranking* speculation (not committed)."""
    if isinstance(res, WalkError):
        return (float("-inf"), False)
    dto = res[2]
    scored = score_record(json.loads(dto.to_json_line()), objective_name)
    return scored if scored is not None else (float("-inf"), bool(dto.identified))


def _wait_any(handles: list) -> bool:
    """Block until any of *handles* finishes.

    Uses the underlying ``concurrent.futures`` futures of :class:`ResilientPool`
    results.  A future that failed (dead worker) also wakes the wait — its
    handle's ``get()`` then runs the pool's recovery.

    :return: ``False`` (without waiting) when no handle exposes a future
        (other pools, test doubles); the caller then blocks on one ``get()``.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    futures = []
    for h in handles:
        fut = getattr(getattr(h, "_task", None), "future", None)
        if fut is not None:
            futures.append(fut)
    if not futures:
        return False
    wait(futures, return_when=FIRST_COMPLETED)
    return True
