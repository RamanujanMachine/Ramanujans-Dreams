"""
Multiprocessing utilities for the Ramanujan Agent pipeline.

This module provides two layers:

1. **Result writer** (``worker_pool``): a context manager yielding a
   ``push(item)`` callable that transforms each item (``worker_fn``) and
   appends it to the shard's JSONL (``writer_fn``) in the calling process.

2. **Stage-specific item functions**: ready-to-use per-item functions
   plugged into ``worker_pool`` by the Search stage:
     - ``sink_item_dto``         — unwraps a search sink tuple to its record.
     - ``write_jsonl_line``     — appends one DTO/patch as a JSON line.

3. **Cross-run dedup helpers**: ``load_seen_trajectories`` and
   ``load_seen_trajectory_ids`` read existing JSONL files so the analyzer
   and search stage can skip already-computed work at the trajectory level.
   ``load_seen_shards`` is a generic helper kept for ad-hoc tools.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional


from dreamer.configs import config
from dreamer.configs.system import sys_config


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------

def search_worker_budget() -> int:
    """Return the number of cores available to search / analysis pools.

    Single source of truth for sizing every search/eval pool:
    ``sys_config.TOTAL_CORES``, else the cores this run was *allocated* (batch
    scheduler request / affinity mask / machine — see
    :func:`dreamer.utils.allocation.available_cores`).  Nothing is
    reserved any more — Tier-2 attributes are computed inside the same pool
    tasks as Tier-1 (there is no separate background queue).

    :return: Worker budget, floored at 1.
    """
    from dreamer.utils.allocation import available_cores
    return available_cores()


# ---------------------------------------------------------------------------
# Config propagation for child processes
# ---------------------------------------------------------------------------

def _init_worker(config_overrides: dict) -> None:
    """Initialise the global config in a freshly-spawned worker process.

    ``mp.Process`` and ``ProcessPoolExecutor`` workers start with a clean
    namespace; calling this at the top of each worker re-applies the same
    config overrides that were active in the parent.
    """
    config.configure(**config_overrides)


def create_process_pool_executor() -> ProcessPoolExecutor:
    """Return a ``ProcessPoolExecutor`` whose workers inherit the current config.

    Sized by :func:`search_worker_budget` — never the executor's default
    ``os.cpu_count()``, which on an unconfined batch node is the whole machine.
    """
    from dreamer.utils.mp_context import pool_context
    return ProcessPoolExecutor(
        max_workers=search_worker_budget(),
        mp_context=pool_context(),
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
    )


def create_pool(name: str = "pool"):
    """Return a process pool whose workers inherit the current config.

    A :class:`~dreamer.utils.resilient_pool.ResilientPool` (survives dead
    workers instead of hanging), sized by :func:`search_worker_budget`.  Used by
    the legacy (V1) searchers and shard encoding; the pipeline stages use a
    stage pool (:class:`~dreamer.utils.stage_pool.StagePool`) and the optimisers
    an eval pool (:func:`~dreamer.search.methods.flatland.parallel_eval.make_eval_pool`).

    :param name: Label shown in the pool's recovery warnings.
    :return: The pool (use as a context manager).
    """
    from dreamer.utils.resilient_pool import ResilientPool
    return ResilientPool(
        search_worker_budget(),
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
        max_tasks_per_child=getattr(sys_config, "POOL_MAX_TASKS_PER_CHILD", None),
        name=name,
    )


# ---------------------------------------------------------------------------
# Generic producer → workers → writer pipeline
# ---------------------------------------------------------------------------

def _identity(item: Any) -> Any:
    return item


@contextmanager
def worker_pool(
    *,
    worker_fn: Optional[Callable[[Any], Any]],
    writer_fn: Callable[[Any, Any], None],
    output_path: str,
) -> Iterator[Callable[[Any], None]]:
    """Context manager yielding a ``push(item)`` callable that writes results.

    Each pushed item flows ``worker_fn → writer_fn`` on the caller's thread and
    is appended to *output_path*, flushed every ``system.WRITER_BATCH_SIZE``
    records and on exit.  No subprocesses: trajectory work runs in the stage /
    eval pools, and this writer — in the main process — is the only writer of
    the per-shard JSONL.

    (Until 2026-10-11 it also had a multi-process mode — worker and writer
    subprocesses behind queues, for the removed Tier-2 queue.  Every caller used
    the direct mode, so that mode was removed.)

    :param worker_fn: Per-item transform (``None`` = identity), e.g.
        :func:`sink_item_dto`.
    :param writer_fn: Per-item sink ``(item, fout)``, e.g. :func:`write_jsonl_line`.
    :param output_path: File to append to (opened once, in append mode).
    :return: Yields ``push(item)``.
    """
    effective_worker_fn = worker_fn if worker_fn is not None else _identity
    batch_size = int(config.system.WRITER_BATCH_SIZE)
    with open(output_path, "a") as fout:
        pending = 0

        def push(item: Any) -> None:
            nonlocal pending
            writer_fn(effective_worker_fn(item), fout)
            pending += 1
            if pending >= batch_size:
                fout.flush()
                pending = 0

        yield push


def sink_item_dto(item):
    """Per-item ``worker_fn`` of the direct-write sinks: unwrap the record.

    Search methods push ``(trajectory_matrix, constant_sympy, dto_or_patch)``
    tuples (Tier-2 attributes are already inside the DTO — they are computed in
    the walk task); the writer only needs the record.

    :param item: A sink tuple, or a bare DTO / patch dict.
    :return: The DTO or patch dict to write.
    """
    if isinstance(item, tuple) and len(item) == 3:
        return item[2]
    return item


#: Structural keys a patch dict may carry for identity / handler reconstruction; a
#: patch with *only* these adds no new metric and is dropped instead of bloating
#: the JSONL with a no-op line.
_PATCH_STRUCTURAL_KEYS = frozenset(
    {"trajectory_id", "constant", "direction", "start_point"}
)


def write_jsonl_line(item, fout) -> None:
    """Per-item writer: serialise *item* as one **flat** JSON line.

    Dicts (patches) go through ``json.dumps``; DTO objects use their
    ``to_json_line()`` method.  The caller's open file handle does the actual
    ``write``.

    A patch carrying only its structural keys (``trajectory_id`` / ``constant``)
    has no new metric — silently drop it.  Full ``TrajectoryDTO`` objects always
    go through: that's the base row for a ``(trajectory, constant)`` pair.
    """
    if isinstance(item, dict):
        if not (set(item.keys()) - _PATCH_STRUCTURAL_KEYS):
            return
        line = json.dumps(item)
    else:
        line = item.to_json_line()
    fout.write(line + "\n")


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def load_seen_trajectories(jsonl_path: str) -> Dict[str, Dict[str, dict]]:
    """Read and merge all ``(trajectory, constant)`` rows from a JSONL file.

    Each line is a **flat** record (see :class:`TrajectoryDTO`).  Rows are keyed by
    the composite ``(trajectory_id, constant)`` and folded into a nested
    ``{trajectory_id: {constant: record}}`` map — the natural shape for the reuse
    decision "have I computed this walk, for this constant, under the current
    config?".  Because the schema is flat, merging a patch into its base row is a
    plain last-write-wins ``dict.update`` (no special-cased nested dict).

    Returns an empty dict if the file does not exist.
    """
    merged: Dict[str, Dict[str, dict]] = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = record.get("trajectory_id")
                const = record.get("constant")
                if tid is None or const is None:
                    continue
                bucket = merged.setdefault(tid, {})
                if const not in bucket:
                    bucket[const] = record
                else:
                    base = bucket[const]
                    fps = record.pop("fingerprints", None)
                    base.update(record)            # flat merge, later wins
                    if fps:                        # per-attribute fps: deep merge
                        base.setdefault("fingerprints", {}).update(fps)
    except FileNotFoundError:
        pass
    return merged


def load_seen_trajectories_for_search(output_path: str, shard_id: str) -> Dict[str, dict]:
    """Load the search-stage trajectory cache for one shard, seeding from a
    separate analysis-stage store when configured.

    By default the analysis and search stages share one per-shard JSONL under
    ``EXPORT_SEARCH_RESULTS``, so the search stage transparently reuses the
    records the analyzer already computed.  When
    ``config.analysis.STORE_TRAJECTORIES_SEPARATELY`` is enabled the analyzer
    writes its records to a parallel per-shard file under
    ``sys_config.EXPORT_ANALYSIS_RESULTS`` instead.  To keep cross-stage cache
    reuse, this helper copies any analysis record for *shard_id* that is not yet
    present in the search file **into** the search file, so the search results
    JSONL stays the self-contained record consumed downstream (summary,
    best-delta reporting) while the analysis store remains a separate pristine
    copy.

    Copied records are de-duplicated by ``trajectory_id``, so re-running the
    search stage never appends an analysis record twice.

    :param output_path: ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl`` path the
        search stage reads and writes.
    :param shard_id: Structural shard id; names the analysis-store file.
    :return: Merged ``{trajectory_id: {constant: record}}`` cache for the shard.
    """
    seen = load_seen_trajectories(output_path)

    if not config.analysis.STORE_TRAJECTORIES_SEPARATELY:
        return seen

    analysis_path = os.path.join(
        sys_config.EXPORT_ANALYSIS_RESULTS, f"{shard_id}.jsonl"
    )
    # Misconfiguration guard: if the analysis store points at the search file
    # there is nothing to carry over.
    if os.path.abspath(analysis_path) == os.path.abspath(output_path):
        return seen

    analysis_seen = load_seen_trajectories(analysis_path)
    # Carry over each (trajectory, constant) row the search file doesn't yet have.
    carryover: list = []
    for tid, by_const in analysis_seen.items():
        for const, rec in by_const.items():
            if const not in seen.get(tid, {}):
                carryover.append(rec)
    if carryover:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "a") as fout:
            for rec in carryover:
                fout.write(json.dumps(rec) + "\n")
                seen.setdefault(rec["trajectory_id"], {})[rec["constant"]] = rec
    return seen


def load_seen_shards(jsonl_path: str) -> Dict[str, dict]:
    """Read JSONL records keyed by ``shard_id`` (last record wins on collisions).

    Generic helper, not used by the active pipeline.  The analyzer now dedups
    per-trajectory (see :func:`load_seen_trajectories`), not per-shard.  This
    function remains available for ad-hoc tools that need shard-level views of
    JSONL outputs.

    Returns an empty dict if the file does not exist.
    """
    merged: Dict[str, dict] = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sid = record.get("shard_id")
                    if sid is not None:
                        merged[sid] = record
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return merged


def load_seen_trajectory_ids(jsonl_path: str) -> set:
    """Return the set of ``trajectory_id`` values in an existing JSONL file.

    Reads ids directly (independent of the ``constant`` column, unlike
    :func:`load_seen_trajectories`) since callers here only need the id set.
    """
    ids: set = set()
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = record.get("trajectory_id")
                if tid is not None:
                    ids.add(tid)
    except FileNotFoundError:
        pass
    return ids
