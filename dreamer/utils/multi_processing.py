"""
Multiprocessing utilities for the Ramanujan Agent pipeline.

This module provides two layers:

1. **Generic parallelism abstraction** (``worker_pool``): a context manager
   that hides all subprocess + queue machinery behind a single ``push(item)``
   callable.  Callers supply a per-item ``worker_fn`` (transform) and a
   per-item ``writer_fn`` (sink).  Setting ``worker_fn=None`` collapses to
   a direct-write loop on the main thread — no subprocesses spawned —
   without changing the producer code.

2. **Stage-specific item functions**: ready-to-use per-item functions
   plugged into ``worker_pool`` by the Search stage:
     - ``compute_tier2_for_item`` — runs ``TIER2_ATTRIBUTES`` for one item.
     - ``write_jsonl_line``     — appends one DTO/patch as a JSON line.

3. **Cross-run dedup helpers**: ``load_seen_trajectories`` and
   ``load_seen_trajectory_ids`` read existing JSONL files so the analyzer
   and search stage can skip already-computed work at the trajectory level.
   ``load_seen_shards`` is a generic helper kept for ad-hoc tools.
"""

import json
import os
import queue
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

import multiprocessing as mp
from multiprocessing import Pool

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.logger import Logger


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------

def search_worker_budget() -> int:
    """Return the number of cores available to search / analysis pools.

    Single source of truth for sizing every search/eval pool.  Starts from
    ``sys_config.TOTAL_CORES`` (``None`` → ``os.cpu_count()``) and reserves cores
    for the Tier-2 background workers **and** their dedicated writer/sink — but
    only when Tier-2 is actually active (``search.TIER2_ATTRIBUTES`` non-empty).
    When Tier-2 is inactive, nothing is reserved and the whole machine is given
    to search/analysis.

    :return: Worker budget, floored at 1.
    """
    total = sys_config.TOTAL_CORES or os.cpu_count() or 1
    if config.search.TIER2_ATTRIBUTES:
        reserved = sys_config.NUM_BACKGROUND_WORKERS + 1  # +1 dedicated writer/sink
    else:
        reserved = 0
    return max(1, total - reserved)


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
    """Return a ``ProcessPoolExecutor`` whose workers inherit the current config."""
    return ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
    )


def create_pool() -> Pool:
    """Return a ``multiprocessing.Pool`` whose workers inherit the current config.

    Sized by :func:`search_worker_budget` so it never oversubscribes the cores
    reserved for the Tier-2 queue + sink (and uses the whole machine when Tier-2
    is inactive).
    """
    return Pool(
        processes=search_worker_budget(),
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
    )


# ---------------------------------------------------------------------------
# Generic producer → workers → writer pipeline
# ---------------------------------------------------------------------------

def _run_worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    results_queue: mp.Queue,
    worker_fn: Callable[[Any], Any],
    config_overrides: dict,
) -> None:
    """Subprocess entry point: pull items, apply *worker_fn*, forward results.

    Stops cleanly on the ``None`` sentinel.  An exception inside
    ``worker_fn`` is logged but does not crash the loop — the original
    item is forwarded so the writer still sees it.
    """
    _init_worker(config_overrides)
    Logger(f"Worker {worker_id} started.", Logger.Levels.debug).log()

    while True:
        try:
            item = task_queue.get(timeout=3)
        except queue.Empty:
            continue
        if item is None:
            Logger(f"Worker {worker_id} stopping.", Logger.Levels.debug).log()
            break
        try:
            result = worker_fn(item)
        except Exception as e:
            Logger(
                f"Worker {worker_id} error: {e}",
                Logger.Levels.warning,
            ).log()
            result = item
        results_queue.put(result)


def _run_writer_loop(
    results_queue: mp.Queue,
    writer_fn: Callable[[Any, Any], None],
    output_path: str,
    config_overrides: dict,
    batch_size: int,
    flush_timeout: float,
) -> None:
    """Subprocess entry point: drain *results_queue* and write each item.

    Opens *output_path* in append mode once and calls ``writer_fn(item, fout)``
    for every item.  Flushes are batched to reduce fsync overhead — an
    explicit ``flush`` is issued every *batch_size* records, or after
    *flush_timeout* seconds of queue inactivity (whichever comes first),
    and one final flush always runs before the writer exits on the
    ``None`` sentinel.  Being the sole writer eliminates file-lock races.
    """
    _init_worker(config_overrides)
    Logger("Writer started.", Logger.Levels.debug).log()

    pending = 0
    with open(output_path, "a") as fout:
        while True:
            try:
                item = results_queue.get(timeout=flush_timeout)
            except queue.Empty:
                # Idle period — flush whatever we've buffered so tail data
                # is durable even if the producer is slow.
                if pending:
                    fout.flush()
                    pending = 0
                continue
            if item is None:
                Logger("Writer stopping.", Logger.Levels.debug).log()
                break
            try:
                writer_fn(item, fout)
                pending += 1
            except Exception as e:
                Logger(f"Writer error: {e}", Logger.Levels.warning).log()
            if pending >= batch_size:
                fout.flush()
                pending = 0
        if pending:
            fout.flush()


def _identity(item: Any) -> Any:
    """Default worker_fn — returns its input unchanged.

    Module-level (not a lambda) so it can be pickled to subprocesses if
    MPMC mode is ever asked for with no explicit transform.
    """
    return item


@contextmanager
def worker_pool(
    *,
    num_workers: int,
    worker_fn: Optional[Callable[[Any], Any]],
    writer_fn: Callable[[Any, Any], None],
    output_path: str,
    config_overrides: dict,
    queue_maxsize: Optional[int] = None,
    parallel: Optional[bool] = None,
) -> Iterator[Callable[[Any], None]]:
    """Context manager yielding a ``push(item)`` callable for a producer.

    The user always supplies a per-item ``worker_fn`` (transform) and a
    per-item ``writer_fn`` (sink).  Items pushed by the producer flow
    ``push → worker_fn → writer_fn`` in **both** modes — the only thing
    that changes is *where* the transform runs:

    * **Direct mode** (``parallel=False``): ``worker_fn`` runs on the
      producer thread, then ``writer_fn`` writes to the open file.  No
      subprocess is created.
    * **MPMC mode** (``parallel=True``): ``num_workers`` subprocess
      workers apply ``worker_fn`` on items pulled from a bounded task
      queue; results flow through a results queue to a single dedicated
      writer subprocess that owns the file.  All sentinels and joins are
      handled on ``__exit__`` (even when the producer raises).

    When ``parallel`` is left as ``None``, it defaults to ``True`` if
    ``worker_fn`` is provided and ``False`` otherwise — this keeps the
    common case ergonomic (call sites that need to skip subprocess
    spin-up just pass ``parallel=False``).

    ``worker_fn=None`` is shorthand for the identity function.

    Parameters
    ----------
    num_workers:
        Number of worker subprocesses to spawn.  Ignored in direct mode.
    worker_fn:
        Per-item transform: takes ``item``, returns the (possibly modified)
        item to be written.  Must be picklable in MPMC mode (typically a
        module-level function).  ``None`` selects an identity transform.
    writer_fn:
        Per-item sink: takes ``(item, fout)`` and writes one record.  Must
        be picklable in MPMC mode.  ``fout`` is an open append-mode handle.
    output_path:
        Destination file path.  Opened on the writer side (or main thread
        in direct mode).
    config_overrides:
        Returned by ``config.export_configurations()`` in the parent;
        propagated to every subprocess via ``_init_worker``.
    queue_maxsize:
        Bounded task-queue capacity.  Defaults to ``max(32, 4 * num_workers)``,
        large enough that the producer rarely stalls waiting for consumers.
    parallel:
        Explicit subprocess on/off.  Defaults to ``True`` when ``worker_fn``
        is provided, ``False`` otherwise.

    Yields
    ------
    A ``push(item)`` callable.  Producer code calls it once per work item.
    """
    effective_worker_fn = worker_fn if worker_fn is not None else _identity

    if parallel is None:
        parallel = worker_fn is not None

    batch_size = int(config.system.WRITER_BATCH_SIZE)
    flush_timeout = float(config.system.WRITER_FLUSH_TIMEOUT_SECONDS)

    if not parallel:
        # --- Direct mode: apply worker_fn inline on the main thread -----
        # Same batch policy as MPMC mode; the file is closed (and therefore
        # flushed) automatically when the ``with`` block exits, so we don't
        # need a tail-timer here.
        with open(output_path, "a") as fout:
            pending = 0

            def push(item: Any) -> None:
                nonlocal pending
                result = effective_worker_fn(item)
                writer_fn(result, fout)
                pending += 1
                if pending >= batch_size:
                    fout.flush()
                    pending = 0
            yield push
        return

    # --- MPMC mode -------------------------------------------------------
    if queue_maxsize is None:
        queue_maxsize = max(32, 4 * num_workers)

    task_queue: mp.Queue = mp.Queue(maxsize=queue_maxsize)
    results_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(
            target=_run_worker_loop,
            args=(i, task_queue, results_queue, effective_worker_fn, config_overrides),
        )
        for i in range(num_workers)
    ]
    writer = mp.Process(
        target=_run_writer_loop,
        args=(
            results_queue, writer_fn, output_path, config_overrides,
            batch_size, flush_timeout,
        ),
    )

    for w in workers:
        w.start()
    writer.start()

    try:
        yield task_queue.put
    finally:
        # Send one sentinel per worker, then signal the writer.  Guaranteed
        # even if the producer raised — prevents hanging subprocesses.
        for _ in workers:
            task_queue.put(None)
        for w in workers:
            w.join()
        results_queue.put(None)
        writer.join()


# ---------------------------------------------------------------------------
# Search-stage per-item worker / writer
# ---------------------------------------------------------------------------

def compute_tier2_for_item(item):
    """Per-item worker: compute the configured Tier-2 attributes.

    *item* is ``(trajectory_matrix, constant, dto_or_patch)`` where
    *dto_or_patch* is either a ``TrajectoryDTO`` (new trajectory) or a patch
    ``dict`` (partial coverage), and *constant* is the sympy expression for
    the target constant (e.g. ``sp.log(2)``) — required by attributes that
    need to compare against the limit (``delta_sequence``, ``limit``, p/q
    identification).  Pass ``None`` when no constant context is available.

    Only attributes listed in ``search.TIER2_ATTRIBUTES`` and **not already
    present** in ``extended_metrics`` are computed.  Per-attribute failures
    are stored as ``<name>_error``; a fatal handler failure is recorded
    under ``worker_error``.  The same dto/patch is returned for the writer.
    """
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
    from dreamer.utils.storage.attribute_registry import compute_attributes, attribute_name
    from dreamer.configs import config

    traj_matrix, constant, dto_or_patch = item
    is_patch = isinstance(dto_or_patch, dict)
    extended_metrics = (
        dto_or_patch["extended_metrics"]
        if is_patch
        else dto_or_patch.extended_metrics
    )
    tid = (
        dto_or_patch.get("trajectory_id", "?")
        if is_patch
        else dto_or_patch.trajectory_id
    )

    attrs_to_compute = config.search.TIER2_ATTRIBUTES
    # Specs may be bare strings or ``(name, predicate)`` tuples; filter on
    # the resolved attribute name so the predicate-skip path still runs
    # inside ``compute_attributes``.
    missing = [
        spec for spec in attrs_to_compute
        if attribute_name(spec) not in extended_metrics
    ]
    if missing and traj_matrix is not None:
        try:
            handler = TrajectoryAttributesHandler(traj_matrix, constant=constant)
            extended_metrics.update(
                compute_attributes(handler, missing, on_error="store")
            )
        except Exception as e:
            Logger(
                f"compute_tier2_for_item error on trajectory {tid}: {e}",
                Logger.Levels.warning,
            ).log()
            extended_metrics["worker_error"] = str(e)
    return dto_or_patch


def write_jsonl_line(item, fout) -> None:
    """Per-item writer: serialise *item* as one JSON line.

    Dicts (patches) go through ``json.dumps``; DTO objects use their
    ``to_json_line()`` method.  The caller's open file handle does the
    actual ``write``.

    A patch with an empty ``extended_metrics`` carries no new information
    (every attribute was either already present or skipped by its predicate)
    — silently drop it instead of bloating the JSONL with no-op lines.
    Full ``TrajectoryDTO`` objects always go through, including when their
    ``extended_metrics`` is empty: that's the base record for a trajectory.
    """
    if isinstance(item, dict):
        if not (item.get("extended_metrics") or {}):
            return
        line = json.dumps(item)
    else:
        line = item.to_json_line()
    fout.write(line + "\n")


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def load_seen_trajectories(jsonl_path: str) -> Dict[str, dict]:
    """Read and merge all trajectory records from an existing JSONL file.

    Records sharing the same ``trajectory_id`` are merged left-to-right so
    that patch records (which contain only ``trajectory_id`` and a partial
    ``extended_metrics`` dict) are folded into the base record.
    ``extended_metrics`` is deep-merged; all other top-level keys are
    shallow-merged with later records winning.

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
                    tid = record.get("trajectory_id")
                    if tid is None:
                        continue
                    if tid not in merged:
                        merged[tid] = record
                    else:
                        existing_metrics = dict(merged[tid].get("extended_metrics") or {})
                        new_metrics = dict(record.get("extended_metrics") or {})
                        merged[tid].update(record)
                        merged[tid]["extended_metrics"] = {**existing_metrics, **new_metrics}
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return merged


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

    Thin wrapper around :func:`load_seen_trajectories` kept for backward
    compatibility with any call site that only needs the id set.
    """
    return set(load_seen_trajectories(jsonl_path).keys())
