"""
Stage-wide process pool and buffered single-writer for per-trajectory work.

Every pipeline stage (analysis, search, post-process) runs one module after the
other; the parallelism lives *inside* a module.  Within a stage the per-trajectory
attribute computations are independent of each other, so they are spread over a
pool of worker processes, while all file I/O stays in the main process.

Design
------
* **One persistent pool per stage** (:class:`StagePool`), spanning every shard the
  stage touches.  The shard list (plus an optional ``extra`` payload such as a
  CMF lookup) is handed to each worker **once**, via the pool initializer; tasks
  then reference a shard by its index, so a task carries only the small
  per-trajectory payload.  This pays the worker start-up cost (under ``spawn``,
  ~2 s to import ``dreamer``) once per stage instead of once per shard.
* **Shared (p, q) cache.**  When the pool is real (>1 worker) each shard gets a
  ``multiprocessing.Manager`` list that every worker's
  :class:`~dreamer.utils.storage.frequency_list.FrequencyList` publishes to and
  folds in from, so a relation identified by one worker is reused by the others.
* **Dynamic load balancing.**  :meth:`StagePool.imap` uses ``Pool.imap`` with a
  small chunk size (``system.STAGE_POOL_CHUNKSIZE``, default 1): an idle worker
  pulls the next task immediately, so a few slow walks (e.g. a high-dimensional
  CMF) cannot stall statically-assigned work.  Results are yielded **in task
  order** (only delivery is ordered, not execution), so the file contents are
  deterministic run to run.
* **Single writer.**  Workers never read or write the JSONL atlas; they return
  results to the main process, which appends them through a
  :class:`BufferedJsonlWriter` (one ``open`` per batch per file).

With ``workers <= 1`` the same task functions run serially in-process — no
subprocess, no manager — which keeps the code path identical for tests and
small machines.

Task functions must be module-level (picklable) and must **not raise**: a raised
exception would abort the whole ``imap`` stream, so failures are returned as a
:class:`TaskError` (carrying the task key) and handled by the caller.
"""

import multiprocessing as mp
import os
import time
from collections import namedtuple
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from dreamer.configs import config

sys_config = config.system


#: Returned by a task function instead of raising, so one failing trajectory
#: degrades gracefully instead of aborting the whole stage stream.
TaskError = namedtuple("TaskError", ["key", "message"])


# Per-process stage context, populated once by the pool initializer (or directly
# by StagePool in serial mode).
_STAGE_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------

def total_core_budget() -> int:
    """Return the number of cores a stage pool may use.

    ``sys_config.TOTAL_CORES`` (``None`` → ``os.cpu_count()``), floored at 1.
    Unlike :func:`dreamer.utils.multi_processing.search_worker_budget` nothing is
    reserved for the Tier-2 queue: use this for stages where that queue is not
    running (analysis, post-process).

    :return: Worker budget (>= 1).
    """
    return max(1, sys_config.TOTAL_CORES or os.cpu_count() or 1)


def resolve_stage_workers(budget: Optional[int] = None) -> int:
    """Apply the ``system.STAGE_POOL_WORKERS`` cap to a core *budget*.

    :param budget: Cores available to the stage (``None`` → :func:`total_core_budget`).
    :return: ``budget`` when the cap is ``None``/``<= 0``, else ``min(cap, budget)``;
        always >= 1.
    """
    budget = total_core_budget() if budget is None else max(1, int(budget))
    cap = getattr(sys_config, "STAGE_POOL_WORKERS", None)
    if cap is None or cap <= 0:
        return budget
    return max(1, min(int(cap), budget))


# ---------------------------------------------------------------------------
# Worker-side state
# ---------------------------------------------------------------------------

def _install_stage_state(shards: Sequence, shared_logs: Optional[Sequence], extra: Optional[dict]) -> None:
    """Install the stage context in the current process.

    :param shards: The stage's shards, addressed by index in tasks.
    :param shared_logs: One Manager list per shard (same order), or ``None`` to
        keep each shard's (p, q) cache process-local.
    :param extra: Optional stage-specific payload (e.g. ``{"cmf_lookup": ...}``).
    """
    if shared_logs is not None:
        from dreamer.utils.storage.frequency_list import FrequencyList
        for shard, log in zip(shards, shared_logs):
            cache = getattr(shard, "cache", None)
            if cache is not None:
                shard.cache = FrequencyList(cache.max_size, shared_log=log)
    _STAGE_STATE.clear()
    _STAGE_STATE["shards"] = list(shards)
    _STAGE_STATE["extra"] = dict(extra or {})


def _stage_pool_init(config_overrides: dict, shards: Sequence, shared_logs, extra) -> None:
    """Pool initializer: re-apply the parent's config and install the stage context."""
    config.configure(**config_overrides)
    _install_stage_state(shards, shared_logs, extra)


def stage_shard(idx: int):
    """Return the stage shard at *idx* (call from inside a task function)."""
    return _STAGE_STATE["shards"][idx]


def stage_extra(key: str, default: Any = None) -> Any:
    """Return an entry of the stage's ``extra`` payload (call from a task function)."""
    return _STAGE_STATE.get("extra", {}).get(key, default)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

class StagePool:
    """Persistent per-stage process pool (context manager).

    Usage::

        with StagePool(shards, workers=total_core_budget()) as pool:
            for result in pool.imap(task_fn, tasks):
                ...

    :param shards: Every shard the stage will reference (by index) in its tasks.
    :param workers: Core budget for the pool; ``None`` → :func:`total_core_budget`.
        The ``system.STAGE_POOL_WORKERS`` cap is applied on top
        (:func:`resolve_stage_workers`).  A resolved count ``<= 1`` runs tasks
        serially in-process.
    :param extra: Optional picklable payload handed to the workers once
        (read with :func:`stage_extra`).
    :param share_pq_cache: Attach a shared (p, q) log per shard (multi-worker only).
    """

    def __init__(
        self,
        shards: Sequence = (),
        *,
        workers: Optional[int] = None,
        extra: Optional[dict] = None,
        share_pq_cache: bool = True,
    ):
        self.shards = list(shards)
        self.workers = resolve_stage_workers(workers)
        self.extra = extra
        self.share_pq_cache = share_pq_cache
        self._pool: Optional[Pool] = None
        self._manager = None

    @property
    def parallel(self) -> bool:
        """``True`` when tasks run in worker processes."""
        return self._pool is not None

    def __enter__(self) -> "StagePool":
        if self.workers <= 1:
            _install_stage_state(self.shards, None, self.extra)
            return self
        shared_logs = None
        if self.share_pq_cache and self.shards:
            self._manager = mp.Manager()
            shared_logs = [self._manager.list() for _ in self.shards]
        self._pool = Pool(
            processes=self.workers,
            initializer=_stage_pool_init,
            initargs=(config.export_configurations(), self.shards, shared_logs, self.extra),
        )
        return self

    def imap(self, fn: Callable, tasks: Iterable) -> Iterator:
        """Apply *fn* to every task; yield results in task order.

        :param fn: Module-level task function (must not raise; return
            :class:`TaskError` on failure).
        :param tasks: Iterable of picklable task payloads.
        """
        if self._pool is None:
            return map(fn, tasks)
        chunksize = max(1, int(getattr(sys_config, "STAGE_POOL_CHUNKSIZE", 1) or 1))
        return self._pool.imap(fn, tasks, chunksize=chunksize)

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._pool is not None:
                if exc_type is None:
                    self._pool.close()
                else:
                    self._pool.terminate()
                self._pool.join()
        finally:
            self._pool = None
            if self._manager is not None:
                self._manager.shutdown()
                self._manager = None
            _STAGE_STATE.clear()
        return False


# ---------------------------------------------------------------------------
# Buffered single writer
# ---------------------------------------------------------------------------

class _PathSink:
    """File-like ``write`` target for one path (lets ``write_jsonl_line`` reuse it)."""

    def __init__(self, writer: "BufferedJsonlWriter", path: str):
        self._writer = writer
        self._path = path

    def write(self, text: str) -> None:
        self._writer.write_text(self._path, text)


class BufferedJsonlWriter:
    """Main-process append-only writer for many JSONL files.

    Lines are buffered per path and appended with **one ``open`` per batch**:
    a path is flushed when it holds ``batch_size`` lines, when
    ``flush_timeout`` seconds passed since the last flush, on :meth:`flush`, and
    on close.  Only the main process writes, so there is a single writer per
    file.

    :param batch_size: Lines per path before a flush (default
        ``system.WRITER_BATCH_SIZE``).
    :param flush_timeout: Seconds after which all buffers are flushed on the next
        write (default ``system.WRITER_FLUSH_TIMEOUT_SECONDS``); bounds the loss
        window on an abrupt exit.
    """

    def __init__(self, batch_size: Optional[int] = None, flush_timeout: Optional[float] = None):
        self.batch_size = max(1, int(batch_size or sys_config.WRITER_BATCH_SIZE))
        self.flush_timeout = float(
            flush_timeout if flush_timeout is not None else sys_config.WRITER_FLUSH_TIMEOUT_SECONDS
        )
        self._buffers: Dict[str, List[str]] = {}
        self._lines: Dict[str, int] = {}
        self._last_flush = time.monotonic()

    def sink(self, path: str) -> _PathSink:
        """Return a file-like object whose ``write`` appends to *path*'s buffer."""
        return _PathSink(self, path)

    def write_line(self, path: str, line: str) -> None:
        """Buffer one JSON *line* (without trailing newline) for *path*."""
        self.write_text(path, line + "\n")

    def write_text(self, path: str, text: str) -> None:
        """Buffer raw *text* for *path* (counted as one record per newline)."""
        self._buffers.setdefault(path, []).append(text)
        self._lines[path] = self._lines.get(path, 0) + max(1, text.count("\n"))
        if self._lines[path] >= self.batch_size:
            self.flush(path)
        elif time.monotonic() - self._last_flush >= self.flush_timeout:
            self.flush()

    def flush(self, path: Optional[str] = None) -> None:
        """Append buffered text to disk — for *path*, or for every path."""
        paths = [path] if path is not None else list(self._buffers)
        for p in paths:
            chunks = self._buffers.pop(p, None)
            self._lines.pop(p, None)
            if not chunks:
                continue
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "a") as fout:
                fout.write("".join(chunks))
        self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush everything."""
        self.flush()

    def __enter__(self) -> "BufferedJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Task functions (module-level so they pickle)
# ---------------------------------------------------------------------------

#: Per-trajectory task payload.  ``key`` is echoed back.
#:
#: * ``constants`` — :class:`Constant` objects that need the full Tier-1 row
#:   (walk + identification); ``extra_metrics`` (attribute specs, e.g. the
#:   configured Tier-2 list) are computed for them **in the same scope**, on the
#:   same handler, so nothing is computed twice.
#: * ``seeded`` — ``(constant, stored_row, specs)`` triples whose Tier-1 row is
#:   still valid: only *specs* are computed, on a handler seeded with the row's
#:   δ / identified / p,q (no LIReC; see ``attributes_patch``).
Tier1Task = namedtuple(
    "Tier1Task",
    ["key", "shard_idx", "trajectory", "start", "constants", "cmf_id", "shard_id",
     "shard_encoding_str", "extra_metrics", "seeded"],
    defaults=((), ()),
)

#: Successful result: the symbolic trajectory matrix (only when requested), one
#: DTO per entry of ``constants`` (same order) and one patch per ``seeded`` entry.
Tier1Result = namedtuple("Tier1Result", ["key", "matrix", "dtos", "patches"], defaults=([],))


def run_tier1_task(
    task: Tier1Task,
    *,
    handler_cls=None,
    build_dtos: Optional[Callable] = None,
    want_matrix: bool = False,
):
    """Compute one trajectory's attributes: full rows and/or seeded patches.

    Mirrors the serial per-trajectory code path: ``handler_cls.from_cmf`` on the
    stage shard, then ``build_dtos`` for ``task.constants`` (with
    ``task.extra_metrics`` computed in the same constant scope), then a seeded
    :func:`attributes_patch` for every ``task.seeded`` entry (reusing the same
    handler when the walk depth matches).  Wrapped in the trajectory watchdog.
    Stage modules wrap this in a module-level task function that passes their
    own ``TrajectoryAttributesHandler`` / ``build_trajectory_dtos`` names.

    :param task: A :class:`Tier1Task`.
    :param handler_cls: Handler class (default ``TrajectoryAttributesHandler``).
    :param build_dtos: DTO builder (default ``build_trajectory_dtos``).
    :param want_matrix: Also return the symbolic trajectory matrix.
    :return: :class:`Tier1Result`, or :class:`TaskError` on failure.
    """
    from dreamer.configs.logging import logging_config
    from dreamer.utils.logger import Logger
    from dreamer.utils.storage import trajectory_attributes as ta

    handler_cls = handler_cls or ta.TrajectoryAttributesHandler
    build_dtos = build_dtos or ta.build_trajectory_dtos
    try:
        shard = stage_shard(task.shard_idx)
        with Logger.watchdog(
            f"Trajectory compute (shard {task.shard_id})",
            logging_config.WATCHDOG_TRAJECTORY_SECONDS,
            detail=lambda: f"start={task.start} direction={task.trajectory}",
        ):
            handler = None
            dtos: List = []
            if task.constants:
                handler = handler_cls.from_cmf(
                    shard.cmf, task.trajectory, task.start,
                    constant=None,  # injected per-constant by the DTO builder
                    searchable=shard,
                )
                kwargs = {"extra_metrics": tuple(task.extra_metrics)} if task.extra_metrics else {}
                dtos = list(build_dtos(
                    handler,
                    cmf_id=task.cmf_id,
                    shard_id=task.shard_id,
                    cmf_name=shard.cmf_name,
                    shard_encoding_str=task.shard_encoding_str,
                    start=task.start,
                    direction=task.trajectory,
                    constants=list(task.constants),
                    **kwargs,
                ))
            patches = []
            for constant, record, specs in task.seeded:
                depth = record.get("walk_depth")
                if handler is None or (depth is not None and int(depth) != handler.walk_depth()):
                    handler = handler_cls.from_cmf(
                        shard.cmf, task.trajectory, task.start,
                        walk_depth=depth, searchable=shard,
                    )
                patches.append(ta.attributes_patch(handler, constant, record, specs))
            if want_matrix and handler is None:
                handler = handler_cls.from_cmf(shard.cmf, task.trajectory, task.start, searchable=shard)
            matrix = handler.trajectory_matrix if want_matrix else None
    except Exception as e:  # noqa: BLE001 — one bad trajectory must not abort the stage
        return TaskError(task.key, str(e))
    return Tier1Result(task.key, matrix, dtos, patches)
