"""
A process pool that survives dead workers.

Why
---
``multiprocessing.Pool`` silently replaces a worker that dies (e.g. killed by
the out-of-memory killer while walking a large CMF) but **loses the task it was
running**, so ``imap`` / ``AsyncResult.get()`` waits forever — a hang with no
error.  This pool is built on :class:`concurrent.futures.ProcessPoolExecutor`,
which *notices* a dead worker (``BrokenProcessPool``), and recovers:

1. **Restart with fewer workers.**  The executor is rebuilt with half the
   workers (never below 1) — if memory pressure killed the worker, fewer
   concurrent tasks usually fit — and every unfinished task is resubmitted, in
   submission order.  Nothing is lost.
2. **Isolate a poison task.**  Each death marks the tasks that were running as
   suspects.  A task suspected twice is run **alone** in a fresh single-worker
   process; if it kills that worker too, it is reported through the caller's
   ``on_death`` hook (e.g. a ``TaskError`` "worker died — probably out of
   memory") and the run continues.  Without ``on_death`` a
   :class:`WorkerDeathError` is raised instead of hanging.
3. **Bounded memory.**  ``imap`` / ``imap_unordered`` keep at most
   ``window × workers`` tasks in flight (``Pool.imap`` queues *every* task up
   front), and ``max_tasks_per_child`` can recycle long-lived workers.

Every recovery is logged as a warning naming the worker count and the task.

The API mirrors the subset of ``multiprocessing.Pool`` the pipeline uses
(``imap``, ``imap_unordered``, ``apply_async(...).get()/.ready()``, ``close``,
``join``, ``terminate``, ``_processes``), so call sites need no other change.
Randomness rules are unchanged (tasks must be deterministic functions of their
arguments — see :mod:`dreamer.utils.rand`).
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional


#: Set in a worker whose initializer raised (see ``_guarded_init``).
_INIT_ERROR: Optional[str] = None


class WorkerInitError(RuntimeError):
    """The pool's worker initializer raised — a code/config bug, not a lost worker."""


def _guarded_init(initializer, initargs, monitor_dir=None):
    """Run the real initializer; on failure keep the worker alive and remember why.

    ``ProcessPoolExecutor`` reports a failed initializer exactly like a killed
    worker (``BrokenProcessPool``), which would be mistaken for out-of-memory
    and retried.  Instead the first task in this worker raises
    :class:`WorkerInitError` with the original traceback.

    With a run monitor active in the parent, ``monitor_dir`` is its run
    directory: the worker writes its own event file there.
    """
    global _INIT_ERROR
    if monitor_dir:
        try:
            from dreamer.utils.monitoring import init_worker
            init_worker(monitor_dir)
        except Exception:      # monitoring must never stop a worker
            pass
    if initializer is None:
        return
    try:
        initializer(*initargs)
    except BaseException:
        import traceback
        _INIT_ERROR = traceback.format_exc()


def _measured_call(fn, args):
    """Run ``fn(*args)`` in a worker and report the worker's peak RSS (MB).

    The peak (``ru_maxrss``: KB on Linux, bytes on macOS) is what sizing
    decisions need — a walk's memory is mostly released after the task, so the
    worker's *current* RSS would hide it.
    """
    import resource
    import sys as _sys
    if _INIT_ERROR is not None:
        raise WorkerInitError(f"worker initializer failed:\n{_INIT_ERROR}")
    value = fn(*args)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (2**20 if _sys.platform == "darwin" else 1024)
    return value, peak_mb


class WorkerDeathError(RuntimeError):
    """A task kept killing its worker process (likely out of memory)."""


_DEATH_MESSAGE = (
    "worker process died while computing this task, also when run alone "
    "(most likely out of memory)"
)


class _Task:
    """One submitted call and its current executor future."""

    __slots__ = ("fn", "args", "future", "suspicions", "result", "done", "seq")

    def __init__(self, fn: Callable, args: tuple, seq: int):
        self.fn = fn
        self.args = args
        self.future = None
        self.suspicions = 0
        self.result = None
        self.done = False
        self.seq = seq


class ResilientResult:
    """``AsyncResult``-like handle returned by :meth:`ResilientPool.apply_async`."""

    def __init__(self, pool: "ResilientPool", task: _Task):
        self._pool = pool
        self._task = task

    def ready(self) -> bool:
        """``True`` once the result is available (never blocks)."""
        t = self._task
        return t.done or (t.future is not None and t.future.done())

    def get(self, timeout: Optional[float] = None):
        """Wait for and return the result, recovering from dead workers."""
        return self._pool._resolve(self._task, timeout)


class ResilientPool:
    """Process pool with dead-worker recovery and memory-adaptive size (see module docstring).

    :param processes: Worker count (>= 1).
    :param initializer: Optional per-worker initializer.
    :param initargs: Arguments for *initializer*.
    :param on_death: Optional ``f(fn, args) -> result`` producing the result of a
        task that kills its worker even when run alone.  ``None`` raises
        :class:`WorkerDeathError`.
    :param max_tasks_per_child: Recycle each worker after this many tasks
        (``None`` = never).  Bounds slow memory growth of long-lived workers at
        the cost of re-spawning (a ``dreamer`` import per new worker).
    :param window: In-flight bound of ``imap`` / ``imap_unordered`` per worker.
    :param name: Label used in the recovery warnings.
    """

    def __init__(
        self,
        processes: int,
        initializer: Optional[Callable] = None,
        initargs: tuple = (),
        *,
        on_death: Optional[Callable[[Callable, tuple], Any]] = None,
        max_tasks_per_child: Optional[int] = None,
        window: int = 2,
        name: str = "pool",
    ):
        from dreamer.configs import config
        from dreamer.utils.memory_budget import memory_worker_cap
        from dreamer.utils.mp_context import start_method
        self._requested = max(1, int(processes))
        self._processes = memory_worker_cap(self._requested)
        # Slow start (memory-aware pools that can grow): measure real per-task
        # peaks with a few workers first, then grow to every core that fits.
        slow = int(getattr(config.system, "POOL_SLOW_START", 0) or 0)
        if (
            slow > 0
            and getattr(config.system, "MEMORY_AWARE_WORKERS", True)
            and start_method() != "fork"
            and self._processes > slow
        ):
            self._processes = slow
        self.peak_worker_mb = 0.0
        self.completed = 0
        self._initializer = initializer
        self._initargs = initargs
        self._on_death = on_death
        self._max_tasks = max_tasks_per_child if max_tasks_per_child else None
        self._window = max(1, int(window))
        self._name = name
        self._seq = 0
        self._outstanding: Dict[int, _Task] = {}
        self._closed = False
        self.deaths = 0
        self.shrinks = 0
        self.grows = 0
        self._last_resize = 0.0
        self._completed_since_resize = 0
        self._submitted_since_check = 0
        self._executor = self._new_executor(self._processes)
        from dreamer.utils import monitoring
        from dreamer.utils.logger import Logger
        self._monitor_handle = monitoring.register_pool(self)
        Logger.event(
            "pool_created",
            f"[{self._name}] {self._processes} worker(s) (requested {self._requested})",
            Logger.Levels.debug if self._processes <= 1 else Logger.Levels.info,
            pool=self._name, workers=self._processes, requested=self._requested,
        )

    def monitor_info(self) -> dict:
        """State for the run monitor (read from the monitor thread; plain reads only)."""
        procs = getattr(self._executor, "_processes", None) or {}
        return {
            "name": self._name, "workers": self._processes, "requested": self._requested,
            "in_flight": self.in_flight(),
            "completed": self.completed, "deaths": self.deaths, "shrinks": self.shrinks,
            "grows": self.grows, "peak_mb": round(self.peak_worker_mb, 1),
            "pids": list(procs), "closed": self._closed,
        }

    def in_flight(self) -> int:
        """Tasks submitted and not yet finished (running or queued)."""
        return sum(
            1 for t in list(self._outstanding.values())
            if not t.done and not (t.future is not None and t.future.done())
        )

    def idle_slots(self) -> int:
        """Workers that would start a newly submitted task right away
        (current worker limit minus tasks in flight; never negative)."""
        return max(0, self._processes - self.in_flight())

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    #: Minimum seconds between two memory-driven resizes.
    _GROW_INTERVAL = 10.0

    def _new_executor(self, workers: int) -> ProcessPoolExecutor:
        from dreamer.utils.mp_context import pool_context
        from dreamer.utils.monitoring import worker_run_dir
        kwargs = dict(
            max_workers=workers,
            mp_context=pool_context(),
            initializer=_guarded_init,
            initargs=(self._initializer, tuple(self._initargs or ()), worker_run_dir()),
        )
        if self._max_tasks and kwargs["mp_context"].get_start_method() != "fork":
            kwargs["max_tasks_per_child"] = self._max_tasks
        return ProcessPoolExecutor(**kwargs)

    def _submit(self, task: _Task) -> None:
        task.future = self._executor.submit(_measured_call, task.fn, task.args)
        self._ensure_workers()
        self._submitted_since_check += 1
        if self._submitted_since_check >= max(4, 2 * self._processes):
            self._submitted_since_check = 0
            self._check_memory()

    def _ensure_workers(self) -> None:
        """Start the worker processes the executor should have but did not start.

        ``ProcessPoolExecutor`` starts a worker on ``submit`` (and replaces a
        recycled one, ``max_tasks_per_child``) only when its *idle-worker
        semaphore* is empty.  That semaphore gains one token per completed task
        and loses one only when a submit finds it non-empty, so tokens pile up
        while the pool runs below its limit (slow start).  After the limit is
        raised in place (:meth:`_check_memory`), or after a worker exits by
        recycling, every submit consumes a stale token, believes a worker is
        idle, and starts nothing: measured 2026-10-15, an 18-worker eval pool
        ran on 2 (look-ahead) or 11 processes for a whole search.

        This tops the executor up to ``min(max_workers, outstanding tasks)``
        processes — never more processes than there is work for, so start-up
        stays lazy.  Only with ``spawn`` / ``forkserver`` (``fork`` must not
        start processes once the executor's thread runs).
        """
        ex = self._executor
        if not getattr(ex, "_safe_to_dynamically_spawn_children", False):
            return
        procs = getattr(ex, "_processes", None)
        spawn = getattr(ex, "_spawn_process", None)
        lock = getattr(ex, "_shutdown_lock", None)
        if procs is None or spawn is None or lock is None:
            return                       # unknown executor internals: keep its behaviour
        want = min(int(ex._max_workers), len(self._outstanding))
        if len(procs) >= want:
            return
        with lock:
            if getattr(ex, "_shutdown_thread", False) or getattr(ex, "_broken", False):
                return
            while len(procs) < want:
                spawn()
        wakeup = getattr(ex, "_executor_manager_thread_wakeup", None)
        if wakeup is not None:           # let the manager thread watch the new workers
            wakeup.wakeup()

    def _check_memory(self) -> None:
        """Adapt the worker count to memory, in both directions, without overlap.

        Called every few submissions.

        * **Shrink** (half) when available memory fell below the reserve: the
          waiting tasks are taken back, the running ones are **drained** (the old
          executor finishes them and its workers exit) and only then a smaller
          executor starts — never more processes than before, so the shrink
          itself cannot push memory higher.
        * **Grow** toward the requested (core) count once the workers are warm
          (``2 x workers`` tasks completed since the last resize, at least one
          measured): the largest
          worker *peak* RSS any task reached (reported by the workers) x 1.25
          is the per-worker cost; if more such workers fit above the reserve —
          after setting aside what live workers need to climb back to that peak — the executor's worker limit is raised
          **in place** (new processes start on the next submissions, as many as
          there are outstanding tasks — see :meth:`_ensure_workers`; no executor
          replacement, no overlap).  Only possible where workers can be
          started dynamically (``spawn`` / ``forkserver``); with ``fork`` the
          conservative start-up cap stands.  At most one step per
          ``_GROW_INTERVAL`` seconds.
        """
        from dreamer.configs import config
        if not getattr(config.system, "MEMORY_AWARE_WORKERS", True):
            return
        import time
        from dreamer.utils.memory_budget import available_memory_mb, rss_mb
        reserve = float(getattr(config.system, "MEMORY_RESERVE_MB", 1024))
        avail = available_memory_mb()
        if avail < reserve and self._processes > 1:
            self.shrinks += 1
            self._shrink_drain(max(1, self._processes // 2), reserve)
            return
        ex = self._executor
        if (
            self._processes >= self._requested
            or not getattr(ex, "_safe_to_dynamically_spawn_children", False)
            or self._completed_since_resize < 2 * self._processes
            or time.time() - self._last_resize < self._GROW_INTERVAL
        ):
            return
        pids = list(getattr(ex, "_processes", {}) or {})
        if len(pids) < self._processes or self.peak_worker_mb <= 0:
            return                       # not all workers started / no measured task yet
        # Per-worker cost = the largest *peak* RSS any task reached (reported by
        # the workers), not the current RSS — walks release most memory after
        # the task, so current RSS would hide the peak and overcommit.
        current = {p: rss_mb([p]) for p in pids}
        per_worker = max(self.peak_worker_mb, max(current.values())) * 1.25
        # Live workers between peaks will climb back to it: keep that room free.
        committed = sum(max(0.0, per_worker - r) for r in current.values())
        room = int((avail - reserve - committed) // per_worker) if per_worker > 0 else 0
        if room < 1:
            return
        from dreamer.utils.logger import Logger
        self.grows += 1
        self._processes = min(self._requested, self._processes + room)
        ex._max_workers = self._processes
        self._last_resize = time.time()
        self._completed_since_resize = 0
        Logger.event(
            "pool_grow",
            f"[{self._name}] warm workers measure ~{per_worker:.0f} MB each and more fit; "
            f"growing to {self._processes} worker(s)",
            Logger.Levels.info,
            pool=self._name, workers=self._processes, per_worker_mb=round(per_worker),
        )

    def _shrink_drain(self, workers: int, reserve: float) -> None:
        """Shrink without ever running more processes than before (see above)."""
        import time
        from dreamer.utils.logger import Logger
        Logger.event(
            "pool_shrink",
            f"[{self._name}] available memory below the {reserve:.0f} MB reserve; "
            f"draining running tasks and shrinking to {workers} worker(s)",
            Logger.Levels.warning,
            pool=self._name, workers=workers, reserve_mb=reserve,
        )
        waiting = [t for t in self._outstanding.values()
                   if not t.done and t.future is not None and not t.future.running() and t.future.cancel()]
        old = self._executor
        old.shutdown(wait=True)          # running tasks complete; their futures keep the results
        self._processes = workers
        self._executor = self._new_executor(workers)
        self._last_resize = time.time()
        self._completed_since_resize = 0
        for t in sorted(waiting, key=lambda t: t.seq):
            t.future = self._executor.submit(_measured_call, t.fn, t.args)

    def _recover(self) -> None:
        """A worker died: restart smaller and resubmit every unfinished task."""
        from dreamer.utils.logger import Logger

        self.deaths += 1
        broken = self._executor
        running = [t for t in self._outstanding.values()
                   if not t.done and t.future is not None and t.future.running()]
        if not running:  # unknown which one: every unfinished task is a suspect
            running = [t for t in self._outstanding.values() if not t.done]
        for t in running:
            t.suspicions += 1
        broken.shutdown(wait=False, cancel_futures=True)
        self._processes = max(1, self._processes // 2)
        unfinished = sum(1 for t in self._outstanding.values() if not t.done)
        Logger.event(
            "worker_died",
            f"[{self._name}] a worker process died (likely out of memory); restarting "
            f"with {self._processes} worker(s) and resubmitting {unfinished} unfinished task(s)",
            Logger.Levels.warning,
            pool=self._name, workers=self._processes, unfinished=unfinished,
            suspects=[t.seq for t in running],
        )
        self._executor = self._new_executor(self._processes)
        for t in sorted(self._outstanding.values(), key=lambda t: t.seq):
            if t.done:
                continue
            if t.suspicions >= 2:
                self._run_isolated(t)
            else:
                self._submit(t)

    def _run_isolated(self, task: _Task) -> None:
        """Run a twice-suspected task alone; a third death marks it as failed."""
        from dreamer.utils.logger import Logger

        Logger.event(
            "task_isolated",
            f"[{self._name}] task {task.seq} was running at two worker deaths; running it alone",
            Logger.Levels.warning, pool=self._name, task=task.seq,
        )
        solo = self._new_executor(1)
        try:
            task.result = solo.submit(_measured_call, task.fn, task.args).result()[0]
        except BrokenProcessPool:
            Logger.event(
                "task_lost",
                f"[{self._name}] task {task.seq} killed its worker again when run "
                f"alone — giving up on it ({_DEATH_MESSAGE})",
                Logger.Levels.warning, pool=self._name, task=task.seq,
            )
            if self._on_death is None:
                self._outstanding.pop(task.seq, None)
                raise WorkerDeathError(f"[{self._name}] {_DEATH_MESSAGE}")
            task.result = self._on_death(task.fn, task.args)
        finally:
            solo.shutdown(wait=True, cancel_futures=True)
        task.done = True
        task.future = None

    def _resolve(self, task: _Task, timeout: Optional[float]):
        """Block until *task* has a result, recovering from dead workers."""
        while not task.done:
            try:
                value, peak_mb = task.future.result(timeout=timeout)
                task.result = value
                task.done = True
                self.completed += 1
                self._completed_since_resize += 1
                self.peak_worker_mb = max(self.peak_worker_mb, peak_mb)
            except BrokenProcessPool:
                self._recover()
        self._outstanding.pop(task.seq, None)
        task.future = None
        return task.result

    # ------------------------------------------------------------------
    # multiprocessing.Pool-compatible API
    # ------------------------------------------------------------------

    def apply_async(self, fn: Callable, args: tuple = ()) -> ResilientResult:
        """Submit ``fn(*args)``; returns a handle with ``get()`` / ``ready()``."""
        if self._closed:
            raise ValueError("pool is closed")
        task = _Task(fn, tuple(args), self._seq)
        self._seq += 1
        self._outstanding[task.seq] = task
        try:
            self._submit(task)
        except BrokenProcessPool:
            self._recover()
        return ResilientResult(self, task)

    def imap(self, fn: Callable, iterable: Iterable, chunksize: int = 1) -> Iterator:
        """Ordered results of ``fn(x)``, with a bounded in-flight window.

        ``chunksize`` is accepted for compatibility; tasks are dispatched one by
        one (dynamic load balancing).
        """
        it = iter(iterable)
        pending: Deque[ResilientResult] = deque()
        exhausted = False
        while True:
            while not exhausted and len(pending) < self._window * self._processes:
                try:
                    x = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(self.apply_async(fn, (x,)))
            if not pending:
                return
            yield pending.popleft().get()

    def imap_unordered(self, fn: Callable, iterable: Iterable, chunksize: int = 1) -> Iterator:
        """Results of ``fn(x)`` as they complete, with a bounded in-flight window."""
        from concurrent.futures import FIRST_COMPLETED, wait

        it = iter(iterable)
        pending: List[ResilientResult] = []
        exhausted = False
        while True:
            while not exhausted and len(pending) < self._window * self._processes:
                try:
                    x = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(self.apply_async(fn, (x,)))
            if not pending:
                return
            ready = [r for r in pending if r.ready()]
            if not ready:
                futures = [r._task.future for r in pending if r._task.future is not None]
                if futures:
                    wait(futures, return_when=FIRST_COMPLETED)
                ready = [r for r in pending if r.ready()] or [pending[0]]
            for r in ready:
                pending.remove(r)
                yield r.get()

    def _retire(self) -> None:
        """Leave the run monitor's pool list (once)."""
        handle = getattr(self, "_monitor_handle", None)
        if handle is not None:
            self._monitor_handle = None
            from dreamer.utils import monitoring
            from dreamer.utils.logger import Logger
            monitoring.unregister_pool(handle)
            Logger.event(
                "pool_closed",
                f"[{self._name}] closed after {self.completed} task(s), {self.deaths} worker death(s)",
                Logger.Levels.debug, pool=self._name, completed=self.completed, deaths=self.deaths,
                shrinks=self.shrinks, grows=self.grows, peak_mb=round(self.peak_worker_mb, 1),
            )

    def close(self) -> None:
        """No new tasks; workers finish what is outstanding."""
        self._closed = True

    def join(self) -> None:
        """Wait for the workers to exit (after :meth:`close`)."""
        self._executor.shutdown(wait=True)
        self._retire()

    def terminate(self) -> None:
        """Stop now, discarding outstanding tasks."""
        self._closed = True
        self._outstanding.clear()
        ex = self._executor
        procs = list(getattr(ex, "_processes", {}).values() or [])
        ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        self._retire()

    def __enter__(self) -> "ResilientPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
            self.join()
        else:
            self.terminate()
        return False
