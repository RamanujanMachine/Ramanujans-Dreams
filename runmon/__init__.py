"""
runmon — lightweight live monitoring for long multi-process Python runs.

Generic (no project imports): a run records a structured, searchable event log
(``events.jsonl``), a progress tree, per-process CPU / memory against its
allocation, hang watchdogs and its batch-job walltime — all maintained by **one
daemon thread** in the main process (:class:`~runmon.hub.Hub`), so no core is
taken from the computation.

Usage in the main process::

    import runmon
    runmon.start("runs/2026-10-15", cores=16, memory_mb=64_000)
    with runmon.task("search", total=len(shards), kind="stage") as stage:
        for shard in shards:
            with runmon.task(f"shard {shard.id}", kind="shard") as node:
                ...
                node.estimate(0.4, T=0.12)     # loops without a fixed count
            stage.advance()
    runmon.event("pool_created", "eval pool with 8 workers", workers=8)
    runmon.stop()

In a worker process (e.g. a pool initializer)::

    runmon.init_worker(run_dir)           # own events.w<pid>.jsonl + watchdog thread
    runmon.event("task_failed", "...", level="WARNING")

Every function is a cheap no-op when monitoring was not started, so library
code can report unconditionally.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from runmon import events as _events
from runmon import lifecycle as _lifecycle
from runmon.events import EventWriter, make_event
from runmon.hub import Hub, HubConfig
from runmon.progress import TaskNode
from runmon.watchdog import DeadlineWatcher

__all__ = [
    "Hub", "HubConfig", "TaskNode", "start", "stop", "active", "hub", "event", "publish",
    "task", "open_task", "current_task", "watch", "register_pool", "unregister_pool",
    "on_grace", "init_worker", "worker_file",
]

_hub: Optional[Hub] = None
_worker_writer: Optional[EventWriter] = None
_worker_watcher: Optional[DeadlineWatcher] = None


class _NullTask:
    """Stand-in returned when monitoring is off: accepts every TaskNode call."""

    id = 0
    name = ""
    total = None
    done = 0.0
    metrics: Dict[str, Any] = {}

    def advance(self, n: float = 1) -> None: ...
    def set(self, done=None, total=None) -> None: ...
    def estimate(self, fraction, **metrics) -> None: ...
    def metric(self, **metrics) -> None: ...
    def close(self, status: str = "done") -> None: ...
    def fraction(self): return None
    def __enter__(self): return self
    def __exit__(self, *exc): return False


NULL_TASK = _NullTask()


def _h() -> Optional[Hub]:
    """The hub of *this* process (a ``fork``-ed child inherits the global, not the thread)."""
    h = _hub
    return h if h is not None and h.pid == os.getpid() else None


# ---------------------------------------------------------------------------
# Main process
# ---------------------------------------------------------------------------

def start(run_dir: str, *, run_id: Optional[str] = None, config: Optional[HubConfig] = None,
          handle_signals: bool = True, **config_kwargs: Any) -> Hub:
    """Start monitoring this process (idempotent per run: a running hub is stopped first).

    :param run_dir: Directory for the run's monitor files.
    :param handle_signals: Install exit/signal handlers that leave a final record
        (:mod:`runmon.lifecycle`); main thread only.
    :param config_kwargs: :class:`HubConfig` fields, if ``config`` is not given.
    """
    global _hub
    if _hub is not None:
        stop("restart")
    cfg = config or HubConfig(**config_kwargs)
    _events.set_role("main")
    _hub = Hub(run_dir, run_id, cfg).start()
    if handle_signals:
        _lifecycle.install(stop, crash_file=os.path.join(_hub.run_dir, "crash.txt"))
    return _hub


def stop(reason: str = "exit") -> None:
    """Stop monitoring and write the final record (idempotent)."""
    global _hub
    h, _hub = _h(), None
    if h is not None:
        if reason.startswith("SIG"):
            h.event("signal_received", f"{reason} received", "WARNING", signal=reason,
                    task=h.tree.active_path(), uptime_s=time.time() - h.started)
        h.stop(reason)
        _lifecycle.uninstall()


def active() -> bool:
    """``True`` when this process has a hub or a worker event file."""
    return _h() is not None or _worker_writer is not None


def hub() -> Optional[Hub]:
    """The running hub (main process), if any."""
    return _h()


def publish(ev: Dict[str, Any]) -> None:
    """Record a prebuilt event (hub in the main process, own file in a worker)."""
    h = _h()
    if h is not None:
        h.publish(ev)
    elif _worker_writer is not None:
        _worker_writer.write(ev)


def event(kind: str, msg: str = "", level: Any = "INFO", **fields: Any) -> None:
    """Record a structured event; no-op when monitoring is off."""
    if _h() is None and _worker_writer is None:
        return
    publish(make_event(kind, msg, level, **fields))


def open_task(name: str, total: Optional[float] = None, kind: str = "task"):
    """Open a progress node under the current one (close it with ``.close()``)."""
    h = _h()
    return h.tree.open(name, total, kind) if h is not None else NULL_TASK


@contextmanager
def task(name: str, total: Optional[float] = None, kind: str = "task") -> Iterator[Any]:
    """``with runmon.task("search", total=n, kind="stage") as node:`` (no-op node when off)."""
    h = _h()
    if h is None:
        yield NULL_TASK
        return
    with h.tree.task(name, total, kind) as node:
        yield node


def current_task():
    """Innermost open progress node of this thread (null node when off / none)."""
    h = _h()
    return (h.tree.current() or NULL_TASK) if h is not None else NULL_TASK


@contextmanager
def watch(label: str, threshold: float, *, detail: Any = None, repeat: bool = True) -> Iterator[None]:
    """Warn (event ``watchdog``) while the block runs past ``threshold`` seconds."""
    h = _h()
    watcher = h.watcher if h is not None else _worker_watcher
    if watcher is None or not threshold or threshold <= 0:
        yield
        return
    with watcher.guard(label, threshold, detail=detail, repeat=repeat):
        yield


def register_pool(pool: Any) -> Optional[int]:
    """Track a pool with ``monitor_info()`` (main process); returns a handle or ``None``."""
    h = _h()
    return h.register_pool(pool) if h is not None else None


def unregister_pool(handle: Optional[int]) -> None:
    """Stop tracking a pool."""
    h = _h()
    if h is not None and handle is not None:
        h.unregister_pool(handle)


def on_grace(callback: Callable[[], None]) -> None:
    """Run ``callback`` when the walltime grace period starts (main process)."""
    h = _h()
    if h is not None:
        h.on_grace(callback)


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def worker_file(run_dir: str, pid: Optional[int] = None) -> str:
    """Path of a worker's own event file (``<run_dir>/workers/events.w<pid>.jsonl``)."""
    return os.path.join(os.path.abspath(run_dir), "workers", f"events.w{pid or os.getpid()}.jsonl")


def init_worker(run_dir: str, role: str = "worker",
                on_watchdog: Optional[Callable[[str, float, Optional[str]], None]] = None) -> None:
    """Set up monitoring in a worker: its own event file + a one-thread watchdog.

    Idempotent per process.  ``on_watchdog`` overrides the default (an event in
    the worker's file).
    """
    global _worker_writer, _worker_watcher
    if _h() is not None:                       # a hub process is never a worker
        return
    path = worker_file(run_dir)
    if _worker_writer is not None and _worker_writer.path == path:
        return
    _events.set_role(role)
    _worker_writer = EventWriter(path, lazy=True)     # no file unless the worker reports

    def _fire(label: str, elapsed: float, detail: Optional[str]) -> None:
        event("watchdog", f"[watchdog] {label} still running after {elapsed:.0f}s"
              + (f" — {detail}" if detail else ""), "WARNING", label=label, elapsed=elapsed)

    _worker_watcher = DeadlineWatcher(on_watchdog or _fire)
    _worker_watcher.start_thread()
