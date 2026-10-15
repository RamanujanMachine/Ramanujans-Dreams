"""
Ramanujan's Dreams ↔ :mod:`runmon` (the live run monitor).

The only module that wires the project into the generic monitor:

* :func:`start_run` / :func:`end_run` — around :meth:`System.run`: one run
  directory per run under ``monitor.RUN_DIR`` (``latest`` points at the newest),
  the run's allocation (:mod:`dreamer.utils.allocation`) as the utilization
  baseline, and the project's log lines copied into the event log.
* :func:`stage` / :func:`shard` — progress nodes for pipeline stages and for
  "this shard × this constant" inside a search / analysis method (the method's
  own loops nest below; :func:`progress` returns the innermost node so a loop
  without a fixed length can report an estimate).
* :func:`worker_run_dir` / :func:`init_worker` — pool workers get their own
  event file (``ResilientPool`` passes the directory to its initializer).
* :func:`watch` — the hang watchdog behind :meth:`Logger.watchdog`.

Everything is a cheap no-op when ``monitor.ENABLED`` is ``False`` or no run is
active, so callers report unconditionally.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import runmon
from runmon.events import JsonlHandler, level_value

from dreamer.configs import config

#: Monitor alerts echoed into run.log / the console (they originate in the hub,
#: not in a log line).
_ECHO_KINDS = {"walltime", "walltime_warning", "walltime_grace", "stall", "stall_cleared",
               "over_allocation", "monitor_error"}

_handler: Optional[logging.Handler] = None
_run_dir: Optional[str] = None


class _EventHandler(JsonlHandler):
    """Copy log records into the event log: structured events always, plain
    lines from ``min_level`` up; never records echoed *from* the monitor."""

    def __init__(self, sink, min_level: int):
        super().__init__(sink, logging.DEBUG)
        self.min_level = min_level

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "runmon_echo", False):
            return
        if record.levelno < self.min_level and not getattr(record, "event", None):
            return
        super().emit(record)


def _sys_logger() -> logging.Logger:
    from dreamer.utils.logger import sys_logger
    return sys_logger


def _log_watchdog(label: str, elapsed: float, detail: Optional[str]) -> None:
    """Watchdog fire → a project warning (console + run.log + event log)."""
    from dreamer.utils.logger import Logger
    Logger.event("watchdog", f"[watchdog] {label} still running after {elapsed:.0f}s"
                 + (f" — {detail}" if detail else ""), Logger.Levels.warning,
                 label=label, elapsed=round(elapsed, 1))


def _echo(event: Dict[str, Any]) -> None:
    """Hub-originated alerts → run.log / console (marked so they are not re-published)."""
    if event.get("kind") not in _ECHO_KINDS or event.get("pid") != os.getpid():
        return
    _sys_logger().log(level_value(event.get("level", "INFO")), f"[monitor] {event.get('msg', '')}",
                      extra={"runmon_echo": True})


# ---------------------------------------------------------------------------
# Run lifecycle (main process)
# ---------------------------------------------------------------------------

def enabled() -> bool:
    """``monitor.ENABLED``."""
    return bool(getattr(config.monitor, "ENABLED", False))


def active() -> bool:
    """A run is being monitored in this process."""
    return runmon.hub() is not None


def run_dir() -> Optional[str]:
    """Directory of the monitored run (``None`` when off)."""
    return _run_dir if active() else None


def _point_latest(base: str, target: str) -> None:
    """Make ``<base>/latest`` point at ``target`` (symlink; text file fallback)."""
    link = os.path.join(base, "latest")
    tmp = f"{link}.tmp{os.getpid()}"
    try:
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(os.path.basename(target), tmp)
        os.replace(tmp, link)
    except OSError:
        try:
            with open(os.path.join(base, "latest.txt"), "w") as f:
                f.write(target + "\n")
        except OSError:
            pass


def start_run(run_id: str, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Start monitoring this run (no-op when disabled); returns the run directory."""
    global _handler, _run_dir
    if not enabled():
        return None
    end_run("restart")
    from dreamer.utils.allocation import allocated_memory_mb, available_cores, describe
    from dreamer.utils.logger import Logger

    cfg = config.monitor
    base = os.path.abspath(cfg.RUN_DIR)
    os.makedirs(base, exist_ok=True)
    directory = os.path.join(base, f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}")
    meta = dict(meta or {})
    meta.setdefault("allocation", describe())
    hub = runmon.start(
        directory, run_id=run_id,
        cores=available_cores(), memory_mb=allocated_memory_mb(),
        sample_s=float(cfg.SAMPLE_SECONDS), heartbeat_s=float(cfg.HEARTBEAT_SECONDS),
        stall_s=float(cfg.STALL_SECONDS), walltime_grace_s=float(cfg.WALLTIME_GRACE_SECONDS),
        on_watchdog=_log_watchdog, listeners=[_echo], grace_callbacks=[_flush_logs], meta=meta,
    )
    _run_dir = hub.run_dir
    _handler = _EventHandler(runmon.publish, level_value(cfg.EVENT_LOG_LEVEL))
    _sys_logger().addHandler(_handler)
    _point_latest(base, directory)
    Logger(f"Run monitor: {directory}  (watch: python -m runmon watch {directory})",
           Logger.Levels.info).log()
    return directory


def end_run(reason: str = "exit") -> None:
    """Stop monitoring (idempotent) and detach the log handler."""
    global _handler
    if _handler is not None:
        _sys_logger().removeHandler(_handler)
        _handler = None
    runmon.stop(reason)


def _flush_logs() -> None:
    """Walltime grace: push every log handler's buffer to disk."""
    for h in list(_sys_logger().handlers):
        try:
            h.flush()
            stream = getattr(h, "stream", None)
            if stream is not None and hasattr(stream, "fileno"):
                os.fsync(stream.fileno())
        except Exception:
            pass


@contextmanager
def monitored_run(run_id: str, meta: Optional[Dict[str, Any]] = None) -> Iterator[Optional[str]]:
    """``with monitored_run(run_id): ...`` — start, and end with the right reason."""
    directory = start_run(run_id, meta)
    reason = "exit"
    try:
        yield directory
    except KeyboardInterrupt:
        reason = "SIGINT"
        raise
    except SystemExit as e:
        reason = f"SystemExit({e.code})"
        raise
    except BaseException as e:
        reason = f"error: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        if directory is not None:
            end_run(reason)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

@contextmanager
def stage(name: str, total: Optional[float] = None) -> Iterator[Any]:
    """A pipeline stage (``loading`` / ``extraction`` / ``analysis`` / ``search`` / ``postprocess``)."""
    with runmon.task(name, total, kind="stage") as node:
        yield node


@contextmanager
def shard(shard_id: Any, constant: Any = None, total: Optional[float] = None,
          label: Optional[str] = None) -> Iterator[Any]:
    """The current shard (× constant) inside a method — the per-shard progress level.

    :param label: Optional prefix naming the activity (e.g. ``"micro-climb"``).
    """
    name = (f"{label} " if label else "") + f"shard {shard_id}" + (
        f" / {getattr(constant, 'name', constant)}" if constant is not None else "")
    with runmon.task(name, total, kind="shard") as node:
        yield node


def progress():
    """Innermost open progress node (a no-op node when off)."""
    return runmon.current_task()


def open_node(name: str, total: Optional[float] = None, kind: str = "loop"):
    """Open a progress node to close later (used by the progress bars)."""
    return runmon.open_task(name, total, kind)


def current_stage() -> Optional[str]:
    """Name of the running stage, if any."""
    h = runmon.hub()
    return h.tree.stage() if h is not None else None


# ---------------------------------------------------------------------------
# Pools and workers
# ---------------------------------------------------------------------------

def register_pool(pool: Any) -> Optional[int]:
    """Track a pool (``monitor_info()``) in the run's snapshots."""
    return runmon.register_pool(pool)


def unregister_pool(handle: Optional[int]) -> None:
    """Stop tracking a pool."""
    runmon.unregister_pool(handle)


def worker_run_dir() -> Optional[str]:
    """Directory workers should write their events to (``None`` when off)."""
    return run_dir()


def init_worker(directory: Optional[str]) -> None:
    """In a pool worker: own event file + watchdog + log lines from ``WORKER_EVENT_LOG_LEVEL``."""
    global _handler
    if not directory or active():
        return
    runmon.init_worker(directory, on_watchdog=_log_watchdog)
    if _handler is None:
        _handler = _EventHandler(runmon.publish, level_value(config.monitor.WORKER_EVENT_LOG_LEVEL))
        _sys_logger().addHandler(_handler)


@contextmanager
def watch(label: str, threshold: float, *, detail: Any = None, repeat: bool = True) -> Iterator[bool]:
    """Watchdog through the monitor; yields ``False`` (and does nothing) when no
    monitor runs in this process, so the caller can fall back."""
    if not runmon.active():
        yield False
        return
    with runmon.watch(label, threshold, detail=detail, repeat=repeat):
        yield True
