"""
The monitor hub: one daemon thread in the monitored (main) process.

It owns the run's monitoring state and does all monitoring work off the hot
path, sleeping between periodic jobs:

=====================  ==========  ===============================================
job                    default     what
=====================  ==========  ===============================================
drain events           0.25 s      write queued events to ``events.jsonl``
tail worker files      0.5 s       merge ``events.w<pid>.jsonl`` written by workers
sample                 1 s         CPU / memory of the process tree (psutil)
watchdog deadlines     on demand   fire overdue :class:`DeadlineWatcher` watches
stall check            10 s        warn when no progress for ``stall_s``
walltime check         5 s         warnings before the deadline; flush in the grace period
status file            5 s         atomically replace ``status.json`` (full snapshot)
heartbeat              30 s        compact snapshot into ``events.jsonl`` (for replay)
=====================  ==========  ===============================================

Code reports through the hub with :meth:`Hub.publish` / :meth:`Hub.event` (any
thread) and the progress tree (:attr:`Hub.tree`).  Ordinary events are queued
(O(1)) and written by the hub thread; ``ERROR`` and above are written
immediately so they survive a crash that follows them.

The hub can be stopped from a signal handler (see :mod:`runmon.lifecycle`):
:meth:`stop` is idempotent and the locks it takes are re-entrant.

Pools register with :meth:`register_pool`: any object with a
``monitor_info() -> dict`` method (``name``, ``workers``, ``pids`` ...) — held
weakly, so a closed pool disappears on its own.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import socket
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from runmon import walltime as wt
from runmon.events import EventWriter, FileTailer, level_value, make_event
from runmon.progress import ProgressTree
from runmon.sampler import ProcessSampler
from runmon.watchdog import DeadlineWatcher

STATUS_FILE = "status.json"


def fmt_duration(seconds: float) -> str:
    """``2d03h`` / ``4h05m`` / ``12m30s`` / ``45s`` (negative → ``0s``)."""
    s = int(max(0, seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"

EVENTS_FILE = "events.jsonl"
WORKER_GLOB = "events.w*.jsonl"


@dataclass
class HubConfig:
    """Hub timing and limits (seconds unless noted)."""

    sample_s: float = 1.0
    drain_s: float = 0.25
    worker_tail_s: float = 0.5
    status_s: float = 5.0
    heartbeat_s: float = 30.0
    stall_s: float = 1800.0
    walltime_check_s: float = 5.0
    walltime_refresh_s: float = 600.0
    walltime_warn_s: Tuple[float, ...] = (3600.0, 900.0, 300.0)
    walltime_grace_s: float = 300.0
    events_ring: int = 500
    cores: Optional[int] = None
    memory_mb: Optional[float] = None
    memory_probe: Optional[Callable[[], float]] = None
    #: ``(label, elapsed, detail) -> None`` for watchdog fires; default publishes an event.
    on_watchdog: Optional[Callable[[str, float, Optional[str]], None]] = None
    #: Event listeners and grace-period callbacks registered *before* the thread
    #: starts (its first tick may already publish or reach the grace period).
    listeners: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)
    grace_callbacks: List[Callable[[], None]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Hub:
    """Monitoring state + the thread that maintains it (see module docstring).

    :param run_dir: Directory for ``events.jsonl``, ``status.json`` and the
        workers' event files (created if missing).
    :param run_id: Identifier stamped on the run (default: ``<host>-<pid>-<time>``).
    :param config: :class:`HubConfig`.
    """

    def __init__(self, run_dir: str, run_id: Optional[str] = None, config: Optional[HubConfig] = None):
        self.cfg = config or HubConfig()
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.started = time.time()
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.run_id = run_id or f"{self.host}-{self.pid}-{int(self.started)}"
        self.tree = ProgressTree()
        self.watcher = DeadlineWatcher(self.cfg.on_watchdog or self._on_watchdog)
        self.sampler = ProcessSampler(cores=self.cfg.cores, memory_mb=self.cfg.memory_mb,
                                      memory_probe=self.cfg.memory_probe)
        self.writer = EventWriter(os.path.join(self.run_dir, EVENTS_FILE))
        self.ring: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.events_ring)
        self._pending: Deque[Dict[str, Any]] = deque()
        self._pools: Dict[int, "weakref.ref"] = {}
        self._pool_ids = itertools.count()
        self._tailers: Dict[str, FileTailer] = {}
        self._grace_callbacks: List[Callable[[], None]] = list(self.cfg.grace_callbacks)
        self._listeners: List[Callable[[Dict[str, Any]], None]] = list(self.cfg.listeners)
        self.last_sample: Optional[Dict[str, Any]] = None
        self.deadline: Optional[wt.Deadline] = None
        self._deadline_checked = 0.0
        self._walltime_warned: set = set()
        self._grace_done = False
        self._stall_alerted = False
        self._over_alert_at = 0.0
        self.alerts: Dict[str, str] = {}
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._stopped = False
        self._stop_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self.ticks = 0
        self.tick_cost_ms = 0.0

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Dict[str, Any]) -> None:
        """Record an event (any thread).  ``ERROR``+ is written immediately."""
        if not event.get("ctx"):
            path = self.tree.active_path()
            if path:
                event["ctx"] = {"task": path}
        self.ring.append(event)
        self._pending.append(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass
        if level_value(event.get("level", "INFO")) >= 40:
            self._drain()

    def event(self, kind: str, msg: str = "", level: Any = "INFO", **fields: Any) -> None:
        """Build and :meth:`publish` an event."""
        self.publish(make_event(kind, msg, level, **fields))

    def subscribe(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Call ``listener(event)`` for every published event (e.g. the server)."""
        self._listeners.append(listener)

    def register_pool(self, pool: Any) -> int:
        """Track a pool exposing ``monitor_info()``; returns its handle."""
        pid = next(self._pool_ids)
        self._pools[pid] = weakref.ref(pool)
        return pid

    def unregister_pool(self, handle: int) -> None:
        """Stop tracking a pool."""
        self._pools.pop(handle, None)

    def on_grace(self, callback: Callable[[], None]) -> None:
        """Run ``callback()`` (on the hub thread) once the walltime grace period starts."""
        self._grace_callbacks.append(callback)

    def _on_watchdog(self, label: str, elapsed: float, detail: Optional[str]) -> None:
        self.event("watchdog", f"[watchdog] {label} still running after {elapsed:.0f}s"
                   + (f" — {detail}" if detail else ""), "WARNING", label=label, elapsed=elapsed)

    # ------------------------------------------------------------------
    # Thread
    # ------------------------------------------------------------------

    def start(self) -> "Hub":
        """Write ``run_start`` and start the hub thread."""
        self.event("run_start", f"run {self.run_id} started on {self.host} (pid {self.pid})",
                   run=self.run_id, host=self.host, run_dir=self.run_dir, meta=self.cfg.meta)
        self._drain()
        self._thread = threading.Thread(target=self._loop, name="runmon-hub", daemon=True)
        self._thread.start()
        return self

    def wake(self) -> None:
        """Make the hub run its due jobs now."""
        self._wake.set()

    def _jobs(self) -> List[List[Any]]:
        c = self.cfg
        now = time.monotonic()
        return [
            [c.drain_s, now, self._drain],
            [c.worker_tail_s, now, self._tail_workers],
            [c.sample_s, now, self._sample],
            [10.0, now + 10.0, self._check_stall],
            [c.walltime_check_s, now, self._check_walltime],
            [c.status_s, now + c.status_s, self.write_status],
            [c.heartbeat_s, now + c.heartbeat_s, self._heartbeat],
        ]

    def _loop(self) -> None:
        jobs = self._jobs()
        while not self._stopping.is_set():
            t0 = time.monotonic()
            for job in jobs:
                if t0 >= job[1]:
                    job[1] = t0 + job[0]
                    try:
                        job[2]()
                    except Exception as e:      # the monitor must never die
                        try:
                            self.event("monitor_error", f"{job[2].__name__}: {e!r}", "WARNING")
                        except Exception:
                            pass
            self.watcher.check()
            self.ticks += 1
            self.tick_cost_ms = (time.monotonic() - t0) * 1000
            waits = [job[1] - time.monotonic() for job in jobs]
            due = self.watcher.next_due()
            if due is not None:
                waits.append(due)
            self._wake.wait(max(0.01, min(waits)))
            self._wake.clear()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _drain(self) -> None:
        batch = []
        while self._pending:
            try:
                batch.append(self._pending.popleft())
            except IndexError:
                break
        self.writer.write_many(batch)

    def _tail_workers(self) -> None:
        for path in glob.glob(os.path.join(self.run_dir, WORKER_GLOB)):
            tailer = self._tailers.get(path)
            if tailer is None:
                tailer = self._tailers[path] = FileTailer(path)
            for ev in tailer.read_new():
                self.ring.append(ev)
                self._pending.append(ev)
                for listener in list(self._listeners):
                    try:
                        listener(ev)
                    except Exception:
                        pass

    def pools_info(self) -> List[Dict[str, Any]]:
        """``monitor_info()`` of every live registered pool."""
        out = []
        for handle, ref in list(self._pools.items()):
            pool = ref()
            if pool is None:
                self._pools.pop(handle, None)
                continue
            try:
                info = dict(pool.monitor_info())
            except Exception as e:
                info = {"name": repr(pool), "error": repr(e)}
            out.append(info)
        return out

    def _sample(self) -> None:
        for info in self.pools_info():
            for pid in info.get("pids", ()) or ():
                self.sampler.label(pid, f"{info.get('name', 'pool')} worker")
        s = self.sampler.sample()
        self.last_sample = s
        t = s["totals"]
        over = []
        if t.get("over_cpu"):
            over.append(f"CPU {t['cpu_percent']:.0f}% > {100 * s['allocation']['cores']}% allocated")
        if t.get("over_mem"):
            over.append(f"memory {t['rss_mb']:.0f} MB > {s['allocation']['memory_mb']:.0f} MB allocated")
        if over:
            self.alerts["over_allocation"] = "; ".join(over)
            if time.time() - self._over_alert_at > 300:
                self._over_alert_at = time.time()
                self.event("over_allocation", "run uses more than its allocation: " + "; ".join(over),
                           "WARNING", **t)
        else:
            self.alerts.pop("over_allocation", None)

    def _check_stall(self) -> None:
        running = any(r.status == "running" for r in list(self.tree.roots))
        idle = time.time() - self.tree.last_progress
        if running and idle > self.cfg.stall_s:
            self.alerts["stall"] = f"no progress for {fmt_duration(idle)} ({self.tree.active_path()})"
            if not self._stall_alerted:
                self._stall_alerted = True
                self.event("stall", f"no progress reported for {idle:.0f}s", "WARNING",
                           task=self.tree.active_path(), idle_s=idle)
        else:
            if self._stall_alerted:
                self.event("stall_cleared", "progress resumed")
            self._stall_alerted = False
            self.alerts.pop("stall", None)

    def _check_walltime(self) -> None:
        now = time.time()
        if now - self._deadline_checked > self.cfg.walltime_refresh_s:
            self._deadline_checked = now
            old = self.deadline
            self.deadline = wt.detect(start=self.started) or self.deadline
            if self.deadline and (old is None or abs(old.end - self.deadline.end) > 60):
                self.event("walltime", f"job deadline {time.ctime(self.deadline.end)} "
                           f"({fmt_duration(self.deadline.remaining(now))} left, from {self.deadline.source})",
                           end=self.deadline.end, source=self.deadline.source, limit=self.deadline.limit)
        if self.deadline is None:
            return
        left = self.deadline.remaining(now)
        for mark in self.cfg.walltime_warn_s:
            if left <= mark and mark not in self._walltime_warned:
                self._walltime_warned.add(mark)
                self.event("walltime_warning", f"{fmt_duration(left)} of walltime left", "WARNING",
                           remaining_s=left)
        if left <= self.cfg.walltime_grace_s and not self._grace_done:
            self._grace_done = True
            self.alerts["walltime"] = f"walltime ends in {fmt_duration(left)}"
            self.event("walltime_grace", f"walltime grace period: {fmt_duration(left)} left — flushing",
                       "WARNING", remaining_s=left)
            for cb in list(self._grace_callbacks):
                try:
                    cb()
                except Exception as e:
                    self.event("monitor_error", f"grace callback failed: {e!r}", "WARNING")
            self._drain()
            self.writer.sync()
            self.write_status()

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self, events: int = 50, depth: int = 8) -> Dict[str, Any]:
        """Full JSON-ready state (what clients and ``status.json`` get)."""
        now = time.time()
        wall = None
        if self.deadline is not None:
            wall = {"end": self.deadline.end, "remaining": self.deadline.remaining(now),
                    "source": self.deadline.source, "limit": self.deadline.limit}
        sample = self.last_sample
        return {
            "v": 1,
            "type": "snapshot",
            "ts": now,
            "run": {"id": self.run_id, "host": self.host, "pid": self.pid, "dir": self.run_dir,
                    "started": self.started, "uptime": now - self.started, "meta": self.cfg.meta,
                    "stopped": self._stopped},
            "sample_age": (now - sample["ts"]) if sample else None,
            "stage": self.tree.stage(),
            "current": self.tree.active_path(),
            "tasks": self.tree.snapshot(depth),
            "resources": sample,
            "pools": self.pools_info(),
            "walltime": wall,
            "alerts": dict(self.alerts),
            "monitor": {"ticks": self.ticks, "tick_ms": round(self.tick_cost_ms, 2)},
            "events": list(self.ring)[-events:] if events else [],
        }

    def write_status(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Atomically replace ``status.json`` with a snapshot."""
        snap = snapshot or self.snapshot()
        path = os.path.join(self.run_dir, STATUS_FILE)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, default=str)
        os.replace(tmp, path)

    def _heartbeat(self) -> None:
        snap = self.snapshot(events=0, depth=3)
        res = snap.get("resources") or {}
        self.event("heartbeat", "", "DEBUG", stage=snap["stage"], current=snap["current"],
                   tasks=snap["tasks"], totals=res.get("totals"), allocation=res.get("allocation"),
                   machine=res.get("machine"), pools=snap["pools"], walltime=snap["walltime"],
                   alerts=snap["alerts"])

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self, reason: str = "exit", timeout: float = 5.0) -> None:
        """Stop the thread and leave a final record (idempotent; signal-safe)."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stopping.set()
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        for node in list(self.tree.roots):
            if node.status == "running":
                node.close("stopped")           # never closed by the program
        try:
            self._tail_workers()
        except Exception:
            pass
        level = "INFO" if reason == "exit" else "WARNING"
        self.event("run_end", f"run {self.run_id} ended ({reason}) after {time.time() - self.started:.0f}s",
                   level, reason=reason)
        self._drain()
        try:
            self.write_status()
        except Exception:
            pass
        self.writer.close()
