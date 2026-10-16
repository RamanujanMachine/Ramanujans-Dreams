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
timeline               60 s        mean / max of the 1 s samples into ``timeline.jsonl``
=====================  ==========  ===============================================

Besides the periodic jobs the hub raises ``memory_pressure`` / ``memory_recovered``
(machine or allocation memory nearly used up) and ``disk_low`` / ``disk_recovered``
(free space where the run directory lives).

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
from runmon.timeline import TIMELINE_FILE, RunSummary, TimelineAggregator, disk_free_mb
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

def _summary_line(s: dict) -> str:
    """``peak CPU 94% / memory 61% of the allocation; jitter CPU 0.31, memory 0.02``."""
    pct = lambda x: "?" if x is None else f"{100 * x:.0f}%"        # noqa: E731
    num = lambda x: "?" if x is None else f"{x:.2f}"               # noqa: E731
    return (f"peak CPU {pct(s['cpu_util_max'])} (mean {pct(s['cpu_util_mean'])}), memory "
            f"{pct(s['mem_util_max'])} (mean {pct(s['mem_util_mean'])}) of the allocation; "
            f"jitter CPU {num(s['cpu_jitter'])}, memory {num(s['rss_jitter'])}")


EVENTS_FILE = "events.jsonl"
WORKER_DIR = "workers"
WORKER_GLOB = "events.w*.jsonl"


@dataclass
class HubConfig:
    """Hub timing and limits (seconds unless noted)."""

    sample_s: float = 1.0
    drain_s: float = 0.25
    worker_tail_s: float = 0.5
    status_s: float = 5.0
    heartbeat_s: float = 30.0
    #: One mean/max summary of the 1 s samples per this many seconds (``timeline.jsonl``).
    timeline_s: float = 60.0
    #: ``memory_pressure`` when machine-available memory is below this fraction of the
    #: machine's memory, or the run uses more than ``mem_high_frac`` of its allocation.
    mem_low_frac: float = 0.08
    mem_high_frac: float = 0.92
    #: ``disk_low`` when the run directory's filesystem has less free space (MB).
    disk_low_mb: float = 2048.0
    stall_s: float = 1800.0
    walltime_check_s: float = 5.0
    walltime_refresh_s: float = 600.0
    walltime_warn_s: Tuple[float, ...] = (3600.0, 900.0, 300.0)
    walltime_grace_s: float = 300.0
    events_ring: int = 500
    #: Serve live snapshots/events over TCP (see :mod:`runmon.server`).
    server: bool = False
    host: str = "auto"
    port: int = 0
    snapshot_s: float = 1.0
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


def _worker_pid(path: str) -> Optional[int]:
    """Pid from ``events.w<pid>.jsonl``."""
    name = os.path.basename(path)
    try:
        return int(name[len("events.w"):-len(".jsonl")])
    except ValueError:
        return None


def _alive(pid: Optional[int]) -> bool:
    """Is the process running (zombies count as gone)?"""
    if pid is None:
        return False
    try:
        import psutil
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


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
        self.tree = ProgressTree(on_change=self._on_task_change)
        self.watcher = DeadlineWatcher(self.cfg.on_watchdog or self._on_watchdog)
        self.sampler = ProcessSampler(cores=self.cfg.cores, memory_mb=self.cfg.memory_mb,
                                      memory_probe=self.cfg.memory_probe)
        self.writer = EventWriter(os.path.join(self.run_dir, EVENTS_FILE))
        self.timeline_writer = EventWriter(os.path.join(self.run_dir, TIMELINE_FILE), lazy=True)
        self.timeline = TimelineAggregator()
        self.summary = RunSummary()
        self._mem_alerted = False
        self._disk_alerted = False
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
        self.server = None
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

    def _on_task_change(self, change: str, node) -> None:
        """Stage nodes leave ``stage_start`` / ``stage_end`` events (timeline markers)."""
        if node.kind != "stage":
            return
        if change == "open":
            ev = make_event("stage_start", f"stage {node.name} started", stage=node.name)
            ev["ts"] = node.started
        else:
            ev = make_event("stage_end", f"stage {node.name} {node.status} after "
                            f"{fmt_duration(node.finished - node.started)}",
                            "INFO" if node.status == "done" else "WARNING",
                            stage=node.name, status=node.status, elapsed=node.finished - node.started)
            ev["ts"] = node.finished
        ev["ctx"] = {"task": node.path()}
        self.publish(ev)

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
        if self.cfg.server:
            try:
                from runmon.server import MonitorServer
                self.server = MonitorServer(self, self.cfg.host, self.cfg.port, self.cfg.snapshot_s)
                self.server.write_info()
                self.event("server", f"monitor server on {self.server.advertised}:{self.server.port}",
                           host=self.server.advertised, port=self.server.port, bind=self.server.bind)
            except Exception as e:       # no server is not a reason to stop the run
                self.server = None
                self.event("monitor_error", f"monitor server failed to start: {e!r}", "WARNING")
        self._drain()
        self._thread = threading.Thread(target=self._loop, name="runmon-hub", daemon=True)
        self._thread.start()
        return self

    def wake(self) -> None:
        """Make the hub run its due jobs now."""
        self._wake.set()
        if self.server is not None:
            self.server.wake()

    def _wait(self, timeout: float) -> None:
        """Sleep until the next job — serving clients meanwhile when there is a server."""
        if self.server is not None and not self.server.closed:
            self.server.poll(timeout)
            return
        self._wake.wait(timeout)
        self._wake.clear()

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
            [c.timeline_s, now + c.timeline_s, self._timeline],
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
            self._wait(max(0.01, min(waits)))

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

    def _tail_workers(self, final: bool = False) -> None:
        """Merge new worker events; delete a worker's file once it is fully merged
        and the worker has exited (its events live on in ``events.jsonl``)."""
        wdir = os.path.join(self.run_dir, WORKER_DIR)
        for path in glob.glob(os.path.join(wdir, WORKER_GLOB)):
            tailer = self._tailers.get(path)
            if tailer is None:
                tailer = self._tailers[path] = FileTailer(path)
            self._merge(tailer.read_new())
            if tailer.caught_up() and not _alive(_worker_pid(path)):
                self._drain()                       # merged copy on disk before removing
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._tailers.pop(path, None)
        if final:
            try:
                os.rmdir(wdir)                      # only if empty
            except OSError:
                pass

    def _merge(self, events: List[Dict[str, Any]]) -> None:
        """Add worker events to the run's log, ring and listeners."""
        for ev in events:
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
        pools = self.pools_info()
        for info in pools:
            for pid in info.get("pids", ()) or ():
                self.sampler.label(pid, f"{info.get('name', 'pool')} worker")
        s = self.sampler.sample()
        # What the program actually allowed itself (pools start small, grow and
        # shrink with memory) — next to the allocation budget.
        open_pools = [p for p in pools if not p.get("closed")]
        s["pool_workers"] = {"workers": sum(int(p.get("workers") or 0) for p in open_pools),
                             "requested": sum(int(p.get("requested") or 0) for p in open_pools),
                             "pools": len(open_pools)}
        self.last_sample = s
        self.timeline.add(s)
        self.summary.add(s)
        self.timeline.add_pools(pools)
        t = s["totals"]
        self._check_memory(s)
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

    def _check_memory(self, s: Dict[str, Any]) -> None:
        """``memory_pressure`` / ``memory_recovered`` with hysteresis (clears at 1.5x the limit)."""
        machine = s.get("machine") or {}
        total, avail = machine.get("mem_total_mb"), machine.get("mem_available_mb")
        alloc = (s.get("allocation") or {}).get("memory_mb")
        rss = (s.get("totals") or {}).get("rss_mb") or 0.0
        low_machine = bool(total) and avail is not None and avail < self.cfg.mem_low_frac * total
        high_alloc = bool(alloc) and rss > self.cfg.mem_high_frac * alloc
        calm = ((not total or avail is None or avail > 1.5 * self.cfg.mem_low_frac * total)
                and (not alloc or rss < self.cfg.mem_high_frac * alloc * 0.95))
        if (low_machine or high_alloc) and not self._mem_alerted:
            self._mem_alerted = True
            why = []
            if low_machine:
                why.append(f"machine has {avail:.0f} MB available of {total:.0f} MB")
            if high_alloc:
                why.append(f"run uses {rss:.0f} MB of its {alloc:.0f} MB allocation")
            self.alerts["memory_pressure"] = "; ".join(why)
            self.event("memory_pressure", "memory nearly used up: " + "; ".join(why), "WARNING",
                       rss_mb=rss, available_mb=avail, total_mb=total, allocation_mb=alloc,
                       biggest=sorted(((p["rss_mb"], p["pid"], p["role"]) for p in s.get("processes") or ()),
                                      reverse=True)[:3])
        elif self._mem_alerted and calm:
            self._mem_alerted = False
            self.alerts.pop("memory_pressure", None)
            self.event("memory_recovered", "memory pressure over", rss_mb=rss, available_mb=avail)

    def _timeline(self, final: bool = False) -> None:
        """Write the mean / max record of the last window; check the disk."""
        free = disk_free_mb(self.run_dir)
        if free is not None:
            if free < self.cfg.disk_low_mb and not self._disk_alerted:
                self._disk_alerted = True
                self.alerts["disk_low"] = f"{free:.0f} MB free"
                self.event("disk_low", f"only {free:.0f} MB free where the run directory lives", "WARNING",
                           free_mb=free, run_dir=self.run_dir)
            elif free > 2 * self.cfg.disk_low_mb and self._disk_alerted:
                self._disk_alerted = False
                self.alerts.pop("disk_low", None)
                self.event("disk_recovered", f"{free:.0f} MB free again", free_mb=free)
        rec = self.timeline.flush(time.time(), stage=self.tree.stage(), disk_free_mb=free)
        if rec is not None:
            self.timeline_writer.write(rec)

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
            self._timeline()                   # the partial window is worth keeping
            self.writer.sync()
            self.timeline_writer.sync()
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
            "summary": self.summary.as_dict(),
            "pools": self.pools_info(),
            "walltime": wall,
            "alerts": dict(self.alerts),
            "monitor": {"ticks": self.ticks, "tick_ms": round(self.tick_cost_ms, 2),
                        "clients": sum(1 for c in list(self.server.clients.values()) if c.authed)
                        if self.server is not None and not self.server.closed else 0},
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
                   pool_workers=res.get("pool_workers"),
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
        self.wake()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        for node in list(self.tree.roots):
            if node.status == "running":
                node.close("stopped")           # never closed by the program
        try:
            self._tail_workers(final=True)
        except Exception:
            pass
        s = self.summary.as_dict()
        if s["samples"]:
            self.event("resource_summary", _summary_line(s), **s)
        level = "INFO" if reason == "exit" else "WARNING"
        self.event("run_end", f"run {self.run_id} ended ({reason}) after {time.time() - self.started:.0f}s",
                   level, reason=reason)
        self._drain()
        try:
            self._timeline(final=True)
        except Exception:
            pass
        try:
            self.write_status()
        except Exception:
            pass
        if self.server is not None:
            try:
                self.server.close(reason)          # pushes run_end, then bye
            except Exception:
                pass
        self.writer.close()
        self.timeline_writer.close()
