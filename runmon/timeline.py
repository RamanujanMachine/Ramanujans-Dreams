"""
The run's resource timeline: one summary record per interval (default 60 s).

The hub samples the process tree every second, but only the *live* views keep
those samples.  Writing them all would be a lot of data; writing one snapshot
per interval (what the heartbeat does) loses every spike in between.  The
timeline is the middle way: every ``interval`` seconds the 1 s samples of that
window are reduced to their **mean** (the shape of the curve) and **max** (the
spikes — what matters for out-of-memory questions) and appended to
``timeline.jsonl`` as one JSON line.  A 14-day run is ~20 000 short lines.

Record fields (``ts`` = window end; all rates are means over the window unless
named ``*_max`` / ``*_min``):

========================  ====================================================
``ts``, ``t0``, ``n``     window end, window start, number of 1 s samples
``cpu`` / ``cpu_max``     total CPU of the process tree, % of one core
``cpu_std``, ``cpu_jitter``  spread of the 1 s samples, and how spiky they are
                          (mean |step| ÷ mean; 0 = flat, see :class:`_Stat`)
``rss_mb`` / ``rss_max``  total resident memory (MB)
``rss_std``, ``rss_jitter``  the same for memory
``cpu_util_max``          highest CPU use as a fraction of the allocated cores
``mem_util_max``          highest memory use as a fraction of the allocated memory
``procs`` / ``procs_max`` processes in the tree
``workers``/``workers_max`` pool workers the program allowed itself
``mem_avail_min``         lowest machine-available memory (MB) in the window
``cores`` / ``memory_mb`` the allocation (budget) the run was given
``tasks``                 pool tasks completed in the window
``task_s``                worker-seconds those tasks ran (``task_s / tasks`` = mean task time)
``spawned``               processes that appeared in the tree (worker restarts / recycling)
``roles``                 per role (``"eval pool worker"``, ``main`` ...):
                          ``{cpu, rss_mb, rss_max, n}``
``stage``                 active stage at the end of the window
``disk_free_mb``          free space where the run directory lives
========================  ====================================================
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Iterable, Iterator, List, Optional

TIMELINE_FILE = "timeline.jsonl"


class _Stat:
    """Running mean / max / min, standard deviation and jitter of a stream of numbers.

    *Jitter* is the mean absolute change between consecutive values divided by
    the mean: 0 for a flat line, ~0.5 when a typical 1 s step is half the level.
    Unlike the standard deviation it ignores slow drifts and measures how spiky
    the signal is from one sample to the next.
    """

    __slots__ = ("n", "total", "hi", "lo", "m2", "mean_", "last", "steps", "step_sum")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.hi = float("-inf")
        self.lo = float("inf")
        self.m2 = 0.0            # Welford: sum of squared deviations
        self.mean_ = 0.0
        self.last: Optional[float] = None
        self.steps = 0
        self.step_sum = 0.0

    def add(self, x: Optional[float]) -> None:
        if x is None:
            return
        self.n += 1
        self.total += x
        self.hi = max(self.hi, x)
        self.lo = min(self.lo, x)
        delta = x - self.mean_
        self.mean_ += delta / self.n
        self.m2 += delta * (x - self.mean_)
        if self.last is not None:
            self.steps += 1
            self.step_sum += abs(x - self.last)
        self.last = x

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.n if self.n else None

    @property
    def max(self) -> Optional[float]:
        return self.hi if self.n else None

    @property
    def min(self) -> Optional[float]:
        return self.lo if self.n else None

    @property
    def std(self) -> Optional[float]:
        return (self.m2 / self.n) ** 0.5 if self.n else None

    @property
    def jitter(self) -> Optional[float]:
        """Mean absolute step ÷ mean (``None`` with fewer than 2 values or a zero mean)."""
        if not self.steps or not self.total:
            return None
        return (self.step_sum / self.steps) / (self.total / self.n)


def _r(x: Optional[float], digits: int = 1) -> Optional[float]:
    return None if x is None else round(x, digits)


class TimelineAggregator:
    """Reduce 1 s samples to one record per window.

    :meth:`add` every sample; :meth:`flush` turns the samples since the last
    flush into a record (``None`` if there were none).
    """

    def __init__(self):
        self._reset(None)
        self._last_completed: Dict[str, int] = {}
        self._last_seconds: Dict[str, float] = {}
        self._known_pids: Optional[set] = None

    def _reset(self, t0: Optional[float]) -> None:
        self.t0 = t0
        self.n = 0
        self.cpu, self.rss, self.procs = _Stat(), _Stat(), _Stat()
        self.cpu_util, self.mem_util = _Stat(), _Stat()
        self.workers, self.avail = _Stat(), _Stat()
        self.roles: Dict[str, Dict[str, _Stat]] = {}
        self.alloc: Dict[str, Any] = {}
        self.tasks = 0
        self.task_s = 0.0
        self.spawned = 0

    def add(self, sample: Dict[str, Any]) -> None:
        """Account one :meth:`runmon.sampler.ProcessSampler.sample` result."""
        if self.t0 is None:
            self.t0 = sample.get("ts")
        self.n += 1
        t = sample.get("totals") or {}
        self.cpu.add(t.get("cpu_percent"))
        self.rss.add(t.get("rss_mb"))
        self.procs.add(t.get("n"))
        self.cpu_util.add(t.get("cpu_util"))
        self.mem_util.add(t.get("mem_util"))
        self.workers.add((sample.get("pool_workers") or {}).get("workers"))
        self.avail.add((sample.get("machine") or {}).get("mem_available_mb"))
        a = sample.get("allocation") or {}
        self.alloc = {"cores": a.get("cores"), "memory_mb": a.get("memory_mb")}
        pids = {p.get("pid") for p in sample.get("processes") or ()}
        if self._known_pids is not None:           # the first sample is the baseline
            self.spawned += len(pids - self._known_pids)
        self._known_pids = pids
        per_role: Dict[str, List[float]] = {}
        for p in sample.get("processes") or ():
            acc = per_role.setdefault(p.get("role") or "child", [0.0, 0.0, 0])
            acc[0] += p.get("cpu") or 0.0
            acc[1] += p.get("rss_mb") or 0.0
            acc[2] += 1
        for role, (cpu, rss, n) in per_role.items():
            stats = self.roles.setdefault(role, {"cpu": _Stat(), "rss": _Stat(), "n": _Stat()})
            stats["cpu"].add(cpu)
            stats["rss"].add(rss)
            stats["n"].add(n)

    def add_pools(self, pools: Iterable[Dict[str, Any]]) -> None:
        """Count tasks completed since the last call (pool counters reset when a
        pool is replaced; a counter that went down counts from zero)."""
        for info in pools:
            name = str(info.get("name"))
            done = int(info.get("completed") or 0)
            last = self._last_completed.get(name, 0)
            self.tasks += done - last if done >= last else done
            self._last_completed[name] = done
            secs = float(info.get("task_seconds") or 0.0)
            last_s = self._last_seconds.get(name, 0.0)
            self.task_s += secs - last_s if secs >= last_s else secs
            self._last_seconds[name] = secs

    def flush(self, now: float, *, stage: Optional[str] = None,
              disk_free_mb: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The record for the window so far, and start a new window."""
        if not self.n:
            self._reset(None)
            return None
        rec = {
            "ts": now, "t0": self.t0, "n": self.n,
            "cpu": _r(self.cpu.mean), "cpu_max": _r(self.cpu.max),
            "cpu_std": _r(self.cpu.std), "cpu_jitter": _r(self.cpu.jitter, 3),
            "rss_mb": _r(self.rss.mean), "rss_max": _r(self.rss.max),
            "rss_std": _r(self.rss.std), "rss_jitter": _r(self.rss.jitter, 3),
            "cpu_util_max": _r(self.cpu_util.max, 3), "mem_util_max": _r(self.mem_util.max, 3),
            "procs": _r(self.procs.mean), "procs_max": self.procs.max,
            "workers": _r(self.workers.mean), "workers_max": self.workers.max,
            "mem_avail_min": self.avail.min,
            **self.alloc,
            "tasks": self.tasks, "task_s": _r(self.task_s), "spawned": self.spawned,
            "roles": {role: {"cpu": _r(s["cpu"].mean), "rss_mb": _r(s["rss"].mean),
                             "rss_max": _r(s["rss"].max), "n": s["n"].max}
                      for role, s in self.roles.items()},
            "stage": stage,
            "disk_free_mb": _r(disk_free_mb, 0),
        }
        self._reset(None)
        return rec


class RunSummary:
    """Whole-run figures from every 1 s sample (never reset): peaks, averages, jitter.

    ``cpu_util`` / ``mem_util`` are fractions of the run's *allocation* (cores,
    memory); ``jitter`` is defined at :class:`_Stat`.  Computed from the same
    samples as the timeline, so it covers the whole run, not just the windows
    that happened to be written.
    """

    def __init__(self):
        self.samples = 0
        self.cpu, self.rss = _Stat(), _Stat()
        self.cpu_util, self.mem_util = _Stat(), _Stat()

    def add(self, sample: Dict[str, Any]) -> None:
        t = sample.get("totals") or {}
        self.samples += 1
        self.cpu.add(t.get("cpu_percent"))
        self.rss.add(t.get("rss_mb"))
        self.cpu_util.add(t.get("cpu_util"))
        self.mem_util.add(t.get("mem_util"))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready summary (``None`` fields when nothing was sampled)."""
        return {
            "samples": self.samples,
            "cpu_util_max": _r(self.cpu_util.max, 3), "cpu_util_mean": _r(self.cpu_util.mean, 3),
            "mem_util_max": _r(self.mem_util.max, 3), "mem_util_mean": _r(self.mem_util.mean, 3),
            "cpu_max": _r(self.cpu.max), "cpu_mean": _r(self.cpu.mean),
            "rss_max": _r(self.rss.max), "rss_mean": _r(self.rss.mean),
            "cpu_jitter": _r(self.cpu.jitter, 3), "rss_jitter": _r(self.rss.jitter, 3),
            "cpu_std": _r(self.cpu.std), "rss_std": _r(self.rss.std),
        }


def disk_free_mb(path: str) -> Optional[float]:
    """Free space (MB) on the filesystem holding ``path`` (``None`` if unknown)."""
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def parse_lines(lines: Iterable[bytes]) -> Iterator[Dict[str, Any]]:
    """Records from raw JSONL lines (a torn or invalid line is skipped)."""
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if isinstance(rec, dict) and "ts" in rec:
            yield rec


def lines_local(path: str) -> Iterator[bytes]:
    """Lines of the timeline of a run directory (or of the file itself); nothing
    if the run has none (older runs)."""
    if os.path.isdir(path):
        path = os.path.join(path, TIMELINE_FILE)
    try:
        with open(path, "rb") as f:
            yield from f
    except OSError:
        return
