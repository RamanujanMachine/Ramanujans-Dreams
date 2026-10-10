"""
Memory-aware worker counts.

Cores say how many workers we *could* run; memory says how many *fit*.  Every
:class:`~dreamer.utils.resilient_pool.ResilientPool` applies::

    workers = min(cores, floor((available_mb - reserve_mb) / per_worker_mb))

* ``available_mb`` — memory the OS can give us now: Linux ``MemAvailable``
  (capped by a cgroup v1/v2 limit, which is what SLURM / containers enforce),
  macOS ``vm_stat`` (free + inactive + speculative pages), else half of physical
  memory.  ``system.MEMORY_LIMIT_MB`` replaces the detection with a budget for
  this run: available = limit − RSS of the main process and its live workers
  (shared nodes, reproducible tests).
* ``per_worker_mb`` — ``system.WORKER_MEMORY_MB`` if set, else the **measured**
  RSS of a fresh worker that imported the trajectory machinery (measured once
  per process, ~130 MB) plus ``system.WORKER_TASK_HEADROOM_MB`` for the task's
  own data (walked matrices, symbolic matrix, caches).
* ``reserve_mb`` — ``system.MEMORY_RESERVE_MB`` kept free for the main process
  and the OS.

During a run the pool also re-checks available memory and **shrinks** before
the OS starts killing workers (see ``ResilientPool``); worker deaths still halve
it as a last resort.  ``system.MEMORY_AWARE_WORKERS = False`` disables all of
this (cores only).
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Iterable, Optional

from dreamer.configs import config

_MB = 1024 * 1024
_measured_worker_mb: Optional[float] = None


# ---------------------------------------------------------------------------
# Available / total memory
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def parse_meminfo_available_mb(text: str) -> Optional[float]:
    """``MemAvailable`` (MB) from ``/proc/meminfo`` contents."""
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.M)
    return int(m.group(1)) / 1024 if m else None


def parse_vm_stat_available_mb(text: str) -> Optional[float]:
    """Free + inactive + speculative pages (MB) from macOS ``vm_stat`` output."""
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    size = int(page.group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"^{key}:\s+(\d+)\.", text, re.M)
        if m:
            pages += int(m.group(1))
    return pages * size / _MB


def cgroup_available_mb(root: str = "/sys/fs/cgroup") -> Optional[float]:
    """Memory left under a cgroup limit (v2 ``memory.max``, v1 ``limit_in_bytes``), MB."""
    for limit_file, usage_file in (
        (f"{root}/memory.max", f"{root}/memory.current"),
        (f"{root}/memory/memory.limit_in_bytes", f"{root}/memory/memory.usage_in_bytes"),
    ):
        limit, usage = _read(limit_file), _read(usage_file)
        if limit is None or usage is None:
            continue
        limit = limit.strip()
        if limit == "max" or not limit.isdigit() or int(limit) >= 1 << 60:
            continue  # no limit set
        return max(0.0, (int(limit) - int(usage.strip())) / _MB)
    return None


def total_memory_mb() -> float:
    """Physical memory of the machine (MB)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _MB
    except (ValueError, OSError, AttributeError):
        return 4096.0


def available_memory_mb() -> float:
    """Memory available to this run now (MB); see the module docstring."""
    override = getattr(config.system, "MEMORY_LIMIT_MB", None)
    if override:
        # A budget for this run: what is left after the main process and its
        # current worker processes.
        import multiprocessing as mp
        used = rss_mb([os.getpid()] + [p.pid for p in mp.active_children()])
        return max(0.0, float(override) - used)
    candidates = []
    meminfo = _read("/proc/meminfo")
    if meminfo is not None:
        v = parse_meminfo_available_mb(meminfo)
        if v is not None:
            candidates.append(v)
        cg = cgroup_available_mb()
        if cg is not None:
            candidates.append(cg)
    else:
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            v = parse_vm_stat_available_mb(out)
            if v is not None:
                candidates.append(v)
        except (OSError, subprocess.SubprocessError):
            pass
    return min(candidates) if candidates else total_memory_mb() / 2


def rss_mb(pids: Iterable[int]) -> float:
    """Total resident memory (MB) of the given processes (0 for vanished ones)."""
    pids = [str(p) for p in pids]
    if not pids:
        return 0.0
    total = 0.0
    for pid in pids:
        status = _read(f"/proc/{pid}/status")
        if status is not None:
            m = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.M)
            total += int(m.group(1)) / 1024 if m else 0.0
            continue
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True, timeout=5).stdout
            total += int(out.strip() or 0) / 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return total


# ---------------------------------------------------------------------------
# Per-worker estimate
# ---------------------------------------------------------------------------

def _probe_worker_rss() -> float:
    """Task run in a fresh worker: import the trajectory machinery, report RSS."""
    import dreamer.utils.stage_pool  # noqa: F401
    import dreamer.utils.storage.trajectory_attributes  # noqa: F401
    import dreamer.search.methods.flatland.parallel_eval_worker  # noqa: F401
    return rss_mb([os.getpid()])


def measured_worker_baseline_mb() -> float:
    """RSS of a fresh worker after importing what tasks need (measured once)."""
    global _measured_worker_mb
    if _measured_worker_mb is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(1, mp_context=mp.get_context()) as ex:
                _measured_worker_mb = float(ex.submit(_probe_worker_rss).result(timeout=120))
        except Exception:
            _measured_worker_mb = 250.0   # conservative fallback
    return _measured_worker_mb


def per_worker_mb() -> float:
    """Memory budget per worker: configured, or measured baseline + task headroom."""
    fixed = getattr(config.system, "WORKER_MEMORY_MB", None)
    if fixed:
        return float(fixed)
    return measured_worker_baseline_mb() + float(getattr(config.system, "WORKER_TASK_HEADROOM_MB", 512))


def memory_worker_cap(requested: int, *, log: bool = True) -> int:
    """Cap a requested worker count by what fits in available memory (>= 1).

    :param requested: Workers the cores would allow.
    :param log: Log the decision when it reduces the count.
    :return: ``min(requested, floor((available - reserve) / per_worker))``,
        floored at 1; *requested* unchanged when ``MEMORY_AWARE_WORKERS`` is off.
    """
    requested = max(1, int(requested))
    if not getattr(config.system, "MEMORY_AWARE_WORKERS", True) or requested <= 1:
        return requested
    avail = available_memory_mb()
    reserve = float(getattr(config.system, "MEMORY_RESERVE_MB", 1024))
    each = per_worker_mb()
    fit = int((avail - reserve) // each) if each > 0 else requested
    cap = max(1, min(requested, fit))
    if log and cap < requested:
        from dreamer.utils.logger import Logger
        Logger(
            f"Memory-aware worker count: {cap} of {requested} (available {avail:.0f} MB, "
            f"reserve {reserve:.0f} MB, ~{each:.0f} MB per worker)",
            Logger.Levels.info,
        ).log()
    return cap
