"""
The cores and memory this run was *allocated* — not what the machine has.

Why
---
On a batch cluster a job is granted part of a node, but the scheduler does not
always confine it.  On the Technion Zeus PBS cluster (measured 2026-10-15) a job
with ``ncpus=4`` on an 80-core node sees ``os.cpu_count() == 80`` *and*
``len(os.sched_getaffinity(0)) == 80`` (no cpuset) and the whole node's memory
(no memory cgroup).  Sizing pools from those numbers starts ~80 workers on a
4-core allocation and plans memory for the whole node — taking CPU and RAM that
belong to other users' jobs.

Resolution order
----------------
Cores (:func:`available_cores`):

1. ``system.TOTAL_CORES`` when set (explicit user choice wins);
2. the scheduler allocation — PBS ``$NCPUS`` (inside a job, ``$PBS_JOBID`` set),
   Slurm ``$SLURM_CPUS_PER_TASK`` / ``$SLURM_JOB_CPUS_PER_NODE``;
3. the CPU affinity mask (honours cpusets / ``taskset`` where they exist);
4. ``os.cpu_count()``.

The smallest of 2-4 that is known is used, so a confined job never exceeds its
mask and an unconfined one never exceeds its request.

Memory (:func:`allocated_memory_mb`): ``system.MEMORY_LIMIT_MB`` when set, else
the scheduler request — PBS ``Resource_List.mem`` (read once with
``qstat -f $PBS_JOBID``), Slurm ``$SLURM_MEM_PER_NODE`` or
``$SLURM_MEM_PER_CPU × cores`` — else ``None`` (no allocation: the machine's
available memory applies, see :mod:`dreamer.utils.memory_budget`).

The PBS lookup costs one ``qstat`` call per process tree: the result is cached
and exported in ``RD_ALLOCATED_MEMORY_MB`` so worker processes inherit it
instead of calling ``qstat`` again.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Optional

#: Environment variable carrying the resolved memory allocation to child processes.
_MEM_ENV = "RD_ALLOCATED_MEMORY_MB"

#: Per-process cache for the scheduler memory lookup (``False`` = not resolved yet).
_mem_cache: object = False

#: Threading variables of numeric libraries that schedulers set to the core count.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_UNIT_MB = {"b": 1 / 2**20, "kb": 1 / 1024, "mb": 1.0, "gb": 1024.0, "tb": 1024.0**2}


# ---------------------------------------------------------------------------
# Scheduler detection
# ---------------------------------------------------------------------------

def scheduler(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Name of the batch scheduler this process runs under (``"pbs"``/``"slurm"``), else ``None``."""
    env = os.environ if env is None else env
    if env.get("PBS_JOBID"):
        return "pbs"
    if env.get("SLURM_JOB_ID"):
        return "slurm"
    return None


def _positive_int(value: Optional[str]) -> Optional[int]:
    """``int(value)`` if it is a positive integer string, else ``None``."""
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _slurm_cpus_per_node(value: Optional[str]) -> Optional[int]:
    """First node's count from ``SLURM_JOB_CPUS_PER_NODE`` (e.g. ``"16"`` or ``"16(x2),8"``)."""
    if not value:
        return None
    m = re.match(r"\s*(\d+)", value)
    return _positive_int(m.group(1)) if m else None


def scheduler_cores(env: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Cores the scheduler granted this job on this node, or ``None`` outside a job."""
    env = os.environ if env is None else env
    kind = scheduler(env)
    if kind == "pbs":
        return _positive_int(env.get("NCPUS"))
    if kind == "slurm":
        return (_positive_int(env.get("SLURM_CPUS_PER_TASK"))
                or _slurm_cpus_per_node(env.get("SLURM_JOB_CPUS_PER_NODE")))
    return None


def affinity_cores() -> Optional[int]:
    """Number of CPUs in this process' affinity mask (``None`` where unsupported, e.g. macOS)."""
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------

def allocated_cores(env: Optional[Dict[str, str]] = None) -> int:
    """Cores this process may use, ignoring ``system.TOTAL_CORES`` (see module docstring)."""
    known = [n for n in (scheduler_cores(env), affinity_cores(), os.cpu_count()) if n]
    return max(1, min(known)) if known else 1


def available_cores() -> int:
    """Core budget for the run's pools: ``system.TOTAL_CORES``, else :func:`allocated_cores`.

    Single source of truth for every process pool (stage pools, eval pools,
    extraction).  Floored at 1.
    """
    from dreamer.configs.system import sys_config
    total = getattr(sys_config, "TOTAL_CORES", None)
    if total:
        return max(1, int(total))
    return allocated_cores()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def parse_pbs_size_mb(text: str) -> Optional[float]:
    """Size in MB from a PBS size string (``"8gb"``, ``"8192mb"``, ``"1048576kb"``, ``"512"`` bytes)."""
    m = re.fullmatch(r"\s*(\d+)\s*([kmgt]?b)?\s*", str(text), re.I)
    if not m:
        return None
    unit = (m.group(2) or "b").lower()
    return int(m.group(1)) * _UNIT_MB[unit]


def parse_qstat_mem_mb(qstat_full: str) -> Optional[float]:
    """``Resource_List.mem`` (MB) from ``qstat -f <jobid>`` output."""
    m = re.search(r"^\s*Resource_List\.mem\s*=\s*(\S+)", qstat_full, re.M)
    return parse_pbs_size_mb(m.group(1)) if m else None


def _pbs_memory_mb(job_id: str) -> Optional[float]:
    """The job's requested memory, read once with ``qstat -f`` (``None`` if unavailable)."""
    try:
        out = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_qstat_mem_mb(out)


def scheduler_memory_mb(env: Optional[Dict[str, str]] = None) -> Optional[float]:
    """Memory (MB) the scheduler granted this job, or ``None`` outside a job / when unknown.

    With the default environment the result is cached per process and exported
    to children in ``RD_ALLOCATED_MEMORY_MB``.
    """
    global _mem_cache
    use_process_env = env is None
    env = os.environ if env is None else env
    inherited = env.get(_MEM_ENV)
    if inherited:
        try:
            return float(inherited) if float(inherited) > 0 else None
        except ValueError:
            pass
    if use_process_env and _mem_cache is not False:
        return _mem_cache  # type: ignore[return-value]

    kind = scheduler(env)
    value: Optional[float] = None
    if kind == "pbs":
        value = _pbs_memory_mb(env["PBS_JOBID"])
    elif kind == "slurm":
        per_node = _positive_int(env.get("SLURM_MEM_PER_NODE"))
        per_cpu = _positive_int(env.get("SLURM_MEM_PER_CPU"))
        if per_node:
            value = float(per_node)
        elif per_cpu:
            value = float(per_cpu * (scheduler_cores(env) or 1))

    if use_process_env:
        _mem_cache = value
        os.environ[_MEM_ENV] = str(value) if value else "0"
    return value


def allocated_memory_mb() -> Optional[float]:
    """Memory budget for the run: ``system.MEMORY_LIMIT_MB``, else the scheduler's grant, else ``None``."""
    from dreamer.configs import config
    limit = getattr(config.system, "MEMORY_LIMIT_MB", None)
    if limit:
        return float(limit)
    return scheduler_memory_mb()


# ---------------------------------------------------------------------------
# Worker threading
# ---------------------------------------------------------------------------

def limit_worker_threads() -> Dict[str, str]:
    """Inside a batch job, make child processes' numeric libraries single-threaded.

    PBS sets ``OMP_NUM_THREADS = $NCPUS``, so each of N pool workers would start
    N BLAS/OpenMP threads (N² threads on N cores).  The pools already provide
    the parallelism, so children get ``1``.  Only applied inside a scheduler
    job and only where the variable is unset or equals the allocation (a value
    the user set deliberately is kept).  Call before starting workers; the main
    process' already-loaded libraries are unaffected.

    :return: The variables that were changed (name → new value).
    """
    kind = scheduler()
    if kind is None:
        return {}
    alloc = str(scheduler_cores() or "")
    changed = {}
    for var in THREAD_ENV_VARS:
        current = os.environ.get(var)
        if current is None or current == alloc:
            if current != "1":
                os.environ[var] = "1"
                changed[var] = "1"
    return changed


def describe() -> str:
    """One-line summary of the resolved allocation, for the run-start log."""
    kind = scheduler() or "none"
    mem = allocated_memory_mb()
    return (
        f"scheduler={kind} cores={available_cores()} "
        f"(scheduler={scheduler_cores()}, affinity={affinity_cores()}, node={os.cpu_count()}) "
        f"memory={'%.0f MB' % mem if mem else 'machine available'}"
    )
