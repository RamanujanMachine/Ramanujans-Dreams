"""
When will the batch scheduler kill this job?

PBS Pro gives a job **no advance warning**: at the walltime limit it kills the
job (on the Zeus cluster, measured 2026-10-15: ``Exit_status = -29``,
``=>> PBS: job killed: walltime 629 exceeded limit 600`` — the check is polled,
so the kill comes up to ~30 s late).  A run that wants to save its state must
know the deadline in advance and act before it.  :func:`detect` finds it:

1. ``RUNMON_WALLTIME_END`` — explicit end time (epoch seconds or ISO-8601);
2. ``PBS_WALLTIME`` — Torque exports the limit in seconds;
3. PBS Pro — ``qstat -f $PBS_JOBID``: ``Resource_List.walltime`` minus
   ``resources_used.walltime`` (works from inside a job on Zeus);
4. Slurm — ``squeue -h -j $SLURM_JOB_ID -o %e`` (the job's end time);
5. otherwise no deadline.

The limit can change while the job runs (``qalter``), so the hub re-detects
periodically.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

Runner = Callable[..., "subprocess.CompletedProcess"]


@dataclass(frozen=True)
class Deadline:
    """A job end time.

    :ivar end: Epoch seconds at which the scheduler ends the job.
    :ivar source: Where it came from (``env``, ``torque``, ``pbs``, ``slurm``).
    :ivar limit: The total walltime limit in seconds, when known.
    """

    end: float
    source: str
    limit: Optional[float] = None

    def remaining(self, now: Optional[float] = None) -> float:
        """Seconds left (negative once passed)."""
        return self.end - (now if now is not None else time.time())


def parse_hms(text: str) -> Optional[float]:
    """Seconds from ``[[HH:]MM:]SS`` (hours may exceed 24, e.g. ``336:00:00``)."""
    if text is None:
        return None
    parts = str(text).strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if not 1 <= len(nums) <= 3 or any(n < 0 for n in nums):
        return None
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def parse_qstat_walltime(text: str) -> Dict[str, Optional[float]]:
    """``{"limit": s, "used": s}`` from ``qstat -f`` output (missing keys → ``None``)."""
    def grab(key):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(\S+)", text, re.M)
        return parse_hms(m.group(1)) if m else None
    return {"limit": grab("Resource_List.walltime"), "used": grab("resources_used.walltime")}


def _parse_end(text: str) -> Optional[float]:
    """Epoch seconds from an epoch number or an ISO-8601 time (local time if naive)."""
    text = str(text).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _run(runner: Runner, cmd) -> Optional[str]:
    try:
        out = runner(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if getattr(out, "returncode", 0) == 0 else None


def detect(env: Optional[Dict[str, str]] = None, *, runner: Runner = subprocess.run,
           now: Optional[float] = None, start: Optional[float] = None) -> Optional[Deadline]:
    """Find this job's deadline (see module docstring); ``None`` outside a batch job.

    :param start: When this process started (epoch), used with Torque's
        ``PBS_WALLTIME`` (default: now — conservative only if called at start).
    """
    env = os.environ if env is None else env
    now = time.time() if now is None else now

    explicit = env.get("RUNMON_WALLTIME_END")
    if explicit:
        end = _parse_end(explicit)
        if end is not None:
            return Deadline(end, "env")

    torque = parse_hms(env.get("PBS_WALLTIME", "")) if env.get("PBS_WALLTIME") else None
    if torque:
        return Deadline((start if start is not None else now) + torque, "torque", torque)

    job = env.get("PBS_JOBID")
    if job:
        out = _run(runner, ["qstat", "-f", job])
        if out:
            w = parse_qstat_walltime(out)
            if w["limit"] is not None:
                used = w["used"] or 0.0
                return Deadline(now + w["limit"] - used, "pbs", w["limit"])

    sjob = env.get("SLURM_JOB_ID")
    if sjob:
        out = _run(runner, ["squeue", "-h", "-j", sjob, "-o", "%e"])
        if out and out.strip() and out.strip().upper() not in ("N/A", "UNLIMITED", "NONE"):
            end = _parse_end(out.strip().splitlines()[0])
            if end is not None:
                return Deadline(end, "slurm")
    return None
