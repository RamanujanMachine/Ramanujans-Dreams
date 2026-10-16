"""
What a run ran on and with — recorded once at the start (``environment`` event).

Months later "which code, which versions, which settings, which machine?" is the
first question about a result.  :func:`collect` gathers it cheaply and never
raises: every field that cannot be read is simply left out.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

#: Packages whose version matters for reproducing a result.
_PACKAGES = ("ramanujantools", "sympy", "mpmath", "numpy", "scipy", "python-flint", "LIReC", "mip")

#: Scheduler variables worth keeping (job identity and shape, never credentials).
_SCHEDULER_VARS = ("PBS_JOBID", "PBS_JOBNAME", "PBS_QUEUE", "PBS_O_HOST", "PBS_NODEFILE",
                   "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
                   "SLURM_NTASKS", "SLURM_MEM_PER_NODE")


def _git(args: list, cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_version(repo: Optional[str] = None) -> Dict[str, Any]:
    """Commit, branch and whether the working tree had uncommitted changes."""
    repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    info: Dict[str, Any] = {}
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return info
    info["commit"] = commit
    info["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    info["dirty"] = bool(status)
    return info


def package_versions() -> Dict[str, str]:
    """Installed versions of :data:`_PACKAGES` (missing ones are left out)."""
    from importlib import metadata
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def machine() -> Dict[str, Any]:
    """Host, OS, cores and memory of the machine."""
    info: Dict[str, Any] = {"host": platform.node(), "platform": platform.platform(),
                            "python": sys.version.split()[0], "executable": sys.executable,
                            "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["memory_mb"] = round(psutil.virtual_memory().total / 2**20)
    except Exception:
        pass
    return info


def scheduler() -> Dict[str, str]:
    """Batch-job identity from the environment (empty outside PBS / Slurm)."""
    return {k: os.environ[k] for k in _SCHEDULER_VARS if k in os.environ}


def settings(exported: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The configuration sections with JSON-friendly values (anything else as ``str``)."""
    def plain(v):
        return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return {section: {k: plain(v) for k, v in values.items()} for section, values in exported.items()}


def collect(exported_config: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Everything above as one JSON-ready dict (the fields of the ``environment`` event)."""
    env: Dict[str, Any] = {"argv": list(sys.argv), "cwd": os.getcwd(), "pid": os.getpid()}
    for key, fn in (("code", code_version), ("packages", package_versions),
                    ("machine", machine), ("scheduler", scheduler)):
        try:
            env[key] = fn()
        except Exception:
            continue
    if exported_config is not None:
        try:
            env["config"] = settings(exported_config)
        except Exception:
            pass
    return env
