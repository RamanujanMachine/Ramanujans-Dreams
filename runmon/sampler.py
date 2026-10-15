"""
Process and memory sampling with ``psutil``.

:meth:`ProcessSampler.sample` measures the monitored process and all its
descendants (pool workers, fork servers, managers): CPU %, resident memory,
state, threads — plus the machine's memory.  ``psutil.Process`` objects are
cached between samples because ``cpu_percent`` is measured *between two calls
on the same object*; a new worker's first sample therefore reads 0 %.

Utilization is reported against the run's **allocation** (``cores``,
``memory_mb``), not the machine: on a batch node that does not confine jobs the
machine's numbers are irrelevant to whether the run uses what it was given —
and the sampler flags a run that uses *more* than it was given.

Cost: one ``children(recursive=True)`` walk plus a few ``/proc`` reads per
process per sample — milliseconds for dozens of workers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import psutil

_MB = 1024 * 1024

#: Command-line markers of multiprocessing's helper processes.
_HELPERS = (("resource_tracker", "resource tracker"), ("forkserver", "fork server"),
            ("multiprocessing.managers", "manager"), ("SyncManager", "manager"))


def _helper_role(proc: "psutil.Process") -> str:
    """Role of an unlabelled child from its command line (``child`` if unknown)."""
    try:
        cmd = " ".join(proc.cmdline())
    except psutil.Error:
        return "child"
    for marker, role in _HELPERS:
        if marker in cmd:
            return role
    return "child"


class ProcessSampler:
    """Sample a process tree.

    :param root_pid: Process whose tree is measured (default: this process).
    :param cores: Allocated cores (for utilization); ``None`` = machine cores.
    :param memory_mb: Allocated memory (MB); ``None`` = machine total.
    :param memory_probe: Optional ``() -> available MB`` overriding psutil's
        machine-available figure (e.g. a cgroup-aware probe).
    """

    def __init__(self, root_pid: Optional[int] = None, *, cores: Optional[int] = None,
                 memory_mb: Optional[float] = None,
                 memory_probe: Optional[Callable[[], float]] = None):
        self.root_pid = root_pid or os.getpid()
        self.cores = cores
        self.memory_mb = memory_mb
        self.memory_probe = memory_probe
        self.labels: Dict[int, str] = {}       # pid -> role label (pool name ...)
        self._auto: Dict[int, str] = {}        # pid -> role guessed once from the command line
        self._procs: Dict[int, psutil.Process] = {}
        self._root = psutil.Process(self.root_pid)
        self._root.cpu_percent(None)           # prime

    def label(self, pid: int, text: str) -> None:
        """Attach a role label (e.g. ``"eval pool"``) to a process."""
        self.labels[pid] = text

    def _tree(self) -> List[psutil.Process]:
        try:
            kids = self._root.children(recursive=True)
        except psutil.Error:
            kids = []
        live = {self.root_pid: self._root}
        for p in kids:
            cached = self._procs.get(p.pid)
            if cached is None:
                p.cpu_percent(None)            # prime: first reading is 0
                cached = p
                self._auto[p.pid] = _helper_role(p)
            live[p.pid] = cached
        self._procs = live
        self._auto = {pid: r for pid, r in self._auto.items() if pid in live}
        return list(live.values())

    def sample(self) -> Dict[str, Any]:
        """One measurement of the tree and the machine (JSON-ready dict)."""
        t0 = time.time()
        procs = []
        for p in self._tree():
            try:
                with p.oneshot():
                    mem = p.memory_info()
                    procs.append({
                        "pid": p.pid,
                        "ppid": p.ppid(),
                        "role": "main" if p.pid == self.root_pid else self.labels.get(p.pid, self._auto.get(p.pid, "child")),
                        "name": p.name(),
                        "cpu": round(p.cpu_percent(None), 1),
                        "rss_mb": round(mem.rss / _MB, 1),
                        "status": p.status(),
                        "threads": p.num_threads(),
                        "started": p.create_time(),
                    })
            except psutil.Error:               # vanished between list and read
                continue
        vm = psutil.virtual_memory()
        machine_avail = vm.available / _MB
        if self.memory_probe is not None:
            try:
                machine_avail = float(self.memory_probe())
            except Exception:
                pass
        machine_cores = psutil.cpu_count() or 1
        cores = self.cores or machine_cores
        cpu_sum = sum(p["cpu"] for p in procs)
        rss_sum = sum(p["rss_mb"] for p in procs)
        mem_budget = self.memory_mb or vm.total / _MB
        try:
            load = os.getloadavg()
        except (AttributeError, OSError):
            load = None
        result = {
            "ts": t0,
            "processes": procs,
            "totals": {
                "n": len(procs),
                "cpu_percent": round(cpu_sum, 1),
                "rss_mb": round(rss_sum, 1),
                "cpu_util": round(cpu_sum / (100.0 * cores), 3),
                "mem_util": round(rss_sum / mem_budget, 3) if mem_budget else None,
                "over_cpu": cpu_sum > 100.0 * cores * 1.10,
                "over_mem": bool(self.memory_mb) and rss_sum > self.memory_mb,
            },
            "allocation": {"cores": cores, "memory_mb": self.memory_mb,
                           "cores_source": "allocation" if self.cores else "machine"},
            "machine": {
                "cores": machine_cores,
                "mem_total_mb": round(vm.total / _MB),
                "mem_available_mb": round(machine_avail),
                "mem_percent": vm.percent,
                "load": load,
            },
        }
        result["cost_ms"] = round((time.time() - t0) * 1000, 2)
        return result
