"""
Process start method for the package's pools, chosen per OS.

* **Linux** → ``forkserver``: children fork from a small, single-threaded
  server process.  Unlike plain ``fork`` it never forks the (multi-threaded)
  main process — no fork-with-threads deadlocks — and, unlike ``fork``, new
  workers can be started at any time, which the memory-adaptive pools need to
  grow.  Workers also start faster and lighter than with ``spawn`` (the server
  pre-imports the worker modules once and children share those pages).
* **macOS** → ``spawn`` (the platform default; ``fork`` is unsafe there with
  system frameworks, and ``spawn`` already starts workers dynamically).

Windows is not supported (``fpylll``).  ``system.POOL_START_METHOD`` overrides
the choice (``"auto"`` = per OS).
"""

from __future__ import annotations

import multiprocessing as mp
import sys

#: Modules the Linux fork server imports once, so every worker starts with them.
_FORKSERVER_PRELOAD = [
    "dreamer.utils.stage_pool",
    "dreamer.utils.storage.trajectory_attributes",
    "dreamer.search.methods.flatland.parallel_eval_worker",
]
_preload_set = False


def start_method() -> str:
    """The start method pools should use (see module docstring)."""
    from dreamer.configs import config
    choice = str(getattr(config.system, "POOL_START_METHOD", "auto") or "auto").lower()
    available = mp.get_all_start_methods()
    if choice != "auto":
        if choice not in available:
            raise ValueError(f"POOL_START_METHOD {choice!r} not available here: {available}")
        return choice
    if sys.platform.startswith("linux") and "forkserver" in available:
        return "forkserver"
    return "spawn"


def pool_context():
    """``multiprocessing`` context for the package's process pools."""
    global _preload_set
    method = start_method()
    ctx = mp.get_context(method)
    if method == "forkserver" and not _preload_set:
        try:
            ctx.set_forkserver_preload(_FORKSERVER_PRELOAD)
        except Exception:
            pass
        _preload_set = True
    return ctx
