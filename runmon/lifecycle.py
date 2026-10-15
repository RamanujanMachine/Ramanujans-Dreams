"""
Leave a complete record however the run ends.

:func:`install` arranges for ``on_exit(reason)`` to run exactly once when the
process ends:

* normal interpreter exit → ``"exit"`` (``atexit``);
* ``SIGTERM`` (``qdel``, a scheduler's kill, ``kill``) → ``"SIGTERM"``, then
  ``SystemExit(143)`` is raised in the main thread so the program's ``finally``
  blocks and context managers still run (unless the program installed its own
  handler, which is then called instead);
* ``SIGINT`` (Ctrl-C) → ``"SIGINT"``, then the previous handler
  (``KeyboardInterrupt``);
* ``SIGHUP`` (terminal closed) / ``SIGUSR1`` → recorded the same way as SIGTERM.

Only the installing process records: a child created with ``fork`` inherits the
handlers but never runs ``on_exit`` (its signals keep their normal effect).

Crashes in C code (segfaults) cannot run Python; :mod:`faulthandler` writes
their tracebacks to ``crash.txt`` instead.  ``SIGKILL`` cannot be caught at all
— that is why events are flushed as they are written and the hub flushes
*before* a known walltime deadline.

Limitation: Python runs signal handlers in the main thread *between bytecodes*;
while the main thread is inside a long C call (a big sympy/flint computation)
the handler waits for it to return.  The hub thread's pre-deadline flush covers
that case for walltime kills.
"""

from __future__ import annotations

import atexit
import faulthandler
import os
import signal
import threading
from typing import Callable, Dict, Optional

_installed = False
_done = False
_lock = threading.Lock()
_on_exit: Optional[Callable[[str], None]] = None
_previous: Dict[int, object] = {}
_crash_file = None
_pid: Optional[int] = None         # process that installed (forked children must not finish)

#: Signals that end the run without a Python exception by default.
_TERMINATING = ("SIGTERM", "SIGHUP", "SIGUSR1")


def _finish(reason: str) -> None:
    global _done
    with _lock:
        if _done or _on_exit is None or os.getpid() != _pid:
            return
        _done = True
    try:
        _on_exit(reason)
    except Exception:
        pass


def _handler(signum, frame):
    name = signal.Signals(signum).name
    prev = _previous.get(signum)        # read first: on_exit may uninstall (clearing it)
    _finish(name)
    if callable(prev):
        prev(signum, frame)                     # e.g. default_int_handler → KeyboardInterrupt
        return
    if prev == signal.SIG_IGN:
        return
    raise SystemExit(128 + signum)


def install(on_exit: Callable[[str], None], *, crash_file: Optional[str] = None) -> bool:
    """Register ``on_exit`` for every way the process can end (see module docstring).

    Must be called from the main thread for the signal handlers (otherwise only
    ``atexit`` and ``faulthandler`` are set).  Calling again replaces ``on_exit``.

    :return: ``True`` if the signal handlers were installed.
    """
    global _installed, _on_exit, _done, _crash_file, _pid
    _on_exit = on_exit
    _done = False
    _pid = os.getpid()
    if crash_file:
        try:
            if _crash_file is not None:
                _crash_file.close()
            _crash_file = open(crash_file, "a")
            faulthandler.enable(_crash_file, all_threads=True)
        except OSError:
            _crash_file = None
    if not _installed:
        atexit.register(_finish, "exit")
        _installed = True
    if threading.current_thread() is not threading.main_thread():
        return False
    for name in ("SIGINT",) + _TERMINATING:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            current = signal.getsignal(sig)
            if current is not _handler:
                _previous[sig] = current
                signal.signal(sig, _handler)
        except (OSError, ValueError):
            continue
    return True


def uninstall() -> None:
    """Restore the previous signal handlers and stop recording (tests, nested runs)."""
    global _on_exit, _crash_file
    _on_exit = None
    if threading.current_thread() is threading.main_thread():
        for sig, prev in list(_previous.items()):
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError, TypeError):
                pass
    _previous.clear()
    if _crash_file is not None:
        try:
            faulthandler.disable()
            _crash_file.close()
        except Exception:
            pass
        _crash_file = None
