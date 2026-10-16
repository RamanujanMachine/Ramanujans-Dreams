"""
Hang detection with one deadline heap instead of one thread per watched call.

``watch(label, threshold)`` registers a deadline; if the watched block is still
running when it passes, ``on_fire(label, elapsed, detail)`` is called (and, with
``repeat=True``, again every ``threshold`` seconds) until the block ends.

Two ways to drive it:

* **Hub-driven** (main process): the monitor hub calls :meth:`check` every tick
  and uses :meth:`next_due` to bound its sleep — no extra thread.
* **Self-driven** (worker processes, no hub): :meth:`start_thread` runs one
  daemon thread per process that sleeps until the next deadline.  It performs
  no computation (``Condition.wait``), consistent with the project's "no
  threads for CPU work" rule.

``detail`` may be a callable — evaluated only if the watchdog fires, so an
expensive description costs nothing on the happy path.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple

FireCallback = Callable[[str, float, Optional[str]], None]


class _Watch:
    __slots__ = ("label", "threshold", "detail", "repeat", "start", "active", "fired")

    def __init__(self, label, threshold, detail, repeat):
        self.label = label
        self.threshold = float(threshold)
        self.detail = detail
        self.repeat = repeat
        self.start = time.monotonic()
        self.active = True
        self.fired = 0


class DeadlineWatcher:
    """Deadline heap firing ``on_fire`` for blocks that overrun (see module docstring)."""

    def __init__(self, on_fire: FireCallback):
        self.on_fire = on_fire
        self._heap: List[Tuple[float, int, _Watch]] = []
        self._ids = itertools.count()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None

    # -- registration -------------------------------------------------------

    def watch(self, label: str, threshold: float, *, detail: Any = None,
              repeat: bool = True) -> Optional[_Watch]:
        """Start watching; returns a handle for :meth:`cancel` (``None`` if disabled)."""
        if not threshold or threshold <= 0:
            return None
        w = _Watch(label, threshold, detail, repeat)
        with self._cond:
            heapq.heappush(self._heap, (w.start + w.threshold, next(self._ids), w))
            self._cond.notify()
        return w

    @staticmethod
    def cancel(handle: Optional[_Watch]) -> None:
        """Stop watching (the heap entry is dropped lazily)."""
        if handle is not None:
            handle.active = False

    @contextmanager
    def guard(self, label: str, threshold: float, *, detail: Any = None,
              repeat: bool = True) -> Iterator[None]:
        """``with watcher.guard("walk", 300): ...``"""
        handle = self.watch(label, threshold, detail=detail, repeat=repeat)
        try:
            yield
        finally:
            self.cancel(handle)

    # -- driving ------------------------------------------------------------

    def next_due(self) -> Optional[float]:
        """Seconds until the next active deadline (``None`` if nothing is watched)."""
        with self._cond:
            while self._heap and not self._heap[0][2].active:
                heapq.heappop(self._heap)
            if not self._heap:
                return None
            return max(0.0, self._heap[0][0] - time.monotonic())

    def check(self) -> int:
        """Fire every overdue watch; returns how many fired."""
        now = time.monotonic()
        due = []
        with self._cond:
            while self._heap and (self._heap[0][0] <= now or not self._heap[0][2].active):
                _, _, w = heapq.heappop(self._heap)
                if not w.active:
                    continue
                due.append(w)
                if w.repeat:                    # next warning one period from now (no catch-up burst)
                    heapq.heappush(self._heap, (now + w.threshold, next(self._ids), w))
                else:
                    w.active = False
        for w in due:
            w.fired += 1
            try:
                msg = w.detail() if callable(w.detail) else w.detail
            except Exception as e:             # never let the monitor crash the run
                msg = f"<detail error: {e}>"
            try:
                self.on_fire(w.label, now - w.start, None if msg is None else str(msg))
            except Exception:
                pass
        return len(due)

    def start_thread(self) -> None:
        """Drive the watcher from one daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return

        def loop():
            while True:
                with self._cond:
                    wait = self._heap[0][0] - time.monotonic() if self._heap else None
                    if wait is None or wait > 0:
                        self._cond.wait(wait)
                self.check()

        self._thread = threading.Thread(target=loop, name="runmon-watchdog", daemon=True)
        self._thread.start()
