"""Tests for :mod:`runmon.watchdog` — one deadline heap, no thread per call.

  - Self-driven (worker) mode: fires while the block overruns, repeats, or
    fires once; never fires for a block that finishes in time.
  - The ``detail`` callable is evaluated only when firing; its errors are
    reported, not raised.
  - Hub-driven mode: ``check`` fires overdue watches, ``next_due`` bounds the
    hub's sleep and ignores cancelled watches.
  - Many concurrent watches share one thread.
"""

import threading
import time

from runmon.watchdog import DeadlineWatcher


def _collector():
    fired = []
    return fired, lambda label, elapsed, detail: fired.append((label, elapsed, detail))


def test_fires_and_repeats_in_thread_mode():
    fired, cb = _collector()
    w = DeadlineWatcher(cb)
    w.start_thread()
    with w.guard("walk", 0.1, detail="shard 3"):
        time.sleep(0.35)
    time.sleep(0.15)                                  # nothing more after the block ends
    n = len(fired)
    assert 2 <= n <= 4
    assert fired[0][0] == "walk" and fired[0][2] == "shard 3" and fired[0][1] >= 0.1
    time.sleep(0.2)
    assert len(fired) == n


def test_once():
    fired, cb = _collector()
    w = DeadlineWatcher(cb)
    w.start_thread()
    with w.guard("x", 0.05, repeat=False):
        time.sleep(0.25)
    assert len(fired) == 1


def test_quick_block_never_fires_and_detail_is_lazy():
    fired, cb = _collector()
    calls = []
    w = DeadlineWatcher(cb)
    w.start_thread()
    with w.guard("fast", 0.2, detail=lambda: calls.append(1) or "expensive"):
        pass
    time.sleep(0.3)
    assert fired == [] and calls == []


def test_detail_errors_are_reported_not_raised():
    fired, cb = _collector()
    w = DeadlineWatcher(cb)
    h = w.watch("x", 0.01, detail=lambda: 1 / 0)
    time.sleep(0.03)
    assert w.check() == 1
    assert "detail error" in fired[0][2]
    w.cancel(h)


def test_hub_driven_check_and_next_due():
    fired, cb = _collector()
    w = DeadlineWatcher(cb)
    assert w.next_due() is None
    h = w.watch("x", 0.05)
    assert 0 < w.next_due() <= 0.05
    assert w.check() == 0
    time.sleep(0.07)
    assert w.check() == 1
    w.cancel(h)
    assert w.next_due() is None                       # cancelled entries are dropped
    assert w.watch("off", 0) is None                  # disabled threshold


def test_one_thread_for_many_watches():
    fired, cb = _collector()
    w = DeadlineWatcher(cb)
    w.start_thread()
    before = threading.active_count()
    handles = [w.watch(f"t{i}", 0.05, repeat=False) for i in range(200)]
    time.sleep(0.2)
    assert threading.active_count() == before
    assert len(fired) == 200
    for h in handles:
        w.cancel(h)
