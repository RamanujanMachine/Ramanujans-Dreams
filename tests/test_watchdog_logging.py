"""Tests for ``Logger.watchdog`` — the background hang-detection context manager.

The watchdog spawns a monitoring-only daemon thread that emits a WARNING while a
wrapped procedure is *still running* past a threshold, so a true stall is located
even though the call never returns.  These tests use very short thresholds so the
suite stays fast.
"""

import threading
import time

from dreamer.configs.logging import logging_config
from dreamer.utils.logger import Logger


def _capture(monkeypatch):
    """Patch ``Logger.log`` to record ``(msg, level)`` for every emitted line.

    ``list.append`` is atomic in CPython, so it is safe to collect from the
    watchdog's monitor thread without an explicit lock.
    """
    records = []

    def _log(self, msg_prefix='', in_function=False, add_stack_trace=False):
        records.append((self.msg, self.level))

    monkeypatch.setattr(Logger, 'log', _log)
    return records


def _warnings(records):
    return [msg for msg, level in records if level == Logger.Levels.warning]


def _wait_for(predicate, timeout=3.0):
    """Busy-wait (releasing the GIL) until *predicate* is true or *timeout* elapses.

    Used instead of a fixed sleep so the firing tests are not sensitive to how
    quickly the monitor daemon thread gets scheduled (cold-import GIL contention
    can otherwise delay its first tick past a short fixed sleep).
    """
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


def test_watchdog_fires_after_threshold(monkeypatch):
    """A block that overruns the threshold emits a warning carrying the detail."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    with Logger.watchdog('slow op', 0.05, detail='traj_id=abc start=(1,) direction=(2,)'):
        _wait_for(lambda: len(_warnings(records)) >= 1)

    warns = _warnings(records)
    assert warns, "expected at least one watchdog warning"
    assert any('slow op' in w and 'traj_id=abc' in w for w in warns)
    assert any('still running after' in w for w in warns)


def test_watchdog_repeat_true_fires_multiple_times(monkeypatch):
    """With repeat=True the watchdog re-warns every interval while stuck."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    with Logger.watchdog('stuck', 0.05, detail='x', repeat=True):
        _wait_for(lambda: len(_warnings(records)) >= 2)

    assert len(_warnings(records)) >= 2


def test_watchdog_repeat_false_fires_once(monkeypatch):
    """With repeat=False the watchdog warns exactly once."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    with Logger.watchdog('stuck', 0.05, detail='x', repeat=False):
        _wait_for(lambda: len(_warnings(records)) >= 1)
        # Stay in the block well past several more intervals; repeat=False must
        # not fire again.
        time.sleep(0.3)

    assert len(_warnings(records)) == 1


def test_watchdog_no_warning_when_block_is_fast(monkeypatch):
    """A block that finishes before the threshold emits no warning."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    with Logger.watchdog('fast op', 5.0, detail='x'):
        pass

    assert _warnings(records) == []


def test_watchdog_noop_when_disabled(monkeypatch):
    """WATCHDOG_ENABLED=False makes the watchdog a no-op even past the threshold."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', False)
    records = _capture(monkeypatch)

    with Logger.watchdog('slow op', 0.05, detail='x'):
        time.sleep(0.2)

    assert _warnings(records) == []


def test_watchdog_noop_when_threshold_nonpositive(monkeypatch):
    """A falsy / non-positive threshold disables the watchdog."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    for threshold in (0, 0.0, -1):
        with Logger.watchdog('slow op', threshold, detail='x'):
            time.sleep(0.1)

    assert _warnings(records) == []


def test_watchdog_detail_callable_not_invoked_on_happy_path(monkeypatch):
    """The detail callable is never built when the watchdog does not fire."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    _capture(monkeypatch)

    calls = []

    def detail():
        calls.append(1)
        return 'expensive'

    with Logger.watchdog('fast op', 5.0, detail=detail):
        pass

    assert calls == [], "detail() must not run on the happy path"


def test_watchdog_detail_callable_invoked_when_fired(monkeypatch):
    """The detail callable is invoked lazily only when the watchdog fires."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    records = _capture(monkeypatch)

    calls = []

    def detail():
        calls.append(1)
        return 'lazy-detail'

    with Logger.watchdog('slow op', 0.05, detail=detail, repeat=False):
        _wait_for(lambda: len(_warnings(records)) >= 1)

    assert calls, "detail() should have been invoked once the watchdog fired"
    assert any('lazy-detail' in w for w in _warnings(records))


def test_watchdog_thread_is_cleaned_up(monkeypatch):
    """The monitor thread stops promptly once the block exits."""
    monkeypatch.setattr(logging_config, 'WATCHDOG_ENABLED', True)
    _capture(monkeypatch)

    before = {t.name for t in threading.enumerate()}
    with Logger.watchdog('op', 0.05, detail='x'):
        time.sleep(0.12)
    # Give the daemon a moment to observe the Event and exit.
    time.sleep(0.1)
    after = {t.name for t in threading.enumerate()}

    assert not any(name.startswith('watchdog:op') for name in after - before)
