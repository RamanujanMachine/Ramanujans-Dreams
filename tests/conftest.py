"""Shared pytest fixtures for Ramanujan Agent tests."""
import signal
import threading

import mpmath
import pytest


@pytest.fixture(autouse=True)
def set_high_precision():
    """Ensure all tests run with sufficient precision."""
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = 200
    yield
    mpmath.mp.dps = old_dps


@pytest.fixture
def reference_constants():
    """High-precision reference values for fundamental constants."""
    mpmath.mp.dps = 500
    return {
        "e": mpmath.e,
        "pi": mpmath.pi,
        "ln2": mpmath.log(2),
        "zeta3": mpmath.zeta(3),
        "sqrt2": mpmath.sqrt(2),
        "phi": (1 + mpmath.sqrt(5)) / 2,  # golden ratio
    }


def pytest_addoption(parser):
    parser.addoption(
        "--test-timeout",
        action="store",
        type=float,
        default=60.0,
        help="Default timeout in seconds per test (set <=0 to disable).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "timeout(seconds): override timeout for an individual test",
    )


@pytest.fixture(autouse=True)
def enforce_test_timeout(request):
    """Guardrail: fail tests that exceed a bounded runtime."""
    # SIGALRM is supported on Linux/WSL test runners used by this project.
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    marker = request.node.get_closest_marker("timeout")
    timeout_seconds = float(marker.args[0]) if marker and marker.args else request.config.getoption("--test-timeout")
    if timeout_seconds <= 0:
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"Test timed out after {timeout_seconds:.1f}s: {request.node.nodeid}")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.fixture(autouse=True)
def serial_stage_pools():
    """Run the per-stage trajectory pools (analysis, hedgehog search, Tier-3)
    serially in-process by default.

    Unit tests monkeypatch module/class attributes that a spawned worker would
    not see, and spawning a full-machine pool per test is slow.  Tests that
    exercise the real multi-process path override ``STAGE_POOL_WORKERS``
    explicitly (see ``tests/test_stage_pool.py``).
    """
    from dreamer.configs.system import sys_config

    original = sys_config.STAGE_POOL_WORKERS
    sys_config.STAGE_POOL_WORKERS = 1
    yield
    sys_config.STAGE_POOL_WORKERS = original


@pytest.fixture(autouse=True)
def memory_unaware_pools():
    """Size pools by cores only in the unit tests, so results do not depend on how
    much memory the test machine happens to have free.  The memory-aware
    behaviour is tested explicitly in ``tests/test_memory_budget.py``."""
    from dreamer.configs.system import sys_config

    original = sys_config.MEMORY_AWARE_WORKERS
    sys_config.MEMORY_AWARE_WORKERS = False
    yield
    sys_config.MEMORY_AWARE_WORKERS = original
