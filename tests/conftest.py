"""Shared pytest fixtures for Ramanujan Agent tests."""
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
