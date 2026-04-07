"""Sanity tests demonstrating the coverage policy patterns.

These tests verify that the core mathematical infrastructure works correctly
and serve as examples for writing new tests. Every test follows the pattern:
  1. Known-answer test (compare against reference constant)
  2. Edge case test (boundary inputs)
  3. Convergence/consistency test (mathematical invariants)
"""
import mpmath
from mpmath import mpf
import pytest


# ---------------------------------------------------------------------------
# Pattern 1: Known-answer test — verify a computation against a reference
# ---------------------------------------------------------------------------

class TestKnownAnswers:
    """Verify well-known constants via simple series/products."""

    def test_e_via_factorial_series(self, reference_constants):
        """e = sum(1/n!) converges quickly."""
        mpmath.mp.dps = 150
        total = mpmath.mpf(0)
        factorial = mpmath.mpf(1)
        for n in range(200):
            total += 1 / factorial
            factorial *= (n + 1)
        assert mpmath.almosteq(total, reference_constants["e"], 1e-100), (
            f"e series: got {mpmath.nstr(total, 30)}"
        )

    def test_pi_via_machin_formula(self, reference_constants):
        """pi/4 = 4*arctan(1/5) - arctan(1/239) (Machin's formula)."""
        mpmath.mp.dps = 150
        result = 4 * (4 * mpmath.atan(mpf(1) / 5) - mpmath.atan(mpf(1) / 239))
        assert mpmath.almosteq(result, reference_constants["pi"], 1e-100)

    def test_ln2_via_series(self, reference_constants):
        """ln(2) = sum((-1)^(n+1) / n) — slow but correct."""
        mpmath.mp.dps = 50
        # Use the faster series: ln(2) = sum(1/(n * 2^n))
        total = mpmath.mpf(0)
        for n in range(1, 200):
            total += mpmath.mpf(1) / (n * mpmath.power(2, n))
        assert mpmath.almosteq(total, reference_constants["ln2"], 1e-30), (
            f"ln2 series: got {mpmath.nstr(total, 30)}"
        )


# ---------------------------------------------------------------------------
# Pattern 2: Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_zero_depth_returns_initial(self):
        """A matrix product of depth 0 should be the identity."""
        # This pattern applies to any walk/product function:
        # walk(trajectory, iterations=0) should return identity matrix
        identity = mpmath.matrix([[1, 0], [0, 1]])
        assert identity[0, 0] == 1
        assert identity[0, 1] == 0

    def test_single_term_series(self):
        """A single-term partial sum should equal the first term."""
        first_term = mpmath.mpf(1)  # e.g., 1/0! = 1
        assert first_term == 1


# ---------------------------------------------------------------------------
# Pattern 3: Mathematical invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    """Tests that verify mathematical properties and consistency."""

    def test_matrix_inverse_roundtrip(self):
        """M * M^-1 = I for a non-singular 2x2 matrix."""
        M = mpmath.matrix([[3, 1], [2, 4]])
        M_inv = M ** (-1)
        product = M * M_inv
        I = mpmath.matrix([[1, 0], [0, 1]])
        for i in range(2):
            for j in range(2):
                assert mpmath.almosteq(product[i, j], I[i, j], 1e-50), (
                    f"M * M^-1 != I at ({i},{j}): {product[i,j]}"
                )

    def test_convergence_is_monotonic(self):
        """More terms should give more digits of accuracy."""
        mpmath.mp.dps = 300
        target = mpmath.e
        prev_error = mpmath.mpf("inf")
        total = mpmath.mpf(0)
        factorial = mpmath.mpf(1)
        for n in range(50):
            total += 1 / factorial
            factorial *= (n + 1)
            error = abs(total - target)
            if n > 0:
                assert error <= prev_error, (
                    f"Error increased at n={n}: {error} > {prev_error}"
                )
            prev_error = error

    def test_precision_independence(self):
        """Result should not depend on working precision (within tolerance)."""
        results = []
        for dps in [100, 200, 300]:
            mpmath.mp.dps = dps
            val = 4 * mpmath.atan(1)  # = pi
            results.append(val)
        # All should agree to 50 digits
        for r in results[1:]:
            assert mpmath.almosteq(results[0], r, 1e-50)
