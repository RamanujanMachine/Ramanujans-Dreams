# Coverage Policy

## The Rule

**Every new public function must be tested** — either by an existing test that already exercises it, or by a new test you write alongside the code.

No exceptions. No "I'll add tests later."

## What Counts as Coverage

### Minimum per function

| Aspect | Requirement |
|--------|-------------|
| **Happy path** | At least one test with typical inputs producing expected output. |
| **Known-answer** | At least one test comparing output against a pre-computed reference value (e.g., a known constant to 50+ digits). |
| **Edge case** | At least one test with boundary inputs: $n=0$, empty input, minimum-size matrix, single-term sequence. |

### Additional for mathematical functions

| Aspect | Requirement |
|--------|-------------|
| **Precision** | Test at multiple precision levels (e.g., 50 digits, 200 digits) to catch precision-dependent bugs. |
| **Convergence** | If the function computes a limit, verify convergence rate matches expectations. |
| **Invariants** | Test mathematical invariants: $M \cdot M^{-1} = I$, coboundary preserves limit, dual of dual equals original. |

### Additional for performance-critical code

| Aspect | Requirement |
|--------|-------------|
| **Correctness at scale** | Test at $N \geq 1000$ (or the intended operational scale). |
| **Regression** | Benchmark must not regress by more than 20% without justification. |

## How to Write Tests

### Location
- Tests live in `tests/`.
- Test files are named `test_<module>.py`.
- Test functions are named `test_<what_it_tests>`.

### Framework
- Use `pytest`.
- Use `mpmath` for high-precision reference values.
- Use `pytest.approx` for floating-point comparisons, or compare `mpmath` values with explicit tolerance.

### Stochastic sampler tests
- Seed randomness (`np.random.seed(...)` or deterministic RNG objects) for repeatable test behavior.
- Assert distribution health with robust statistics (e.g., nearest-neighbor angular gap medians, per-axis variance), not exact point sets.
- Keep thresholds tolerant and physics-based to avoid flaky failures across CPU/OS differences.

### Example Pattern

```python
"""Test a PCF convergent computation."""
import mpmath
from mpmath import mpf

def test_e_continued_fraction_converges():
    """The simple continued fraction for e should converge to e."""
    mpmath.mp.dps = 150
    # Compute the first 200 convergents of the CF for e
    # (implementation depends on your code)
    result = compute_e_convergent(depth=200)
    expected = mpmath.e
    # Verify to 100 digits
    assert mpmath.almosteq(result, expected, 1e-100), (
        f"e CF at depth 200: got {mpmath.nstr(result, 30)}, "
        f"expected {mpmath.nstr(expected, 30)}"
    )

def test_e_continued_fraction_edge_case():
    """Depth=0 should return the initial convergent."""
    result = compute_e_convergent(depth=0)
    assert result is not None  # Should not crash

def test_e_convergence_rate():
    """Convergence rate should be approximately 1 digit per term."""
    mpmath.mp.dps = 500
    digits_at_100 = count_correct_digits(compute_e_convergent(100), mpmath.e)
    digits_at_200 = count_correct_digits(compute_e_convergent(200), mpmath.e)
    rate = (digits_at_200 - digits_at_100) / 100
    assert rate > 0.5, f"Convergence rate {rate:.2f} is too low"
```

## When Adding Code Without Tests Is Acceptable

**Never for public functions.**

For private helper functions (prefixed with `_`), tests are optional if:
- The helper is trivially simple (one-liner, no branching).
- The helper is fully exercised by tests of the public function that calls it.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_sanity.py -v
```

## Coverage Targets

- **New code**: 100% of public functions tested.
- **Overall project**: Aim for 80%+ line coverage, but prioritize meaningful tests over coverage numbers.
- **Mathematical correctness tests are more valuable than line coverage.** A test that verifies a formula to 100 digits is worth more than 10 tests that check argument types.
