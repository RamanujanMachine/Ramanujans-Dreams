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
- Files that do not follow `test_*.py` are considered diagnostics/scratch only and are **not** part of CI coverage.
- Keep diagnostics under `tests/diagnostics/` (or rename to `test_*.py`) to avoid silent coverage gaps.

### Framework
- Use `pytest`.
- Use `mpmath` for high-precision reference values.
- Use `pytest.approx` for floating-point comparisons, or compare `mpmath` values with explicit tolerance.
- Runtime guardrail: tests default to a 60-second timeout each (configurable via `--test-timeout`).
- Use `@pytest.mark.timeout(<seconds>)` to override timeout for known heavy tests.

### Stochastic sampler tests
- Seed randomness (`np.random.seed(...)` or deterministic RNG objects) for repeatable test behavior.
- Assert distribution health with robust statistics (e.g., nearest-neighbor angular gap medians, per-axis variance), not exact point sets.
- Keep thresholds tolerant and physics-based to avoid flaky failures across CPU/OS differences.
- Validate across at least 2 seeds for sampler smoke tests and 3+ seeds for changes that alter sampling kernels.

### Challenge rubric (required in PR notes for non-trivial features)

Score each touched public class/module on a 0/1 basis per row:

| Dimension | Requirement |
|---|---|
| **Failure-path coverage** | At least one test asserts the expected exception/fallback path. |
| **Boundary stress** | At least one test hits strict boundaries (e.g., equality edge, zero-dim, empty constraints). |
| **Known-answer/invariant** | At least one test checks a mathematical identity or known-answer target. |
| **Stochastic robustness** | For randomized code, test statistical health instead of exact values. |
| **Regression trap** | Include a test that would fail on the bug fixed in the PR. |

Minimum bar for acceptance: **3/5** for ordinary changes, **4/5** for extraction/sampling/search changes.

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
pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing

# Override default timeout guardrail (seconds per test)
pytest tests/ -v --test-timeout=90

# Run a specific test file
pytest tests/test_sanity.py -v
```

For PR transparency, include the exact coverage command output (or CI link) and list touched files with line/branch coverage deltas.

## Coverage Targets

- **New code**: 100% of public functions tested.
- **Changed files**: target **90%+ line** and **80%+ branch** coverage; if lower, explain why and add follow-up tasks. But prioritize meaningful tests over coverage numbers.
- **Overall project**: maintain **80%+ line** and **65%+ branch** coverage. But prioritize meaningful tests over coverage numbers.
- **Critical math/search paths** (`dreamer/extraction`, `dreamer/search`): maintain **85%+ line** and **75%+ branch** coverage. But prioritize meaningful tests over coverage numbers.
- **Mathematical correctness tests are more valuable than line coverage.** A test that verifies a formula to 100 digits is worth more than 10 tests that check argument types.
