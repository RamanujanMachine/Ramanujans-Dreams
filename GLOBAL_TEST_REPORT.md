# Global Test Report

Last full-suite run: **2026-10-10** (`rama` env, Python 3.12.15, macOS, 18 cores).

| Metric | Value |
|---|---|
| Command | `python -m pytest tests -q --ignore=tests/pocs` |
| Collected | 916 |
| Passed | 914 |
| Failed | 0 |
| Skipped | 2 |
| Warnings | 2 |
| Line+branch coverage (`--cov=dreamer --cov-branch`) | 69% (10517 statements, 3698 branches) |

## Failing tests

None.

## Warnings

1. `LIReC/db/models.py:10` — SQLAlchemy `MovedIn20Warning` (`declarative_base`); third-party, harmless.
2. `tests/test_attributes_management.py::TestKamideltaEigenvalueSelection::test_prediction_tracks_delta`
   — `PytestRemovedIn10Warning`: class-scoped fixture defined as an instance method. Fix by
   making the fixture a `@classmethod` before pytest 10.

## Uncovered cases / known gaps

- `stage_pool._stage_pool_init` runs only inside spawned workers, so coverage does not trace
  it (it is exercised by the 2-process tests in `tests/test_stage_pool.py`).
- The optimiser searchers' multi-process eval pools are exercised only lightly in unit tests.
