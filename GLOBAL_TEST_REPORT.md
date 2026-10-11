# Global Test Report

Last full-suite run: **2026-10-11** (`rama` env, Python 3.12.15, sympy 1.14.0,
python-flint 0.9.0, mpmath 1.3.0, macOS, 18 cores).

| Metric | Value |
|---|---|
| Command | `python -m pytest tests -v --ignore=tests/pocs --cov=dreamer --cov-branch --cov-report=term-missing` |
| Collected | 1146 |
| Passed | 1144 |
| Failed | 0 |
| Skipped | 2 |
| Warnings | 2 |
| Line+branch coverage (`--cov=dreamer --cov-branch`) | 72% (11468 statements, 4002 branches) |

## Failing tests

None.

## Warnings

1. `LIReC/db/models.py:10` — SQLAlchemy `MovedIn20Warning` (`declarative_base`); third-party, harmless.
2. `tests/test_attributes_management.py::TestKamideltaEigenvalueSelection::test_prediction_tracks_delta`
   — `PytestRemovedIn10Warning`: class-scoped fixture defined as an instance method. Fix by
   making the fixture a `@classmethod` before pytest 10.

## Uncovered cases / known gaps

- `dreamer/utils/schemes/searchable.py` (legacy `DataManager` path) is at 54%: the δ code
  changed on 2026-10-11 is covered (`tests/test_high_precision.py::test_legacy_calc_delta_*`),
  but `compute_trajectory_data`'s `find_limit` / `find_eigen_values` / no-LIReC options and
  the "should never happen" δ guardrail logs are not. Follow-up: tests for those options or
  retire the legacy path.
- `mp_context.pool_context`: the `set_forkserver_preload` failure branch (defensive) is untested.
- `task_memory.release_task_memory`: the branch for a missing `ramanujantools` is untested
  (defensive import guard).
- `stage_pool._stage_pool_init` and the `ResilientPool._guarded_init` / `_measured_call`
  bodies run only inside spawned workers, so coverage does not trace them; they are
  exercised by the multi-process tests in `tests/test_stage_pool.py` and
  `tests/test_resilient_pool.py`.
- The optimiser searchers' multi-process eval pools are exercised only lightly in unit tests.
