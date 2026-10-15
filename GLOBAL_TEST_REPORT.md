# Global Test Report

Last full-suite run: **2026-10-16** (branch `fix/memory-and-idle-cores`) (`rama` env, Python 3.12.15, sympy 1.14.0,
python-flint 0.9.0, mpmath 1.3.0, macOS, 18 cores).

| Metric | Value |
|---|---|
| Command | `python -m pytest tests -q --cov=dreamer --cov-branch --cov-report=term-missing` |
| Collected | 1427 |
| Passed | 1423 |
| Failed | 0 |
| Skipped | 4 |
| Warnings | 2 |
| Line+branch coverage (`--cov=dreamer --cov-branch`) | 74% (12462 statements, 4302 branches) |

## Failing tests

None.

## Warnings

1. `LIReC/db/models.py:10` — SQLAlchemy `MovedIn20Warning` (`declarative_base`); third-party, harmless.
2. `tests/test_attributes_management.py::TestKamideltaEigenvalueSelection::test_prediction_tracks_delta`
   — `PytestRemovedIn10Warning`: class-scoped fixture defined as an instance method. Fix by
   making the fixture a `@classmethod` before pytest 10.

## Uncovered cases / known gaps

- `ResilientPool._ensure_workers`: the early returns for `fork`, unknown executor internals and a
  shutting-down executor are defensive and untested.
- `parallel_eval._wait_any`: the real-future path is exercised only by the real-pool SA test and the
  benchmarks; unit tests use the in-process pool double (fallback path).

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
- `search/searchers/spsa_adam_mod.py` is at 32%: its shard loop (`_run_shard`) has no test (pre-existing gap;
  the other four optimiser modules are at 98%). The `EvalStats` wiring there is checked structurally
  (`tests/test_eval_stats.py::TestWiring`).
