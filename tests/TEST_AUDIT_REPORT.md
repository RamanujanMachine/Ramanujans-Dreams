# Test Audit Report (2026-10-10, b)

Bottom line: simulated annealing gained an asynchronous speculative look-ahead
(`ANNEAL_LOOKAHEAD`, `WalkPrefetcher`); the stage templates in `examples/` were fixed.
The look-ahead is proven output-identical (rows, order, `seen`, RNG state, Metropolis
draws) in-process and through a real 2-process pool. Full suite: `914 passed, 2 skipped`.

## Touched Modules (Detailed Review)

| Touched module | Changes made | Coverage evidence | Challenge rubric (/5) | Regression evidence |
|---|---|---|---:|---|
| `dreamer/search/methods/flatland/parallel_eval.py` | `WalkPrefetcher` (idle-capacity, group-wise async speculation; claim/take; age eviction); `evaluate_batch(prefetcher=, speculate=)`. | `tests/test_anneal_lookahead.py` (25 tests); 88%. | 5 | Uncommitted-speculation, capacity, dedup, eviction, claim-before-speculate regression, prefetched ≡ direct commit tests. |
| `dreamer/search/methods/annealing/annealing_scan.py` | Look-ahead wiring, `lookahead_enabled` policy, `_next_step_groups`. | 85%; policy parametrised (off/on/auto incl. d_flat = 11 → 44 workers, Tier-2 gate, invalid mode). | 5 | Real-walk SA equivalence in accept- and reject-dominated regimes (asserts Metropolis draws happened and the look-ahead served walks); byte-identical JSONL via `SimulatedAnnealingMod` on a real pool. |
| `dreamer/configs/search.py` | `ANNEAL_LOOKAHEAD` (`auto`). | Policy tests. | 4 | — |
| `examples/search.py`, `examples/analysis.py` | Fixed to `build_trajectory_dtos`, flat rows, nested cache. | `tests/test_example_templates.py` (2 end-to-end tests). | 4 | Template API drift now fails CI. |

Challenge rubric breakdown: failure path (eviction race regression, invalid mode), boundary
(capacity edge, empty groups, 1-worker pool, d_flat thresholds), invariant (on ≡ off outputs),
stochastic robustness (two temperature regimes, RNG-state equality), regression trap (byte-identical files).

Benchmark (18 cores, pFq(2,1) log 2, SA 100 iterations): Tier-2 off — SA 214 s → 181 s (−15 %),
byte-identical files; Tier-2 on — +11 % (speculation competes with the Tier-2 bottleneck), so
`auto` stays off while the Tier-2 queue is active.

## Non-Touched Modules (Repository-Wide Summary)

| Area | Status | Risk / follow-up |
|---|---|---|
| Tier-2 queue | Green, unchanged. | Bottleneck + Tier-1 redo (design pending). With Tier-2 on, row *order* in search files is nondeterministic (MPMC writer); content identical. |
| Other optimisers | Green, unchanged. | Look-ahead not applied (micro-climb Phase A is a candidate). |
| Repository | 69% line+branch coverage. | — |

---

# Test Audit Report (2026-10-10, a)

Bottom line: the analyzer, `SearcherModV1` (Tier-1) and the Tier-3 post-process now
run their per-trajectory work on one stage-wide process pool
(`dreamer/utils/stage_pool.py`) with the main process as the single JSONL writer.
Serial and 2-process runs write byte-identical files. Full suite green:
`887 passed, 2 skipped, 2 warnings` (baseline before the change: `857 passed, 2 skipped`).

## Touched Modules (Detailed Review)

| Touched module | Changes made | Coverage evidence | Challenge rubric (/5) | Regression evidence |
|---|---|---|---:|---|
| `dreamer/utils/stage_pool.py` (new) | `StagePool` (persistent per-stage pool, shards via initializer, shared (p,q) logs, ordered `imap`), `BufferedJsonlWriter`, `run_tier1_task`, budget helpers. | `tests/test_stage_pool.py` (30 tests); 90% line+branch. Uncovered: the worker-side initializer body (runs only in spawned processes, exercised by the 2-process tests but not traced by coverage). | 5 | Ordered-delivery, context-in-worker, error-isolation, terminate-on-exception and batching tests fail on any contract drift. |
| `dreamer/analysis/analyzers/serial_scan/analyzer_mod.py` | Split into plan → compute (stage pool) → finish; main-process buffered writer. | Existing analyzer tests (11) + equivalence, cache-rerun and failed-walk tests; 92%. | 5 | Serial vs 2-process: identical rows, file order and priorities; fully cached rerun must not create a pool. |
| `dreamer/search/searchers/hedgehog_scan_mod.py` | One stage-wide Tier-1 pool for all shards; producer plans then dispatches; matrix shipped only when Tier-2 is active. | Existing producer tests + parallel equivalence + failed-walk test; 94%. | 4 | Serial vs 2-process search files byte-identical. |
| `dreamer/post_process/tier3_post_process_mod.py` | Tier-3 rows planned in main, computed on a stage pool (CMF lookup sent once), patches written by main. | Existing Tier-3 tests + parallel equivalence; 84%. | 4 | Serial vs 2-process patches byte-identical. |
| `dreamer/configs/system.py` | New `STAGE_POOL_WORKERS`, `STAGE_POOL_CHUNKSIZE`. | 100%; cap resolution parametrised over None/0/negative/below/above budget. | 4 | `TestBudget` fails on cap-semantics drift. |
| `tests/conftest.py` | Autouse `serial_stage_pools` fixture (stage pools serial by default in unit tests). | Whole suite. | — | Keeps monkeypatch-based tests valid; real multi-process paths are tested explicitly. |

Challenge rubric breakdown (this task):
- Failure-path coverage: yes (task errors in serial/parallel streams, failing walks in analyzer + search, exception inside the pool context).
- Boundary stress: yes (empty task list, no writes, cap edge values, single worker).
- Known-answer / invariant: yes (`run_tier1_task` rows equal the direct serial walk; serial ≡ parallel files).
- Stochastic robustness: n/a for the pool (sampling stays in the main process, seeded as before).
- Regression trap: yes (byte-identical equivalence tests for all three stages).

## Non-Touched Modules (Repository-Wide Summary)

| Area | Status from latest run | Risk / follow-up |
|---|---|---|
| Optimiser searchers (GA, SA, annealing, gradient, SPSA) | Green; unchanged. | Still limited by per-step batch size and per-shard pools — next iteration. |
| Tier-2 queue (`worker_pool`, `compute_tier2_for_item`) | Green; unchanged. | Now the search-stage bottleneck (fixed `NUM_BACKGROUND_WORKERS`); worker rebuilds a matrix-only handler at depth 1500. Parked for design discussion. |
| `examples/search.py`, `examples/analysis.py` | **Fail to import** (pre-existing: use the removed `build_trajectory_dto`). | Fix templates to `build_trajectory_dtos`. |
| Repository total | 68% line+branch coverage (10426 stmts). | Coverage debt unchanged outside touched modules. |

## Executed Test Evidence

```bash
python -m pytest tests -q --ignore=tests/pocs                     # 887 passed, 2 skipped
python -m pytest tests -q --ignore=tests/pocs --cov=dreamer --cov-branch --cov-report=term-missing
python -m pytest tests/test_stage_pool.py -q                      # 30 passed
```

Performance evidence (18-core Mac, pFq(2,1) log 2, 12 shards, 300 analysis / 600 search
trajectories per shard, A/B against commit `dfed794`): analysis 23 s @ 1.0 core → 6 s @
10.6 cores; Tier-3 13 s @ 3.6 → 4 s @ 12.9; search 12 s → 13 s (Tier-2-bound; 3 s @ 9.6
cores with Tier-2 off); total 52 s → 26 s.

---

# Previous report

## Test Audit Report (2026-04-24)

Bottom line: DataManager now carries its searchable space context, a new `JSONable` contract was introduced, and Shard/DataManager JSON export-import paths were implemented and validated; full suite and policy coverage runs are green (`173 passed`, `1 warning`).

## Touched Modules (Detailed Review)

| Touched module | Changes made | Coverage evidence | Challenge rubric (/5) | Regression evidence |
|---|---|---|---:|---|
| `dreamer/utils/schemes/jsonable.py` | Added abstract `JSONable` attribute class enforcing `to_json` implementation. | Covered by `tests/test_storage_objects.py::test_jsonable_requires_to_json_implementation`. | 3 | Instantiation fails for subclass without `to_json`, trapping contract drift. |
| `dreamer/utils/storage/storage_objects.py` | Extended `DataManager` with `searchable_space`; added `to_json`/`from_json_obj` support for nested JSONable payloads and backward-compatible `to_json_obj`. | Covered by `tests/test_storage_objects.py::test_data_manager_json_roundtrip_preserves_searchable_space_and_entries`; included in full suite + coverage run. | 4 | Roundtrip test fails if searchable context or entries are dropped during JSON serialization. |
| `dreamer/extraction/shard.py` | Implemented `JSONable` for `Shard` via `to_json`/`from_json_obj` + `to_json_obj` alias for exporter compatibility. | Covered by `tests/test_shard.py::TestShardJsonSerialization::test_shard_json_roundtrip_via_exporter_importer`; module now `89%` line coverage (`76` stmts, `7` missed). | 4 | Test fails if JSON export/import no longer restores a functional Shard instance. |
| `dreamer/utils/storage/exporter.py` | Added recursive JSON conversion helper so nested objects/lists/dicts of JSONable payloads export correctly. | Exercised through Shard/DataManager JSON export tests and system JSON export tests. | 4 | JSON priority export tests fail if list-of-Shard payloads are not JSON-converted recursively. |
| `dreamer/utils/storage/importer.py` | Added recursive JSON restoration for `DataManager` and `Shard` payloads. | Exercised through Shard roundtrip and system JSON import tests; module now `68%` line coverage (`49` stmts, `13` missed). | 4 | CMF-filtered JSON searchable import test fails if nested payload restoration regresses. |
| `dreamer/configs/system.py` | Added configurable formats: `EXPORT_SEARCHABLES_FORMAT` and `EXPORT_ANALYSIS_PRIORITIES_FORMAT`. | Indirectly validated by system JSON export/import tests that set these fields. | 3 | JSON system tests fail if config-driven format routing is removed. |
| `dreamer/extraction/extractor.py` + `dreamer/system/system.py` | Switched searchable/priorities export-import paths from pickle-only to config-driven `Formats(...)`. | Validated by `tests/test_system_priorities_io.py` JSON + pickle scenarios (`5/5` passing). | 4 | Both JSON and pickle path tests fail on extension/format mismatches. |
| `dreamer/search/methods/genetic.py` + `dreamer/search/methods/hedgehog_scan.py` | Ensure internally created DataManagers include `searchable_space=self.space`. | Covered by full suite regression paths in `tests/test_search_genetic.py` and related module tests. | 3 | Search module behavior would regress if DataManager construction signature drifted. |
| `tests/test_storage_objects.py`, `tests/test_shard.py`, `tests/test_system_priorities_io.py` | Added targeted tests for JSONable contract, DataManager searchable-space roundtrip, Shard JSON roundtrip, and system-level JSON export/import configuration paths. | Targeted run: `36 passed`; full suite + coverage run: `173 passed`. | 5 | New tests directly trap the new feature surface and config-driven IO behavior. |

Challenge rubric breakdown (this task):
- Failure-path coverage: yes (`JSONable` abstract-contract failure path explicitly asserted).
- Boundary stress: yes (CMF-filtered import against mixed wanted/other CMF directories for JSON path).
- Known-answer / invariant: yes (DataManager/Shard JSON roundtrip invariants on identity-critical fields).
- Stochastic robustness: not applicable (deterministic serialization/IO).
- Regression trap: yes (new tests fail if JSON routing, restoration, or DataManager context persistence drifts).

## Non-Touched Modules (Repository-Wide Summary)

| Area | Status from latest run | Risk / follow-up |
|---|---|---|
| Existing extraction/search tests outside touched files | All pre-existing tests remain green under the new JSON/config changes. | Low immediate regression risk. |
| Remaining runtime modules not directly touched | Full suite and coverage command passed (`173 passed`, `1 warning`). | Moderate ongoing risk is still coverage debt in low-covered modules. |

## Executed Test Evidence

Commands executed in this cycle:

```bash
python -m pytest -q tests/test_storage_objects.py tests/test_shard.py tests/test_system_priorities_io.py tests/test_search_genetic.py
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing
```

Observed outcomes:
- Targeted regression run: `36 passed`, `1 warning`.
- Full suite run: `173 passed`, `1 warning`.
- Full suite + coverage run: `173 passed`, `1 warning`.

## Coverage Command Output Snapshot

Required command status:
- `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` -> **executed successfully**.

Coverage highlights from this run:
- `dreamer/extraction/shard.py`: `89%` line coverage (`76` stmts, `7` missed), branch partials present.
- `dreamer/utils/storage/storage_objects.py`: `66%` line coverage (`99` stmts, `25` missed), branch partials present.
- `dreamer/utils/storage/importer.py`: `68%` line coverage (`49` stmts, `13` missed), branch partials present.
- Overall project coverage: `56%` line coverage.

## Notes / Remaining Risks

1. Workspace remains in a pre-existing dirty state; this report evaluates touched files plus repository-wide test outcomes.
2. The LIReC SQLAlchemy deprecation warning persists and is unrelated to this feature set.
3. Coverage policy thresholds remain below target globally and in critical paths (`dreamer/extraction`, `dreamer/search`) despite green tests.
