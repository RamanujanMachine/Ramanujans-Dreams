# Test Audit Report (2026-04-24)

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
