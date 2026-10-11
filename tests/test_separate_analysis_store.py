"""
Tests for the optional separate analysis-trajectory store
(``analysis.STORE_TRAJECTORIES_SEPARATELY`` + ``EXPORT_ANALYSIS_RESULTS``).

Covers the cross-stage carry-over helper
``multi_processing.load_seen_trajectories_for_search`` in isolation (pure
JSONL I/O — no CMF / walk machinery), which is the single chokepoint every
search-stage module uses to seed its per-shard cache.
"""

import json
import os

import pytest

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.multi_processing import (
    load_seen_trajectories,
    load_seen_trajectories_for_search,
)


def _write_jsonl(path, records):
    """Write *records* (list of dicts) as one JSON line each."""
    with open(path, "w") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")


def _read_ids(path):
    """Return the list of ``trajectory_id`` values in *path* (in file order)."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["trajectory_id"])
    return ids


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Distinct search-results and analysis directories wired into config."""
    search_dir = tmp_path / "search"
    analysis_dir = tmp_path / "analysis"
    search_dir.mkdir()
    analysis_dir.mkdir()
    monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(search_dir))
    monkeypatch.setattr(sys_config, "EXPORT_ANALYSIS_RESULTS", str(analysis_dir))
    return search_dir, analysis_dir


class TestLoadSeenForSearch:

    def test_flag_off_ignores_analysis_store(self, stores, monkeypatch):
        """With the flag off the helper matches plain ``load_seen_trajectories``."""
        search_dir, analysis_dir = stores
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)

        shard_id = "shardA"
        search_path = search_dir / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "s1", "constant": "e"}])
        # An analysis-store record that must be IGNORED while the flag is off.
        _write_jsonl(analysis_dir / f"{shard_id}.jsonl", [{"trajectory_id": "a1", "constant": "e"}])

        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        assert set(seen) == {"s1"}
        assert _read_ids(search_path) == ["s1"], "Search file must be untouched"

    def test_flag_on_copies_analysis_records_into_search_file(self, stores, monkeypatch):
        """Analysis records absent from the search file are copied in and merged."""
        search_dir, analysis_dir = stores
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)

        shard_id = "shardB"
        search_path = search_dir / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "s1", "constant": "e", "delta": 0.5}])
        _write_jsonl(
            analysis_dir / f"{shard_id}.jsonl",
            [
                {"trajectory_id": "a1", "constant": "e", "delta": 1.0},
                {"trajectory_id": "a2", "constant": "e", "delta": 2.0},
            ],
        )

        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        # Returned cache is the union (nested {tid: {const: record}}).
        assert set(seen) == {"s1", "a1", "a2"}
        assert seen["a2"]["e"]["delta"] == 2.0
        # The analysis records were physically copied into the search file.
        assert set(_read_ids(search_path)) == {"s1", "a1", "a2"}

    def test_does_not_overwrite_existing_search_record(self, stores, monkeypatch):
        """A trajectory present in both stores keeps the search file's version."""
        search_dir, analysis_dir = stores
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)

        shard_id = "shardC"
        search_path = search_dir / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "t1", "constant": "e", "delta": 9.9}])
        _write_jsonl(
            analysis_dir / f"{shard_id}.jsonl",
            [{"trajectory_id": "t1", "constant": "e", "delta": 0.0}],
        )

        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        assert seen["t1"]["e"]["delta"] == 9.9
        # No duplicate line appended for the already-present id.
        assert _read_ids(search_path) == ["t1"]

    def test_rerun_does_not_duplicate_copied_records(self, stores, monkeypatch):
        """Running the helper twice appends each analysis record at most once."""
        search_dir, analysis_dir = stores
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)

        shard_id = "shardD"
        search_path = search_dir / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "s1", "constant": "e"}])
        _write_jsonl(analysis_dir / f"{shard_id}.jsonl", [{"trajectory_id": "a1", "constant": "e"}])

        load_seen_trajectories_for_search(str(search_path), shard_id)
        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        assert sorted(_read_ids(search_path)) == ["a1", "s1"]
        assert set(seen) == {"s1", "a1"}

    def test_missing_analysis_file_is_noop(self, stores, monkeypatch):
        """No analysis file for the shard → behaves like the plain loader."""
        search_dir, _analysis_dir = stores
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)

        shard_id = "shardE"
        search_path = search_dir / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "s1", "constant": "e"}])

        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        assert set(seen) == {"s1"}
        assert _read_ids(search_path) == ["s1"]

    def test_same_dir_misconfig_is_noop(self, tmp_path, monkeypatch):
        """If the analysis store points at the search file, nothing is copied."""
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(shared))
        monkeypatch.setattr(sys_config, "EXPORT_ANALYSIS_RESULTS", str(shared))
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)

        shard_id = "shardF"
        search_path = shared / f"{shard_id}.jsonl"
        _write_jsonl(search_path, [{"trajectory_id": "s1", "constant": "e"}])

        seen = load_seen_trajectories_for_search(str(search_path), shard_id)

        assert set(seen) == {"s1"}
        assert _read_ids(search_path) == ["s1"]
