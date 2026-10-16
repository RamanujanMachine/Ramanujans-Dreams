"""Tests for *slim* loading of the per-shard trajectory JSONL.

Background (measured 2026-10-15): with ``('relation', 'if_identified')`` every
identified row gets a Tier-3 recurrence ``relation`` of 0.3-2 MB of text; one
shard's JSONL reached 945 MB, and a full ``load_seen_trajectories`` kept all of
it in the main process for the whole search.  The stages that only need Tier-1
values, scalar metrics and *which* attributes exist now load slim.

These tests pin that a slim load:
  - replaces only large, non-Tier-1 values by ``Omitted`` and keeps the key;
  - keeps Tier-1 core fields (even long p/q vectors), fingerprints and scalars;
  - gives the same staleness answers (``stale_attributes``) and the same summary
    statistics as a full load;
  - merges patch rows exactly as a full load does;
  - can never be written back to JSON silently (``Omitted`` is not serialisable);
  - is what the search loader returns, while the analysis carry-over still writes
    the full analysis rows to the search file.
"""

import json

import pytest

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.multi_processing import (
    SLIM_VALUE_CHARS,
    Omitted,
    load_seen_trajectories,
    load_seen_trajectories_for_search,
)
from dreamer.utils.storage.attribute_fingerprints import stale_attributes

BIG = "x" * (SLIM_VALUE_CHARS + 1)


def _base(tid="t1", const="c", **extra):
    row = {
        "trajectory_id": tid, "constant": const, "cmf_id": "cmf", "shard_id": "s",
        "start_point": [1, 2], "direction": [1, 1], "identified": True, "delta": 0.25,
        "p_vector": [10 ** 1000, 1], "q_vector": [3, 4], "walk_depth": 1000,
    }
    row.update(extra)
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestSlimLoad:
    def test_large_extension_value_is_omitted_key_kept(self, tmp_path):
        """A big Tier-3 value becomes ``Omitted`` (with its size); the key stays."""
        f = tmp_path / "s.jsonl"
        _write(f, [_base(relation=BIG, convergence_rate=1.5, small_list=[1, 2, 3])])
        rec = load_seen_trajectories(str(f), omit_values_over=SLIM_VALUE_CHARS)["t1"]["c"]
        assert isinstance(rec["relation"], Omitted)
        assert rec["relation"].size == len(BIG)
        assert rec["convergence_rate"] == 1.5 and rec["small_list"] == [1, 2, 3]

    def test_tier1_core_fields_are_never_omitted(self, tmp_path):
        """Long p/q vectors are Tier-1 seed data (Tier-3 needs them): kept."""
        f = tmp_path / "s.jsonl"
        _write(f, [_base()])
        rec = load_seen_trajectories(str(f), omit_values_over=10)["t1"]["c"]
        assert rec["p_vector"][0] == 10 ** 1000 and rec["delta"] == 0.25

    def test_default_load_is_full(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write(f, [_base(relation=BIG)])
        assert load_seen_trajectories(str(f))["t1"]["c"]["relation"] == BIG

    def test_patch_rows_merge_like_a_full_load(self, tmp_path):
        """A base row plus a patch row (as Tier-3 appends) merges the same way:
        last write wins, fingerprints deep-merged."""
        f = tmp_path / "s.jsonl"
        _write(f, [
            _base(fingerprints={"a": "1"}),
            {"trajectory_id": "t1", "constant": "c", "relation": BIG, "fingerprints": {"relation": "fp"}},
        ])
        full = load_seen_trajectories(str(f))["t1"]["c"]
        slim = load_seen_trajectories(str(f), omit_values_over=SLIM_VALUE_CHARS)["t1"]["c"]
        assert slim.keys() == full.keys()
        assert slim["fingerprints"] == full["fingerprints"] == {"a": "1", "relation": "fp"}
        assert {k: v for k, v in slim.items() if k != "relation"} == \
               {k: v for k, v in full.items() if k != "relation"}

    def test_staleness_is_unchanged(self, tmp_path, monkeypatch):
        """Presence-based reuse decisions are identical for slim and full rows."""
        from dreamer.utils.storage import attribute_fingerprints as af
        f = tmp_path / "s.jsonl"
        fp = af.attribute_fingerprint("relation", 1000)
        _write(f, [_base(relation=BIG, fingerprints={"relation": fp})])
        full = load_seen_trajectories(str(f))["t1"]["c"]
        slim = load_seen_trajectories(str(f), omit_values_over=SLIM_VALUE_CHARS)["t1"]["c"]
        names = ["relation", "eigenvalues"]
        assert stale_attributes(slim, names, 1000) == stale_attributes(full, names, 1000) == ["eigenvalues"]

    def test_omitted_cannot_be_written_back(self):
        with pytest.raises(TypeError):
            json.dumps({"relation": Omitted(10)})


class TestSlimConsumers:
    def test_summary_statistics_match_full_load(self, tmp_path):
        """The run summary reads only scalars: slim and full give the same stats."""
        from dreamer.utils.storage.summary import _ShardStats
        f = tmp_path / "s.jsonl"
        _write(f, [_base(tid=f"t{i}", delta=0.1 * i, relation=BIG) for i in range(5)])
        objective = sys_config.OPTIMIZATION_OBJECTIVE
        stats = []
        for kw in ({}, {"omit_values_over": SLIM_VALUE_CHARS}):
            st = _ShardStats("s", "cmf", "c", objective)
            for by_const in load_seen_trajectories(str(f), **kw).values():
                st.add(by_const["c"])
            stats.append((st.trajectories, st.identified, st.positive_delta,
                          st.best_score, st.best_trajectory_id))
        assert stats[0] == stats[1]

    def test_search_loader_is_slim_and_carryover_writes_full_rows(self, tmp_path, monkeypatch):
        """``load_seen_trajectories_for_search`` returns slim rows; analysis rows
        it copies into the search file are written in full."""
        search_dir, analysis_dir = tmp_path / "search", tmp_path / "analysis"
        search_dir.mkdir()
        analysis_dir.mkdir()
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", True)
        monkeypatch.setattr(sys_config, "EXPORT_ANALYSIS_RESULTS", str(analysis_dir))
        out = search_dir / "s.jsonl"
        _write(out, [_base(tid="t1", relation=BIG)])
        _write(analysis_dir / "s.jsonl", [_base(tid="t2", relation=BIG)])
        seen = load_seen_trajectories_for_search(str(out), "s")
        assert isinstance(seen["t1"]["c"]["relation"], Omitted)
        assert isinstance(seen["t2"]["c"]["relation"], Omitted)
        written = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["trajectory_id"] for r in written] == ["t1", "t2"]
        assert written[1]["relation"] == BIG                 # carried over in full
