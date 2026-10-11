"""End-to-end reuse behaviour of the search stage (per-attribute fingerprints).

Runs ``SearcherModV1`` on a real 2F1 log(2) shard (serial stage pool) and pins:

  - a rerun with an unchanged config does **no** work (no new lines, no LIReC);
  - adding a Tier-2 attribute computes **only** that attribute, as seeded
    patches — no new Tier-1 rows and no LIReC call;
  - rows written by the analysis stage (Tier-1 only) get their Tier-2
    attributes by seeding, not by recomputing Tier-1;
  - Tier-2 values from a fresh row and from a seeded patch agree;
  - ``REUSE_POLICY='keep'`` reuses stale rows; ``'recompute'`` redoes rows of
    an earlier run but reuses rows of the current one.
"""

import json

import pytest

from dreamer import log
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids

LOG2 = log(2)


@pytest.fixture(scope="module")
def shard():
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq

    old = (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
           config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS)
    config.search.DEFAULT_USES_INV_T = False
    config.extraction.STRATEGY = "heuristic"
    config.extraction.LOAD_SHARD_CACHE = False
    sys_config.EXPORT_CMFS = None
    try:
        shards = ShardExtractor([LOG2], pFq(LOG2, 2, 1, -1).to_cmf()).extract()
    finally:
        (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
         config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS) = old
    for sh in shards:
        if tuple(int(v) for v in sh.get_interior_point().values()) == (1, 1, 3):
            return sh
    raise AssertionError("log(2) shard (1,1,3) not found")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated store, small search, deterministic sampling, LIReC counter."""
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    monkeypatch.setattr(config.search, "NUM_TRAJECTORIES_FROM_DIM", lambda d: 12)
    monkeypatch.setattr(config.analysis, "NUM_TRAJECTORIES_FROM_DIM", lambda d: 12)
    monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
    monkeypatch.setattr(config.analysis, "IDENTIFY_THRESHOLD", -1)
    monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", None)
    monkeypatch.setattr(sys_config, "REUSE_POLICY", "strict")
    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (("eigenvalues", "if_identified"),))
    calls = {"lirec": 0}
    real = ta.db.identify

    def counting(*a, **k):
        calls["lirec"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ta.db, "identify", counting)
    return tmp_path, calls


def _search(shard):
    from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
    SearcherModV1({LOG2: [shard]}, use_LIReC=True).execute()


def _analyse(shard):
    from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
    AnalyzerModV1({LOG2: [shard]}).execute()


def _lines(tmp_path, shard):
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = tmp_path / f"{shard_id}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _identified(rows):
    return [r for r in rows if r.get("identified")]


@pytest.mark.timeout(600)
class TestSearchReuse:
    def test_rerun_does_nothing(self, shard, env):
        tmp, calls = env
        _search(shard)
        first = _lines(tmp, shard)
        assert _identified(first), "no identified rows — test is vacuous"
        assert all("eigenvalues" in r for r in _identified(first))
        calls["lirec"] = 0
        _search(shard)
        assert _lines(tmp, shard) == first
        assert calls["lirec"] == 0

    def test_adding_an_attribute_computes_only_it(self, shard, env, monkeypatch):
        tmp, calls = env
        _search(shard)
        n_rows = len(_lines(tmp, shard))
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES",
                            (("eigenvalues", "if_identified"), ("gcd_slope", "if_identified")))
        calls["lirec"] = 0
        _search(shard)
        new = _lines(tmp, shard)[n_rows:]
        assert new, "no patches written"
        for line in new:
            assert "delta" not in line                     # patch, not a new Tier-1 row
            assert "eigenvalues" not in line               # fresh value not recomputed
            assert set(line["fingerprints"]) == {"gcd_slope"}
        assert calls["lirec"] == 0                          # seeded: no identification
        merged = load_seen_trajectories(str(next(tmp.glob("*.jsonl"))))
        for by_c in merged.values():
            rec = by_c[LOG2.name]
            if rec.get("identified"):
                assert "gcd_slope" in rec and set(rec["fingerprints"]) >= {"eigenvalues", "gcd_slope"}

    def test_analysis_rows_get_tier2_by_seeding(self, shard, env):
        tmp, calls = env
        _analyse(shard)
        rows_after_analysis = len(_lines(tmp, shard))
        calls["lirec"] = 0
        _search(shard)
        lines = _lines(tmp, shard)[rows_after_analysis:]
        patches = [l for l in lines if "delta" not in l]
        assert patches, "analysis rows were not patched with Tier-2"
        assert all("eigenvalues" in p for p in patches)
        # LIReC only for trajectories search sampled that analysis did not.
        new_rows = [l for l in lines if "delta" in l]
        assert calls["lirec"] <= 2 * len(new_rows)

    def test_fresh_and_seeded_tier2_agree(self, shard, env, monkeypatch):
        tmp, _ = env
        _search(shard)                                       # Tier-1 + eigenvalues in one task
        fresh = {r["trajectory_id"]: r for r in _identified(_lines(tmp, shard))}
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp / "b"))
        (tmp / "b").mkdir()
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        _search(shard)                                       # Tier-1 only
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (("eigenvalues", "if_identified"),))
        _search(shard)                                       # eigenvalues by seeding
        merged = load_seen_trajectories(str(next((tmp / "b").glob("*.jsonl"))))
        compared = 0
        for tid, row in fresh.items():
            seeded = merged.get(tid, {}).get(LOG2.name)
            if seeded and "eigenvalues" in seeded:
                assert seeded["eigenvalues"] == row["eigenvalues"]
                compared += 1
        assert compared >= 2


@pytest.mark.timeout(600)
class TestPolicies:
    def test_keep_reuses_stale_rows(self, shard, env, monkeypatch):
        tmp, calls = env
        _search(shard)
        before = _lines(tmp, shard)
        monkeypatch.setattr(config.search, "LIMIT_DIFF_ERROR_BOUND", config.search.LIMIT_DIFF_ERROR_BOUND * 10)
        monkeypatch.setattr(sys_config, "REUSE_POLICY", "keep")
        calls["lirec"] = 0
        _search(shard)
        assert _lines(tmp, shard) == before and calls["lirec"] == 0
        monkeypatch.setattr(sys_config, "REUSE_POLICY", "strict")
        _search(shard)                                       # strict: Tier-1 is stale now
        assert len(_lines(tmp, shard)) > len(before)

    def test_recompute_redoes_earlier_runs_only(self, shard, env, monkeypatch):
        tmp, _ = env
        monkeypatch.setattr(sys_config, "RUN_ID", "run-A")
        _search(shard)
        n_a = len(_lines(tmp, shard))
        monkeypatch.setattr(sys_config, "REUSE_POLICY", "recompute")
        monkeypatch.setattr(sys_config, "RUN_ID", "run-B")
        _search(shard)
        n_b = len(_lines(tmp, shard))
        assert n_b == 2 * n_a                                # every row of run-A redone
        _search(shard)                                       # same run id → reused
        assert len(_lines(tmp, shard)) == n_b


@pytest.mark.timeout(600)
def test_attribute_report_counts(shard, env, monkeypatch):
    """The report sees fresh values after a search, stale ones after a relevant
    knob change, and missing ones for an attribute not computed yet."""
    from dreamer.utils.storage.attribute_report import attribute_report, format_attribute_report

    tmp, _ = env
    _search(shard)
    rep = attribute_report(str(tmp), ["eigenvalues", "gcd_slope"])
    n = rep["tier1"]["fresh"]
    assert n > 0 and rep["tier1"]["stale"] == 0
    assert rep["eigenvalues"]["fresh"] > 0 and rep["eigenvalues"]["stale"] == 0
    assert rep["gcd_slope"]["missing"] == n
    monkeypatch.setattr(config.search, "LIMIT_DIFF_ERROR_BOUND", config.search.LIMIT_DIFF_ERROR_BOUND * 10)
    rep = attribute_report(str(tmp), ["eigenvalues"])
    assert rep["tier1"]["stale"] == n
    assert rep["eigenvalues"]["stale"] == 0          # spectral values do not depend on it
    assert "attribute" in format_attribute_report(rep)
