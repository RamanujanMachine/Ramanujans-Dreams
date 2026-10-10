"""Tests for computing Tier-2/3 attributes from a stored row (seeded handler).

A stored row keeps δ (float64), ``identified``, the exact p/q vectors and the
projection column — not the walk.  ``attributes_patch`` rebuilds the handler at
the row's depth and **seeds** those values instead of recomputing them.  These
tests pin that this is both *correct* and *cheaper*:

  - Known answer: every default Tier-2 attribute (and the walk-based
    ``digits_computed``) computed from the seeded row equals the value computed
    on the fresh handler that produced the row (eigenvalues: identical strings;
    floats: within 1e-9 relative).
  - LIReC is **never** called on the seeded path (``db.identify`` is replaced
    by a function that fails the test).
  - Eigenvalue-only attributes do not walk at all; the full-depth walk happens
    only for attributes that genuinely need it.
  - Predicates read the seeded ``identified`` (a non-identified row gets no
    gated attribute); patches carry per-attribute fingerprints; the JSONL loader
    deep-merges fingerprint dicts across patch rows.
"""

import json
import math

import numpy as np
import pytest

from dreamer import log
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.attribute_fingerprints import attribute_fingerprint
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    attributes_patch,
    build_trajectory_dtos,
    derive_cmf_and_shard_ids,
)

LOG2 = log(2)

TIER2 = (
    ("eigenvalues", "if_identified"), ("eigenvalue_errors", "if_identified"),
    ("spectral_gap", "if_identified"), ("delta_prediction", "if_identified"),
    ("gcd_slope", "if_identified"), ("error_formula_ratio", "if_identified"),
    ("approximated_digits_per_step", "if_identified"), ("digits_approximation", "if_identified"),
    ("convergence_rate", "if_identified"), ("digits_computed", "if_identified"),
    ("avg_computed_digits_per_step", "if_identified"),
)


@pytest.fixture(scope="module")
def shard():
    """2F1(z=-1) log(2) shard with interior (1, 1, 3) — identifies on most rays."""
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


@pytest.fixture(autouse=True)
def _walk_type(monkeypatch):
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)


def _directions(shard):
    syms = list(shard.cmf.matrices.keys())
    from ramanujantools import Position
    return [Position(dict(zip(syms, d))) for d in ((1, 1, 2), (2, 1, 3), (1, 2, 4))]


def _fresh_row_and_values(shard, direction):
    """Tier-1 row + Tier-2 values computed on one fresh handler (the reference)."""
    start = shard.get_interior_point()
    cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
    h = TrajectoryAttributesHandler.from_cmf(shard.cmf, direction, start, searchable=shard)
    (dto,) = build_trajectory_dtos(
        h, cmf_id=cmf_id, shard_id=shard_id, cmf_name=shard.cmf_name,
        shard_encoding_str=enc, start=start, direction=direction,
        constants=[LOG2], extra_metrics=TIER2,
    )
    row = json.loads(dto.to_json_line())
    reference = {k: row[k] for k in row if k in {a for a, _ in TIER2}}
    base = {k: v for k, v in row.items() if k not in reference and k != "fingerprints"}
    return base, reference


def _seeded_handler(shard, row):
    from ramanujantools import Position
    syms = list(shard.cmf.matrices.keys())
    start = Position(dict(zip(syms, row["start_point"])))
    direction = Position(dict(zip(syms, row["direction"])))
    return TrajectoryAttributesHandler.from_cmf(
        shard.cmf, direction, start, walk_depth=row["walk_depth"], searchable=shard,
    )


def _no_lirec(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LIReC called on the seeded path")
    monkeypatch.setattr(ta.db, "identify", boom)


def _close(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.timeout(600)
class TestSeededEqualsFresh:
    def test_all_default_tier2_attributes_match_without_lirec(self, shard, monkeypatch):
        checked = 0
        for direction in _directions(shard):
            row, reference = _fresh_row_and_values(shard, direction)
            if not row["identified"]:
                continue
            _no_lirec(monkeypatch)
            patch = attributes_patch(_seeded_handler(shard, row), LOG2, row, TIER2)
            monkeypatch.undo()
            monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
            for name, value in reference.items():
                assert name in patch, f"{name} missing from seeded patch"
                assert _close(patch[name], value), (name, patch[name], value)
            checked += 1
        assert checked >= 2, "too few identified trajectories — test is vacuous"

    def test_patch_carries_fingerprints(self, shard):
        row, _ = _fresh_row_and_values(shard, _directions(shard)[0])
        patch = attributes_patch(_seeded_handler(shard, row), LOG2, row, (("gcd_slope", "if_identified"),))
        assert patch["fingerprints"] == {"gcd_slope": attribute_fingerprint("gcd_slope", row["walk_depth"])}
        assert patch["trajectory_id"] == row["trajectory_id"] and patch["constant"] == row["constant"]


@pytest.mark.timeout(600)
class TestSeededCost:
    def test_eigenvalue_only_attributes_do_not_walk(self, shard, monkeypatch):
        row, _ = _fresh_row_and_values(shard, _directions(shard)[0])
        h = _seeded_handler(shard, row)
        walked = []
        real = TrajectoryAttributesHandler._walked_matrix

        def spy(self, depth):
            walked.append(depth)
            return real(self, depth)

        monkeypatch.setattr(TrajectoryAttributesHandler, "_walked_matrix", spy)
        _no_lirec(monkeypatch)
        patch = attributes_patch(h, LOG2, row, (("eigenvalues", "if_identified"), ("spectral_gap", "if_identified")))
        assert "eigenvalues" in patch and walked == []

    def test_full_walk_only_for_walk_attributes(self, shard, monkeypatch):
        row, _ = _fresh_row_and_values(shard, _directions(shard)[0])
        h = _seeded_handler(shard, row)
        walked = []
        real = TrajectoryAttributesHandler._walked_matrix

        def spy(self, depth):
            walked.append(depth)
            return real(self, depth)

        monkeypatch.setattr(TrajectoryAttributesHandler, "_walked_matrix", spy)
        _no_lirec(monkeypatch)
        attributes_patch(h, LOG2, row, (("gcd_slope", "if_identified"),))
        assert all(d < row["walk_depth"] for d in walked)       # short walks only
        attributes_patch(h, LOG2, row, (("digits_computed", "if_identified"),))
        assert row["walk_depth"] in walked                      # now the full walk


class TestSeedSemantics:
    def test_unidentified_row_gets_no_gated_attributes(self, shard, monkeypatch):
        row, _ = _fresh_row_and_values(shard, _directions(shard)[0])
        row = dict(row, identified=False, delta=float("-inf"), p_vector=None, q_vector=None)
        _no_lirec(monkeypatch)
        patch = attributes_patch(_seeded_handler(shard, row), LOG2, row, TIER2)
        assert set(patch) == {"trajectory_id", "constant"}

    def test_loader_deep_merges_fingerprints(self, tmp_path):
        path = tmp_path / "s.jsonl"
        rows = [
            {"trajectory_id": "t", "constant": "c", "delta": 1.0, "fingerprints": {"a": "1"}, "a": 1},
            {"trajectory_id": "t", "constant": "c", "b": 2, "fingerprints": {"b": "2"}},
            {"trajectory_id": "t", "constant": "c", "a": 3, "fingerprints": {"a": "3"}},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        rec = load_seen_trajectories(str(path))["t"]["c"]
        assert rec["fingerprints"] == {"a": "3", "b": "2"}
        assert (rec["a"], rec["b"], rec["delta"]) == (3, 2, 1.0)
