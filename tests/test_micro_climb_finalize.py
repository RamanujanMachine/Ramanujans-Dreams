"""
Tests for the universal micro-hill-climb finalization (search-stage assurance).

Coverage:
  - ``_best_records_for_constant``: picks the max-δ record(s), including every
    trajectory tied with it to two decimal places; ignores unidentified /
    non-finite / other-constant records.
  - ``finalize_best_trajectories``:
      * no-op (never touches disk) when ENABLE_MICRO_HILL_CLIMB is off.
      * climbs each *distinct* tied-best trajectory once, dedups trajectories
        sharing a primitive ray (no double work), and skips non-best ones.
"""

import json

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.discrete_local_max import primitive_ray_key
import dreamer.search.searchers.micro_climb_finalize as fin
from dreamer.search.searchers.micro_climb_finalize import (
    _best_records_for_constant,
    finalize_best_trajectories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cmf():
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def whole_space_shard(simple_cmf, symbols):
    zero_shift = Position({s: sp.Integer(0) for s in symbols})
    return Shard(simple_cmf, e, [], [], zero_shift)


def _rec(tid, direction, delta, identified=True, const="e", **extra):
    # One flat per-(trajectory, constant) row; metrics are top-level columns.
    return {
        "trajectory_id": tid,
        "constant": const,
        "start_point": [0, 0],
        "direction": list(direction),
        "delta": delta,
        "identified": identified,
        "config_fingerprint": "fp",
        **extra,
    }


def _seen(*recs):
    """Nest flat records into the ``{trajectory_id: {constant: record}}`` shape."""
    out = {}
    for r in recs:
        out.setdefault(r["trajectory_id"], {})[r["constant"]] = r
    return out


# ---------------------------------------------------------------------------
# _best_records_for_constant
# ---------------------------------------------------------------------------

class TestBestRecordsForConstant:
    def test_selects_ties_within_two_decimals(self):
        seen = _seen(
            _rec("a", (3, 1), 0.204),   # max
            _rec("b", (1, 3), 0.196),   # ties at 0.20
            _rec("c", (1, 1), 0.101),   # 0.10 — not tied
        )
        max_delta, best = _best_records_for_constant(seen, "e", "delta")
        assert max_delta == pytest.approx(0.204)
        ids = {r["trajectory_id"] for r in best}
        assert ids == {"a", "b"}

    def test_ignores_unidentified_and_other_constants(self):
        seen = _seen(
            _rec("a", (3, 1), 0.5, identified=False),     # not identified
            _rec("b", (1, 3), 0.4, const="pi"),           # other constant
            _rec("c", (1, 1), 0.3),                       # the only valid one
        )
        max_delta, best = _best_records_for_constant(seen, "e", "delta")
        assert max_delta == pytest.approx(0.3)
        assert [r["trajectory_id"] for r in best] == ["c"]

    def test_empty_when_no_records(self):
        max_delta, best = _best_records_for_constant({}, "e", "delta")
        assert best == []
        assert max_delta == float("-inf")

    def test_ranks_by_objective_column_when_objective_active(self):
        """Under a non-δ objective, ranking follows the convergence_rate column —
        the record with the worse δ but better convergence_rate wins."""
        seen = _seen(
            _rec("a", (3, 1), 0.9, convergence_rate=0.10),
            _rec("b", (1, 3), 0.2, convergence_rate=0.50),
        )
        max_score, best = _best_records_for_constant(seen, "e", "convergence_rate")
        assert max_score == pytest.approx(0.50)
        assert [r["trajectory_id"] for r in best] == ["b"]

    def test_ranks_by_delta_under_delta_objective(self):
        seen = _seen(_rec("a", (1, 1), 0.7), _rec("b", (2, 1), 0.3))
        max_score, best = _best_records_for_constant(seen, "e", "delta")
        assert max_score == pytest.approx(0.7)
        assert [r["trajectory_id"] for r in best] == ["a"]


# ---------------------------------------------------------------------------
# finalize_best_trajectories
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_noop_when_disabled(self, whole_space_shard, monkeypatch, tmp_path):
        monkeypatch.setattr(config.search, "ENABLE_MICRO_HILL_CLIMB", False, raising=False)

        def boom(*a, **k):
            raise AssertionError("disabled finalization must not climb")
        monkeypatch.setattr(fin, "parallel_micro_climb", boom)

        # Returns immediately without reading the (absent) JSONL or climbing.
        finalize_best_trajectories(
            shard=whole_space_shard, identified_consts=[e],
            geom=FlatlandGeometry(whole_space_shard),
            start=whole_space_shard.get_interior_point(), eval_pool=None,
            cmf_id="", shard_id="sid", shard_encoding_str="",
            output_path=str(tmp_path / "missing.jsonl"),
            num_workers=0, config_overrides={},
        )

    def test_climbs_distinct_best_trajectories_with_dedup(
        self, whole_space_shard, symbols, monkeypatch, tmp_path
    ):
        geom = FlatlandGeometry(whole_space_shard)  # Z_reduced = I -> z == direction
        monkeypatch.setattr(config.search, "ENABLE_MICRO_HILL_CLIMB", True, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", [], raising=False)

        # Flushed JSONL: two distinct tied bests (0.20), one duplicate ray of the
        # first best (must NOT be climbed twice), and a non-best record.
        path = tmp_path / "shard.jsonl"
        records = [
            _rec("a", (3, 1), 0.204),   # best
            _rec("b", (1, 3), 0.196),   # tied best, distinct ray
            _rec("a2", (3, 1), 0.20),   # tied best, SAME ray as "a" -> dedup
            _rec("c", (1, 1), 0.10),    # not tied -> skipped
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Anchor evaluation (batched cache hits in production): identified, finite.
        monkeypatch.setattr(
            fin, "evaluate_neighbours", lambda zs, ctx, pool: [(0.20, True) for _ in zs]
        )

        # The concurrent climb receives one anchor per distinct tied-best ray.
        climbed_keys = []
        def fake_parallel(anchors, **kw):
            for z, cur_delta in anchors:
                climbed_keys.append(primitive_ray_key(z, kw["geom"]))
            return [(np.asarray(z), cur_delta) for z, cur_delta in anchors]
        monkeypatch.setattr(fin, "parallel_micro_climb", fake_parallel)

        finalize_best_trajectories(
            shard=whole_space_shard, identified_consts=[e], geom=geom,
            start=whole_space_shard.get_interior_point(), eval_pool=None,
            cmf_id="", shard_id="sid", shard_encoding_str="",
            output_path=str(path), num_workers=0, config_overrides={},
        )

        # Exactly the two distinct best rays were climbed; the duplicate and the
        # non-best record were not.
        assert sorted(climbed_keys) == [(1, 3), (3, 1)]
