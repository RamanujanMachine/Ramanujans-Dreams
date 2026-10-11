"""
Tests for three connected, critical pieces of the search pipeline:

* **Task 1 — config-aware recomputation.**  Trajectory records carry a
  ``config_fingerprint`` of every config knob that influences their Tier-1
  values.  When a re-run changes such a knob (e.g. a deeper walk), the cached
  record must be treated as stale and recomputed rather than silently reused.

* **Task 2 — Small Angle evaluates *all* neighbours.**  The hill-climb must
  probe every in-cone perturbation and re-center on the global best, not stop
  on the first improvement.

* **Task 3 — trajectory length in real shard space.**  Length caps must be
  measured in the shard's real basis (via ``traj_norm`` / ``traj_norm_many``),
  not the flatland LLL basis.
"""

import json

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e, pi
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.gradient_ascent.lattice import snap_to_trajectory
from dreamer.search.methods.small_angle import SmallAngleSearch
from dreamer.utils.storage.dtos import TrajectoryDTO
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.trajectory_attributes import (
    _position_to_tuple,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)

search_config = config.search


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_small_angle_search.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cmf():
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def zero_shift(symbols):
    return Position({s: sp.Integer(0) for s in symbols})


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


# ===========================================================================
# Task 1 — config fingerprint
# ===========================================================================

class TestTier1Fingerprint:
    """``tier1_config_fingerprint`` must change iff a Tier-1-affecting knob does."""

    def test_stable_for_same_inputs(self):
        assert tier1_config_fingerprint(100) == tier1_config_fingerprint(100)

    def test_depth_changes_fingerprint(self):
        assert tier1_config_fingerprint(100) != tier1_config_fingerprint(200)

    def test_walk_type_changes_fingerprint(self, monkeypatch):
        monkeypatch.setattr(ta.search_config, "DEFAULT_USES_INV_T", True)
        fp_inv = tier1_config_fingerprint(100)
        monkeypatch.setattr(ta.search_config, "DEFAULT_USES_INV_T", False)
        fp_direct = tier1_config_fingerprint(100)
        assert fp_inv != fp_direct

    def test_identify_threshold_changes_fingerprint(self, monkeypatch):
        fp_a = tier1_config_fingerprint(100)
        monkeypatch.setattr(ta.search_config, "IDENTIFY_CHECK_THRESHOLD", 1e-3)
        fp_b = tier1_config_fingerprint(100)
        assert fp_a != fp_b

    def test_constant_precision_changes_fingerprint(self, monkeypatch):
        fp_a = tier1_config_fingerprint(100)
        monkeypatch.setattr(ta.search_config, "CONSTANT_NO_DIGITS_LOW_RES", 1234)
        fp_b = tier1_config_fingerprint(100)
        assert fp_a != fp_b

    def test_unrelated_knob_does_not_change_fingerprint(self, monkeypatch):
        """A purely-algorithmic knob (e.g. GA learning rate) must NOT invalidate
        Tier-1 records — it does not affect δ / identification of a trajectory."""
        fp_a = tier1_config_fingerprint(100)
        monkeypatch.setattr(ta.search_config, "GRAD_LR", 12.34)
        fp_b = tier1_config_fingerprint(100)
        assert fp_a == fp_b

    def test_walk_depth_for_uses_config_callable(self, monkeypatch, simple_cmf, symbols):
        direction = Position({symbols[0]: sp.Integer(3), symbols[1]: sp.Integer(4)})
        monkeypatch.setattr(ta.search_config, "DEPTH_FROM_TRAJECTORY_LEN", lambda L, d: 777)
        assert walk_depth_for(simple_cmf, direction) == 777

    def test_fingerprint_is_memoised(self):
        """Repeated calls with the same config + depth hit the LRU cache rather
        than re-serialising / re-hashing."""
        ta._tier1_fingerprint_for_key.cache_clear()
        first = tier1_config_fingerprint(100)
        second = tier1_config_fingerprint(100)
        assert first == second
        info = ta._tier1_fingerprint_for_key.cache_info()
        # Second identical call must be served from the cache.
        assert info.hits >= 1


class TestDtoFingerprintRoundTrip:
    def test_walk_depth_and_fingerprint_round_trip(self):
        dto = TrajectoryDTO(
            trajectory_id="t", cmf_id="c", shard_id="s", constant="e",
            start_point=(0, 0), direction=(1, 2),
            delta=2.0, identified=True,
            walk_depth=321, config_fingerprint="abc123",
        )
        back = TrajectoryDTO.from_dict(json.loads(dto.to_json_line()))
        assert back.walk_depth == 321
        assert back.config_fingerprint == "abc123"

    def test_record_without_fingerprint_parses_none(self):
        """A flat row without the walk-metadata columns parses with None."""
        rec = {
            "trajectory_id": "t", "cmf_id": "c", "shard_id": "s", "constant": "e",
            "start_point": [0, 0], "direction": [1, 2], "delta": 2.0, "identified": True,
        }
        back = TrajectoryDTO.from_dict(rec)
        assert back.walk_depth is None
        assert back.config_fingerprint is None


class TestEvaluatorRespectsFingerprint:
    """``evaluate_in_flatland`` Case A must only short-circuit on a fingerprint match."""

    def _setup(self, shard, symbols):
        method = SmallAngleSearch(shard, e, use_LIReC=False)
        geom = FlatlandGeometry(shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        z = geom.to_flatland(t)
        start = shard.get_interior_point()
        start_t = _position_to_tuple(start)
        dir_t = _position_to_tuple(geom.to_real_primitive(z))
        tid = derive_trajectory_id("sid", shard.cmf_name, "", start_t, dir_t)
        fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, geom.to_real_primitive(z)))
        return method, geom, z, start, tid, fp

    def test_matching_fingerprint_short_circuits(self, whole_space_shard, symbols):
        method, geom, z, start, tid, fp = self._setup(whole_space_shard, symbols)
        seen = {tid: {e.name: {"trajectory_id": tid, "constant": e.name,
                               "delta": 2.5, "identified": True,
                               "config_fingerprint": fp}}}
        built = []
        from dreamer.search.methods.small_angle import small_angle_scan as sas
        orig = sas.TrajectoryAttributesHandler.from_cmf
        sas.TrajectoryAttributesHandler.from_cmf = staticmethod(
            lambda *a, **k: built.append(1) or orig(*a, **k))
        try:
            delta, ided = method._evaluate(
                z, geom=geom, start=start, constant=e, cmf_id="", shard_id="sid",
                shard_encoding_str="", sink=lambda x: None,
                seen_trajectories=seen, handler_cache={},
            )
        finally:
            sas.TrajectoryAttributesHandler.from_cmf = staticmethod(orig)
        assert (delta, ided) == (2.5, True)
        assert not built  # short-circuited: no recompute

    def test_stale_fingerprint_forces_recompute(self, whole_space_shard, symbols):
        method, geom, z, start, tid, _ = self._setup(whole_space_shard, symbols)
        # Record stored under a DIFFERENT (stale) config → must not be reused.
        seen = {tid: {e.name: {"trajectory_id": tid, "constant": e.name,
                               "delta": 2.5, "identified": True,
                               "config_fingerprint": "STALE"}}}
        built = []
        from dreamer.search.methods.small_angle import small_angle_scan as sas
        orig = sas.TrajectoryAttributesHandler.from_cmf

        def _spy(*a, **k):
            built.append(1)
            raise RuntimeError("recompute attempted")  # short-circuit the expensive walk

        sas.TrajectoryAttributesHandler.from_cmf = staticmethod(_spy)
        try:
            delta, ided = method._evaluate(
                z, geom=geom, start=start, constant=e, cmf_id="", shard_id="sid",
                shard_encoding_str="", sink=lambda x: None,
                seen_trajectories=seen, handler_cache={},
            )
        finally:
            sas.TrajectoryAttributesHandler.from_cmf = staticmethod(orig)
        assert built, "stale fingerprint must trigger a recompute (Case C)"
        assert delta == float("-inf")  # the spy raised → caught → sentinel


# ===========================================================================
# Task 2 — Small Angle evaluates all neighbours
# ===========================================================================

class TestEvaluatesAllNeighbours:
    def test_best_inside_perturbation_probes_every_neighbour(self, whole_space_shard, symbols):
        """All in-cone perturbations are evaluated and the GLOBAL best is kept,
        even when an earlier neighbour already improved on the centre."""
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)
        z = np.array([3, 3], dtype=np.int64)[: geom.d_flat]

        in_cone = [c for c in geom.perturbations(z) if geom.is_inside(c)]
        assert len(in_cone) >= 3  # sanity: multiple neighbours to choose among

        evaluated = []
        # δ increasing with index so the LAST neighbour is the global best — a
        # first-improvement climber would wrongly stop earlier.
        order = {c.tobytes(): i for i, c in enumerate(in_cone)}

        def fake_eval(cand, **ctx):
            evaluated.append(cand.tobytes())
            return float(order.get(cand.tobytes(), -1)), True

        method._evaluate = fake_eval
        ctx = dict(geom=geom)
        best_z, best_score = method._best_inside_perturbation(z, ctx)

        # Every in-cone neighbour was evaluated (no early stop).
        assert set(evaluated) == {c.tobytes() for c in in_cone}
        # The global best (highest δ) was selected.
        assert best_score == max(order.values())
        assert best_z.tobytes() == max(order, key=order.get)


# ===========================================================================
# Task 3 — trajectory length measured in real shard space
# ===========================================================================

class TestShardSpaceNorm:
    def test_traj_norm_many_matches_scalar(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        Z = np.array([[2, 0], [1, 1], [3, -2]], dtype=np.int64)[:, : geom.d_flat]
        for norm in ("linf", "l1", "l2"):
            many = geom.traj_norm_many(Z, norm)
            scalar = np.array([geom.traj_norm(z, norm) for z in Z])
            assert np.allclose(many, scalar)

    def test_traj_norm_is_basis_invariant_real_space(self, simple_shard):
        """traj_norm measures the real-space (shard) vector, not the flatland z."""
        geom = FlatlandGeometry(simple_shard)
        z = np.array([1, 1], dtype=np.int64)[: geom.d_flat]
        v = np.array([float(geom.to_real_primitive(z)[s]) for s in geom.symbols])
        assert geom.traj_norm(z, "l2") == pytest.approx(np.linalg.norm(v))

    def test_snap_respects_shard_space_cap(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        d = np.array([1.0, 1.0])[: geom.d_flat]
        max_norm = 3.0
        z = snap_to_trajectory(d, geom, max_norm, "l2")
        assert z is not None
        assert geom.traj_norm(z, "l2") <= max_norm + 1e-9

    def test_snap_returns_none_when_cap_excludes_all(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        d = np.array([1.0, 1.0])[: geom.d_flat]
        # A cap below the shortest primitive ray length admits nothing.
        z = snap_to_trajectory(d, geom, 0.5, "l2")
        assert z is None


# ===========================================================================
# Closed recession cone — the zero direction is never inside (not a trajectory)
# ===========================================================================

class TestZeroDirectionExcluded:
    def test_is_inside_rejects_zero(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        zero = np.zeros(geom.d_flat, dtype=np.int64)
        nonzero = np.ones(geom.d_flat, dtype=np.int64)
        assert not geom.is_inside(zero)
        assert geom.is_inside(nonzero)  # whole space: any non-zero direction is inside

    def test_is_inside_many_rejects_zero_row(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        Z = np.array([np.zeros(geom.d_flat), np.ones(geom.d_flat)], dtype=np.int64)
        mask = geom.is_inside_many(Z)
        assert mask.tolist() == [False, True]
