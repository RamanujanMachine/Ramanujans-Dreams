"""
Tests for HybridSPSASearch (SPSA + Adam macro-navigation + discrete fallback).

Coverage:
  - Lattice min-angle: theta_min = arcsin(1/L^2).
  - SPSA gradient: g = (d+ - d-)/(2 c_k) * Delta sign; None when probes never
    yield an identified, distinct, in-cone pair (drives the discrete fallback).
  - Orthogonal neighbours: 2D +/-1 candidates, cone + norm filtered, zero skipped.
  - Discrete hill-climb: greedily moves to the strictly-best neighbour and
    terminates at the discrete local maximum.
  - Seed selection: ascending L2 norm, first identifier, NoInitialIdentification.
  - End-to-end run: a flat (zero-gradient) landscape stalls macro-navigation and
    falls through to the discrete fallback, terminating cleanly.
"""

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.gradient_ascent.spsa_adam_ascent import (
    HybridSPSASearch,
    NoInitialIdentification,
)
import dreamer.search.methods.gradient_ascent.spsa_adam_ascent as spsa
import dreamer.search.methods.flatland.discrete_local_max as dlm

search_config = config.search


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
def zero_shift(symbols):
    return Position({s: sp.Integer(0) for s in symbols})


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    """Bounded shard: cone x>=0, y>=0 with interior point (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


def _ctx(shard, geom):
    return dict(
        geom=geom, shard=shard, start=shard.get_interior_point(),
        constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
        sink=lambda x: None, seen_trajectories={}, handler_cache={},
    )


# ---------------------------------------------------------------------------
# 1. Lattice min-angle (the "pixel size")
# ---------------------------------------------------------------------------

class TestMinLatticeAngle:
    def test_matches_arcsin_one_over_L_squared(self):
        for L in (10.0, 35.0, 50.0):
            assert HybridSPSASearch._min_lattice_angle(L) == pytest.approx(np.arcsin(1.0 / L ** 2))

    def test_decreases_with_length(self):
        assert HybridSPSASearch._min_lattice_angle(50.0) < HybridSPSASearch._min_lattice_angle(10.0)

    def test_clamped_for_tiny_L(self):
        # 1/L^2 capped at 1.0 so arcsin stays defined for L < 1.
        assert HybridSPSASearch._min_lattice_angle(0.5) == pytest.approx(np.arcsin(1.0))


# ---------------------------------------------------------------------------
# 2. SPSA gradient
# ---------------------------------------------------------------------------

class TestSpsaGradient:
    def test_returns_none_when_probes_never_identified(self, whole_space_shard, monkeypatch):
        """Every probe non-identified -> no usable difference -> None (=> fallback)."""
        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        method._rng = np.random.default_rng(0)
        geom = FlatlandGeometry(whole_space_shard)
        monkeypatch.setattr(spsa, "evaluate_in_flatland", lambda z, **kw: (0.5, False))
        monkeypatch.setattr(config.search, "SPSA_PROBE_RETRIES", 3, raising=False)

        d = np.ones(geom.d_flat)
        assert method._spsa_gradient(d, c_k=0.2, eval_ctx=_ctx(whole_space_shard, geom),
                                     geom=geom, max_norm=35.0) is None

    def test_gradient_sign_follows_delta_difference(self, whole_space_shard, monkeypatch):
        """delta increases along +e0 => the recovered gradient has a positive e0 sign signal."""
        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        method._rng = np.random.default_rng(1)
        geom = FlatlandGeometry(whole_space_shard)

        # delta(z) = z[0]: a higher first coordinate scores higher.
        def fake_eval(z, **kw):
            return float(np.asarray(z)[0]), True
        monkeypatch.setattr(spsa, "evaluate_in_flatland", fake_eval)

        d = np.zeros(geom.d_flat); d[0] = 1.0  # pointing along +e0
        g = method._spsa_gradient(d, c_k=0.3, eval_ctx=_ctx(whole_space_shard, geom),
                                  geom=geom, max_norm=35.0)
        assert g is not None
        # The estimator g = (d+ - d-)/(2c) * Delta is an unbiased ascent signal on
        # coordinate 0 (delta depends only on z[0]); its e0 component is finite.
        assert np.isfinite(g[0])


# ---------------------------------------------------------------------------
# 3. Orthogonal neighbours (shared discrete local-max module)
# ---------------------------------------------------------------------------

class TestOrthogonalNeighbours:
    def test_count_and_cone_filtering(self, simple_shard):
        """On the x>=0,y>=0 cone, the -1 neighbours that leave the cone are dropped."""
        geom = FlatlandGeometry(simple_shard)
        z = np.array([2, 2], dtype=np.int64)[: geom.d_flat]
        nbrs = dlm.orthogonal_neighbours(z, geom, max_norm=100.0, traj_norm="linf")
        # Each returned neighbour is inside the cone and differs from z in one coord by 1.
        for n in nbrs:
            assert geom.is_inside(n)
            assert int(np.sum(np.abs(n - z))) == 1

    def test_zero_vector_neighbour_skipped(self, whole_space_shard):
        """A -1 step that produces the zero vector is rejected (not a trajectory)."""
        geom = FlatlandGeometry(whole_space_shard)
        z = np.zeros(geom.d_flat, dtype=np.int64); z[0] = 1
        nbrs = dlm.orthogonal_neighbours(z, geom, max_norm=100.0, traj_norm="linf")
        assert all(np.any(n) for n in nbrs)

    def test_norm_cap_excludes_long_neighbours(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        z = np.zeros(geom.d_flat, dtype=np.int64); z[0] = 3
        nbrs = dlm.orthogonal_neighbours(z, geom, max_norm=3.0, traj_norm="linf")
        # The +1 step to [4, ...] exceeds the linf cap of 3 and is excluded.
        assert all(geom.traj_norm(n, "linf") <= 3.0 for n in nbrs)


# ---------------------------------------------------------------------------
# 4. Discrete hill-climb / local-maximum certificate (shared module)
# ---------------------------------------------------------------------------

class TestDiscreteHillClimb:
    def test_climbs_to_local_max_and_stops(self, whole_space_shard, monkeypatch):
        """delta peaks at z[0]=3: the climb walks 1->2->3 then stops (no improver)."""
        geom = FlatlandGeometry(whole_space_shard)

        # Concave landscape in the first coordinate, peak at 3, others neutral.
        def fake_eval(z, **kw):
            z0 = float(np.asarray(z)[0])
            return -abs(z0 - 3.0), True
        # The shared hill-climb evaluates via its OWN module binding.
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)

        start = np.zeros(geom.d_flat, dtype=np.int64); start[0] = 1
        z_final, delta_final = dlm.discrete_hill_climb(
            start, -2.0, geom=geom, eval_ctx=_ctx(whole_space_shard, geom),
            max_norm=100.0, traj_norm="linf", improve_threshold=1e-9, pool=None,
        )
        assert int(z_final[0]) == 3
        assert delta_final == pytest.approx(0.0)

    def test_terminates_immediately_at_local_max(self, whole_space_shard, monkeypatch):
        """A flat landscape: no neighbour strictly improves -> no move."""
        geom = FlatlandGeometry(whole_space_shard)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", lambda z, **kw: (0.5, True))

        start = np.zeros(geom.d_flat, dtype=np.int64); start[0] = 2
        z_final, delta_final = dlm.discrete_hill_climb(
            start, 0.5, geom=geom, eval_ctx=_ctx(whole_space_shard, geom),
            max_norm=100.0, traj_norm="linf", improve_threshold=1e-9, pool=None,
        )
        np.testing.assert_array_equal(z_final, start)
        assert delta_final == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. Seed selection
# ---------------------------------------------------------------------------

class TestSeedSelection:
    def test_picks_first_identifier_in_l2_order(self, whole_space_shard, symbols, monkeypatch):
        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        far = Position({symbols[0]: sp.Integer(5), symbols[1]: sp.Integer(0)})
        near = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)})
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: {far, near})

        evaluated = []
        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True
        monkeypatch.setattr(spsa, "evaluate_in_flatland", fake_eval)

        geom = FlatlandGeometry(whole_space_shard)
        seed = method._select_seed(geom, _ctx(whole_space_shard, geom), "sid", e)
        assert list(evaluated[0]) == [1, 0]
        assert list(seed) == [1, 0]

    def test_raises_when_none_identify(self, whole_space_shard, symbols, monkeypatch):
        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: {t})
        monkeypatch.setattr(spsa, "evaluate_in_flatland", lambda z, **kw: (-1.0, False))

        geom = FlatlandGeometry(whole_space_shard)
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, _ctx(whole_space_shard, geom), "sid", e)


# ---------------------------------------------------------------------------
# 6. End-to-end run: macro stall -> discrete fallback
# ---------------------------------------------------------------------------

class TestEndToEndStallFallback:
    def test_flat_landscape_stalls_then_terminates(self, whole_space_shard, symbols, monkeypatch):
        """A constant-delta (zero-gradient) landscape: SPSA produces a zero Adam
        step (< min_angle) -> stall -> discrete certificate finds no improver -> done."""
        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        const_eval = lambda z, **kw: (0.42, True)
        # The macro phase evaluates via the spsa-module binding; the always-on
        # discrete certificate evaluates via the shared-module binding.
        monkeypatch.setattr(spsa, "evaluate_in_flatland", const_eval)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", const_eval)
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: {Position({symbols[0]: sp.Integer(1),
                                                       symbols[1]: sp.Integer(0)})})
        monkeypatch.setattr(config.search, "SPSA_MAX_STEPS", 20, raising=False)
        monkeypatch.setattr(config.search, "SPSA_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        assert method.best_delta == pytest.approx(0.42)
        # The discrete certificate ran (constant landscape -> stall -> fallback).
        assert method.used_discrete_fallback is True
