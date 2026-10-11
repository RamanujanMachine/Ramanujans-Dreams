"""
Tests for the discrete micro-hill-climb endgame (Phase A + Phase B).

Coverage:
  - ``primitive_ray_key``: scaled copies share the canonical primitive-ray key.
  - ``doublings_to_resolution``: the doubling count is purely geometric —
    ``K = ⌈log2(max_norm/|z|)⌉`` (≥ 1), bounded only by the max-length radius.
  - ``resolution_probe_rays`` (Phase B fan): ``2^j z ± e_i`` re-snapped to
    in-cone, length-capped integer rays, excluding the current center, deduped,
    and pruned against a shared ``visited`` set.
  - ``discrete_micro_climb``:
      * reduces to the Phase-A ±1 certificate when the fan finds no improver.
      * a flat (constant-δ) landscape terminates at the start and fires the
        local-max callback exactly once.
      * Phase B accepts a superior interstitial ray, recenters, re-runs Phase A,
        and stops once the max-length-resolution fan yields no improvement.
      * the recenter loop is bounded by improvement, not a round count.
      * the shared ``visited`` set is populated so repeated work is skipped.
"""

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
import dreamer.search.methods.flatland.discrete_local_max as dlm
from dreamer.search.methods.flatland.discrete_local_max import (
    discrete_hill_climb,
    discrete_micro_climb,
    doublings_to_resolution,
    primitive_ray_key,
    resolution_probe_rays,
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
# primitive_ray_key
# ---------------------------------------------------------------------------

class TestPrimitiveRayKey:
    def test_scaled_copies_share_key(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)  # Z_reduced = I
        z = np.array([2, 3], dtype=np.int64)
        assert primitive_ray_key(z, geom) == primitive_ray_key(2 * z, geom)
        assert primitive_ray_key(z, geom) == (2, 3)

    def test_gcd_reduced(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        assert primitive_ray_key(np.array([4, 6]), geom) == (2, 3)


# ---------------------------------------------------------------------------
# doublings_to_resolution (geometric, max-length-bounded loop count)
# ---------------------------------------------------------------------------

class TestDoublingsToResolution:
    def test_log2_of_length_ratio(self):
        # |center| = 1, cap = 32 -> log2(32) = 5 doublings reach the resolution.
        assert doublings_to_resolution(1.0, 32.0) == 5
        assert doublings_to_resolution(2.0, 32.0) == 4

    def test_at_least_one_when_center_at_or_over_cap(self):
        # Already at/over the cap -> still one (final max-length) round.
        assert doublings_to_resolution(40.0, 35.0) == 1
        assert doublings_to_resolution(0.0, 35.0) == 1

    def test_grows_with_cap(self):
        # The max-length radius is the only bound: a bigger cap -> more doublings.
        assert doublings_to_resolution(1.0, 1024.0) == 10
        assert doublings_to_resolution(1.0, 1024.0) > doublings_to_resolution(1.0, 32.0)


# ---------------------------------------------------------------------------
# resolution_probe_rays (Phase B coarse->finest fan)
# ---------------------------------------------------------------------------

class TestResolutionProbeRays:
    def test_rays_in_cone_distinct_and_exclude_center(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        z = np.array([3, 3], dtype=np.int64)
        rays = resolution_probe_rays(z, geom, max_norm=35.0, traj_norm="l2")

        assert rays, "expected at least one finer-angle ray"
        z_key = primitive_ray_key(z, geom)
        keys = set()
        for ray in rays:
            assert geom.is_inside(ray)                     # cone-safe
            assert geom.traj_norm(ray, "l2") <= 35.0 + 1e-9  # within the cap
            k = primitive_ray_key(ray, geom)
            assert k != z_key                              # never the center itself
            assert k not in keys                           # deduped
            keys.add(k)

    def test_spans_multiple_resolutions(self, simple_shard):
        """A larger cap (more doublings) yields a richer fan than a tight cap."""
        geom = FlatlandGeometry(simple_shard)
        z = np.array([1, 1], dtype=np.int64)
        few = resolution_probe_rays(z, geom, max_norm=4.0, traj_norm="l2")
        many = resolution_probe_rays(z, geom, max_norm=60.0, traj_norm="l2")
        assert len(many) >= len(few)

    def test_visited_rays_are_pruned(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        z = np.array([3, 3], dtype=np.int64)
        rays = resolution_probe_rays(z, geom, max_norm=35.0, traj_norm="l2")
        assert rays

        skip = primitive_ray_key(rays[0], geom)
        rays2 = resolution_probe_rays(z, geom, max_norm=35.0, traj_norm="l2",
                                      visited={skip})
        assert all(primitive_ray_key(r, geom) != skip for r in rays2)


# ---------------------------------------------------------------------------
# discrete_micro_climb
# ---------------------------------------------------------------------------

class TestDiscreteMicroClimb:
    def test_reduces_to_phase_a_when_no_finer_improver(self, whole_space_shard, monkeypatch):
        """When the resolution fan surfaces nothing better, the result matches the
        plain ±1 hill-climb (Phase A) — Phase B is a pure no-op refinement."""
        geom = FlatlandGeometry(whole_space_shard)

        # delta(z) = -|z[0] - 3|: a 1-D ridge peaking at z[0] = 3 (global max delta 0).
        def fake_eval(z, **kw):
            return -abs(float(np.asarray(z)[0]) - 3.0), True
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)

        start = np.array([1, 1], dtype=np.int64)
        d0, _ = fake_eval(start)
        ctx = _ctx(whole_space_shard, geom)

        a_z, a_d = discrete_hill_climb(
            start.copy(), d0, geom=geom, eval_ctx=ctx, max_norm=35.0,
            traj_norm="l2", improve_threshold=1e-9,
        )
        m_z, m_d = discrete_micro_climb(
            start.copy(), d0, geom=geom, eval_ctx=ctx, max_norm=35.0,
            traj_norm="l2", improve_threshold=1e-9,
        )
        assert np.array_equal(m_z, a_z)
        assert m_d == pytest.approx(a_d)
        assert int(m_z[0]) == 3  # climbed the ridge to the peak

    def test_flat_landscape_terminates_and_calls_callback(self, whole_space_shard, monkeypatch):
        geom = FlatlandGeometry(whole_space_shard)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", lambda z, **kw: (0.42, True))

        seen_max = []
        start = np.array([2, 2], dtype=np.int64)
        z_out, d_out = discrete_micro_climb(
            start, 0.42, geom=geom, eval_ctx=_ctx(whole_space_shard, geom),
            max_norm=35.0, traj_norm="l2", improve_threshold=1e-9,
            on_local_max=lambda z, dlt: seen_max.append((z, dlt)),
        )
        assert d_out == pytest.approx(0.42)
        assert np.array_equal(z_out, start)
        assert len(seen_max) == 1  # callback fires exactly once at the end

    def test_phase_b_accepts_superior_ray_and_reclimbs(self, whole_space_shard, monkeypatch):
        """Phase A sits on a plateau; the resolution fan reveals a strictly better ray."""
        geom = FlatlandGeometry(whole_space_shard)
        better = np.array([5, 5], dtype=np.int64)

        def fake_eval(z, **kw):
            return (1.0, True) if np.array_equal(np.asarray(z), better) else (0.0, True)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)

        # First fan surfaces `better`; after recentering, the fan finds nothing new.
        calls = {"n": 0}
        def fake_rays(z, g, max_norm, traj_norm, visited=None):
            calls["n"] += 1
            return [better] if calls["n"] == 1 else []
        monkeypatch.setattr(dlm, "resolution_probe_rays", fake_rays)

        z_out, d_out = discrete_micro_climb(
            np.array([1, 1], dtype=np.int64), 0.0,
            geom=geom, eval_ctx=_ctx(whole_space_shard, geom), max_norm=35.0,
            traj_norm="l2", improve_threshold=1e-9,
        )
        assert d_out == pytest.approx(1.0)
        assert np.array_equal(z_out, better)
        # One accept (recenter) + one empty fan that stops the loop.
        assert calls["n"] == 2

    def test_recenters_through_successive_improvers(self, whole_space_shard, monkeypatch):
        """The recenter loop is bounded by *improvement* (not a round count): it
        walks every successive improver the fan surfaces and stops on a dry fan."""
        geom = FlatlandGeometry(whole_space_shard)
        # Isolated peaks spaced by 2 so Phase-A ±1 climbs never bridge them; each
        # advance must come from a fresh fan.
        ladder = [np.array([2 * (k + 1), 0], dtype=np.int64) for k in range(4)]
        score = {tuple(v): float(i + 1) for i, v in enumerate(ladder)}

        def fake_eval(z, **kw):
            return score.get(tuple(np.asarray(z)), 0.0), True
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)

        rounds = {"n": 0}
        def fake_rays(z, g, max_norm, traj_norm, visited=None):
            i = rounds["n"]
            rounds["n"] += 1
            return [ladder[i]] if i < len(ladder) else []
        monkeypatch.setattr(dlm, "resolution_probe_rays", fake_rays)

        z_out, d_out = discrete_micro_climb(
            np.array([1, 1], dtype=np.int64), 0.0,
            geom=geom, eval_ctx=_ctx(whole_space_shard, geom), max_norm=1e9,
            traj_norm="l2", improve_threshold=1e-9,
        )
        # 4 improvers consumed + 1 empty fan that terminates the loop.
        assert rounds["n"] == 5
        assert d_out == pytest.approx(4.0)
        assert np.array_equal(z_out, ladder[-1])

    def test_visited_set_is_populated(self, whole_space_shard, monkeypatch):
        geom = FlatlandGeometry(whole_space_shard)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", lambda z, **kw: (0.0, True))
        visited = set()
        start = np.array([2, 2], dtype=np.int64)
        discrete_micro_climb(
            start, 0.0, geom=geom, eval_ctx=_ctx(whole_space_shard, geom),
            max_norm=35.0, traj_norm="l2", improve_threshold=1e-9, visited=visited,
        )
        assert primitive_ray_key(start, geom) in visited


# ---------------------------------------------------------------------------
# parallel_micro_climb — must equal sequential discrete_micro_climb per anchor
# ---------------------------------------------------------------------------

class TestParallelMicroClimbEquivalence:
    """The concurrent (batched-across-ties) climb is a pure scheduling change: for
    every anchor it must return exactly what running :func:`discrete_micro_climb`
    on that anchor alone (with its own fresh ``visited``) returns.  δ is a fixed
    function of the direction here, so the only thing the restructure changes is
    the order evaluations are dispatched — the refined result must be identical."""

    def _fake_eval(self):
        # A ridge in z[0] (peak at 7) with a gentle z[1] tilt (peak at 2): Phase A
        # climbs, Phase B probes finer rays.  Pure function of z ⇒ cache-independent.
        def fake_eval(z, **kw):
            z = np.asarray(z, dtype=float)
            return -abs(z[0] - 7.0) - 0.1 * abs(z[1] - 2.0), True
        return fake_eval

    def test_matches_sequential_per_anchor(self, whole_space_shard, monkeypatch):
        geom = FlatlandGeometry(whole_space_shard)
        fake_eval = self._fake_eval()
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        ctx = _ctx(whole_space_shard, geom)
        kw = dict(geom=geom, eval_ctx=ctx, max_norm=35.0, traj_norm="l2",
                  improve_threshold=1e-9)

        starts = [np.array([1, 1]), np.array([3, 5]), np.array([9, 1]),
                  np.array([4, 8])]
        starts = [np.asarray(z, dtype=np.int64) for z in starts]

        # Sequential: each anchor climbed independently (fresh visited each).
        sequential = [
            dlm.discrete_micro_climb(z.copy(), fake_eval(z)[0], **kw) for z in starts
        ]

        # Concurrent: all anchors climbed in lockstep, one batch per round.
        anchors = [(z.copy(), fake_eval(z)[0]) for z in starts]
        concurrent = dlm.parallel_micro_climb(anchors, pool=None, **kw)

        assert len(concurrent) == len(sequential)
        for (pz, pd), (sz, sd) in zip(concurrent, sequential):
            assert tuple(int(v) for v in pz) == tuple(int(v) for v in sz)
            assert pd == pytest.approx(sd)

    def test_single_anchor_matches_micro_climb(self, whole_space_shard, monkeypatch):
        geom = FlatlandGeometry(whole_space_shard)
        fake_eval = self._fake_eval()
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        ctx = _ctx(whole_space_shard, geom)
        kw = dict(geom=geom, eval_ctx=ctx, max_norm=35.0, traj_norm="l2",
                  improve_threshold=1e-9)
        z = np.array([1, 1], dtype=np.int64)
        seq_z, seq_d = dlm.discrete_micro_climb(z.copy(), fake_eval(z)[0], **kw)
        (par_z, par_d), = dlm.parallel_micro_climb([(z.copy(), fake_eval(z)[0])], pool=None, **kw)
        assert tuple(int(v) for v in par_z) == tuple(int(v) for v in seq_z)
        assert par_d == pytest.approx(seq_d)
