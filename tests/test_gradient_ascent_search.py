"""
Tests for GradientAscentSearch (optimizers + lattice realization + method + module).

Coverage:
  - Optimizer strategies (VanillaGrad / Momentum / RMSprop / Adam): step() math,
    reset(), and the optimizer_for factory + UnknownGradVariant.
  - Lattice realization: rotate_toward (norm-preserving, parallel no-op) and
    snap_to_trajectory (in-cone, norm-capped, angle-best, None on no candidate).
  - Gradient estimate: forward differences in angle space; non-identified probes skipped.
  - Seed selection: ascending L2 norm, first identifier, NoInitialIdentification.
  - Convergence stop: vanishing gradient terminates the ascent.
  - Recovery ladder: skip -> length-doubling -> diffraction -> SearchStalled.
  - GradientAscentMod orchestration + NoInitialIdentification / SearchStalled caught.
"""

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
from dreamer.search.methods.gradient_ascent.optimizers import (
    VanillaGrad,
    Momentum,
    RMSprop,
    Adam,
    optimizer_for,
    UnknownGradVariant,
)
from dreamer.search.methods.gradient_ascent.lattice import rotate_toward, snap_to_trajectory
from dreamer.search.methods.gradient_ascent.grad_ascent_scan import (
    GradientAscentSearch,
    NoInitialIdentification,
    SearchStalled,
)

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


# ---------------------------------------------------------------------------
# 1. Optimizer strategies
# ---------------------------------------------------------------------------

class TestVanillaGrad:
    def test_returns_raw_gradient(self):
        opt = VanillaGrad(3)
        g = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(opt.step(g), g)

    def test_reset_is_noop(self):
        opt = VanillaGrad(2)
        opt.reset()  # must not raise
        np.testing.assert_allclose(opt.step(np.array([3.0, 4.0])), [3.0, 4.0])


class TestMomentum:
    def test_accumulates_velocity(self):
        """v <- beta*v + g; with beta=0.5 and constant g=[1,0]: 1, 1.5, 1.75."""
        opt = Momentum(2, beta=0.5)
        g = np.array([1.0, 0.0])
        np.testing.assert_allclose(opt.step(g), [1.0, 0.0])
        np.testing.assert_allclose(opt.step(g), [1.5, 0.0])
        np.testing.assert_allclose(opt.step(g), [1.75, 0.0])

    def test_reset_clears_velocity(self):
        opt = Momentum(1, beta=0.9)
        opt.step(np.array([1.0]))
        opt.reset()
        np.testing.assert_allclose(opt.step(np.array([2.0])), [2.0])


class TestRMSprop:
    def test_normalizes_by_rms(self):
        """First step: s=(1-b2)*g^2; update = g/(sqrt(s)+eps) ~ 1/sqrt(1-b2)."""
        b2, eps = 0.9, 1e-12
        opt = RMSprop(1, beta2=b2, epsilon=eps)
        out = opt.step(np.array([2.0]))[0]
        s = (1 - b2) * 4.0
        expected = 2.0 / (np.sqrt(s) + eps)
        assert out == pytest.approx(expected)

    def test_reset_clears_state(self):
        opt = RMSprop(1, beta2=0.9)
        opt.step(np.array([5.0]))
        opt.reset()
        # After reset the first-step magnitude is recovered.
        assert opt.step(np.array([5.0]))[0] == pytest.approx(5.0 / np.sqrt((1 - 0.9) * 25.0), rel=1e-6)


class TestAdam:
    def test_first_step_is_lr_scale_invariant(self):
        """Adam's first step magnitude is ~1 regardless of gradient scale (bias-corrected)."""
        opt = Adam(1, beta1=0.9, beta2=0.999, epsilon=1e-12)
        out = opt.step(np.array([10.0]))[0]
        # m_hat = g, s_hat = g^2 => update = g/|g| = 1 (sign of gradient).
        assert out == pytest.approx(1.0, rel=1e-4)

    def test_step_count_advances_bias_correction(self):
        opt = Adam(1)
        opt.step(np.array([1.0]))
        assert opt._t == 1
        opt.step(np.array([1.0]))
        assert opt._t == 2

    def test_reset_clears_moments_and_step(self):
        opt = Adam(2)
        opt.step(np.array([1.0, 2.0]))
        opt.reset()
        assert opt._t == 0
        np.testing.assert_allclose(opt._m, [0.0, 0.0])
        np.testing.assert_allclose(opt._s, [0.0, 0.0])


class TestOptimizerFactory:
    @pytest.mark.parametrize("name,cls", [
        ("vanilla", VanillaGrad), ("momentum", Momentum),
        ("rmsprop", RMSprop), ("adam", Adam),
    ])
    def test_builds_each_variant(self, name, cls):
        assert isinstance(optimizer_for(name, 3, search_config), cls)

    def test_case_insensitive(self):
        assert isinstance(optimizer_for("AdAm", 2, search_config), Adam)

    def test_unknown_raises(self):
        with pytest.raises(UnknownGradVariant):
            optimizer_for("nesterov", 2, search_config)


# ---------------------------------------------------------------------------
# 2. Lattice realization
# ---------------------------------------------------------------------------

class TestRotateToward:
    def test_preserves_norm(self):
        d = np.array([3.0, 4.0])  # norm 5
        rotated = rotate_toward(d, 1, 0.3)
        assert np.linalg.norm(rotated) == pytest.approx(5.0)

    def test_parallel_axis_is_noop(self):
        d = np.array([2.0, 0.0])  # parallel to axis 0
        np.testing.assert_allclose(rotate_toward(d, 0, 0.5), d)

    def test_rotation_moves_toward_axis(self):
        """Rotating toward axis 1 increases the y-component (for a small positive angle)."""
        d = np.array([5.0, 0.0])
        rotated = rotate_toward(d, 1, 0.2)
        assert rotated[1] > 0.0

    def test_zero_vector_returns_zero(self):
        np.testing.assert_allclose(rotate_toward(np.zeros(2), 0, 0.4), [0.0, 0.0])


class TestPrimitiveRayAndBatchCone:
    """Geometry helpers added for the perf pass: primitive ray + batched cone test."""

    def test_to_real_primitive_collapses_scaled_directions(self, simple_shard):
        """z and 2z must map to the SAME primitive real ray (one cached walk)."""
        geom = FlatlandGeometry(simple_shard)
        z = np.zeros(geom.d_flat, dtype=np.int64)
        z[0] = 1
        p1 = geom.to_real_primitive(z)
        p2 = geom.to_real_primitive(2 * z)
        assert [int(p1[s]) for s in simple_shard.symbols] == [int(p2[s]) for s in simple_shard.symbols]

    def test_to_real_primitive_is_gcd_one(self, whole_space_shard):
        """The returned real ray is primitive (GCD of components == 1)."""
        import math
        geom = FlatlandGeometry(whole_space_shard)
        z = np.zeros(geom.d_flat, dtype=np.int64)
        z[0] = 4  # 4*e0 -> primitive e0
        p = geom.to_real_primitive(z)
        comps = [abs(int(p[s])) for s in whole_space_shard.symbols if int(p[s]) != 0]
        g = comps[0]
        for c in comps[1:]:
            g = math.gcd(g, c)
        assert g == 1

    def test_is_inside_many_matches_is_inside(self, simple_shard):
        """Batched cone test agrees with the scalar one row-by-row."""
        geom = FlatlandGeometry(simple_shard)
        Z = np.array([[1, 0], [0, 1], [-1, 0], [1, 1]], dtype=np.int64)[:, : geom.d_flat]
        batch = geom.is_inside_many(Z)
        scalar = np.array([geom.is_inside(z) for z in Z])
        np.testing.assert_array_equal(batch, scalar)


class TestSnapToTrajectory:
    def test_returns_in_cone_capped_integer(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        d = geom.to_flatland(Position({s: sp.Integer(1) for s in simple_shard.symbols})).astype(float)
        z = snap_to_trajectory(d, geom, max_norm=20.0)
        assert z is not None
        assert geom.is_inside(z)
        assert np.linalg.norm(z) <= 20.0

    def test_angle_best_aligns_with_direction(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        d = np.zeros(geom.d_flat)
        d[0] = 3.0  # ask for a direction along axis 0
        z = snap_to_trajectory(d, geom, max_norm=10.0)
        assert z is not None
        # The best integer direction should be (nearly) parallel to e_0.
        cos = np.dot(d, z) / (np.linalg.norm(d) * np.linalg.norm(z))
        assert cos == pytest.approx(1.0, abs=1e-9)

    def test_zero_direction_returns_none(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        assert snap_to_trajectory(np.zeros(geom.d_flat), geom, max_norm=10.0) is None

    def test_respects_norm_cap(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        d = np.zeros(geom.d_flat)
        d[0] = 1.0
        z = snap_to_trajectory(d, geom, max_norm=3.0)
        assert z is not None
        assert np.linalg.norm(z) <= 3.0


# ---------------------------------------------------------------------------
# 3. Gradient estimate (forward differences in angle)
# ---------------------------------------------------------------------------

class TestGradientEstimate:
    def test_skips_non_identified_probes(self, whole_space_shard, monkeypatch):
        """A probe whose trajectory is not identified contributes 0 and is not counted."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)

        # Every probe returns identified=False -> gradient all zero, usable=0.
        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (0.5, False))

        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        d = np.zeros(geom.d_flat); d[0] = 5.0
        grad, usable = method._estimate_gradient(d, base_delta=0.0, eval_ctx=ctx, geom=geom)
        assert usable == 0
        np.testing.assert_allclose(grad, np.zeros(geom.d_flat))

    def test_forward_difference_sign(self, whole_space_shard, monkeypatch):
        """g_i = (delta_probe - base)/h: a probe with higher delta yields a positive component."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)

        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (1.0, True))
        monkeypatch.setattr(config.search, "GRAD_FD_ANGLE", 0.1, raising=False)

        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        d = np.ones(geom.d_flat)
        grad, usable = method._estimate_gradient(d, base_delta=0.0, eval_ctx=ctx, geom=geom)
        # base=0, probe=1, h=0.1 => each usable component = 10.0.
        assert usable >= 1
        for i in range(geom.d_flat):
            assert grad[i] in (0.0, pytest.approx(10.0))


# ---------------------------------------------------------------------------
# 4. Seed selection
# ---------------------------------------------------------------------------

class TestSeedSelection:
    def test_picks_first_identifier_in_l2_order(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        far = Position({symbols[0]: sp.Integer(5), symbols[1]: sp.Integer(0)})
        near = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)})
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories", lambda self, n: {far, near},
        )

        evaluated = []
        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True
        monkeypatch.setattr(gs, "evaluate_in_flatland", fake_eval)

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        seed = method._select_seed(geom, ctx, "sid", e)
        assert list(evaluated[0]) == [1, 0]
        assert list(seed) == [1, 0]

    def test_raises_when_none_identify(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories", lambda self, n: {t},
        )
        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (-1.0, False))

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, ctx, "sid", e)


# ---------------------------------------------------------------------------
# 5. Convergence stop
# ---------------------------------------------------------------------------

class TestConvergenceStop:
    def test_no_improving_move_stops(self, whole_space_shard, monkeypatch):
        """A constant-delta landscape has no improving lattice move, so the ascent
        stops via the no-move break (zero gradient -> zero Adam step -> snap leaves
        ``z`` unchanged), not via any manual gradient-magnitude threshold.

        Termination is resolution-driven: ``GRAD_GRAD_TOL`` was removed, and
        ``GRAD_MAX_STEPS`` is only a safety ceiling — it must not be the reason the
        loop ends here (it stops on the very first step)."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        import dreamer.search.methods.flatland.discrete_local_max as dlm
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)

        # Seed identifies; every evaluation returns a constant delta -> zero gradient.
        eval_count = [0]

        def fake_eval(z, **kw):
            eval_count[0] += 1
            return 0.5, True

        # The ascent evaluates via the gs binding; the always-on local-maximum
        # certificate evaluates via the shared-module binding.
        monkeypatch.setattr(gs, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        # A large safety ceiling: the run must end via the no-move break long before
        # exhausting it (proving termination is not the step budget).
        monkeypatch.setattr(config.search, "GRAD_MAX_STEPS", 1000, raising=False)
        monkeypatch.setattr(config.search, "GRAD_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        assert method.best_delta == pytest.approx(0.5)
        # The no-move break fires on the first step: seed + one zero-gradient probe
        # pass + the one-sweep local-max certificate.  Far below GRAD_MAX_STEPS
        # (1000) worth of evaluations, proving termination is resolution-driven.
        assert eval_count[0] < 20

    def test_certificate_climbs_neighbour_continuous_phase_missed(self, whole_space_shard, monkeypatch):
        """The final discrete certificate moves to an improving ±1 neighbour even
        when the continuous ascent stopped (here: a delta spike one step away that
        the zero-gradient continuous phase cannot see)."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        import dreamer.search.methods.flatland.discrete_local_max as dlm
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)

        # Flat everywhere (delta 0) except a spike at the +e1 neighbour of the seed
        # (1,0,...) -> (1,1,0,...) — a genuinely distinct ray (not collinear with
        # the seed).  The continuous gradient is ~0 (every angle probe snaps back to
        # the seed), so the ascent no-move-stops at the seed; only the discrete
        # ±1-neighbour certificate reaches the spike.
        def fake_eval(z, **kw):
            z = np.asarray(z)
            spike = z[0] == 1 and z[1] == 1 and not np.any(z[2:])
            return (1.0 if spike else 0.0), True

        monkeypatch.setattr(gs, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        monkeypatch.setattr(config.search, "GRAD_RESERVOIR_SIZE", 1, raising=False)
        # Improvement threshold below the spike so the move counts as strict.
        monkeypatch.setattr(config.search, "GRAD_IMPROVE_THRESHOLD", 1e-6, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        # Without the certificate this would be 0.0 (seed delta); the certificate
        # climbs to the neighbouring spike.
        assert method.best_delta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Recovery ladder
# ---------------------------------------------------------------------------

class TestRecoveryLadder:
    def test_skip_keeps_last_identified(self, whole_space_shard, monkeypatch):
        """Within the skip budget, _recover returns the last identified direction."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)
        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (0.3, True))
        monkeypatch.setattr(config.search, "GRAD_SKIP_LIMIT", 3, raising=False)

        last = np.zeros(geom.d_flat, dtype=np.int64); last[0] = 1
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        cur_z, cur_delta, d, li, skip, dbl = method._recover(
            geom, ctx, "sid", e, last, skip_count=0, doubling_count=0, max_norm=100.0)
        np.testing.assert_array_equal(cur_z, last)
        assert cur_delta == pytest.approx(0.3)

    def test_diffraction_stalls_raises(self, whole_space_shard, monkeypatch):
        """When skip + doubling are exhausted and diffraction never identifies, SearchStalled."""
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs

        method = GradientAscentSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)
        # Everything is non-identified so diffraction can never succeed.
        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (0.0, False))
        monkeypatch.setattr(config.search, "GRAD_SKIP_LIMIT", 3, raising=False)
        monkeypatch.setattr(config.search, "GRAD_MAX_DOUBLINGS", 0, raising=False)
        monkeypatch.setattr(config.search, "GRAD_DIFFRACT_TRIES", 4, raising=False)

        last = np.zeros(geom.d_flat, dtype=np.int64); last[0] = 1
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        with pytest.raises(SearchStalled):
            method._recover(geom, ctx, "sid", e, last,
                            skip_count=3, doubling_count=0, max_norm=100.0)


# ---------------------------------------------------------------------------
# 7. Module orchestration
# ---------------------------------------------------------------------------

class TestGradientAscentMod:
    def test_runs_once_per_constant_and_catches_errors(self, simple_shard, monkeypatch, tmp_path):
        from dreamer.search.searchers.gradient_ascent_mod import GradientAscentMod
        from dreamer.search.searchers import gradient_ascent_mod as mod_module
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)

        run_calls = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None,
                     geom=None, start=None, pool=None):
            run_calls.append(constant)
            if constant.name == "pi":
                raise NoInitialIdentification(shard_id, constant)
            if constant.name == "e":
                raise SearchStalled(shard_id, constant, 10)

        monkeypatch.setattr(mod_module.GradientAscentSearch, "run", fake_run)

        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = GradientAscentMod(priorities, use_LIReC=False)
        searcher.execute()  # both errors must be swallowed

        names = sorted(c.name for c in run_calls)
        assert names == ["e", "pi"]

    def test_geometry_built_once_per_shard(self, simple_shard, monkeypatch, tmp_path):
        """FlatlandGeometry (LLL/BKZ) is constructed once per shard, reused
        across the shard's identified constants."""
        from dreamer.search.searchers.gradient_ascent_mod import GradientAscentMod
        from dreamer.search.searchers import gradient_ascent_mod as mod_module
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)
        monkeypatch.setattr(config.search, "GRAD_NUM_EVAL_WORKERS", 0, raising=False)

        construct_count = [0]
        real_geom = mod_module.FlatlandGeometry

        def counting_geom(shard):
            construct_count[0] += 1
            return real_geom(shard)

        monkeypatch.setattr(mod_module, "FlatlandGeometry", counting_geom)

        received = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None,
                     geom=None, start=None, pool=None):
            received.append(geom)

        monkeypatch.setattr(mod_module.GradientAscentSearch, "run", fake_run)

        priorities = {e: [simple_shard], pi: [simple_shard]}
        GradientAscentMod(priorities, use_LIReC=False).execute()

        assert construct_count[0] == 1
        assert len(received) == 2 and received[0] is received[1] is not None

    def test_empty_searchables_is_noop(self, monkeypatch, tmp_path):
        from dreamer.search.searchers.gradient_ascent_mod import GradientAscentMod
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        searcher = GradientAscentMod({}, use_LIReC=False)
        searcher.execute()  # must not raise
