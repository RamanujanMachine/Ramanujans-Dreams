"""
Tests for the user-supplied *initial trajectory* search/analysis seed.

A trajectory paired with a selected start point is retained on the shard
(``Shard.selected_trajectory``) and used as the initial seed of every search
method (instead of a reservoir-sampled seed).  Policy (see
``dreamer/search/methods/flatland/seed.py``):

  * geometrically invalid (maps to 0 / outside the cone) → WARN + fall back to
    the default reservoir seed;
  * valid but does not identify the constant → WARN + use it anyway.

This module covers the shared helpers plus the per-method wiring (Gradient
Ascent / SPSA / Small Angle / Simulated Annealing / Genetic).
"""

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.configs import config
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland import seed as seed_mod
from dreamer.search.methods.flatland.seed import trajectory_to_seed, resolve_injected_seed
from dreamer.utils.logger import Logger

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


def _traj(symbols, coords):
    return Position({s: sp.Integer(c) for s, c in zip(symbols, coords)})


# ---------------------------------------------------------------------------
# 1. trajectory_to_seed
# ---------------------------------------------------------------------------

class TestTrajectoryToSeed:
    def test_valid_direction_maps_to_flatland_z(self, whole_space_shard, symbols):
        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        z = trajectory_to_seed(geom, traj)
        assert z is not None
        assert list(z) == list(geom.to_flatland(traj))

    def test_zero_direction_is_invalid(self, whole_space_shard, symbols):
        geom = FlatlandGeometry(whole_space_shard)
        assert trajectory_to_seed(geom, _traj(symbols, [0, 0])) is None

    def test_out_of_cone_direction_is_invalid(self, whole_space_shard, symbols, monkeypatch):
        geom = FlatlandGeometry(whole_space_shard)
        # Force the cone test to reject — emulates an out-of-cone direction on a
        # bounded shard regardless of the (here trivial) whole-space geometry.
        monkeypatch.setattr(geom, "is_inside", lambda z: False)
        assert trajectory_to_seed(geom, _traj(symbols, [2, 0])) is None


# ---------------------------------------------------------------------------
# 2. resolve_injected_seed (policy + user-facing messaging)
# ---------------------------------------------------------------------------

class _FakeLogger:
    Levels = Logger.Levels
    records = []

    def __init__(self, msg, level=None):
        type(self).records.append((msg, level))

    def log(self):
        return None


@pytest.fixture
def capture_logs(monkeypatch):
    _FakeLogger.records = []
    monkeypatch.setattr(seed_mod, "Logger", _FakeLogger)
    return _FakeLogger.records


class TestResolveInjectedSeed:
    def test_none_trajectory_returns_none_silently(self, whole_space_shard, capture_logs):
        geom = FlatlandGeometry(whole_space_shard)
        assert resolve_injected_seed(geom, None, "sid", e) is None
        assert capture_logs == []

    def test_valid_identifying_seed_used_no_warning(self, whole_space_shard, symbols, capture_logs):
        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        z = resolve_injected_seed(geom, traj, "sid", e, identify_fn=lambda z: True)
        assert list(z) == list(geom.to_flatland(traj))
        assert capture_logs == []

    def test_valid_non_identifying_seed_used_with_warning(self, whole_space_shard, symbols, capture_logs):
        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        z = resolve_injected_seed(geom, traj, "sid", e, identify_fn=lambda z: False)
        assert list(z) == list(geom.to_flatland(traj))
        assert any("does not identify" in msg for msg, _ in capture_logs)
        assert all(lvl == Logger.Levels.warning for _, lvl in capture_logs)

    def test_invalid_seed_falls_back_with_warning(self, whole_space_shard, symbols, capture_logs):
        geom = FlatlandGeometry(whole_space_shard)
        # Zero direction is geometrically invalid → None + a fallback warning.
        out = resolve_injected_seed(geom, _traj(symbols, [0, 0]), "sid", e, identify_fn=lambda z: True)
        assert out is None
        assert any("not a valid recession direction" in msg for msg, _ in capture_logs)


# ---------------------------------------------------------------------------
# 3. Per-method run() wiring
# ---------------------------------------------------------------------------

class TestGradientAscentSeed:
    def _make(self, shard):
        from dreamer.search.methods.gradient_ascent.grad_ascent_scan import GradientAscentSearch
        return GradientAscentSearch(shard, e, use_LIReC=False)

    def test_uses_injected_seed_and_skips_reservoir(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        import dreamer.search.methods.flatland.discrete_local_max as dlm

        method = self._make(whole_space_shard)
        evaluated, sampled = [], [0]

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 0.5, True

        monkeypatch.setattr(gs, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: sampled.__setitem__(0, sampled[0] + 1) or {_traj(symbols, [1, 0])})
        monkeypatch.setattr(config.search, "GRAD_MAX_STEPS", 1000, raising=False)

        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={}, initial_trajectory=traj)

        assert list(evaluated[0]) == list(geom.to_flatland(traj))
        assert sampled[0] == 0  # reservoir seeding never invoked

    def test_invalid_injected_seed_falls_back_to_reservoir(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.gradient_ascent.grad_ascent_scan as gs
        import dreamer.search.methods.flatland.discrete_local_max as dlm

        method = self._make(whole_space_shard)
        sampled = [0]

        monkeypatch.setattr(gs, "evaluate_in_flatland", lambda z, **kw: (0.5, True))
        monkeypatch.setattr(dlm, "evaluate_in_flatland", lambda z, **kw: (0.5, True))
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: sampled.__setitem__(0, sampled[0] + 1) or {_traj(symbols, [1, 0])})
        monkeypatch.setattr(config.search, "GRAD_MAX_STEPS", 1000, raising=False)
        monkeypatch.setattr(config.search, "GRAD_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={},
                   initial_trajectory=_traj(symbols, [0, 0]))  # invalid (zero)
        assert sampled[0] >= 1  # fell back to reservoir seeding


class TestSPSASeed:
    def test_uses_injected_seed_and_skips_reservoir(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.gradient_ascent.spsa_adam_ascent as spsa
        import dreamer.search.methods.flatland.discrete_local_max as dlm
        from dreamer.search.methods.gradient_ascent.spsa_adam_ascent import HybridSPSASearch

        method = HybridSPSASearch(whole_space_shard, e, use_LIReC=False)
        evaluated, sampled = [], [0]

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 0.42, True

        monkeypatch.setattr(spsa, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(dlm, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: sampled.__setitem__(0, sampled[0] + 1) or {_traj(symbols, [1, 0])})
        monkeypatch.setattr(config.search, "SPSA_MAX_STEPS", 20, raising=False)

        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={}, initial_trajectory=traj)

        assert list(evaluated[0]) == list(geom.to_flatland(traj))
        assert sampled[0] == 0


class TestSmallAngleSeed:
    def test_uses_injected_seed_and_skips_reservoir(self, whole_space_shard, symbols, monkeypatch):
        from dreamer.search.methods.small_angle.small_angle_scan import SmallAngleSearch

        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        evaluated, sampled = [], [0]

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 0.5, True  # flat → stops on patience quickly

        method._evaluate = fake_eval
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: sampled.__setitem__(0, sampled[0] + 1) or {_traj(symbols, [1, 0])})
        monkeypatch.setattr(search_config, "SA_MAX_DEPTH", 5, raising=False)
        monkeypatch.setattr(search_config, "SA_PATIENCE", 2, raising=False)

        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={}, initial_trajectory=traj)

        assert list(evaluated[0]) == list(geom.to_flatland(traj))
        assert sampled[0] == 0


class TestAnnealingSeed:
    def test_uses_injected_seed_and_skips_reservoir(self, whole_space_shard, symbols, monkeypatch):
        import dreamer.search.methods.annealing.annealing_scan as ann

        from dreamer.search.methods.annealing.annealing_scan import SimulatedAnnealingSearch
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        evaluated, sampled = [], [0]

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 0.5, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: sampled.__setitem__(0, sampled[0] + 1) or {_traj(symbols, [1, 0])})
        # Terminate quickly.
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 1, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 5, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 5, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)

        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={}, initial_trajectory=traj)

        assert list(evaluated[0]) == list(geom.to_flatland(traj))
        assert sampled[0] == 0


class TestGeneticSeed:
    def test_init_population_includes_injected_seed(self, whole_space_shard, symbols, monkeypatch):
        from dreamer.search.methods.genetic_search.genetic_scan import GeneticSearch
        from dreamer.utils.rand import derive_py_random

        method = GeneticSearch(whole_space_shard, e, use_LIReC=False)
        method._rng_py = derive_py_random("sid", "genetic", str(e))
        monkeypatch.setattr(ShardSamplingOrchestrator, "sample_trajectories",
                            lambda self, n: {_traj(symbols, [1, 0]), _traj(symbols, [0, 1])})

        geom = FlatlandGeometry(whole_space_shard)
        traj = _traj(symbols, [2, 0])
        pop = method._init_population(geom, 5, "sid", e, initial_trajectory=traj)

        expected = list(geom.to_flatland(traj))
        assert any(list(g) == expected for g in pop)
        # Injected seed is the first genome.
        assert list(pop[0]) == expected
