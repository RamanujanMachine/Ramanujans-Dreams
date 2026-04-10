"""DEPRECATED: legacy SA tests kept for reference only (excluded from active suite)."""

import pytest
import numpy as np
import sympy as sp
from ramanujantools import Position
from contextlib import contextmanager
from typing import cast

import dreamer.search.methods.sa as sa_mod
from dreamer.search.methods.sa import SimulatedAnnealingSearchMethod
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import SearchData, SearchVector


pytestmark = pytest.mark.skip(reason="Deprecated: simulated annealing test module is not maintained")


# --- Picklable Dummy Classes ---
class DummyCMF:
    def __init__(self):
        self.x, self.y = sp.symbols('x y')
        self.symbols = [self.x, self.y]

    def dim(self):
        return 2


class DummySpace:
    def __init__(self):
        self.cmf = DummyCMF()
        self.is_whole_space = True
        self.A = None
        self.symbols = self.get_symbols()

    def get_symbols(self):
        return list(self.cmf.symbols)

    def is_unconstrained(self):
        return True

    def get_interior_point(self):
        x, y = self.cmf.symbols
        return Position({x: 1, y: 1})

    def sample_trajectories(self, compute_n_samples):
        # Return a set with one sample trajectory
        x, y = self.cmf.symbols
        return {Position({x: 1, y: 2})}

    def in_space(self, pos):
        return True

    def compute_trajectory_data(self, traj, start, **kwargs):
        # Deterministic score keeps tests stable across runs.
        sd = SearchData(SearchVector(start, traj))
        x, y = self.cmf.symbols
        sd.delta = float(abs(traj[x]) + abs(traj[y]))
        return sd


@pytest.fixture
def mock_space():
    """Provides a picklable 2D searchable space."""
    return DummySpace()


@pytest.fixture(autouse=True)
def _mock_shard_sampler(monkeypatch, mock_space):
    x, y = mock_space.cmf.symbols

    def _fake_sample(_self, _compute_n_samples):
        return {Position({x: 1, y: 2})}

    monkeypatch.setattr("dreamer.extraction.sampler.shard_sampler.ShardSampler.sample_trajectories", _fake_sample)


def test_sa_initialization(mock_space):
    """Tests if hyperparameters and DataManager are initialized correctly."""
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, iterations=50, t0=5.0, tmin=0.1
    )
    assert method.iterations == 50
    assert method.t0 == 5.0
    assert method.data_manager is not None


def test_flatland_projection_unconstrained(mock_space):
    """Tests the bidirectional projection into a 2D flatland space without hyperplanes."""
    method = SimulatedAnnealingSearchMethod(cast(Searchable, cast(object, mock_space)), constant=None)
    method._setup_flatland()

    # Unconstrained space should yield an Identity basis
    assert method.dim_flat == 2
    np.testing.assert_array_equal(method.Z, np.eye(2))

    # Test projection -> flatland
    x, y = mock_space.cmf.symbols
    orig_pos = Position({x: 3, y: 7})
    flat_v = method._to_flatland(orig_pos)
    np.testing.assert_array_equal(flat_v, np.array([3, 7]))

    # Test projection -> original
    proj_pos = method._to_original(flat_v)
    assert proj_pos[x] == 3 and proj_pos[y] == 7


def test_annealing_execution_loop(mock_space):
    """Tests that the search executes concurrently and populates the DataManager."""
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, iterations=10, cores=2
    )

    result_data = method.search()

    # Verify that the DataManager collected evaluated trajectories
    assert len(result_data) > 0


def test_sa_search_respects_iteration_budget(monkeypatch, mock_space):
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, iterations=6, t0=10.0, tmin=1e-9
    )

    calls = {"neighbor_batches": 0}

    def fake_neighbors(_cur, _start, num_samples=10):
        calls["neighbor_batches"] += 1
        return [np.array([1, 1], dtype=int)]

    def fake_eval(traj_flat, start):
        x, y = mock_space.cmf.symbols
        traj_orig = Position({x: int(traj_flat[0]), y: int(traj_flat[1])})
        sv = SearchVector(start, traj_orig)
        sd = SearchData(sv)
        sd.delta = 1.0
        return {"traj_flat": traj_flat, "traj_orig": traj_orig, "delta": 1.0, "sd": sd, "sv": sv}

    class _DummyPool:
        def map(self, func, trajs, starts):
            return [func(t, s) for t, s in zip(trajs, starts)]

    @contextmanager
    def _dummy_pool_context():
        yield _DummyPool()

    monkeypatch.setattr(method, "_get_neighbors_flatland", fake_neighbors)
    monkeypatch.setattr(method, "_evaluate_trajectory", fake_eval)
    monkeypatch.setattr(sa_mod, "create_pool", _dummy_pool_context)

    method.search()

    assert calls["neighbor_batches"] == method.iterations


def test_log_schedule_temperature_is_finite_and_decreasing(mock_space):
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, schedule_type="log", t0=5.0
    )

    t1 = method._get_temp(1)
    t2 = method._get_temp(2)

    assert np.isfinite(t1)
    assert np.isfinite(t2)
    assert t2 < t1
