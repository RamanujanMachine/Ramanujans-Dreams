from contextlib import contextmanager
from typing import cast

import numpy as np
import pytest
import sympy as sp
from ramanujantools import Position

import dreamer.search.methods.sa as sa_mod
from dreamer.search.methods.sa import SimulatedAnnealingSearchMethod
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import SearchData, SearchVector


pytestmark = pytest.mark.skip(reason="Simulated annealing tests are currently deprecated")


class DummyCMF:
    def __init__(self):
        self.x, self.y = sp.symbols("x y")
        self.symbols = [self.x, self.y]

    def dim(self):
        return 2


class DummySpace:
    def __init__(self):
        self.cmf = DummyCMF()
        self.is_whole_space = True
        self.A = None
        self.symbols = list(self.cmf.symbols)

    def get_interior_point(self):
        x, y = self.cmf.symbols
        return Position({x: 1, y: 1})

    def in_space(self, _pos):
        return True

    def is_valid_trajectory(self, _traj):
        return True

    def compute_trajectory_data(self, traj, start, **_kwargs):
        sd = SearchData(SearchVector(start, traj))
        x, y = self.cmf.symbols
        sd.delta = float(abs(traj[x]) + abs(traj[y]))
        return sd


@pytest.fixture
def mock_space():
    return DummySpace()


@pytest.fixture(autouse=True)
def _mock_shard_sampler(monkeypatch, mock_space):
    x, y = mock_space.cmf.symbols

    def _fake_sample(_self, _compute_n_samples):
        return {Position({x: 1, y: 2})}

    monkeypatch.setattr(sa_mod.ShardSamplingOrchestrator, "sample_trajectories", _fake_sample)


def test_sa_initialization(mock_space):
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, iterations=20, t0=5.0, tmin=0.1
    )
    assert method.iterations == 20
    assert method.t0 == 5.0
    assert method.data_manager is not None


def test_flatland_projection_unconstrained(mock_space):
    method = SimulatedAnnealingSearchMethod(cast(Searchable, cast(object, mock_space)), constant=None)
    method._setup_flatland()

    assert method.dim_flat == 2
    np.testing.assert_array_equal(method.Z, np.eye(2))

    x, y = mock_space.cmf.symbols
    orig_pos = Position({x: 3, y: 7})
    flat_v = method._to_flatland(orig_pos)
    np.testing.assert_array_equal(flat_v, np.array([3, 7]))

    proj_pos = method._to_original(flat_v)
    assert proj_pos[x] == 3 and proj_pos[y] == 7


def test_sa_search_early_exit_when_no_neighbors(monkeypatch, mock_space):
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, iterations=10, cores=1
    )

    monkeypatch.setattr(method, "_get_neighbors_flatland", lambda *_args, **_kwargs: [])

    class _DummyPool:
        def map(self, func, trajs, starts):
            return [func(t, s) for t, s in zip(trajs, starts)]

    @contextmanager
    def _dummy_pool_context():
        yield _DummyPool()

    monkeypatch.setattr(sa_mod, "create_pool", _dummy_pool_context)

    result_data = method.search()
    assert len(result_data) >= 1


def test_log_schedule_temperature_is_finite_and_decreasing(mock_space):
    method = SimulatedAnnealingSearchMethod(
        cast(Searchable, cast(object, mock_space)), constant=None, schedule_type="log", t0=5.0
    )

    t1 = method._get_temp(1)
    t2 = method._get_temp(2)

    assert np.isfinite(t1)
    assert np.isfinite(t2)
    assert t2 < t1
