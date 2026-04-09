import pytest
import numpy as np
from ramanujantools import Position

from dreamer.search.methods.sa import SimulatedAnnealingSearchMethod
from dreamer.utils.storage.storage_objects import SearchData, SearchVector


# --- Picklable Dummy Classes ---
class DummyCMF:
    def __init__(self):
        self.symbols = ['x', 'y']

    def dim(self):
        return 2


class DummySpace:
    def __init__(self):
        self.cmf = DummyCMF()
        self.is_whole_space = True
        self.A = None

    def get_interior_point(self):
        return Position({'x': 1, 'y': 1})

    def sample_trajectories(self, compute_n_samples):
        # Return a set with one sample trajectory
        return {Position({'x': 1, 'y': 2})}

    def in_space(self, pos):
        return True

    def compute_trajectory_data(self, traj, start, **kwargs):
        # Simulate computing a delta
        sd = SearchData(SearchVector(start, traj))
        sd.delta = float(np.random.uniform(1.0, 10.0))
        return sd


@pytest.fixture
def mock_space():
    """Provides a picklable 2D searchable space."""
    return DummySpace()


def test_sa_initialization(mock_space):
    """Tests if hyperparameters and DataManager are initialized correctly."""
    method = SimulatedAnnealingSearchMethod(
        mock_space, constant=None, iterations=50, t0=5.0, tmin=0.1
    )
    assert method.iterations == 50
    assert method.t0 == 5.0
    assert method.data_manager is not None


def test_flatland_projection_unconstrained(mock_space):
    """Tests the bidirectional projection into a 2D flatland space without hyperplanes."""
    method = SimulatedAnnealingSearchMethod(mock_space, constant=None)
    method._setup_flatland()

    # Unconstrained space should yield an Identity basis
    assert method.dim_flat == 2
    np.testing.assert_array_equal(method.Z, np.eye(2))

    # Test projection -> flatland
    orig_pos = Position({'x': 3, 'y': 7})
    flat_v = method._to_flatland(orig_pos)
    np.testing.assert_array_equal(flat_v, np.array([3, 7]))

    # Test projection -> original
    proj_pos = method._to_original(flat_v)
    assert proj_pos['x'] == 3 and proj_pos['y'] == 7


def test_annealing_execution_loop(mock_space):
    """Tests that the search executes concurrently and populates the DataManager."""
    method = SimulatedAnnealingSearchMethod(
        mock_space, constant=None, iterations=10, cores=2
    )

    result_data = method.search()

    # Verify that the DataManager collected evaluated trajectories
    assert len(result_data) > 0
