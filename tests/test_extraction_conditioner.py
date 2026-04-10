import numpy as np

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner


def test_extract_constraints_detects_equalities():
    a = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
    ])
    conditioner = HyperSpaceConditioner(a)

    e, b = conditioner._extract_constraints()

    assert e.shape == (2, 2)
    assert b.shape == (1, 2)


def test_compute_integer_basis_without_equalities_is_identity():
    conditioner = HyperSpaceConditioner(np.eye(3))
    basis = conditioner._compute_integer_basis(np.empty((0, 3)))

    assert np.array_equal(basis, np.eye(3, dtype=np.int64))


def test_compute_integer_basis_for_x_equals_y_plane():
    conditioner = HyperSpaceConditioner(np.eye(2))
    basis = conditioner._compute_integer_basis(np.array([[1.0, -1.0]]))

    assert basis.shape == (2, 1)
    assert np.array_equal(np.abs(basis[:, 0]), np.array([1, 1]))


def test_transform_bounds_with_no_inequalities_returns_empty():
    conditioner = HyperSpaceConditioner(np.eye(2))
    transformed = conditioner._transform_bounds(np.empty((0, 2)), np.eye(2), np.eye(2))

    assert transformed.shape == (0, 2)


def test_process_can_run_with_monkeypatched_reduction(monkeypatch):
    a = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
    ])
    conditioner = HyperSpaceConditioner(a)

    def _fake_ratchet(z):
        return z, np.eye(z.shape[1], dtype=np.int64)

    monkeypatch.setattr(conditioner, "_ratchet_lattice_reduction", _fake_ratchet)

    z_reduced, b_reduced, u = conditioner.process()

    assert z_reduced.shape == (2, 1)
    assert b_reduced.shape == (1, 1)
    assert np.array_equal(u, np.eye(1, dtype=np.int64))
