import numpy as np
import pytest

from dreamer.extraction.utils import initial_points


def test_decode_signatures_returns_expected_signs():
    """Assumption: little-endian bit packing; failure mode: incorrect sign decoding order."""
    # 0b101 over three hyperplanes => [+1, -1, +1]
    decoded = initial_points.decode_signatures([(5,)], 3)
    assert np.array_equal(decoded, np.array([[1, -1, 1]], dtype=np.int8))


def test_decode_signatures_empty_input_returns_empty_matrix():
    """Assumption: empty inputs are valid; failure mode: shape mismatch for downstream matrix ops."""
    decoded = initial_points.decode_signatures([], 4)
    assert decoded.shape == (0, 4)


def test_decode_signatures_rejects_negative_hyperplane_count():
    """Assumption: hyperplane count is non-negative; failure mode: silent invalid shape construction."""
    with pytest.raises(ValueError, match=r"M must be non-negative"):
        initial_points.decode_signatures([(1,)], -1)


def test_filter_symmetrical_cones_deduplicates_points():
    """Assumption: equivalent pFq cones reduce to one representative; failure mode: duplicate shard work."""
    mapping = {
        (1,): np.array([3, 1, 4]),
        (2,): np.array([1, 3, 4]),
        (3,): np.array([2, 5, 6]),
    }
    filtered = initial_points.filter_symmetrical_cones(mapping, p=2, q=1, shift=[0, 0, 0])

    assert len(filtered) == 2


def test_filter_symmetrical_cones_validates_dimensions():
    """Assumption: p+q must equal shift dimension; failure mode: incorrect symmetry partitioning."""
    with pytest.raises(ValueError, match=r"p \+ q must be the dimension"):
        initial_points.filter_symmetrical_cones({(1,): np.array([1, 2])}, p=1, q=2, shift=[0, 0])


class _DummyPool:
    """Sequential Pool replacement to keep tests deterministic and process-safe."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, tasks):
        return [func(*task) for task in tasks]


def test_compute_mapping_selects_closest_point_per_signature(monkeypatch):
    """Assumption: all points share one signature; failure mode: first-seen point leaks through instead of nearest."""
    monkeypatch.setattr(initial_points.mp, "Pool", lambda *_args, **_kwargs: _DummyPool())
    monkeypatch.setattr(initial_points.mp, "cpu_count", lambda: 1)

    D = 1
    S = 5
    A = np.array([[1]], dtype=np.int64)
    b = np.array([10], dtype=np.int64)

    mapping = initial_points.compute_mapping(D=D, S=S, A=A, b=b, prefix_dims=1)

    assert len(mapping) == 1
    selected = next(iter(mapping.values()))
    assert np.array_equal(selected, np.array([0], dtype=np.int64))


def test_compute_mapping_tie_breaks_lexicographically(monkeypatch):
    """Assumption: equal norms must be deterministic; failure mode: nondeterministic shard representatives across runs."""

    def fake_worker(fixed_prefix, D, S, A, b, filter_func=None):
        point = np.array([0, -1], dtype=np.int64) if int(fixed_prefix[0]) == 0 else np.array([-1, 0], dtype=np.int64)
        result = {(1,): point}
        return filter_func(result) if filter_func else result

    monkeypatch.setattr(initial_points.mp, "Pool", lambda *_args, **_kwargs: _DummyPool())
    monkeypatch.setattr(initial_points.mp, "cpu_count", lambda: 1)
    monkeypatch.setattr(initial_points, "__worker_wrapper", fake_worker)

    A = np.array([[1, 0]], dtype=np.int64)
    b = np.array([1], dtype=np.int64)
    mapping = initial_points.compute_mapping(D=2, S=2, A=A, b=b, prefix_dims=1)

    assert np.array_equal(mapping[(1,)], np.array([-1, 0], dtype=np.int64))


def test_compute_mapping_validates_shapes():
    """Assumption: API validates array rank/shape; failure mode: numba crashes from malformed linear system input."""
    with pytest.raises(ValueError, match=r"A second dimension must equal D"):
        initial_points.compute_mapping(
            D=2,
            S=3,
            A=np.array([[1, 2, 3]], dtype=np.int64),
            b=np.array([1], dtype=np.int64),
        )

