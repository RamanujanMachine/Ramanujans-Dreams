import numpy as np
import pytest

from dreamer.extraction.utils import initial_points


def test_decode_signatures_returns_expected_signs():
    # 0b101 over three hyperplanes => [+1, -1, +1]
    decoded = initial_points.decode_signatures([(5,)], 3)
    assert np.array_equal(decoded, np.array([[1, -1, 1]], dtype=np.int8))


def test_decode_signatures_empty_input_returns_empty_matrix():
    decoded = initial_points.decode_signatures([], 4)
    assert decoded.shape == (0, 4)


def test_filter_symmetrical_cones_deduplicates_points():
    mapping = {
        (1,): np.array([3, 1, 4]),
        (2,): np.array([1, 3, 4]),
        (3,): np.array([2, 5, 6]),
    }
    filtered = initial_points.filter_symmetrical_cones(mapping, p=2, q=1, shift=[0, 0, 0])

    assert len(filtered) == 2


def test_filter_symmetrical_cones_validates_dimensions():
    with pytest.raises(ValueError, match=r"p \+ q must be the dimension"):
        initial_points.filter_symmetrical_cones({(1,): np.array([1, 2])}, p=1, q=2, shift=[0, 0])
