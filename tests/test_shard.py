"""Tests for the Shard class geometry and construction.

Covers:
- in_space() for interior and exterior points
- Whole-space shards (A=None, b=None)
- generate_matrices() from hyperplane sign vectors
- Interior point retrieval
"""
import numpy as np
import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_cmf():
    """A 1F1(z=1) CMF — 2 symbols."""
    return rt_pFq(1, 1, 1)


@pytest.fixture
def const_e():
    from dreamer import e
    return e


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


# ---------------------------------------------------------------------------
# 1. Bounded shards
# ---------------------------------------------------------------------------
class TestBoundedShard:

    def test_interior_point_is_inside(self, simple_cmf, const_e, symbols):
        """A known interior point must satisfy Ax < b."""
        # Shard: x0 > 0, x1 > 0, x0 + x1 < 10
        A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
        b = np.array([0, 0, 10], dtype=float)
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, A, b,
                      Position({s0: 0, s1: 0}), symbols)

        assert shard.in_space(Position({s0: 3, s1: 3}))

    def test_exterior_point_is_outside(self, simple_cmf, const_e, symbols):
        A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
        b = np.array([0, 0, 10], dtype=float)
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, A, b,
                      Position({s0: 0, s1: 0}), symbols)

        # (-1, 3): violates x0 > 0
        assert not shard.in_space(Position({s0: -1, s1: 3}))

    def test_boundary_point_is_outside(self, simple_cmf, const_e, symbols):
        """Points exactly on the boundary (Ax = b) are excluded (strict inequality)."""
        A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
        b = np.array([0, 0, 10], dtype=float)
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, A, b,
                      Position({s0: 0, s1: 0}), symbols)

        # (5, 5) is on boundary: x0+x1 = 10
        assert not shard.in_space(Position({s0: 5, s1: 5}))

    def test_sum_constraint_violation(self, simple_cmf, const_e, symbols):
        A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
        b = np.array([0, 0, 10], dtype=float)
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, A, b,
                      Position({s0: 0, s1: 0}), symbols)

        # (6, 6): sum = 12 > 10
        assert not shard.in_space(Position({s0: 6, s1: 6}))


# ---------------------------------------------------------------------------
# 2. Whole-space shards
# ---------------------------------------------------------------------------
class TestWholeSpaceShard:

    def test_whole_space_accepts_any_point(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, None, None,
                      Position({s0: 0, s1: 0}), symbols)

        assert shard.is_whole_space
        assert shard.in_space(Position({s0: 1000, s1: -999}))


# ---------------------------------------------------------------------------
# 3. Interior point retrieval
# ---------------------------------------------------------------------------
class TestInteriorPoint:

    def test_get_interior_point_returns_position(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        A = np.array([[-1, 0], [0, -1]], dtype=float)
        b = np.array([0, 0], dtype=float)
        ip = Position({s0: 5, s1: 5})
        shard = Shard(simple_cmf, const_e, A, b, ip, symbols)

        result = shard.get_interior_point()
        assert isinstance(result, Position)

    def test_no_interior_point_returns_zero(self, simple_cmf, const_e, symbols):
        """If no interior_point is given, a zero position is returned."""
        s0, s1 = symbols
        A = np.array([[1, 0]], dtype=float)
        b = np.array([10], dtype=float)
        shard = Shard(simple_cmf, const_e, A, b,
                      Position({s0: 0, s1: 0}), symbols,
                      interior_point=None)

        result = shard.get_interior_point()
        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# 4. generate_matrices
# ---------------------------------------------------------------------------
class TestGenerateMatrices:

    def test_basic_generation(self):
        from dreamer.extraction.hyperplanes import Hyperplane
        x, y = sp.symbols("x y")
        hps = [
            Hyperplane(x, symbols=[x, y]),
            Hyperplane(y, symbols=[x, y]),
        ]
        indicator = (1, 1)  # both above
        A, b, syms = Shard.generate_matrices(hps, indicator)

        assert A.shape[0] == 2  # 2 hyperplanes
        assert A.shape[1] == 2  # 2 dimensions
        assert len(b) == 2

    def test_invalid_indicator_raises(self):
        from dreamer.extraction.hyperplanes import Hyperplane
        x, y = sp.symbols("x y")
        hps = [Hyperplane(x, symbols=[x, y])]
        with pytest.raises(ValueError, match="Indicators vector must be 1"):
            Shard.generate_matrices(hps, (0,))
