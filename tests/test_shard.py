"""Tests for Shard geometry, shifted coordinates, and matrix generation."""
import numpy as np
import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_cmf():
    """A 1F1(z=1) CMF — 2 symbols."""
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def const_e():
    from dreamer import e
    return e


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


def _point(symbols, values):
    return Position({sym: sp.sympify(v) for sym, v in zip(symbols, values)})


def _build_shard(simple_cmf, const_e, hyperplanes, shift, interior_point):
    # Match extractor behavior: sign is selected by evaluation at a known interior point.
    encoding = []
    interior_subs = {s: interior_point[s] for s in symbols_from_hyperplanes(hyperplanes, simple_cmf)}
    for hp in hyperplanes:
        value = hp.expr.subs(interior_subs)
        if value == 0:
            raise ValueError("Interior point cannot lie on shard boundary")
        encoding.append(1 if value > 0 else -1)
    return Shard(simple_cmf, const_e, hyperplanes, encoding, shift, interior_point)


def symbols_from_hyperplanes(hyperplanes, cmf):
    if hyperplanes:
        return list(hyperplanes[0].symbols)
    return list(cmf.matrices.keys())


# ---------------------------------------------------------------------------
# 1. Bounded shards
# ---------------------------------------------------------------------------
class TestBoundedShard:

    def test_interior_point_is_inside(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [
            Hyperplane(s0, symbols),
            Hyperplane(s1, symbols),
            Hyperplane(s0 + s1 - 10, symbols),
        ]
        inside = _point(symbols, [3, 3])
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), inside)

        assert shard.in_space(inside)

    def test_exterior_point_is_outside(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 10, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), _point(symbols, [3, 3]))

        assert not shard.in_space(_point(symbols, [-1, 3]))

    def test_boundary_point_is_outside(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 10, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), _point(symbols, [3, 3]))

        assert not shard.in_space(_point(symbols, [5, 5]))

    def test_sum_constraint_violation(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 10, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), _point(symbols, [3, 3]))

        assert not shard.in_space(_point(symbols, [6, 6]))

    def test_position_key_order_does_not_change_membership(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 10, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), _point(symbols, [3, 3]))

        p_ordered = Position({s0: sp.Integer(3), s1: sp.Integer(2)})
        p_reversed = Position({s1: sp.Integer(2), s0: sp.Integer(3)})
        assert shard.in_space(p_ordered)
        assert shard.in_space(p_reversed)


class TestShiftedPoints:

    def test_shifted_constraints_accept_inside_absolute_point(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shift = _point(symbols, [2, -3])
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 5, symbols)]
        inside_abs = _point(symbols, [3, -2])
        shard = _build_shard(simple_cmf, const_e, hps, shift, inside_abs)

        assert shard.in_space(inside_abs)

    def test_shifted_constraints_reject_boundary_absolute_point(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shift = _point(symbols, [2, -3])
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 5, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, shift, _point(symbols, [3, -2]))

        # The boundary of the absolute shifted space includes s0=0.
        # [0, -2] sits exactly on the s0=0 boundary line, so it should be rejected.
        boundary_abs = _point(symbols, [0, -2])
        assert not shard.in_space(boundary_abs)

    def test_shifted_constraints_reject_outside_absolute_point(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shift = _point(symbols, [2, -3])
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 5, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, shift, _point(symbols, [3, -2]))

        assert not shard.in_space(_point(symbols, [7, 0]))

    def test_shifted_point_order_is_robust(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shift = _point(symbols, [2, -3])
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols), Hyperplane(s0 + s1 - 5, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, shift, _point(symbols, [3, -2]))

        assert shard.in_space(Position({s1: sp.Integer(-2), s0: sp.Integer(3)}))

    def test_fractional_shift_inside_outside_and_boundary(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        # Using sp.Rational strictly asserts we can handle proper fractions in shift computation
        shift = _point(symbols, [sp.Rational(1, 2), sp.Rational(-1, 3)])
        hps = [
            Hyperplane(s0 + s1 - 1, symbols),
            Hyperplane(s0 - s1, symbols),
        ]

        inside_abs = _point(symbols, [sp.Rational(3, 2), sp.Rational(1, 6)])
        shard = _build_shard(simple_cmf, const_e, hps, shift, inside_abs)

        boundary_abs = _point(symbols, [sp.Rational(5, 6), sp.Rational(1, 6)])
        outside_abs = _point(symbols, [sp.Rational(0), sp.Rational(0)])

        assert shard.in_space(inside_abs)
        assert not shard.in_space(boundary_abs)
        assert not shard.in_space(outside_abs)

    def test_explicit_rational_shift_calculations(self, simple_cmf, const_e, symbols):
        """Dedicated test ensuring complex rational shifts don't cause floating point/type errors."""
        s0, s1 = symbols
        shift = _point(symbols, [sp.Rational(3, 7), sp.Rational(-4, 5)])

        # Test planes: s0 - 1 > 0 and s1 + 1 > 0
        hps = [
            Hyperplane(s0 - 1, symbols),
            Hyperplane(s1 + 1, symbols)
        ]

        inside = _point(symbols, [2, 0])
        shard = _build_shard(simple_cmf, const_e, hps, shift, inside)

        # Checking points defined fully via rationals
        valid_point = _point(symbols, [sp.Rational(3, 2), sp.Rational(-1, 2)])
        boundary_point = _point(symbols, [sp.Rational(1, 1), sp.Rational(-1, 2)])

        assert shard.in_space(valid_point)
        assert not shard.in_space(boundary_point)


# ---------------------------------------------------------------------------
# 2. Whole-space shards
# ---------------------------------------------------------------------------
class TestWholeSpaceShard:

    def test_whole_space_accepts_any_point(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        shard = Shard(simple_cmf, const_e, [], [], _point(symbols, [0, 0]))

        assert shard.is_whole_space
        assert shard.in_space(_point(symbols, [1000, -999]))

    def test_empty_hyperplanes_produce_cmf_symbol_order(self, simple_cmf, const_e, symbols):
        shard = Shard(simple_cmf, const_e, [], [], _point(symbols, [0, 0]))
        # Accessing `symbols` property directly as the class doesn't have a `get_symbols()` method.
        assert shard.symbols == symbols


# ---------------------------------------------------------------------------
# 3. Interior point retrieval
# ---------------------------------------------------------------------------
class TestInteriorPoint:

    def test_get_interior_point_returns_position(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols)]
        ip = _point(symbols, [5, 5])
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), ip)

        result = shard.get_interior_point()
        assert isinstance(result, Position)

    def test_no_interior_point_returns_zero(self, simple_cmf, const_e, symbols):
        """If no interior_point is given, a zero position is returned."""
        s0, _ = symbols
        hps = [Hyperplane(s0, symbols)]
        shard = Shard(simple_cmf, const_e, hps, [1], _point(symbols, [0, 0]), interior_point=None)

        result = shard.get_interior_point()
        assert all(v == 0 for v in result.values())

    def test_interior_point_symbol_order_is_preserved(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols)]
        interior = Position({s1: sp.Integer(11), s0: sp.Integer(7)})
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), interior)

        result = shard.get_interior_point()
        assert result[s0] == 7
        assert result[s1] == 11


class TestTrajectoryValidity:

    def test_is_valid_trajectory_checks_cone_direction(self, simple_cmf, const_e, symbols):
        s0, s1 = symbols
        hps = [Hyperplane(s0, symbols), Hyperplane(s1, symbols)]
        shard = _build_shard(simple_cmf, const_e, hps, _point(symbols, [0, 0]), _point(symbols, [1, 1]))

        assert shard.is_valid_trajectory(_point(symbols, [1, 2]))
        assert not shard.is_valid_trajectory(_point(symbols, [-1, 2]))



# ---------------------------------------------------------------------------
# 4. generate_matrices
# ---------------------------------------------------------------------------
class TestGenerateMatrices:

    def test_basic_generation(self):
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
        x, y = sp.symbols("x y")
        hps = [Hyperplane(x, symbols=[x, y])]
        with pytest.raises(ValueError, match="Indicators vector must be 1"):
            Shard.generate_matrices(hps, (0,))

    def test_generation_matches_expected_vectors(self):
        x, y = sp.symbols("x y")
        hps = [
            Hyperplane(x + y - 2, symbols=[x, y]),
            Hyperplane(x - y, symbols=[x, y]),
        ]

        # First as_above (1), second as_below (-1)
        A, b, syms = Shard.generate_matrices(hps, (1, -1))

        assert syms == [x, y]

        # as_above_vector representation of `x + y - 2 > 0`
        # Formatted to < bounding constraints: -x - y < -2
        assert np.array_equal(A[0], np.array([-1, -1]))
        assert b[0] == -2

        # as_below_vector representation of `x - y < 0`
        assert np.array_equal(A[1], np.array([1, -1]))
        assert b[1] == 0
