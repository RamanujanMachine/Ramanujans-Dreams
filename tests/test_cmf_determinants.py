"""Closed-form Meijer-G axis determinants (``dreamer.extraction.meijer_g_determinant``).

Same strategy as RT's ``pfq_test.py``: compare the closed form with the actual
``Matrix.det()`` — symbolically where ``det()`` is cheap, and exactly at random
rational points for MG55 / MG66 (where a symbolic ``det()`` does not finish).
"""
import random

import pytest
import sympy as sp
from sympy.abc import z
from ramanujantools.cmf import MeijerG

from dreamer.extraction.meijer_g_determinant import meijer_g_axis_sign, meijer_g_determinant


def _axes(cmf):
    return list(cmf.matrices.keys())


@pytest.mark.timeout(300)
@pytest.mark.parametrize("args", [
    (1, 1, 2, 2, z), (2, 1, 2, 2, -1), (1, 2, 2, 3, z), (2, 0, 3, 2, 1), (1, 1, 2, 1, z),
    (2, 2, 2, 2, 1),
])
def test_symbolic_match(args):
    """Small Meijer-G CMFs, incl. degree drops (z = ±1 cancels the top coefficient)."""
    cmf = MeijerG(*args)
    for axis in _axes(cmf):
        expected = sp.factor(cmf.matrices[axis].det())
        assert sp.simplify(meijer_g_determinant(*args, axis) - expected) == 0, axis


def _generic_point(symbols, rng):
    """Rational point with pairwise-distinct, non-integer-offset coordinates (avoids poles)."""
    primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163]
    return {s: sp.Rational(rng.randint(-50, 50), 1) + sp.Rational(1, primes[i]) for i, s in enumerate(symbols)}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("args", [(2, 2, 3, 3, 1), (3, 1, 3, 3, z), (5, 2, 5, 5, 1), (6, 2, 6, 6, 1)])
def test_numeric_match(args):
    """MG33 / MG55 / MG66 (MG55 and MG66 at z = 1 drop one degree): exact values at random points."""
    rng = random.Random(7)
    cmf = MeijerG(*args)
    for axis in _axes(cmf):
        M = cmf.matrices[axis]
        closed = meijer_g_determinant(*args, axis)
        symbols = sorted(M.free_symbols | closed.free_symbols, key=str)
        for _ in range(2):
            pt = _generic_point(symbols, rng)
            assert M.subs(pt).det() == closed.subs(pt), (axis, pt)


def test_axis_sign_matches_rt():
    # RT: a_i -> -1 if i >= n else 1;  b_j -> 1 if j >= m else -1.
    a0, a3, b0, b4 = sp.symbols("a0 a3 b0 b4")
    assert [meijer_g_axis_sign(2, 5, s) for s in (a0, a3, b0, b4)] == [1, -1, -1, -1]
    assert meijer_g_axis_sign(2, 1, b4) == 1


def test_rejects_non_axis():
    with pytest.raises(ValueError):
        meijer_g_determinant(1, 1, 2, 2, z, sp.Symbol("x0"))


@pytest.mark.timeout(300)
def test_mg55_hyperplane_extraction_finishes():
    """Was > 4 min for the first axis alone with a symbolic det(); now seconds."""
    from ramanujantools import Position
    from dreamer.extraction.extractor import extract_cmf_hyperplanes
    from dreamer.utils.types import CMFData

    cmf = MeijerG(5, 2, 5, 5, 1)
    data = CMFData(cmf, Position({s: 0 for s in _axes(cmf)}), None, False, True, "mg55")
    exprs = {str(h.expr) for h in extract_cmf_hyperplanes(data)}
    a0, b0 = sp.symbols("a0 b0")
    # zeros/poles a_i = b_j, a_i = b_j + 1 and the rank-reduction wall Σa − Σb = 5 (z = 1)
    assert {"a0 - b0", "a0 - b0 - 1", "a0 + a1 + a2 + a3 + a4 - b0 - b1 - b2 - b3 - b4 - 5"} <= exprs
