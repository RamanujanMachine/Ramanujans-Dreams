"""Closed-form determinants of CMF axis matrices (for hyperplane extraction).

A generic symbolic ``Matrix.det()`` does not finish for large D-finite CMFs (the
first axis of ``MeijerG(5, 2, 5, 5, 1)`` ran for more than 4 minutes).  For D-finite
CMFs the determinant follows from the differential equation and the contiguous
relations instead:

1. the differential equation P(θ) = 0 (θ = z d/dz) of degree d;
2. the contiguous relation of each axis, expressing θ through the shift S of that
   axis (RT builds the axis matrix from the companion matrix C of the monic P);
3. substituting (2) into (1) gives the characteristic polynomial of S; its free
   coefficient times (−1)^d is det(S).

For ``MeijerG`` (``ramanujantools.cmf.meijer_g``), with C = companion(P / LC(P)),
s = ±1 the axis sign of ``MeijerG.construct_matrix`` and d = deg P:

* b axes (raising):  M_b = s (C − b I)  ⇒  det M_b = (−s)^d · (P / LC)(θ = b);
* a axes (stored as the inverse of the lowering matrix s (C − (a−1) I), shifted
  a → a+1):  det M_a = 1 / [ (−s)^d · ((P / LC)|_{a → a+1})(θ = a) ].

The shift a → a+1 applies to LC too: when z = ±1 makes the top coefficient
cancel (e.g. MG55 at z = 1, where d drops from 5 to 4) the new leading coefficient
depends on the a's.  pFq uses RT's own hard-coded ``pFq.determinant``.
Verified against ``det()`` in ``tests/test_cmf_determinants.py``.
"""
from __future__ import annotations

import sympy as sp
from ramanujantools.cmf import MeijerG as rt_MeijerG
from ramanujantools.cmf import pFq as rt_pFq
from ramanujantools.cmf.d_finite import theta


def meijer_g_axis_sign(n: int, m: int, axis: sp.Symbol) -> int:
    """The sign RT's ``MeijerG.construct_matrix`` multiplies the axis matrix by."""
    index = int(axis.name[1:])
    if axis.name.startswith("a"):
        return -1 if index >= n else 1
    return 1 if index >= m else -1


def meijer_g_determinant(m: int, n: int, p: int, q: int, z, axis: sp.Symbol) -> sp.Expr:
    """Factored determinant of the ``MeijerG(m, n, p, q, z)`` axis matrix of *axis*.

    :param axis: ``a_i`` or ``b_j`` symbol of the CMF.
    :raises ValueError: for a symbol that is not an a/b axis.
    """
    name = axis.name
    if not (name[:1] in ("a", "b") and name[1:].isdigit()):
        raise ValueError(f"not a Meijer-G axis: {axis}")
    P = rt_MeijerG.differential_equation(m, n, p, q, z)
    monic = P.as_expr() / P.LC()
    d = P.degree()
    factor = (-meijer_g_axis_sign(n, m, axis)) ** d
    if name.startswith("b"):
        return sp.factor(factor * monic.subs(theta, axis))
    lowered = monic.subs(axis, axis + 1).subs(theta, axis)
    return sp.factor(1 / (factor * lowered))


def axis_determinant(cmf, axis: sp.Symbol) -> sp.Expr:
    """Determinant of ``cmf.matrices[axis]``: closed form when known, else ``det()``."""
    if isinstance(cmf, rt_pFq):
        return rt_pFq.determinant(cmf.p, cmf.q, cmf.z, axis)
    if isinstance(cmf, rt_MeijerG):
        return meijer_g_determinant(cmf.m, cmf.n, cmf.p, cmf.q, cmf.z, axis)
    return cmf.matrices[axis].det()
