"""
High-precision evaluation of constants with FLINT's arb (ball arithmetic).

``sympy``'s ``expr.evalf(digits)`` goes through mpmath, whose special-function
algorithms (and caches) grow roughly quadratically with the precision: ζ(3) to
80 000 digits takes ~7 s and ~2.8 GB, and mpmath keeps that memory for the rest
of the process.  δ needs exactly such precisions on long trajectories (the
estimate p/q can agree with the constant to 10⁵ digits), so every worker
computing δ used to hold gigabytes.  arb computes the same value in ~0.1 s and a
few MB.

:func:`high_precision` converts the constant's sympy expression into an arb
computation, checks that the resulting ball is accurate to the requested
precision (retrying with more guard bits otherwise), and returns a sympy
``Float`` with the **same binary precision** that ``expr.evalf(digits)`` would
produce, so every downstream formula is unchanged.  Anything the converter does
not support falls back to ``expr.evalf(digits)``.

Supported nodes: integers, rationals, floats, π, e, γ (Euler), Catalan, the
golden ratio, ``+``, ``×``, powers (integer exponents; rational exponents of
positive bases), ``exp``, ``log``, ``zeta``, ``Ei``, ``polylog``, ``gamma``,
``Abs``, ``sin``/``cos``/``atan`` and generalized hypergeometric ``hyper``.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp
from mpmath.libmp import dps_to_prec, normalize, round_nearest

#: Extra working bits above the target precision for the first attempt.
_GUARD_BITS = 64
#: Attempts (doubling the guard bits each time) before falling back to evalf.
_MAX_ATTEMPTS = 4


class Unsupported(Exception):
    """The expression contains a node the arb converter does not handle."""


def _to_arb(expr: sp.Expr):
    """Evaluate a numeric sympy expression with arb at the current ``ctx.prec``."""
    from flint import arb, fmpq, fmpz

    if expr.is_Integer:
        return arb(fmpz(int(expr)))
    if expr.is_Rational:
        return arb(fmpq(int(expr.p), int(expr.q)))
    if expr.is_Float:
        sign, man, exp, _ = expr._mpf_
        if man == 0:
            return arb(0)
        value = arb(fmpz(int(man))) * arb(2) ** int(exp)     # exact
        return -value if sign else value
    if expr is sp.pi:
        return arb.pi()
    if expr is sp.E:
        return arb.const_e()
    if expr is sp.EulerGamma:
        return arb.const_euler()
    if expr is sp.Catalan:
        return arb.const_catalan()
    if expr is sp.GoldenRatio:
        return (1 + arb(5).sqrt()) / 2

    if expr.is_Add:
        result = arb(0)
        for arg in expr.args:
            result += _to_arb(arg)
        return result
    if expr.is_Mul:
        result = arb(1)
        for arg in expr.args:
            result *= _to_arb(arg)
        return result
    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Integer:
            return _to_arb(base) ** int(exponent)
        b = _to_arb(base)
        if exponent.is_Rational and b > 0:
            return b ** _to_arb(exponent)
        raise Unsupported(f"power {expr}")

    func = expr.func
    args = expr.args
    if func is sp.exp:
        return _to_arb(args[0]).exp()
    if func is sp.log and len(args) == 1:
        x = _to_arb(args[0])
        if not x > 0:
            raise Unsupported(f"log of a non-positive value {expr}")
        return x.log()
    if func is sp.zeta and len(args) == 1:
        return _to_arb(args[0]).zeta()
    if func is sp.Ei:
        return _to_arb(args[0]).ei()
    if func is sp.polylog:
        return _to_arb(args[1]).polylog(_to_arb(args[0]))
    if func is sp.gamma:
        return _to_arb(args[0]).gamma()
    if func is sp.Abs:
        return abs(_to_arb(args[0]))
    if func is sp.sin:
        return _to_arb(args[0]).sin()
    if func is sp.cos:
        return _to_arb(args[0]).cos()
    if func is sp.atan:
        return _to_arb(args[0]).atan()
    if func is sp.hyper:
        ap, bq, z = args
        return _to_arb(z).hypgeom([_to_arb(a) for a in ap], [_to_arb(b) for b in bq])
    raise Unsupported(f"{func.__name__} in {expr}")


def _ball_to_float(ball, prec: int) -> sp.Float:
    """The ball's midpoint, rounded to *prec* bits, as a sympy ``Float``."""
    man, exp = ball.mid().man_exp()
    man, exp = int(man), int(exp)
    if man == 0:
        return sp.Float._new((0, 0, 0, 0), prec)
    mpf = normalize(int(man < 0), abs(man), exp, abs(man).bit_length(), prec, round_nearest)
    return sp.Float._new(mpf, prec)


def arb_value(expr: sp.Expr, digits: int) -> sp.Float:
    """Evaluate *expr* to *digits* decimal digits with arb (no fallback).

    :raises Unsupported: the expression cannot be evaluated with arb, or the
        ball did not reach the requested accuracy (e.g. the value is 0).
    """
    from flint import ctx

    target = dps_to_prec(int(digits))
    expr = sp.sympify(expr)
    old_prec = ctx.prec
    try:
        guard = _GUARD_BITS
        for _ in range(_MAX_ATTEMPTS):
            ctx.prec = target + guard
            ball = _to_arb(expr)
            if ball.is_finite() and ball.rel_accuracy_bits() >= target + 4:
                return _ball_to_float(ball, target)
            guard *= 2
    finally:
        ctx.prec = old_prec
    raise Unsupported(f"arb could not reach {digits} digits for {expr}")


@lru_cache(maxsize=16)
def high_precision(expr: sp.Expr, digits: int) -> sp.Float:
    """``expr.evalf(digits)``, computed with arb whenever possible.

    Same result type and binary precision as ``expr.evalf(digits)``; values
    agree to the requested precision (verified against sympy at 80 000 digits
    for every ready-made constant).  Cached per process (16 entries, a few
    tens of KB each at 10⁵ digits).
    """
    try:
        return arb_value(expr, digits)
    except Unsupported:
        return sp.sympify(expr).evalf(int(digits))


#: An error |estimate − C| is trusted only when it exceeds the constant's own
#: rounding (1 ulp of C) by at least this many bits, so δ is accurate to ~12 digits.
ERROR_MARGIN_BITS = 40


def error_is_resolved(err: sp.Expr, reference: sp.Float, margin_bits: int = ERROR_MARGIN_BITS) -> bool:
    """Whether ``err = |estimate − reference|`` is a real error, not rounding noise.

    *reference* is the constant rounded to ``reference._prec`` bits, so it is
    within about 1 ulp of the true value (sympy's ``evalf`` rounds twice and can
    be 1 ulp off; arb is correctly rounded).  An ``err`` within a few ulps of
    that — in particular exactly 0 — says nothing about the estimate: the
    estimate agrees with the constant *at least* to the reference's precision.
    Requiring ``err >= 2**margin_bits`` ulps makes the relative uncertainty of
    ``err`` at most ~``2**(1 - margin_bits)``.
    """
    e = sp.Float(err, reference._prec)._mpf_
    if e[1] == 0:                                    # exactly zero (Float(0) != 0 in sympy)
        return False
    r = reference._mpf_
    if r[1] == 0:
        return True                                  # reference 0: any nonzero error is real
    log2_err = e[2] + e[3]                           # exponent + bit count ~ floor(log2 |err|) + 1
    log2_ulp = r[2] + r[3] - reference._prec         # log2 of one ulp of the reference
    return log2_err - log2_ulp >= margin_bits


def resolved_error(estimate: sp.Expr, constant: sp.Expr, digits: int, max_digits: int):
    """``|estimate − constant|`` at the first precision (doubling from *digits*,
    capped at *max_digits*) where it is a real error (:func:`error_is_resolved`).

    :return: ``(err, digits_used, reference)``; ``err`` is ``None`` when even
        *max_digits* digits cannot resolve it (the estimate agrees with the
        constant beyond *max_digits* digits).
    """
    digits = int(min(digits, max_digits))
    while True:
        reference = high_precision(constant, digits)
        err = sp.Abs(estimate - reference)
        if error_is_resolved(err, reference):
            return err, digits, reference
        if digits >= max_digits:
            return None, digits, reference
        digits = int(min(digits * 2, max_digits))
