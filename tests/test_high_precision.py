"""Tests for the arb-based high-precision constant evaluator.

``high_precision(expr, digits)`` replaces ``expr.evalf(digits)`` in the δ code
(mpmath needed GBs for 10⁵-digit constants).  These tests pin that it:

  - agrees with sympy to the requested precision for every ready-made constant
    (3 000 digits here; 80 000 digits verified once by
    ``scratchpad/verify80k.py``, see roadmap 2026-10-11);
  - returns a ``Float`` with exactly evalf's binary precision;
  - falls back to ``evalf`` for unsupported expressions and zero values;
  - leaves δ unchanged on a real trajectory, including the precision-escalation
    path;
  - (regression) a handler switching constants does not reuse the previous
    constant's high-precision value.
"""

import math

import pytest
import sympy as sp

from dreamer.utils.constants import ready_made_consts as R
from dreamer.utils.constants.high_precision import Unsupported, arb_value, high_precision

DIGITS = 3000

CONSTANTS = {
    "e": R.e.value_sympy,
    "pi": R.pi.value_sympy,
    "euler_gamma": R.euler_gamma.value_sympy,
    "pi_squared": R.pi_squared.value_sympy,
    "catalan": R.catalan.value_sympy,
    "gompertz": R.gompertz.value_sympy,
    "calegary": R.calegary.value_sympy,
    "zeta2": R.zeta(2).value_sympy,
    "zeta3": R.zeta(3).value_sympy,
    "zeta5": R.zeta(5).value_sympy,
    "log2": R.log(2).value_sympy,
    "sqrt2": R.sqrt(2).value_sympy,
    "pi_cubed": R.power(R.pi, 3).value_sympy,
    "golden": sp.GoldenRatio,
    "mixed": 3 * sp.zeta(3) / 2 - sp.pi ** 2 / 7 + sp.log(3) * sp.sqrt(5),
    "polylog": sp.polylog(3, sp.Rational(1, 2)),
    "gamma_quarter": sp.gamma(sp.Rational(1, 4)),
    "trig": sp.atan(sp.Rational(1, 3)) + sp.cos(1) - sp.sin(2),
    "negative": -sp.exp(2) + sp.Rational(1, 7),
}


def agreeing_digits(a: sp.Float, b: sp.Float) -> float:
    """Relative agreement of two Floats in decimal digits (inf if identical)."""
    diff = abs(a - b)
    if diff == 0:
        return math.inf
    return float(-sp.log(diff / abs(b), 10).evalf(20))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", sorted(CONSTANTS))
def test_agrees_with_sympy(name):
    expr = CONSTANTS[name]
    ours = arb_value(expr, DIGITS)            # arb only — no silent fallback
    reference = expr.evalf(DIGITS + 30)
    assert agreeing_digits(ours, reference) >= DIGITS
    assert ours._prec == expr.evalf(DIGITS)._prec


def test_float_input_is_exact():
    x = sp.Float("1.25", 50)
    assert arb_value(x * sp.pi, 200) == (sp.Rational(5, 4) * sp.pi).evalf(200)


@pytest.mark.parametrize("expr", [sp.besselj(0, 1), sp.I + 1, sp.log(-2), (-2) ** sp.Rational(1, 3)])
def test_unsupported_falls_back_to_evalf(expr):
    with pytest.raises(Unsupported):
        arb_value(expr, 50)
    high_precision.cache_clear()
    assert high_precision(expr, 50) == expr.evalf(50)


def test_zero_value_falls_back():
    expr = sp.pi - sp.pi * 1        # sympy folds to 0; also try a non-folded zero
    assert high_precision(sp.Integer(0), 30) == 0
    zero = sp.Add(sp.zeta(2), -sp.pi ** 2 / 6, evaluate=False)
    with pytest.raises(Unsupported):
        arb_value(zero, 30)         # a ball around 0 has no relative accuracy
    assert expr == 0


def test_cached_and_precision_restored():
    from flint import ctx

    before = ctx.prec
    high_precision.cache_clear()
    a = high_precision(sp.zeta(3), 500)
    b = high_precision(sp.zeta(3), 500)
    assert a is b
    assert ctx.prec == before


# ---------------------------------------------------------------------------
# δ is unchanged on real trajectories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def log2_shard():
    """2F1(z=-1) log(2) shard with interior (1, 1, 3) — identifies on most rays."""
    from dreamer.configs import config
    from dreamer.configs.system import sys_config
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq

    const = R.log(2)
    old = (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
           config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS)
    config.search.DEFAULT_USES_INV_T = False
    config.extraction.STRATEGY = "heuristic"
    config.extraction.LOAD_SHARD_CACHE = False
    sys_config.EXPORT_CMFS = None
    try:
        shards = ShardExtractor([const], pFq(const, 2, 1, -1).to_cmf()).extract()
    finally:
        (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
         config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS) = old
    for sh in shards:
        if tuple(int(v) for v in sh.get_interior_point().values()) == (1, 1, 3):
            return sh
    raise AssertionError("log(2) shard (1,1,3) not found")


def _handler(shard, direction=(2, 1, 3)):
    from ramanujantools import Position
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

    syms = list(shard.cmf.matrices.keys())
    return TrajectoryAttributesHandler.from_cmf(
        shard.cmf, Position(dict(zip(syms, direction))), shard.get_interior_point(), searchable=shard,
    )


@pytest.fixture(autouse=True)
def _walk_type(monkeypatch):
    from dreamer.configs import config
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)


def _delta_with(monkeypatch, shard, evaluator, high_res_digits, constant=None):
    from dreamer.configs import config
    from dreamer.utils.storage import trajectory_attributes as ta

    monkeypatch.setattr(ta, "high_precision", evaluator)
    monkeypatch.setattr(config.search, "CONSTANT_NO_DIGITS_HIGH_RES", high_res_digits)
    h = _handler(shard)
    h._constant = constant if constant is not None else R.log(2).value_sympy
    return h.delta(), h._constant_resolution


_evalf = lambda expr, digits: sp.sympify(expr).evalf(int(digits))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("high_res_digits", [10_000, 20])
def test_delta_identical_to_evalf(monkeypatch, log2_shard, high_res_digits):
    """Default precision, and a tiny start precision that forces the
    escalation loop (doubling) — δ and the final precision match evalf's."""
    ref, ref_res = _delta_with(monkeypatch, log2_shard, _evalf, high_res_digits)
    high_precision.cache_clear()
    ours, our_res = _delta_with(monkeypatch, log2_shard, high_precision, high_res_digits)
    assert math.isfinite(ref)
    if high_res_digits == 20:
        assert ref_res > 20                      # the escalation really happened
    assert ours == ref and our_res == ref_res


@pytest.mark.timeout(300)
def test_constant_swap_resets_high_precision_value(log2_shard):
    """Regression: ``compute_for_constant`` with a second constant on the same
    handler used the first constant's high-precision value for δ."""
    c1 = R.log(2).value_sympy
    c2 = 2 * R.log(2).value_sympy            # same p/q family → also identified
    shared = _handler(log2_shard)
    shared.compute_for_constant(c1)
    swapped = shared.compute_for_constant(c2)[0]
    fresh = _handler(log2_shard).compute_for_constant(c2)[0]
    assert math.isfinite(fresh)
    assert swapped == fresh
