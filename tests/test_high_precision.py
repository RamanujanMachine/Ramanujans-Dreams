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


# ---------------------------------------------------------------------------
# δ precision escalation: rounding noise is never taken as the error
# ---------------------------------------------------------------------------

from dreamer.utils.constants.high_precision import error_is_resolved, resolved_error


def _true_error_digits(estimate, constant, digits=5000):
    """-log10 |estimate - constant| with the constant far beyond the agreement."""
    return float(-sp.log(abs(estimate - arb_value(constant, digits)), 10).evalf(15))


class TestErrorResolution:
    def test_zero_and_ulp_level_errors_are_noise(self):
        c = high_precision(sp.zeta(3), 100)
        ulp = sp.Float(2, c._prec) ** (c._mpf_[2])          # one unit in the last place
        assert not error_is_resolved(sp.Float(0), c)
        assert not error_is_resolved(abs(ulp), c)
        assert not error_is_resolved(abs(ulp) * 2 ** 30, c)
        assert error_is_resolved(abs(ulp) * 2 ** 45, c)

    def test_escalates_past_the_agreement(self):
        """An estimate agreeing with ζ(3) to ~300 digits: starting at 50 digits
        the resolution must climb past 300, and the error must be the true one."""
        estimate = sp.Rational(sp.zeta(3).evalf(300)) + sp.Rational(1, 10 ** 300)
        err, digits, ref = resolved_error(estimate, sp.zeta(3), 50, 10_000)
        assert digits > 300
        true_digits = _true_error_digits(estimate, sp.zeta(3))
        assert abs(float(-sp.log(err, 10).evalf(15)) - true_digits) < 1e-9

    def test_gives_up_at_max(self):
        estimate = sp.Rational(sp.zeta(3).evalf(2000))
        err, digits, _ = resolved_error(estimate, sp.zeta(3), 50, 400)
        assert err is None and digits == 400

    def test_old_rule_was_fooled_by_one_ulp(self):
        """The case seen on 4F3: sympy's evalf rounds twice and can be 1 ulp off;
        the old rule (escalate only on err == 0) accepted that ulp as the error."""
        c = sp.zeta(3).evalf(200)
        ulp = sp.Float(2, c._prec) ** (c._mpf_[2])
        estimate = sp.Rational(c) + sp.Rational(abs(ulp))   # agrees to ~200 digits
        old_err = sp.Abs(estimate - c)
        assert old_err != 0                                  # old rule: "resolved"
        err, digits, _ = resolved_error(estimate, sp.zeta(3), 200, 10_000)
        assert digits > 200 and err is not None


@pytest.mark.timeout(300)
@pytest.mark.parametrize("high_res_digits", [10_000, 20])
def test_delta_equals_true_delta(monkeypatch, log2_shard, high_res_digits):
    """δ from the handler equals δ computed from the true error (constant to
    far more digits than the estimate agrees with)."""
    from dreamer.configs import config

    monkeypatch.setattr(config.search, "CONSTANT_NO_DIGITS_HIGH_RES", high_res_digits)
    h = _handler(log2_shard)
    h._constant = R.log(2).value_sympy
    delta = h.delta()
    (p, q), _ = h._pq_vector()
    col = sp.Matrix(h._effective_walk_values(h.walk_depth())[0])
    estimate = sp.Abs(sp.Rational(sp.Matrix(p).T.dot(col), sp.Matrix(q).T.dot(col)))
    digits = _true_error_digits(estimate, R.log(2).value_sympy, 20_000)
    true_delta = -1 + digits / float(sp.log(estimate.q, 10).evalf(30))
    assert delta == pytest.approx(true_delta, rel=1e-9)


# ---------------------------------------------------------------------------
# Guardrails, precision levels, remaining node types
# ---------------------------------------------------------------------------

import mpmath


class TestGuardrails:
    @pytest.mark.parametrize("bad", [0, -5, 2.5, True, "10", None])
    def test_digits_must_be_positive_int(self, bad):
        """Public entry points reject non-positive / non-integer precisions."""
        high_precision.cache_clear()
        with pytest.raises(ValueError):
            high_precision(sp.pi, bad)
        with pytest.raises(ValueError):
            arb_value(sp.pi, bad)
        with pytest.raises(ValueError):
            resolved_error(sp.Rational(22, 7), sp.pi, bad, 100)
        with pytest.raises(ValueError):
            resolved_error(sp.Rational(22, 7), sp.pi, 10, bad)

    def test_one_digit_is_a_valid_boundary(self):
        """digits=1 is the smallest precision: still the evalf precision."""
        assert arb_value(sp.pi, 1)._prec == sp.pi.evalf(1)._prec

    def test_start_above_max_is_clamped(self):
        """A start precision above the cap is clamped to the cap (no escalation)."""
        err, digits, ref = resolved_error(sp.Rational(22, 7), sp.pi, 500, 100)
        assert digits == 100 and err is not None


class TestKnownAnswersAtSeveralPrecisions:
    """Known answers vs mpmath at 50 and 200 digits (reference computed with
    30 extra digits), per the coverage policy's multi-precision rule."""

    @pytest.mark.parametrize("digits", [50, 200])
    @pytest.mark.parametrize("expr, reference", [
        (sp.zeta(3), lambda: mpmath.zeta(3)),
        (sp.pi, lambda: mpmath.pi),
        (sp.Catalan, lambda: mpmath.catalan),
        (sp.EulerGamma, lambda: mpmath.euler),
        (-sp.E * sp.Ei(-1), lambda: -mpmath.e * mpmath.ei(-1)),
    ])
    def test_matches_mpmath(self, digits, expr, reference):
        with mpmath.workdps(digits + 30):
            ref = reference()
            ours = mpmath.mpf(arb_value(expr, digits)._mpf_)
            assert abs(ours - ref) <= abs(ref) * mpmath.mpf(10) ** (-digits)

    def test_cancellation_needs_more_guard_bits(self):
        """exp(π√163) − 640320³ − 744 ≈ −7.5·10⁻¹³ cancels ~30 digits (> the
        64 guard bits of the first attempt): the retry loop must still deliver
        the full 100 digits."""
        expr = sp.Add(sp.exp(sp.pi * sp.sqrt(163)), -640320 ** 3 - 744, evaluate=False)
        with mpmath.workdps(160):
            ref = mpmath.exp(mpmath.pi * mpmath.sqrt(163)) - 640320 ** 3 - 744
        ours = mpmath.mpf(arb_value(expr, 100)._mpf_)
        assert ref < 0 and abs(ours - ref) <= abs(ref) * mpmath.mpf(10) ** -100


class TestNodeCoverage:
    def test_float_zero_and_unevaluated_abs(self):
        """Exact zero Float input, and an unevaluated Abs of a negative value."""
        expr = sp.Add(sp.Float(0), sp.pi, evaluate=False)
        assert arb_value(expr, 60) == sp.pi.evalf(60)
        absval = sp.Abs(sp.Add(sp.Integer(3), -sp.pi, evaluate=False), evaluate=False)
        assert agreeing_digits(arb_value(absval, 80), (sp.pi - 3).evalf(110)) >= 80

    def test_log_of_non_positive_is_unsupported(self):
        """log of a value ≤ 0 is not real: Unsupported (→ evalf fallback)."""
        with pytest.raises(Unsupported):
            arb_value(sp.log(sp.Add(sp.Integer(3), -sp.pi, evaluate=False), evaluate=False), 30)

    def test_zero_reference_counts_any_nonzero_error(self):
        """With a zero reference, any nonzero error is real (no ulp scale)."""
        zero = sp.Float(0, 100)
        assert error_is_resolved(sp.Float("1e-50", 100), zero)
        assert not error_is_resolved(sp.Float(0, 100), zero)


# ---------------------------------------------------------------------------
# The legacy δ path (Searchable.calc_delta) uses the same rule
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_legacy_calc_delta_matches_handler(monkeypatch, log2_shard):
    """``Searchable.calc_delta`` (DataManager path) and the handler compute the
    same δ for the same trajectory, walk type and depth."""
    import numpy as np
    from ramanujantools import Position
    from dreamer.configs import config

    syms = list(log2_shard.cmf.matrices.keys())
    traj = Position(dict(zip(syms, (2, 1, 3))))
    start = log2_shard.get_interior_point()
    monkeypatch.setattr(log2_shard, "use_inv_t", False)
    traj_len = float(np.sqrt(sum(float(v) ** 2 for v in traj.values())))
    m = log2_shard.cmf.trajectory_matrix(traj, start)
    legacy, pq, _ = log2_shard.calc_delta(m, traj, start, R.log(2).value_sympy, traj_len)

    depth = config.search.DEPTH_FROM_TRAJECTORY_LEN(traj_len, log2_shard.dim)
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
    h = TrajectoryAttributesHandler(m, R.log(2).value_sympy, walk_depth=depth, walk_type=2,
                                    searchable=log2_shard)
    assert legacy is not None and pq is not None
    assert legacy == pytest.approx(h.delta(), rel=1e-9)


@pytest.mark.timeout(300)
def test_legacy_calc_delta_gives_up_beyond_max(monkeypatch, log2_shard):
    """If the estimate agrees with the constant beyond MAX_CONSTANT_RESOLUTION,
    the legacy path skips the trajectory instead of returning a noise δ."""
    import numpy as np
    from ramanujantools import Position
    from dreamer.configs import config

    monkeypatch.setattr(config.search, "CONSTANT_NO_DIGITS_HIGH_RES", 5)
    monkeypatch.setattr(config.search, "MAX_CONSTANT_RESOLUTION", 10)
    monkeypatch.setattr(log2_shard, "use_inv_t", False)
    syms = list(log2_shard.cmf.matrices.keys())
    traj = Position(dict(zip(syms, (2, 1, 3))))
    start = log2_shard.get_interior_point()
    traj_len = float(np.sqrt(sum(float(v) ** 2 for v in traj.values())))
    m = log2_shard.cmf.trajectory_matrix(traj, start)
    assert log2_shard.calc_delta(m, traj, start, R.log(2).value_sympy, traj_len) == (None, None, None)
