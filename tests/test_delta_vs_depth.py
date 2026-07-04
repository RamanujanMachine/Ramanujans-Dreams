"""
Tests for the δ-vs-depth graphing script (``graphs/delta_vs_depth.py``).

Scope is the pure input-parsing / grid / tail-error-bound layer (the δ /
kamidelta computation itself exercises the production attribute handler + LIReC
and is covered by the attribute-management tests; the error-bound statistics are
tested here against a fake handler).  The Agg backend is forced so no display is
needed.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "graphs" / "delta_vs_depth.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("delta_vs_depth", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dvd = _load_module()


class TestParsePoint:
    def test_brackets_and_spaces(self):
        assert dvd.parse_point("(-3, 1, -1)") == (-3, 1, -1)
        assert dvd.parse_point("-3,1,-1") == (-3, 1, -1)
        assert dvd.parse_point("[2, 0, 0, 5]") == (2, 0, 0, 5)

    def test_integer_floats_ok(self):
        assert dvd.parse_point("(1.0, 2.0, -3.0)") == (1, 2, -3)

    def test_rationals(self):
        import sympy as sp
        out = dvd.parse_point("(0, 1, 3, -1/2, 5, 7, 7/2)")
        assert out == (0, 1, 3, sp.Rational(-1, 2), 5, 7, sp.Rational(7, 2))
        # the fractions are genuine sympy Rationals, not floats
        assert out[3] == sp.Rational(-1, 2) and out[3].q == 2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            dvd.parse_point("(abc, 2, 3)")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            dvd.parse_point("()")


class TestParseTrajectories:
    def test_list(self):
        out = dvd.parse_trajectories(["(-1, 3, 0)", "(-1, 2, -1)"])
        assert out == [(-1, 3, 0), (-1, 2, -1)]


class TestDepthGrid:
    def test_sorted_unique_within_bounds(self):
        g = dvd.depth_grid(200, 20)
        assert g[0] >= 2 and g[-1] == 200
        assert g == sorted(g)
        assert len(g) == len(set(g))

    def test_small_max_depth_dedupes(self):
        # Few distinct integers between 2 and 5 — must dedupe, stay sorted.
        g = dvd.depth_grid(5, 20)
        assert g == sorted(set(g))
        assert g[0] >= 2 and g[-1] == 5

    def test_max_depth_too_small_raises(self):
        with pytest.raises(ValueError):
            dvd.depth_grid(1, 10)


class TestResolveConstant:
    def test_registered_name(self):
        c = dvd._resolve_constant("pi", None)
        assert c.name == "pi"

    def test_sympy_expression(self):
        import sympy as sp
        c = dvd._resolve_constant("zeta(2)", None)
        assert c.name == "zeta(2)"
        assert sp.simplify(c.value_sympy - sp.pi**2 / 6) == 0

    def test_log2_shortcut(self):
        import sympy as sp
        c = dvd._resolve_constant("log-2", None)
        assert sp.simplify(c.value_sympy - sp.log(2)) == 0

    def test_free_symbol_rejected(self):
        with pytest.raises(SystemExit):
            dvd._resolve_constant("not_a_constant_xyz", None)


class TestBuildCmfFromExpr:
    def test_pfq_builds(self):
        cd = dvd.build_cmf_from_expr("pFq(2, 1, -1)", use_inv_t=False)
        assert len(cd.cmf.matrices) == 3  # p + q
        assert cd.use_inv_t is False
        assert "pFq(2, 1, -1)" == cd.cmf_name
        # All shift coordinates default to 0.
        assert all(int(v) == 0 for v in cd.shift.values())

    def test_use_inv_t_flag(self):
        cd = dvd.build_cmf_from_expr("pFq(2, 1, 1)", use_inv_t=True)
        assert cd.use_inv_t is True

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            dvd.build_cmf_from_expr("pFq 2 1 -1", use_inv_t=False)

    def test_wrong_argcount_raises(self):
        with pytest.raises(ValueError):
            dvd.build_cmf_from_expr("pFq(2, 1)", use_inv_t=False)

    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError):
            dvd.build_cmf_from_expr("MeijerG(1, 2)", use_inv_t=False)


class TestFloatHelpers:
    def test_as_float(self):
        assert dvd._as_float(0.5) == 0.5
        assert np.isnan(dvd._as_float("not a number"))
        assert dvd._as_float(float("-inf")) == float("-inf")

    def test_finite_or_nan(self):
        assert dvd._finite_or_nan(1.2) == 1.2
        assert np.isnan(dvd._finite_or_nan(float("-inf")))
        assert np.isnan(dvd._finite_or_nan(float("nan")))


class TestFastKamidelta:
    def test_nan_actual_short_circuits(self):
        # When δ is not identified (NaN), the fast path returns all-NaN without
        # touching the handler (so a None handler is safe here).
        out = dvd._fast_kamidelta(None, [2, 5, 10], float("nan"))
        assert out.shape == (3,)
        assert np.all(np.isnan(out))


class _FakeHandler:
    """Minimal handler exposing ``delta_sequence`` / ``delta`` for the tail
    error-bound layer — returns a preset δ per depth via a callable."""

    def __init__(self, fn):
        self._fn = fn

    def delta_sequence(self, depths):
        return [self._fn(d) for d in depths]

    def delta(self, d):
        return self._fn(d)


class TestComputeErrorBound:
    def test_matches_numpy_on_converged_tail(self):
        # A flat tail with small zero-mean noise: mean/std/sem must equal the
        # numpy reference over exactly the last-`window` integer depths.
        rng = np.random.default_rng(0)
        noise = {d: float(rng.normal(0, 0.01)) for d in range(1, 501)}
        h = _FakeHandler(lambda d: 0.30 + noise[d])
        eb = dvd.compute_error_bound(h, 500, 100)
        tail = np.array([0.30 + noise[d] for d in range(401, 501)])
        assert eb["n"] == 100
        assert eb["depth_lo"] == 401 and eb["depth_hi"] == 500
        assert eb["mean"] == pytest.approx(float(tail.mean()))
        assert eb["std"] == pytest.approx(float(tail.std(ddof=1)))
        assert eb["sem"] == pytest.approx(float(tail.std(ddof=1)) / np.sqrt(100))
        assert eb["converged"] is True

    def test_trending_tail_flagged_not_converged(self):
        # A strong linear drift makes total drift exceed the scatter → the
        # bound is reported but marked not-converged.
        rng = np.random.default_rng(1)
        h = _FakeHandler(lambda d: 0.30 + 0.002 * d + float(rng.normal(0, 0.001)))
        eb = dvd.compute_error_bound(h, 500, 100)
        assert eb["drift"] > eb["std"]
        assert eb["converged"] is False

    def test_window_below_two_disables(self):
        h = _FakeHandler(lambda d: 0.3)
        assert dvd.compute_error_bound(h, 500, 0) is None
        assert dvd.compute_error_bound(h, 500, 1) is None

    def test_nonfinite_deltas_filtered(self):
        # -inf / NaN tail values are dropped; n counts only the finite ones.
        h = _FakeHandler(lambda d: float("-inf") if d % 2 == 0 else 0.3)
        eb = dvd.compute_error_bound(h, 500, 100)
        assert eb["n"] == sum(1 for d in range(401, 501) if d % 2 == 1)

    def test_clamps_low_depth_to_two(self):
        # window larger than max_depth starts the tail block at depth 2, not 1.
        h = _FakeHandler(lambda d: 0.3)
        eb = dvd.compute_error_bound(h, 10, 100)
        assert eb["depth_lo"] == 2 and eb["depth_hi"] == 10


def _eb(mean, std, converged=True):
    """A minimal error-bound dict for summary / render tests."""
    return {"mean": mean, "std": std, "sem": std / 10.0, "slope": 0.0,
            "drift": 0.0 if converged else 10 * std, "last": mean, "n": 100,
            "depth_lo": 3901, "depth_hi": 4000, "converged": converged}


class TestErrorSummaryLines:
    def test_one_line_per_traj_and_kind(self):
        # δ (index 3) + kamiδ (index 4) both present → two lines for one traj.
        results = [((1, 2), None, None, _eb(0.3, 0.01), _eb(0.29, 0.02))]
        lines = dvd._error_summary_lines(results, [dvd._KIND_DELTA, dvd._KIND_KAMI])
        assert len(lines) == 2
        assert "δ" in lines[0] and "kamiδ" in lines[1]
        assert "SEM" in lines[0] and "depths 3901–4000" in lines[0]

    def test_missing_bound_skipped(self):
        # kamiδ bound absent → only the δ line is emitted.
        results = [((1, 2), None, None, _eb(0.3, 0.01), None)]
        lines = dvd._error_summary_lines(results, [dvd._KIND_DELTA, dvd._KIND_KAMI])
        assert len(lines) == 1 and "kamiδ" not in lines[0]

    def test_not_converged_flagged(self):
        results = [((1, 2), None, None, _eb(0.3, 0.01, converged=False), None)]
        lines = dvd._error_summary_lines(results, [dvd._KIND_DELTA])
        assert "not converged" in lines[0]


class TestPlotCurvesRenders:
    def _results(self):
        depths = dvd.depth_grid(500, 8, 2)
        delta = np.linspace(0.30, 0.31, len(depths))
        kami = delta - 0.05
        results = [((11, 13, 2), delta, kami, _eb(0.305, 0.01), _eb(0.255, 0.01))]
        return depths, results

    def test_overlay_has_footer_legend_and_text(self):
        depths, results = self._results()
        fig = dvd.plot_curves(depths, results, constant_name="zeta(2)",
                              title="t", show_error_bound=True)
        # A figure-level legend exists (the footer legend) ...
        assert fig.legends, "expected a figure-level footer legend"
        # ... and a footer text block carrying the error description.
        texts = [t.get_text() for t in fig.texts]
        assert any("SEM" in t and "±" in t for t in texts)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_separate_mode_two_axes(self):
        depths, results = self._results()
        fig = dvd.plot_curves(depths, results, constant_name="zeta(2)",
                              separate=True, show_error_bound=True)
        # δ axis + kamidelta axis + colorbar axis.
        assert len(fig.axes) >= 2
        assert fig.legends
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_disabled_error_bound_no_footer_text(self):
        depths, results = self._results()
        fig = dvd.plot_curves(depths, results, constant_name="zeta(2)",
                              show_error_bound=False)
        texts = [t.get_text() for t in fig.texts]
        assert not any("SEM" in t for t in texts)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestFastKamideltaMatchesProduction:
    """The graph's fast kamidelta (single-walk prefix fits) must match the
    production ``delta_prediction`` (per-depth slow path) — same eigenvalue-pair
    selection and the same δ-consistent ``gcd_slope`` walk.  Uses a real handler
    (LIReC + walk) on a small, quick-to-identify log(2) trajectory."""

    def test_fast_matches_slow(self):
        import sympy as sp
        from ramanujantools import Position
        from ramanujantools.cmf import pFq
        from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

        cmf = pFq(2, 1, -1)
        syms = list(cmf.matrices.keys())
        start = Position({s: sp.Rational(v) for s, v in zip(syms, (-1, 2, 1))})
        direction = Position({s: sp.Rational(v) for s, v in zip(syms, (-4, 8, 5))})
        h = TrajectoryAttributesHandler.from_cmf(
            cmf, direction, start, sp.log(2), walk_depth=280, walk_type=1)

        depths = [200, 240, 280]
        fast = dvd._fast_kamidelta(h, depths, h.delta(280))
        assert fast is not None
        for d, kf in zip(depths, fast):
            slow = h.delta_prediction(d)
            assert slow is not None
            assert abs(float(kf) - float(slow["predicted_delta"])) < 1e-6
