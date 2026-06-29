"""
Tests for the δ-vs-depth graphing script (``graphs/delta_vs_depth.py``).

Scope is the pure input-parsing / grid layer (the δ / kamidelta computation
itself exercises the production attribute handler + LIReC and is covered by the
attribute-management tests).  The Agg backend is forced so no display is needed.
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
