"""
Tests for the JSONL δ-on-sphere graphing script
(``graphs/shard_delta_sphere_jsonl.py``).

Scope is the pure data/logic layer (no matplotlib display required — the
Agg backend is forced):

  - ProjectionSpec: identity guard, from_layout (valid + invalid), free_to_full
    embedding, project (subspace filtering + unit normalisation).
  - Value extractors: delta_value, field_value (+transform), extended_metric_value
    (missing → None), sympy_attribute_value (eval / missing / complex-skip /
    top-level container).
  - CLI parsers: _parse_subs, _parse_layout, _resolve_value_fn.
  - JSONL loaders against a temp shard file: load_shard_trajectories,
    shard_has_identified, load_path_directions.
  - Colorbar norm helpers: _nice_step, _make_norm (incl. constant-field widening).
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # no display needed

import numpy as np
import pytest
import sympy as sp

from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Import the script as a module.  It uses ``from __future__ import annotations``
# + dataclasses, so it must be registered in sys.modules *before* exec for the
# dataclass field-type resolution to succeed.
# ---------------------------------------------------------------------------

_MOD_PATH = Path(__file__).resolve().parents[1] / "graphs" / "shard_delta_sphere_jsonl.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("shard_delta_sphere_jsonl", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sds = _load_module()


@pytest.fixture(scope="module")
def const():
    """A registered log(2) constant matching the example data's key."""
    return Constant.registry.get("log-2") or Constant("log-2", sp.log(2))


# ===========================================================================
# ProjectionSpec
# ===========================================================================

class TestProjectionSpec:
    def test_identity_requires_three_dims(self):
        assert sds.ProjectionSpec.identity(3).axes == (0, 1, 2)
        with pytest.raises(ValueError):
            sds.ProjectionSpec.identity(2)

    def test_from_layout_valid(self):
        spec = sds.ProjectionSpec.from_layout(["x", "y", "z", (1, -1, 0)])
        assert spec.axes == (0, 1, 2)
        assert spec.dependent == {3: (1.0, -1.0, 0.0)}

    def test_from_layout_reordered_free_axes(self):
        # (f, x, y, z) variant: dependent coordinate first.
        spec = sds.ProjectionSpec.from_layout([(2, 0, 0), "x", "y", "z"])
        assert spec.axes == (1, 2, 3)
        assert spec.dependent == {0: (2.0, 0.0, 0.0)}

    def test_from_layout_missing_axis_raises(self):
        with pytest.raises(ValueError):
            sds.ProjectionSpec.from_layout(["x", "x", "z"])  # no y
        with pytest.raises(ValueError):
            sds.ProjectionSpec.from_layout(["x", "y", "z", (1, -1)])  # bad coeffs

    def test_ignoring_picks_remaining_axes(self):
        spec = sds.ProjectionSpec.ignoring(5, [0, 4])
        assert spec.axes == (1, 2, 3)
        assert spec.dependent == {} and spec.dim == 5

    def test_ignoring_drops_coords_in_projection(self):
        spec = sds.ProjectionSpec.ignoring(5, [0, 4])
        # direction (3, 2, 0, 4, 7): the ignored 3 and 7 must not affect the unit
        # vector — it is the normalisation of (dir[1], dir[2], dir[3]) = (2, 0, 4).
        dirs = np.array([[3.0, 2.0, 0.0, 4.0, 7.0]])
        unit, mask = spec.project(dirs, tol=1e-6)
        assert mask.all()
        expected = np.array([2.0, 0.0, 4.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(unit[0], expected)

    def test_ignoring_free_to_full_is_full_dim(self):
        spec = sds.ProjectionSpec.ignoring(5, [0, 4])
        full = spec.free_to_full(np.array([[0.6, 0.0, 0.8]]))
        assert full.shape == (1, 5)
        # ignored coords (0, 4) stay 0; kept coords carry x/y/z.
        np.testing.assert_allclose(full[0], [0.0, 0.6, 0.0, 0.8, 0.0])

    def test_ignoring_wrong_count_raises(self):
        with pytest.raises(ValueError):
            sds.ProjectionSpec.ignoring(5, [0])      # leaves 4
        with pytest.raises(ValueError):
            sds.ProjectionSpec.ignoring(5, [9])      # out of range

    def test_ignores_coords_flag(self):
        # ignoring drops coords → True (cone trim must be skipped in surface mode)
        assert sds.ProjectionSpec.ignoring(5, [0, 4]).ignores_coords is True
        # identity / full layout cover every coord → False (cone trim valid)
        assert sds.ProjectionSpec.identity(3).ignores_coords is False
        assert sds.ProjectionSpec.from_layout(["x", "y", "z", (1, 0, 0)]).ignores_coords is False

    def test_free_to_full_embedding(self):
        spec = sds.ProjectionSpec.from_layout(["x", "y", "z", (1, -1, 0)])
        xyz = np.array([[0.3, 0.4, 0.5]])
        full = spec.free_to_full(xyz)
        assert full.shape == (1, 4)
        np.testing.assert_allclose(full[0], [0.3, 0.4, 0.5, 0.3 - 0.4])

    def test_from_constraints_folds_dependents(self):
        syms = ["x0", "x1", "x2", "y0", "y1"]
        spec = sds.ProjectionSpec.from_constraints(5, syms, {"x0": 12, "x1": 14, "y1": 28})
        # anchor x0 kept as an axis with the two free coords; x1, y1 dependent on it.
        assert spec.axes == (0, 2, 3) and spec.dim == 5
        assert spec.dependent[1] == pytest.approx((14 / 12, 0.0, 0.0))
        assert spec.dependent[4] == pytest.approx((28 / 12, 0.0, 0.0))
        assert spec.ignores_coords is False  # 3 axes + 2 dependent = 5 = dim

    def test_from_constraints_wrong_count_raises(self):
        syms = ["x0", "x1", "x2", "y0", "y1"]
        with pytest.raises(ValueError):  # only 2 fixed → 4 effective axes
            sds.ProjectionSpec.from_constraints(5, syms, {"x0": 12, "y1": 28})

    def test_from_constraints_effective_normal_embedding(self):
        # The hyperplane great circles use A @ free_to_full(eye3).T; check the
        # embedding folds the dependent (ratio) coords onto the anchor axis.
        syms = ["x0", "x1", "x2", "y0", "y1"]
        spec = sds.ProjectionSpec.from_constraints(5, syms, {"x0": 12, "x1": 14, "y1": 28})
        emb = spec.free_to_full(np.eye(3))  # (3, 5)
        np.testing.assert_allclose(emb[0], [1.0, 14 / 12, 0.0, 0.0, 28 / 12])
        np.testing.assert_allclose(emb[1], [0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(emb[2], [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_project_filters_and_normalises(self):
        spec = sds.ProjectionSpec.from_layout(["x", "y", "z", (1, 0, 0)])
        # row 0 in-subspace (d3 == d0); row 1 out (d3 != d0).
        dirs = np.array([
            [3.0, 0.0, 4.0, 3.0],
            [3.0, 0.0, 4.0, 99.0],
        ])
        unit, mask = spec.project(dirs, tol=1e-6)
        assert mask.tolist() == [True, False]
        assert unit.shape == (1, 3)
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), [1.0])
        np.testing.assert_allclose(unit[0], [0.6, 0.0, 0.8])

    def test_project_identity_keeps_all(self):
        spec = sds.ProjectionSpec.identity(3)
        dirs = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        unit, mask = spec.project(dirs, tol=1e-6)
        assert mask.all()
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), [1.0, 1.0])


# ===========================================================================
# Value extractors
# ===========================================================================

class TestValueExtractors:
    def test_delta_value(self, const):
        fn = sds.delta_value(const)
        assert fn({"constant": const.name, "delta": 0.28}) == pytest.approx(0.28)
        assert fn({"constant": const.name}) is None       # no δ column
        assert fn({"constant": "other", "delta": 0.28}) is None  # different constant
        assert fn({}) is None

    def test_field_value(self):
        fn = sds.field_value("limit_value")
        assert fn({"limit_value": -2.43}) == pytest.approx(-2.43)
        assert fn({}) is None
        fn2 = sds.field_value("recurrence_order", transform=lambda v: v * 2)
        assert fn2({"recurrence_order": 3}) == 6

    def test_extended_metric_value(self):
        fn = sds.extended_metric_value("digits_per_step")
        assert fn({"digits_per_step": 1.5}) == pytest.approx(1.5)   # flat column
        assert fn({}) is None
        assert fn({"digits_per_step": None}) is None

    def test_sympy_attribute_value(self):
        fn = sds.sympy_attribute_value("asymptotics", {"n": 1e6})
        assert fn({"asymptotics": "3*n + 1"}) == pytest.approx(3e6 + 1)  # flat column
        assert fn({}) is None
        # Complex / non-real → skipped.
        assert fn({"asymptotics": "sqrt(-1)"}) is None
        # Non-string → skipped.
        assert fn({"asymptotics": 5}) is None

    def test_sympy_attribute_top_level(self):
        fn = sds.sympy_attribute_value("expr", {"n": 2}, in_extended_metrics=False)
        assert fn({"expr": "n**3"}) == pytest.approx(8.0)

    def test_eigenvalue_lognorm_value(self):
        fn0 = sds.eigenvalue_lognorm_value(0)
        fn1 = sds.eigenvalue_lognorm_value(1)
        rec = {"direction": [3, 4, 0],  # ||v|| = 5
               "eigenvalues": ["2.0", "0.5"]}
        assert fn0(rec) == pytest.approx(math.log(2.0) / 5.0)
        assert fn1(rec) == pytest.approx(math.log(0.5) / 5.0)
        # Negative / complex eigenvalue → log of magnitude (well-defined).
        rec_neg = {"direction": [1, 0, 0], "eigenvalues": ["-2"]}
        assert fn0(rec_neg) == pytest.approx(math.log(2.0))
        # Missing / too-short list, zero vector, or λ=0 → skipped.
        assert fn1({"direction": [1, 0, 0], "eigenvalues": ["2"]}) is None
        assert fn0({}) is None
        assert fn0({"direction": [0, 0, 0], "eigenvalues": ["2"]}) is None
        assert fn0({"direction": [1, 0, 0], "eigenvalues": ["0"]}) is None


# ===========================================================================
# CLI parsers
# ===========================================================================

class TestCliParsers:
    def test_parse_subs(self):
        assert sds._parse_subs("n=1e6,m=2") == {"n": 1e6, "m": 2.0}
        assert sds._parse_subs("") == {}
        assert sds._parse_subs(None) == {}

    def test_parse_layout(self):
        assert sds._parse_layout(None) is None
        spec = sds._parse_layout("x,y,z,1:-1:0")
        assert spec.axes == (0, 1, 2)
        assert spec.dependent == {3: (1.0, -1.0, 0.0)}

    @staticmethod
    def _args(**kw):
        base = dict(field=None, metric=None, sympy_attr=None, eigen_lognorm=None,
                    subs=None, value_label=None,
                    numeric_convergence_rate=None, neg_numeric_convergence_rate=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_resolve_value_fn_default_is_delta(self):
        fn, label = sds._resolve_value_fn(self._args())
        assert fn is None and label is None  # δ default

    def test_resolve_value_fn_field(self):
        fn, label = sds._resolve_value_fn(self._args(field="limit_value"))
        assert label == "limit_value"
        assert fn({"limit_value": 1.0}) == 1.0

    def test_resolve_value_fn_sympy(self):
        fn, label = sds._resolve_value_fn(
            self._args(sympy_attr="asymptotics", subs="n=10", value_label="asy"))
        assert label == "asy"
        assert fn({"asymptotics": "n+5"}) == pytest.approx(15.0)

    def test_resolve_value_fn_eigen(self):
        fn, label = sds._resolve_value_fn(self._args(eigen_lognorm=1))
        assert "lambda_1" in label
        rec = {"direction": [3, 4, 0], "eigenvalues": ["2.0"]}
        assert fn(rec) == pytest.approx(math.log(2.0) / 5.0)


# ===========================================================================
# JSONL loaders (temp shard file)
# ===========================================================================

def _fake_shard(symbols=("x", "y", "z")):
    """A minimal shard-like object accepted by the loaders' id derivation."""
    return SimpleNamespace(
        cmf_name="testcmf",
        encoding=(1, -1, 1),
        symbols=list(symbols),
        A=None,
    )


def _write_shard_jsonl(root: Path, shard, records):
    """Write shard trajectory records, flattening any legacy per-constant dicts
    (``delta_estimate`` / ``identified`` / ``extended_metrics``) into the current
    flat per-(trajectory, constant) schema (constant ``"log-2"``)."""
    _, shard_id, _ = sds.derive_cmf_and_shard_ids(shard)
    path = root / f"{shard_id}.jsonl"
    with open(path, "w") as f:
        for i, rec in enumerate(records):
            rec = dict(rec)
            rec.setdefault("trajectory_id", f"{shard_id}__t{i}")
            de = rec.pop("delta_estimate", None)
            idd = rec.pop("identified", None)
            em = rec.pop("extended_metrics", None)
            rec["constant"] = "log-2"
            if isinstance(de, dict) and "log-2" in de:
                rec["delta"] = de["log-2"]
            if isinstance(idd, dict):
                rec["identified"] = bool(idd.get("log-2", False))
            if isinstance(em, dict):
                rec.update(em)   # flatten metric columns to top level
            f.write(json.dumps(rec) + "\n")
    return path


class TestJsonlLoaders:
    def test_merge_keeps_flat_metric_columns(self, tmp_path):
        # A base row and a Tier-2 patch (flat columns, any order) merge without
        # wiping each other; keyed by (trajectory_id, constant).
        path = tmp_path / "s.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"trajectory_id": "t1", "constant": "log-2",
                                "direction": [1, 0, 0], "gcd_slope": 14.5}) + "\n")
            f.write(json.dumps({"trajectory_id": "t1", "constant": "log-2",
                                "direction": [1, 0, 0]}) + "\n")
        merged = sds._merge_jsonl(path)
        assert merged[("t1", "log-2")]["gcd_slope"] == pytest.approx(14.5)

    def test_load_shard_trajectories(self, tmp_path, const):
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [0, 0, 1], "delta_estimate": {"log-2": 0.2}},
            {"direction": [0, 1, 1], "delta_estimate": {"log-2": 0.3}},
            # skipped: non-finite δ
            {"direction": [1, 1, 1], "delta_estimate": {"log-2": float("-inf")}},
            # skipped: no direction
            {"delta_estimate": {"log-2": 0.5}},
        ])
        dirs, vals = sds.load_shard_trajectories(
            shard, sds.delta_value(const), str(tmp_path)
        )
        assert dirs.shape == (2, 3)
        np.testing.assert_allclose(sorted(vals), [0.2, 0.3])

    def test_load_shard_trajectories_missing_file(self, tmp_path, const):
        dirs, vals = sds.load_shard_trajectories(
            _fake_shard(), sds.delta_value(const), str(tmp_path)
        )
        assert dirs.shape == (0, 3) and vals.shape == (0,)

    def test_shard_has_identified(self, tmp_path, const):
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [0, 0, 1], "identified": {"log-2": False}},
            {"direction": [0, 1, 1], "identified": {"log-2": True}},
        ])
        assert sds.shard_has_identified(shard, const, str(tmp_path)) is True

    def test_shard_has_identified_none(self, tmp_path, const):
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [0, 0, 1], "identified": {"log-2": False}},
        ])
        assert sds.shard_has_identified(shard, const, str(tmp_path)) is False
        # absent file → False
        assert sds.shard_has_identified(_fake_shard(("a", "b")), const, str(tmp_path / "x")) is False

    def test_load_shard_samples_delta(self, tmp_path, const):
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [0, 0, 1], "delta_estimate": {"log-2": 0.3},
             "identified": {"log-2": True}},
            {"direction": [0, 1, 1], "delta_estimate": {"log-2": float("-inf")},
             "identified": {"log-2": False}},
            {"direction": [1, 1, 1], "identified": {"log-2": False}},  # no delta
        ])
        dirs, vals, ident = sds.load_shard_samples(
            shard, sds.delta_value(const), const, str(tmp_path))
        # All samples kept (unlike load_shard_trajectories, which drops non-finite).
        assert dirs.shape == (3, 3)
        assert ident.tolist() == [True, False, False]
        # Identified sample keeps its δ; the others are NaN (-inf / missing).
        assert vals[0] == pytest.approx(0.3)
        assert np.isnan(vals[1]) and np.isnan(vals[2])

    def test_load_shard_samples_uses_value_fn(self, tmp_path, const):
        # Colour value must come from value_fn (e.g. a metric), not δ.
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [0, 0, 1], "delta_estimate": {"log-2": 0.3},
             "identified": {"log-2": True},
             "extended_metrics": {"digits_per_step": 19.0}},
        ])
        _, vals, ident = sds.load_shard_samples(
            shard, sds.extended_metric_value("digits_per_step"), const, str(tmp_path))
        assert vals[0] == pytest.approx(19.0)  # the metric, not δ (0.3)
        assert ident.tolist() == [True]

    def test_load_shard_samples_missing_file(self, tmp_path, const):
        dirs, vals, ident = sds.load_shard_samples(
            _fake_shard(), sds.delta_value(const), const, str(tmp_path / "nope")
        )
        assert dirs.shape == (0, 3) and vals.shape == (0,) and ident.shape == (0,)

    def test_load_path_directions_order(self, tmp_path):
        shard = _fake_shard()
        _write_shard_jsonl(tmp_path, shard, [
            {"direction": [1, 0, 0]},
            {"direction": [2, 0, 0]},
            {"direction": [3, 0, 0]},
        ])
        path = sds.load_path_directions(shard, str(tmp_path))
        assert path.shape == (3, 3)
        np.testing.assert_allclose(path[:, 0], [1, 2, 3])


# ===========================================================================
# Colorbar norm helpers
# ===========================================================================

class TestNormHelpers:
    def test_nice_step_positive(self):
        assert sds._nice_step(1.0) > 0
        assert sds._nice_step(0.0) == 1.0  # degenerate range

    def test_make_norm_snaps_to_step(self):
        vals = np.array([-0.55, 0.31])
        norm, vmin, vmax, step = sds._make_norm(vals, 0.2)
        assert step == 0.2
        assert vmin <= -0.55 and vmax >= 0.31
        # snapped to multiples of 0.2
        assert vmin == pytest.approx(-0.6)
        assert vmax == pytest.approx(0.4)

    def test_make_norm_constant_field_widens(self):
        vals = np.array([2.0, 2.0, 2.0])
        norm, vmin, vmax, step = sds._make_norm(vals, None)
        assert vmin < vmax  # widened so the colorbar is valid
