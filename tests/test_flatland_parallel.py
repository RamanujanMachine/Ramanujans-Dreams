"""
Tests for the shared process-based batch evaluator
:func:`dreamer.search.methods.flatland.parallel_eval.evaluate_batch`.

The main-process orchestration (dispatch / scaled-ray dedupe / Case-A reuse /
WalkError → −∞ / valid_fn filtering / serial fallback) is exercised with an
in-process dummy pool and a stubbed worker walk, so no real subprocesses spawn.
"""

import numpy as np
import pytest
import sympy as sp
from types import SimpleNamespace

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.evaluator import (
    _score_from_record,
    flatland_trajectory_key,
)
from dreamer.search.methods.flatland import parallel_eval as pe


@pytest.fixture
def simple_cmf():
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def zero_shift(symbols):
    return Position({s: sp.Integer(0) for s in symbols})


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


class _DummyPool:
    """Runs pool.map in-process (no real worker processes)."""
    def map(self, fn, args):
        return [fn(a) for a in args]


def _make_ctx(shard, constant):
    geom = FlatlandGeometry(shard)
    start = shard.get_interior_point()
    sink_items = []
    ctx = dict(
        geom=geom, shard=shard, start=start, constant=constant,
        cmf_id="c", shard_id="s", shard_encoding_str="",
        sink=lambda item: sink_items.append(item),
        seen_trajectories={}, handler_cache={},
    )
    return ctx, sink_items


def _stub_walk(monkeypatch):
    """Stub the pool worker: δ = sum of the primitive direction's coords."""
    def fake_pool_walk(args):
        direction, constant, *_ = args
        val = float(sum(int(v) for v in direction.values()))
        dto = SimpleNamespace(
            delta_estimate={constant.name: val},
            identified={constant.name: True},
            objective_name="delta",
            objective_value={constant.name: val},
        )
        return ("MATRIX", constant.value_sympy, dto)

    monkeypatch.setattr(pe, "_pool_walk", fake_pool_walk)


def _deltas(results):
    return [d for d, _ in results]


class TestScoreFromRecord:
    """The evaluator's objective-aware cache reader."""

    def test_reads_objective_value_under_matching_objective(self):
        rec = {
            "objective_name": "convergence_rate",
            "objective_value": {"e": 0.4},
            "identified": {"e": True},
            "delta_estimate": {"e": 0.9},
        }
        assert _score_from_record(rec, "e", "convergence_rate") == (0.4, True)

    def test_delta_fallback_for_legacy_record(self):
        # No objective fields at all → still scores under the delta objective.
        rec = {"delta_estimate": {"e": 1.7}, "identified": {"e": True}}
        assert _score_from_record(rec, "e", "delta") == (1.7, True)

    def test_no_fallback_for_non_delta_objective(self):
        # A record lacking objective_value cannot be scored for convergence_rate.
        rec = {"delta_estimate": {"e": 1.7}, "identified": {"e": True}}
        assert _score_from_record(rec, "e", "convergence_rate") is None

    def test_stale_objective_name_falls_through(self):
        rec = {
            "objective_name": "delta",
            "objective_value": {"e": 1.0},
            "identified": {"e": True},
        }
        # Active objective differs and no delta_estimate to fall back on → None.
        assert _score_from_record(rec, "e", "convergence_rate") is None

    def test_none_raw_maps_to_worst_score(self):
        rec = {
            "objective_name": "convergence_rate",
            "objective_value": {"e": None},   # unavailable for this trajectory
            "identified": {"e": False},
        }
        assert _score_from_record(rec, "e", "convergence_rate") == (float("-inf"), False)


class TestEvaluateBatch:
    def test_dispatch_dedupes_scaled_rays(self, whole_space_shard, monkeypatch):
        """z and 2z share a primitive ray → one walk, both get the same δ."""
        _stub_walk(monkeypatch)
        ctx, sink_items = _make_ctx(whole_space_shard, e)
        d = ctx["geom"].d_flat
        z = np.array([1, 0], dtype=np.int64)[:d]
        batch = [z.copy(), (2 * z).copy(), z.copy()]
        results = pe.evaluate_batch(batch, eval_ctx=ctx, pool=_DummyPool())
        assert _deltas(results) == [1.0, 1.0, 1.0]
        assert len(sink_items) == 1  # deduped: one walk / one sink emission

    def test_identified_flag_propagated(self, whole_space_shard, monkeypatch):
        """evaluate_batch returns (delta, identified) per direction (dispatched
        path; the stub marks every walk identified)."""
        _stub_walk(monkeypatch)
        ctx, _ = _make_ctx(whole_space_shard, e)
        d = ctx["geom"].d_flat
        z = np.array([2, 0], dtype=np.int64)[:d]
        # >1 element so the parallel/dispatch path runs (not the serial walk).
        results = pe.evaluate_batch([z, z.copy()], eval_ctx=ctx, pool=_DummyPool())
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert results[0][1] is True

    def test_case_a_skips_dispatch(self, whole_space_shard, monkeypatch):
        """A direction whose δ is already cached (matching fingerprint) is not
        re-walked."""
        _stub_walk(monkeypatch)
        ctx, sink_items = _make_ctx(whole_space_shard, e)
        d = ctx["geom"].d_flat
        z = np.array([3, 0], dtype=np.int64)[:d]
        _, tid, fp = flatland_trajectory_key(
            z, geom=ctx["geom"], shard=whole_space_shard, start=ctx["start"],
            shard_id="s", shard_encoding_str="",
        )
        ctx["seen_trajectories"][tid] = {
            "delta_estimate": {e.name: 42.0},
            "identified": {e.name: True},
            "config_fingerprint": fp,
        }
        # Two genomes so the parallel path runs; both share the cached ray.
        results = pe.evaluate_batch([z.copy(), z.copy()], eval_ctx=ctx, pool=_DummyPool())
        assert _deltas(results) == [42.0, 42.0]
        assert sink_items == []  # no walk dispatched

    def test_walk_error_degrades_to_neg_inf(self, whole_space_shard, monkeypatch):
        """A worker WalkError maps to −∞ without aborting the batch / emitting."""
        monkeypatch.setattr(pe, "_pool_walk", lambda args: pe.WalkError("boom"))
        ctx, sink_items = _make_ctx(whole_space_shard, e)
        d = ctx["geom"].d_flat
        batch = [np.array([1, 0], dtype=np.int64)[:d],
                 np.array([0, 1], dtype=np.int64)[:d]]
        results = pe.evaluate_batch(batch, eval_ctx=ctx, pool=_DummyPool())
        assert _deltas(results) == [float("-inf"), float("-inf")]
        assert sink_items == []

    def test_valid_fn_filters_without_walk(self, simple_shard, monkeypatch):
        """valid_fn=False directions get (−∞, False) and are never dispatched."""
        _stub_walk(monkeypatch)
        ctx, sink_items = _make_ctx(simple_shard, e)
        d = ctx["geom"].d_flat
        good = np.array([1, 1], dtype=np.int64)[:d]
        bad = np.array([-5, -5], dtype=np.int64)[:d]
        results = pe.evaluate_batch(
            [good, bad], eval_ctx=ctx, pool=_DummyPool(),
            valid_fn=ctx["geom"].is_inside,
        )
        assert results[1] == (float("-inf"), False)
