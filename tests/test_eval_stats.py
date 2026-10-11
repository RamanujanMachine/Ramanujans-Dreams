"""Tests for the search methods' reuse counts (``EvalStats``).

Every optimiser (SA, GA, Small Angle, Gradient Ascent, SPSA) and the micro-climb
evaluate through ``evaluate_in_flatland`` / ``evaluate_batch``; ``EvalStats``
counts how each evaluation was served and the search modules log one summary
line per ``(shard, constant)`` — the same "reused · new" vocabulary as the
analysis and hedgehog plan lines.  These tests pin that every evaluation lands
in exactly one bucket:

  - ``reused`` (fresh row from the store, i.e. present before the run) vs
    ``revisited`` (row computed earlier in this run, or a duplicate ray in one
    batch);
  - ``new`` walks, with ``stale`` counting those that replace a stored row whose
    fingerprint no longer matches;
  - ``failed`` walks and directions ``outside`` the cone;
  - the serial and the batched (pool) paths count identically;
  - every method's ``run()`` and every search module pass the counter on.

The batched path uses the in-process dummy pool and stubbed worker walk of
``tests/test_flatland_parallel.py``, copied locally (no subprocesses).
"""

import inspect

import numpy as np
import pytest

from dreamer import e
from dreamer.search.methods.flatland import parallel_eval as pe
from dreamer.search.methods.flatland.evaluator import EvalStats, flatland_trajectory_key
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry


# Local copies of the tests/test_flatland_parallel.py helpers: importing another
# test module fails under bare ``pytest`` (CI), where ``tests`` is not importable.
@pytest.fixture
def whole_space_shard():
    """1F1 CMF over the whole space (no hyperplanes) — every direction is in-cone."""
    cmf = rt_pFq(1, 1, sp.Integer(1))
    syms = list(cmf.matrices.keys())
    return Shard(cmf, e, [], [], Position({s: sp.Integer(0) for s in syms}))


class _DummyPool:
    """Runs the pool API in-process (no worker processes)."""

    def imap_unordered(self, fn, args, chunksize=1):
        return [fn(a) for a in args]


def _make_ctx(shard, constant):
    """Evaluation context as the optimisers build it (empty caches)."""
    sink_items = []
    ctx = dict(
        geom=FlatlandGeometry(shard), shard=shard, start=shard.get_interior_point(),
        constant=constant, cmf_id="c", shard_id="s", shard_encoding_str="",
        sink=lambda item: sink_items.append(item),
        seen_trajectories={}, handler_cache={},
    )
    return ctx, sink_items


def _z(ctx, *coords):
    return np.array(coords, dtype=np.int64)[: ctx["geom"].d_flat]


def _fresh_dto(shard, direction, constant, delta=1.0):
    """A DTO whose row is *fresh* under the current config (real fingerprint),
    so a later lookup scores it instead of re-walking."""
    from dreamer.utils.storage.dtos import TrajectoryDTO
    from dreamer.utils.storage.trajectory_attributes import tier1_config_fingerprint, walk_depth_for

    return TrajectoryDTO(
        trajectory_id="t", cmf_id="c", shard_id="s", constant=constant.name,
        start_point=(), direction=(), identified=True, delta=delta,
        config_fingerprint=tier1_config_fingerprint(walk_depth_for(shard.cmf, direction)),
    )


def _stub_walk(monkeypatch, shard):
    """Pool worker stub returning a fresh row (no real walk)."""
    def fake_pool_walk(args):
        idx, direction, constant, *_ = args
        return idx, ("MATRIX", constant.value_sympy, _fresh_dto(shard, direction, constant))

    monkeypatch.setattr(pe, "_pool_walk", fake_pool_walk)


class TestEvalStatsUnit:
    def test_buckets_and_summary(self):
        """Stored ids count as reused; others as revisited; stale ⊂ new."""
        stats = EvalStats({"stored": {"e": {}}, "other-constant": {"pi": {}}}, "e")
        stats.scored("stored")
        stats.scored("fresh-this-run")
        stats.walked("stored", cached_walk=False, ok=True)      # replaces a stale row
        stats.walked("brand-new", cached_walk=False, ok=True)
        stats.walked("x", cached_walk=True, ok=True)
        stats.walked("y", cached_walk=False, ok=False)
        stats.add("outside", 2)
        c = stats.counts
        assert (c["reused"], c["revisited"], c["new"], c["stale"]) == (1, 1, 2, 1)
        assert (c["cached_walk"], c["failed"], c["outside"]) == (1, 1, 2)
        assert stats.total == 8                                  # stale not double-counted
        line = stats.summary()
        assert line.startswith("8 evaluations → 1 reused · 1 revisited")
        assert "2 new (1 replacing stale rows)" in line and "2 outside the cone" in line

    def test_empty_summary(self):
        """Boundary: nothing evaluated still gives a well-formed line."""
        assert EvalStats({}, "e").summary() == "0 evaluations → 0 reused · 0 revisited · 0 new"


class TestBatchCounting:
    def test_new_then_revisited_then_reused(self, whole_space_shard, monkeypatch):
        """First batch walks (duplicates in the batch are revisits); a second
        batch in the same run revisits; a later run reuses the stored rows."""
        _stub_walk(monkeypatch, whole_space_shard)
        ctx, _ = _make_ctx(whole_space_shard, e)
        z = _z(ctx, 1, 0)
        ctx["stats"] = EvalStats(ctx["seen_trajectories"], e.name)
        pe.evaluate_batch([z, 2 * z, z.copy()], eval_ctx=ctx, pool=_DummyPool())
        assert ctx["stats"].counts["new"] == 1 and ctx["stats"].counts["revisited"] == 2
        pe.evaluate_batch([z, z.copy()], eval_ctx=ctx, pool=_DummyPool())
        assert ctx["stats"].counts["revisited"] == 4 and ctx["stats"].counts["reused"] == 0

        later = EvalStats(ctx["seen_trajectories"], e.name)   # next run: rows are stored
        ctx["stats"] = later
        pe.evaluate_batch([z, z.copy()], eval_ctx=ctx, pool=_DummyPool())
        assert later.counts["reused"] == 2 and later.counts["new"] == 0

    def test_stale_row_counts_as_new_and_stale(self, whole_space_shard, monkeypatch):
        """A stored row with an old fingerprint is re-walked: new + stale."""
        _stub_walk(monkeypatch, whole_space_shard)
        ctx, _ = _make_ctx(whole_space_shard, e)
        z = _z(ctx, 3, 0)
        _, tid, _ = flatland_trajectory_key(
            z, geom=ctx["geom"], shard=whole_space_shard, start=ctx["start"],
            shard_id="s", shard_encoding_str="",
        )
        ctx["seen_trajectories"][tid] = {e.name: {
            "trajectory_id": tid, "constant": e.name, "delta": 1.0,
            "identified": True, "config_fingerprint": "old",
        }}
        ctx["stats"] = EvalStats(ctx["seen_trajectories"], e.name)
        pe.evaluate_batch([z, z.copy()], eval_ctx=ctx, pool=_DummyPool())
        c = ctx["stats"].counts
        assert (c["new"], c["stale"], c["revisited"], c["reused"]) == (1, 1, 1, 0)

    def test_failed_and_outside(self, whole_space_shard, monkeypatch):
        """Failure path: a WalkError counts every direction of the group as
        failed; directions rejected by valid_fn are outside (both paths)."""
        monkeypatch.setattr(pe, "_pool_walk", lambda args: (args[0], pe.WalkError("boom")))
        ctx, _ = _make_ctx(whole_space_shard, e)
        ctx["stats"] = EvalStats({}, e.name)
        a, b = _z(ctx, 1, 0), _z(ctx, 0, 1)
        pe.evaluate_batch([a, b, -a], eval_ctx=ctx, pool=_DummyPool(),
                          valid_fn=lambda z: bool(z.sum() > 0))
        c = ctx["stats"].counts
        assert (c["failed"], c["outside"]) == (2, 1)
        pe.evaluate_batch([-a], eval_ctx=ctx, pool=None, valid_fn=lambda z: False)
        assert ctx["stats"].counts["outside"] == 2              # serial path too

    def test_serial_path_counts_like_batch(self, whole_space_shard, monkeypatch):
        """Invariant: the serial evaluator classifies exactly like the batch."""
        from dreamer.search.methods.flatland import evaluator as ev
        class _H:
            def release_walks(self):
                pass

        def fake_build(handler, **kw):
            return [_fresh_dto(whole_space_shard, kw["direction"], e, 2.0)]

        monkeypatch.setattr(ev.TrajectoryAttributesHandler, "from_cmf", staticmethod(lambda *a, **k: _H()))
        monkeypatch.setattr(ev, "build_trajectory_dtos", fake_build)
        ctx, _ = _make_ctx(whole_space_shard, e)
        ctx["stats"] = EvalStats({}, e.name)
        z = _z(ctx, 2, 0)
        ev.evaluate_in_flatland(z, **ctx)                        # new walk
        ev.evaluate_in_flatland(z, **ctx)                        # row now fresh → revisited
        c = ctx["stats"].counts
        assert (c["new"], c["revisited"]) == (1, 1)


class TestWiring:
    @pytest.mark.parametrize("cls_path", [
        "dreamer.search.methods.annealing.annealing_scan:SimulatedAnnealingSearch",
        "dreamer.search.methods.genetic_search.genetic_scan:GeneticSearch",
        "dreamer.search.methods.gradient_ascent.grad_ascent_scan:GradientAscentSearch",
        "dreamer.search.methods.gradient_ascent.spsa_adam_ascent:HybridSPSASearch",
        "dreamer.search.methods.small_angle.small_angle_scan:SmallAngleSearch",
    ])
    def test_every_method_accepts_stats(self, cls_path):
        import importlib
        mod, name = cls_path.split(":")
        cls = getattr(importlib.import_module(mod), name)
        assert "stats" in inspect.signature(cls.run).parameters

    @pytest.mark.parametrize("mod", [
        "annealing_mod", "genetic_search_mod", "gradient_ascent_mod",
        "small_angle_mod", "spsa_adam_mod", "micro_climb_finalize",
    ])
    def test_every_search_module_logs_the_summary(self, mod):
        """Regression trap: a module that stops passing / logging the counter."""
        import importlib
        src = inspect.getsource(importlib.import_module(f"dreamer.search.searchers.{mod}"))
        assert "EvalStats(" in src and "stats=stats" in src and "log_eval_stats(" in src
