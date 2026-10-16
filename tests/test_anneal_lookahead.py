"""Tests for the speculative look-ahead of simulated annealing.

The look-ahead (``WalkPrefetcher`` + ``evaluate_batch(prefetcher=, speculate=)``)
must change only **when** walks run, never what the search does.  These tests pin:

  - ``WalkPrefetcher``: capacity / all-or-nothing groups, skipping of known
    directions, take-once semantics, eviction, and that speculative results are
    never committed (no sink, no ``seen`` row) until requested.
  - ``evaluate_batch`` with a prefetcher returns the same values and commits the
    same sink items in the same order as without one.
  - ``lookahead_enabled`` policy (off / on / auto thresholds, incl. d_flat = 11)
    and ``_next_step_groups`` candidate generation.
  - **End-to-end equivalence**: simulated annealing on a real 2F1 log(2) shard
    (d_flat = 3), with real
    walks, emits the identical sequence of rows, the identical ``seen`` cache and
    leaves its RNG in the identical state with the look-ahead on and off — across
    temperature regimes (mostly-accept / mostly-reject) — while the look-ahead
    actually serves walks (guards against a vacuous pass).
  - The same equivalence through ``SimulatedAnnealingMod`` on a real 2-process
    eval pool (byte-identical JSONL).

The in-process ``_InlinePool`` runs every task synchronously in the test process
(after the real ``_pool_init``), so the walks are real but deterministic.
"""

import json

import numpy as np
import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e, log
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.search.methods.annealing import annealing_scan as an
from dreamer.search.methods.annealing.annealing_scan import (
    SimulatedAnnealingSearch,
    _next_step_candidates,
    _next_step_groups,
    _ranked_next_groups,
    lookahead_enabled,
)
from dreamer.search.methods.flatland import parallel_eval as pe
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.parallel_eval import WalkPrefetcher, evaluate_batch
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


# ---------------------------------------------------------------------------
# In-process pool double
# ---------------------------------------------------------------------------

class _Done:
    """A finished ``AsyncResult`` stand-in."""

    def __init__(self, value):
        self._value = value

    def ready(self):
        return True

    def get(self):
        return self._value


class _Pending(_Done):
    """An ``AsyncResult`` stand-in that reports not-ready until released."""

    def __init__(self, value):
        super().__init__(value)
        self.released = False

    def ready(self):
        return self.released


class _InlinePool:
    """Runs pool tasks synchronously in-process (real ``_pool_walk``).

    :param shard, start: handed to the real ``_pool_init`` once, like a pool
        initializer would in each worker.
    :param processes: reported worker count (drives look-ahead capacity).
    :param pending: when True, ``apply_async`` results report not-ready until
        ``release_all`` (exercises the capacity accounting).
    """

    def __init__(self, shard, start, processes=8, pending=False):
        pe._pool_init(config.export_configurations(), shard, start)
        self._processes = processes
        self.pending = pending
        self.async_results = []
        self.calls = 0

    def apply_async(self, fn, args):
        self.calls += 1
        res = (_Pending if self.pending else _Done)(fn(*args))
        self.async_results.append(res)
        return res

    def imap_unordered(self, fn, iterable, chunksize=1):
        for a in iterable:
            self.calls += 1
            yield fn(a)

    def release_all(self):
        for r in self.async_results:
            r.released = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_shard():
    """1F1(z=1) bounded shard (x >= 0, y >= 0) with interior point (1, 1)."""
    cmf = rt_pFq(1, 1, sp.Integer(1))
    syms = list(cmf.matrices.keys())
    zero = Position({s: sp.Integer(0) for s in syms})
    hps = [Hyperplane(syms[0], syms), Hyperplane(syms[1], syms)]
    interior = Position({syms[0]: sp.Integer(1), syms[1]: sp.Integer(1)})
    return Shard(cmf, e, hps, [1, 1], zero, interior)


@pytest.fixture
def shard():
    return _make_shard()


def _ctx(shard, sink_items, seen=None):
    """Build an ``eval_ctx`` collecting sink items into *sink_items*."""
    geom = FlatlandGeometry(shard)
    cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
    return dict(
        geom=geom, shard=shard, start=shard.get_interior_point(), constant=e,
        cmf_id=cmf_id, shard_id=shard_id, shard_encoding_str=enc,
        sink=sink_items.append, seen_trajectories={} if seen is None else seen,
        handler_cache={},
    )


def _lines(items):
    """Sink items → their DTO JSON lines (the exact rows that would be written)."""
    return [dto.to_json_line() for _m, _v, dto in items]


def _vec(*xs):
    return np.array(xs, dtype=np.int64)


# ---------------------------------------------------------------------------
# WalkPrefetcher
# ---------------------------------------------------------------------------

class TestWalkPrefetcher:
    def test_speculation_is_not_committed(self, shard):
        """Speculative walks run but produce no sink item and no ``seen`` row."""
        sink = []
        ctx = _ctx(shard, sink)
        pool = _InlinePool(shard, ctx["start"], processes=8)
        pf = WalkPrefetcher(pool, ctx)
        pf.speculate([[_vec(1, 2), _vec(2, 1)]])
        assert pf.stats["submitted"] == 2 and pool.calls == 2
        assert sink == [] and ctx["seen_trajectories"] == {}

    def test_groups_fill_idle_capacity_partially_and_in_order(self, shard):
        """Groups are taken in order and may start partially: with 3 idle workers
        the first group gets 3 walks and the next group none; a freed slot is
        filled by the next waiting walk."""
        ctx = _ctx(shard, [])
        pool = _InlinePool(shard, ctx["start"], processes=3, pending=True)
        pf = WalkPrefetcher(pool, ctx)
        first = [_vec(1, 2), _vec(2, 1), _vec(1, 3), _vec(3, 1)]
        assert pf.speculate([first, [_vec(2, 3), _vec(3, 2)]]) == 3
        assert pf.outstanding() == 3 and pf.free_slots() == 0
        assert pf.speculate([[_vec(1, 4), _vec(4, 1)]]) == 0   # no idle worker
        pool.release_all()
        assert pf.outstanding() == 0 and pf.free_slots() == 3
        # The unstarted rest of the first group comes first when slots free up.
        assert pf.speculate([first, [_vec(2, 3)]]) == 2
        assert pf.stats["submitted"] == 5

    def test_capacity_counts_every_task_in_flight(self, shard):
        """With a pool that reports its idle workers (``ResilientPool``), the
        step's own walks count against the capacity, not only speculation."""
        ctx = _ctx(shard, [])
        pool = _InlinePool(shard, ctx["start"], processes=8)
        pool.idle_slots = lambda: 1            # 7 of 8 workers busy with the step
        pf = WalkPrefetcher(pool, ctx)
        assert pf.speculate([[_vec(1, 2), _vec(2, 1), _vec(1, 3)]]) == 1

    def test_worker_count_is_read_live(self, shard):
        """The pool can grow after the prefetcher is created (slow start)."""
        ctx = _ctx(shard, [])
        pool = _InlinePool(shard, ctx["start"], processes=2)
        pf = WalkPrefetcher(pool, ctx)
        pool._processes = 12
        assert pf.workers == 12

    def test_skips_known_and_duplicate_directions(self, shard):
        """Already-started, primitive-equal and already-scored directions are not walked again."""
        sink = []
        ctx = _ctx(shard, sink)
        pool = _InlinePool(shard, ctx["start"], processes=16)
        pf = WalkPrefetcher(pool, ctx)
        pf.speculate([[_vec(1, 2), _vec(2, 4)]])   # (2,4) is the same primitive ray
        assert pf.stats["submitted"] == 1
        pf.speculate([[_vec(1, 2)]])               # already started
        assert pf.stats["submitted"] == 1
        evaluate_batch([_vec(1, 3), _vec(3, 1)], eval_ctx=ctx, pool=pool)  # scored → seen
        pf.speculate([[_vec(1, 3), _vec(3, 1)]])
        assert pf.stats["submitted"] == 1

    def test_take_once_and_eviction(self, shard):
        """``take`` pops the entry; unclaimed finished entries age out."""
        ctx = _ctx(shard, [])
        pool = _InlinePool(shard, ctx["start"], processes=16)
        pf = WalkPrefetcher(pool, ctx, max_age=1)
        pf.speculate([[_vec(1, 2)]])
        tid = next(iter(pf._entries))
        assert pf.has(tid)
        payload = pf.take(tid)
        assert not pf.has(tid) and not isinstance(payload, pe.WalkError)
        pf.speculate([[_vec(2, 1)]])
        pf.speculate([])
        pf.speculate([])                            # refills within a step do not age entries
        assert len(pf._entries) == 1
        pf.next_step()
        pf.next_step()                              # step advanced past max_age
        assert pf._entries == {} and pf.stats["dropped"] == 1

    def test_empty_groups(self, shard):
        ctx = _ctx(shard, [])
        pf = WalkPrefetcher(_InlinePool(shard, ctx["start"]), ctx)
        pf.speculate([])
        pf.speculate([[]])
        assert pf.stats["submitted"] == 0


class TestEvaluateBatchWithPrefetcher:
    def test_prefetched_commit_matches_direct(self):
        """Values, sink rows (order + content) and ``seen`` are identical whether the
        walks were prefetched or dispatched directly."""
        batch = [_vec(1, 2), _vec(2, 1), _vec(1, 1), _vec(2, 4)]

        sink_a = []
        shard_a = _make_shard()
        ctx_a = _ctx(shard_a, sink_a)
        pool_a = _InlinePool(shard_a, ctx_a["start"])
        direct = evaluate_batch(batch, eval_ctx=ctx_a, pool=pool_a)

        sink_b = []
        shard_b = _make_shard()
        ctx_b = _ctx(shard_b, sink_b)
        pool_b = _InlinePool(shard_b, ctx_b["start"])
        pf = WalkPrefetcher(pool_b, ctx_b)
        pf.speculate([batch[:2]])
        assert sink_b == []
        via_pf = evaluate_batch(batch, eval_ctx=ctx_b, pool=pool_b, prefetcher=pf)

        assert pf.stats["taken"] == 2
        assert via_pf == direct
        assert _lines(sink_b) == _lines(sink_a)
        assert ctx_b["seen_trajectories"] == ctx_a["seen_trajectories"]

    def test_speculate_only_when_batch_fully_cached(self, shard):
        """A batch with nothing to walk still forwards its speculation."""
        sink = []
        ctx = _ctx(shard, sink)
        pool = _InlinePool(shard, ctx["start"])
        evaluate_batch([_vec(1, 2), _vec(2, 1)], eval_ctx=ctx, pool=pool)
        pf = WalkPrefetcher(pool, ctx)
        evaluate_batch([_vec(1, 2), _vec(2, 1)], eval_ctx=ctx, pool=pool,
                       prefetcher=pf, speculate=[[_vec(1, 3)]])
        assert pf.stats["submitted"] == 1 and len(sink) == 2


# ---------------------------------------------------------------------------
# Policy + candidate generation
# ---------------------------------------------------------------------------

class _P:
    def __init__(self, n):
        self._processes = n


class TestLookaheadPolicy:
    @pytest.mark.parametrize("mode, workers, d, expected", [
        ("off", 100, 3, False),
        ("on", 2, 11, True),
        ("on", 1, 3, False),        # single worker = serial → nothing idle
        ("auto", 3, 3, False),      # needs more than d_flat workers
        ("auto", 4, 3, True),
        ("auto", 11, 11, False),
        ("auto", 12, 11, True),
        ("auto", 18, 7, True),      # the 2026-10-15 run: 18 cores, d_flat = 7
        ("AUTO", 80, 11, True),     # case-insensitive
    ])
    def test_modes(self, monkeypatch, mode, workers, d, expected):
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", mode)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        assert lookahead_enabled(_P(workers), d) is expected

    def test_auto_independent_of_tier2(self, monkeypatch):
        """Tier-2 runs inside the walk tasks now (no separate queue competing for
        CPU), so ``auto`` no longer depends on ``TIER2_ATTRIBUTES``."""
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (("eigenvalues", "if_identified"),))
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", "auto")
        assert lookahead_enabled(_P(80), 3) is True
        assert lookahead_enabled(_P(3), 3) is False

    def test_auto_uses_requested_not_slow_start_size(self, monkeypatch):
        """The decision is made when the pool still runs its slow-start workers;
        it must use the size the pool will grow to."""
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", "auto")
        pool = _P(2)
        pool._requested = 18
        assert lookahead_enabled(pool, 7) is True

    def test_no_pool_disables(self, monkeypatch):
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", "on")
        assert lookahead_enabled(None, 3) is False

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", "sometimes")
        with pytest.raises(ValueError):
            lookahead_enabled(_P(8), 3)


class TestNextStepGroups:
    def test_groups_cover_accept_and_reject_branches(self, shard):
        geom = FlatlandGeometry(shard)
        cur = _vec(2, 3)
        nbs = [c for c in geom.perturbations(cur, reduce=False) if geom.is_inside(c)]
        groups = _next_step_groups(cur, nbs, [cur.tobytes()], geom, 1e9, "linf",
                                   doubling_allowed=True)
        assert len(groups) == len(nbs) + 1
        tabu = {cur.tobytes()} | {n.tobytes() for n in nbs}
        for g in groups:
            assert all(geom.is_inside(c) and c.tobytes() not in tabu for c in g)
        doubled = {c.tobytes() for c in geom.perturbations(cur * 2, reduce=False) if geom.is_inside(c)}
        assert {c.tobytes() for c in groups[-1]} == doubled - tabu

    def test_no_doubling_group_when_not_allowed_and_length_cap(self, shard):
        geom = FlatlandGeometry(shard)
        cur = _vec(2, 3)
        nbs = [c for c in geom.perturbations(cur, reduce=False) if geom.is_inside(c)]
        groups = _next_step_groups(cur, nbs, [], geom, 1e9, "linf", doubling_allowed=False)
        assert len(groups) == len(nbs)
        assert _next_step_groups(cur, nbs, [], geom, 0.5, "linf", doubling_allowed=True) == []


class TestRankedNextGroups:
    """Order of the speculative neighbourhoods (affects only which walks start first)."""

    CANDS = [(0, ["g0"]), (1, ["g1"]), (2, ["g2"]), (-1, ["gd"])]

    def test_nothing_known_keeps_neighbour_order_then_doubling(self):
        rank = _ranked_next_groups(self.CANDS, cur_delta=0.0)
        assert rank([None, None, None]) == [["g0"], ["g1"], ["g2"], ["gd"]]

    def test_best_known_first_when_it_improves(self):
        """Best neighbour >= current: it is the likely next center; unknown
        neighbours (could still win) next; worse known ones; doubling last."""
        rank = _ranked_next_groups(self.CANDS, cur_delta=0.0)
        partial = [(-1.0, True), None, (0.5, True)]
        assert rank(partial) == [["g2"], ["g1"], ["g0"], ["gd"]]

    def test_doubling_second_when_best_known_is_worse(self):
        """Best neighbour so far < current: a rejection (doubling) is likely."""
        rank = _ranked_next_groups(self.CANDS, cur_delta=1.0)
        partial = [(-1.0, True), None, (0.5, True)]
        assert rank(partial) == [["g2"], ["gd"], ["g1"], ["g0"]]

    def test_without_doubling_candidate(self):
        rank = _ranked_next_groups(self.CANDS[:3], cur_delta=1.0)
        assert rank([(0.2, True), None, None]) == [["g0"], ["g1"], ["g2"]]

    def test_candidates_keep_center_keys(self, shard):
        geom = FlatlandGeometry(shard)
        cur = _vec(2, 3)
        nbs = [c for c in geom.perturbations(cur, reduce=False) if geom.is_inside(c)]
        cands = _next_step_candidates(cur, nbs, [cur.tobytes()], geom, 1e9, "linf",
                                      doubling_allowed=True)
        assert [k for k, _g in cands] == list(range(len(nbs))) + [-1]
        as_bytes = lambda groups: [[c.tobytes() for c in g] for g in groups]  # noqa: E731
        assert as_bytes(g for _k, g in cands) == as_bytes(_next_step_groups(
            cur, nbs, [cur.tobytes()], geom, 1e9, "linf", doubling_allowed=True))


class TestTailFilling:
    def test_speculation_is_reranked_as_walks_finish(self, shard):
        """``speculate`` as a callable is consulted after the step's walks are
        queued and again after each finishes, with the results known so far;
        the batch's values and committed rows equal the plain path."""
        batch = [_vec(1, 2), _vec(2, 1), _vec(1, 3)]
        ref_sink = []
        ref_ctx = _ctx(shard, ref_sink)
        ref = evaluate_batch(batch, eval_ctx=ref_ctx,
                             pool=_InlinePool(shard, ref_ctx["start"], processes=8))

        sink = []
        ctx = _ctx(shard, sink)
        pool = _InlinePool(shard, ctx["start"], processes=8, pending=True)
        pf = WalkPrefetcher(pool, ctx)
        seen_partials = []

        def speculate(partial):
            seen_partials.append(list(partial))
            return [[_vec(3, 1)], [_vec(2, 3)]]

        out = evaluate_batch(batch, eval_ctx=ctx, pool=pool, prefetcher=pf, speculate=speculate)
        assert out == ref
        assert _lines(sink) == _lines(ref_sink)                 # same rows, same order
        assert seen_partials[0] == [None, None, None]           # before any walk finished
        known = [sum(p is not None for p in part) for part in seen_partials]
        assert known == sorted(known) and known[-1] == 3       # filled in as walks finished
        assert pf.stats["submitted"] == 2                       # both speculative walks started
        assert ctx["seen_trajectories"].keys() == ref_ctx["seen_trajectories"].keys()


# ---------------------------------------------------------------------------
# End-to-end: identical SA with the look-ahead on and off
# ---------------------------------------------------------------------------

LOG2 = log(2)


def _log2_shard(monkeypatch):
    """A real 2F1(z=-1) shard for log(2) — the ``examples/main_example.py`` CMF.

    Extracted fresh (heuristic, seeded ⇒ deterministic) so every run gets its own
    Shard object and an empty (p, q) cache; the shard with interior point
    (1, 1, 3) identifies log(2) on all sampled trajectories (d_flat = 3).
    """
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq

    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    monkeypatch.setattr(config.extraction, "STRATEGY", "heuristic")
    monkeypatch.setattr(config.extraction, "IGNORE_DUPLICATE_SEARCHABLES", True)
    monkeypatch.setattr(config.extraction, "LOAD_SHARD_CACHE", False)
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", None)
    shards = ShardExtractor([LOG2], pFq(LOG2, 2, 1, -1).to_cmf()).extract()
    for sh in shards:
        if tuple(int(v) for v in sh.get_interior_point().values()) == (1, 1, 3):
            return sh
    raise AssertionError("expected log(2) shard with interior (1, 1, 3) not found")


def _run_sa(mode, monkeypatch, *, t0, iters=12, processes=64):
    """Run one SA search (real walks, inline pool); return its observable outputs."""
    monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", mode)
    monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", iters)
    monkeypatch.setattr(config.search, "ANNEAL_T0", t0)
    monkeypatch.setattr(config.search, "ANNEAL_TMIN", 1e-12)  # let low-T0 runs actually step
    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 30)  # bound the reject regime
    shard = _log2_shard(monkeypatch)
    sink = []
    seen = {}
    geom = FlatlandGeometry(shard)
    start = shard.get_interior_point()
    cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
    pool = _InlinePool(shard, start, processes=processes)
    method = SimulatedAnnealingSearch(shard, LOG2, use_LIReC=True)
    # Count Metropolis draws (one per worse best-neighbour) so the test can assert
    # the reject / length-doubling branch really ran in the low-temperature regime.
    draws = [0]
    real_derive = an.derive_py_random

    def counting_derive(*parts):
        rng = real_derive(*parts)
        real_random = rng.random

        def random():
            draws[0] += 1
            return real_random()

        rng.random = random
        return rng

    monkeypatch.setattr(an, "derive_py_random", counting_derive)
    method.run(
        constant=LOG2, cmf_id=cmf_id, shard_id=shard_id, shard_encoding_str=enc,
        sink=sink.append, seen_trajectories=seen, handler_cache={},
        geom=geom, start=start, pool=pool,
    )
    return _lines(sink), seen, method._rng_py.getstate(), method.lookahead_stats, draws[0]


@pytest.mark.timeout(600)
class TestAnnealingEquivalence:
    @pytest.mark.parametrize("t0", [1.0, 1e-3])  # mostly accept / mostly reject (+ doubling)
    def test_identical_path_rows_and_rng(self, monkeypatch, t0):
        rows_off, seen_off, rng_off, stats_off, draws_off = _run_sa("off", monkeypatch, t0=t0)
        rows_on, seen_on, rng_on, stats_on, draws_on = _run_sa("on", monkeypatch, t0=t0)
        assert stats_off is None and stats_on is not None
        assert draws_on == draws_off
        if t0 < 1:
            assert draws_off > 0, "low-T run made no Metropolis draw — reject branch untested"
        assert rows_off, "SA emitted no rows — test is vacuous"
        assert rows_on == rows_off            # same rows, same order
        assert seen_on == seen_off
        assert rng_on == rng_off              # same number of Metropolis draws
        assert stats_on["taken"] > 0, "look-ahead never served a walk — test is vacuous"

    def test_auto_disabled_on_small_pool_is_plain_loop(self, monkeypatch):
        """``auto`` with too few workers (<= d_flat = 3) creates no prefetcher and matches ``off``."""
        rows_off, _, rng_off, _, _ = _run_sa("off", monkeypatch, t0=1.0, processes=3)
        rows_auto, _, rng_auto, stats, _ = _run_sa("auto", monkeypatch, t0=1.0, processes=3)
        assert stats is None
        assert rows_auto == rows_off and rng_auto == rng_off


@pytest.mark.timeout(600)
def test_annealing_mod_real_pool_lookahead_matches(tmp_path, monkeypatch):
    """Through ``SimulatedAnnealingMod`` on a real 2-process eval pool, the JSONL
    written with the look-ahead on is byte-identical to the one written with it off."""
    from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod

    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 8)
    monkeypatch.setattr(config.search, "ANNEAL_NUM_EVAL_WORKERS", 2)
    monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
    monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
    texts = {}
    for mode in ("off", "on"):
        out = tmp_path / mode
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out))
        monkeypatch.setattr(config.search, "ANNEAL_LOOKAHEAD", mode)
        shard = _log2_shard(monkeypatch)
        SimulatedAnnealingMod({LOG2: [shard]}, use_LIReC=True).execute()
        _, shard_id, _ = derive_cmf_and_shard_ids(shard)
        texts[mode] = (out / f"{shard_id}.jsonl").read_text()
    assert texts["off"].strip(), "no rows written — test is vacuous"
    assert texts["on"] == texts["off"]


def test_claimed_entry_survives_eviction_during_same_batch(shard):
    """Regression: an old prefetched walk requested by a batch must not be evicted
    by the speculation that the same batch submits (claim-before-speculate)."""
    ctx = _ctx(shard, [])
    pool = _InlinePool(shard, ctx["start"], processes=16)
    pf = WalkPrefetcher(pool, ctx, max_age=0)
    pf.speculate([[_vec(1, 2)]])           # gen 1; the batch's speculate → gen 2 evicts gen < 2
    out = evaluate_batch([_vec(1, 2), _vec(2, 1)], eval_ctx=ctx, pool=pool,
                         prefetcher=pf, speculate=[[_vec(1, 3)]])
    assert len(out) == 2 and pf.stats["taken"] == 1
