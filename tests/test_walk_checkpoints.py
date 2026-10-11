"""Tests for the handler's walk checkpoints (one walk per trajectory).

Tier-1 used to walk the trajectory matrix 2-3 times from 0 (identification at
min(IDENTIFY_DEPTH, D), the convergence limits at 0.9D/0.95D/D (+ the steps
before each), δ at D).  The handler now stores the products W(d) and continues
from the deepest stored one.  These tests pin that:

  - every W(d) equals ``Matrix.walk`` from 0 exactly, including continued
    products and depth 0;
  - Tier-1 values (δ, p, q, identified, projection column) and Tier-2 values
    are identical to the old "walk from 0 per request" behaviour, for walk
    type 1 and 2, D = 1000 and D > 1000;
  - walk cost: an identified trajectory walks D steps in total; an
    unidentified one never walks more than before (D vs IDENTIFY_DEPTH + D);
  - ``release_walks`` drops the products and the trajectory still computes.
"""

import math

import pytest
import sympy as sp
from sympy.abc import n

from dreamer.configs import config
from dreamer.utils.constants import ready_made_consts as R
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler, build_trajectory_dtos

LOG2 = R.log(2)

TIER2 = (
    ("eigenvalues", "if_identified"), ("spectral_gap", "if_identified"),
    ("gcd_slope", "if_identified"), ("digits_computed", "if_identified"),
    ("convergence_rate", "if_identified"), ("avg_computed_digits_per_step", "if_identified"),
)


@pytest.fixture(scope="module")
def shard():
    """2F1(z=-1) log(2) shard with interior (1, 1, 3)."""
    from dreamer.configs.system import sys_config
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq

    old = (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
           config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS)
    config.search.DEFAULT_USES_INV_T = False
    config.extraction.STRATEGY = "heuristic"
    config.extraction.LOAD_SHARD_CACHE = False
    sys_config.EXPORT_CMFS = None
    try:
        shards = ShardExtractor([LOG2], pFq(LOG2, 2, 1, -1).to_cmf()).extract()
    finally:
        (config.search.DEFAULT_USES_INV_T, config.extraction.STRATEGY,
         config.extraction.LOAD_SHARD_CACHE, sys_config.EXPORT_CMFS) = old
    for sh in shards:
        if tuple(int(v) for v in sh.get_interior_point().values()) == (1, 1, 3):
            return sh
    raise AssertionError("log(2) shard (1,1,3) not found")


def _handler(shard, direction=(2, 1, 3), depth=None, walk_type=2):
    from ramanujantools import Position

    syms = list(shard.cmf.matrices.keys())
    return TrajectoryAttributesHandler.from_cmf(
        shard.cmf, Position(dict(zip(syms, direction))), shard.get_interior_point(),
        walk_depth=depth, walk_type=walk_type, searchable=shard,
    )


def _old_raw_walks(self, depths):
    """The previous behaviour: every request walks from 0 (reference)."""
    wanted = sorted({int(d) for d in depths})
    return dict(zip(wanted, self._traj.walk({n: 1}, wanted, {n: 0})))


def _dto_values(shard, handler):
    start = shard.get_interior_point()
    (dto,) = build_trajectory_dtos(
        handler, cmf_id="c", shard_id="s", cmf_name="c", shard_encoding_str="e",
        start=start, direction=handler._trajectory, constants=[LOG2], extra_metrics=TIER2,
    )
    row = dto.__dict__.copy()
    return {k: row[k] for k in ("delta", "identified", "p_vector", "q_vector")} | {
        k: v for k, v in (row.get("extra") or {}).items() if k != "fingerprints"
    }


class _CountingWalk:
    """Wraps ``NumericMatrix.walk``; records (start, max iteration) per call."""

    def __init__(self, monkeypatch):
        from ramanujantools.flint_core import NumericMatrix

        self.calls = []
        original = NumericMatrix.walk

        def counted(matrix, trajectory, iterations, start):
            its = [iterations] if isinstance(iterations, int) else list(iterations)
            self.calls.append((int(start[n]), max(its)))
            return original(matrix, trajectory, iterations, start)

        monkeypatch.setattr(NumericMatrix, "walk", staticmethod(counted))

    @property
    def steps(self):
        return sum(m for _, m in self.calls)


# ---------------------------------------------------------------------------
# Exactness of the products
# ---------------------------------------------------------------------------

class TestExactProducts:
    def test_checkpoints_equal_walk_from_zero(self, shard):
        h = _handler(shard, depth=300)
        depths = [0, 1, 7, 150, 151, 299, 300]
        got = h._raw_walks([150])                 # first a middle depth...
        got = h._raw_walks(depths)                # ...then continue / reuse
        ref = _old_raw_walks(h, depths)
        for d in depths:
            assert got[d] == ref[d], d

    def test_continuation_beyond_stored_depth(self, shard):
        h = _handler(shard, depth=200)
        h._raw_walks([200])
        got = h._raw_walks([260, 333])            # continues from 200
        ref = _old_raw_walks(h, [260, 333])
        assert got[260] == ref[260] and got[333] == ref[333]


    def test_depth_zero_is_identity(self, shard):
        """Boundary: W(0) is the identity (empty product), with no walk."""
        from ramanujantools import Matrix

        h = _handler(shard, depth=100)
        w0 = h._raw_walks([0])[0]
        assert w0 == Matrix.eye(h._traj.rows)

    def test_symbolic_fallback_for_extra_symbols(self):
        """A matrix with a symbol besides n cannot be walked numerically: the
        handler falls back to ``Matrix.walk`` (no continuation) — same result."""
        from ramanujantools import Matrix

        k = sp.Symbol("k")
        m = Matrix([[0, -(n ** 2) * k], [1, n + k]])
        h = TrajectoryAttributesHandler(m, walk_depth=20, walk_type=2)
        assert not h._numeric_walk_possible()
        got = h._raw_walks([5, 9])
        ref = m.walk({n: 1}, [5, 9], {n: 0})
        assert got[5] == ref[0] and got[9] == ref[1]
        assert h._raw_walks([9])[9] == ref[1]          # served from the stored result


# ---------------------------------------------------------------------------
# Identical results to the old walk-per-request behaviour
# ---------------------------------------------------------------------------

@pytest.mark.timeout(600)
@pytest.mark.parametrize("walk_type", [1, 2])
@pytest.mark.parametrize("direction", [(2, 1, 3), (1, 2, 4), (1, 1, 2)])
@pytest.mark.parametrize("depth", [1000, 1200])
def test_tier1_and_tier2_identical(monkeypatch, shard, walk_type, direction, depth):
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", walk_type == 1)
    new = _dto_values(shard, _handler(shard, direction, depth, walk_type))
    shard.cache.clear() if hasattr(shard.cache, "clear") else None
    monkeypatch.setattr(TrajectoryAttributesHandler, "_raw_walks", _old_raw_walks)
    old = _dto_values(shard, _handler(shard, direction, depth, walk_type))
    assert new.keys() == old.keys()
    for k in new:
        a, b = new[k], old[k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == b, (k, a, b)


# ---------------------------------------------------------------------------
# Walk cost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth, expected_calls", [(1000, 1), (1200, 2)])
def test_identified_trajectory_walks_depth_steps_once(monkeypatch, shard, depth, expected_calls):
    counter = _CountingWalk(monkeypatch)
    h = _handler(shard, depth=depth)
    h._constant = LOG2.value_sympy
    assert h.identified()
    assert counter.steps == depth                 # was 2·D (+ IDENTIFY_DEPTH when D > it)
    assert len(counter.calls) == expected_calls   # D > IDENTIFY_DEPTH: identify, then continue


@pytest.mark.parametrize("depth", [1000, 1200])
def test_unidentified_trajectory_never_walks_more_than_before(monkeypatch, shard, depth):
    """Unidentified: identification walks IDENTIFY_DEPTH and (by design) retries
    at the full depth D.  New: D steps in total (the retry continues from the
    first walk); old: IDENTIFY_DEPTH + D.  No convergence/δ walk either way."""
    from ramanujantools import Matrix

    def run(raw_walks=None):
        if raw_walks is not None:
            monkeypatch.setattr(TrajectoryAttributesHandler, "_raw_walks", raw_walks)
        Matrix._walk_inner.cache_clear()
        counter = _CountingWalk(monkeypatch)
        h = _handler(shard, depth=depth)
        h._constant = R.pi.value_sympy            # not this shard's constant
        assert not h.identified()
        return counter.steps

    monkeypatch.setattr(ta.db, "identify", lambda values: [])   # LIReC finds nothing
    new = run()
    old = run(_old_raw_walks)
    assert new == depth
    assert old == (depth if depth == int(config.search.IDENTIFY_DEPTH) else 1000 + depth)
    assert new <= old


def test_release_walks(shard):
    h = _handler(shard, depth=400)
    h._constant = LOG2.value_sympy
    delta = h.delta()
    assert h._checkpoints and any(k.startswith("walked_") for k in h._utility_cache)
    h.release_walks()
    assert not h._checkpoints and not h._rt_checkpoints
    assert not any(k.startswith(("walked_", "effective_walk_")) for k in h._utility_cache)
    assert h.delta() == delta                     # cached value kept
    h.clear_cache()
    assert h.delta() == delta                     # recomputed from a fresh walk
