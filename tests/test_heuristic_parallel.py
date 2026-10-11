"""Tests for the parallel / per-batch-seeded heuristic ray shooter.

What must hold (and why):

  - ``_summarize`` + ``_merge_summary`` is exactly the old per-point
    ``_collect_unique_cells_into`` (min-L1 witness, earliest on ties, sign-0
    points dropped, Good-Turing ``f1``/``n``) — checked against a verbatim copy
    of the previous per-point loop on random points with heavy ties, with and
    without symmetry, and across several batches.
  - **Worker-count invariance**: 1 worker, N workers, and a pool that starts
    mid-phase all return the identical ``{cell: witness}`` map (generic +
    face-aligned phases, with and without symmetry). Each batch / face subset is
    a pure function of ``(seed, phase, index)`` and merging is in index order.
  - Seeding: same seed ⇒ same result; different seeds ⇒ different ray streams;
    ``None`` runs (nondeterministic) without error.
  - ``_resolve_workers`` maps ``None``/``0`` to the core budget.
  - Coverage sanity against exact ground truth (3D complete; 5D within the
    known exact count and ≥ 93 % of it, like the pre-change heuristic).
"""

import numpy as np
import pytest
import sympy as sp

from dreamer import log, zeta
from dreamer.configs.system import sys_config
from dreamer.extraction.extractor import extract_cmf_hyperplanes
from dreamer.extraction.v2.ray_extractor import (
    RayShootingExtractor,
    _GTStats,
    _resolve_workers,
)
from dreamer.extraction.v2.symmetry import symmetry_for_cmf
from dreamer.loading import pFq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrangement(p, q, z, const):
    """Shifted hyperplanes + symmetry of a pFq CMF (as the extractor builds them)."""
    cd = pFq(const, p, q, z).to_cmf()
    hps = [hp.apply_shift(cd.shift) for hp in extract_cmf_hyperplanes(cd)]
    return hps, symmetry_for_cmf(cd.cmf, list(cd.shift.values()))


@pytest.fixture(scope="module")
def arr3():
    return _arrangement(2, 1, -1, log(2))


@pytest.fixture(scope="module")
def arr5():
    return _arrangement(3, 2, 1, zeta(2))


def _old_collect(ex, points, A, c, out, counts, stats):
    """Verbatim copy of the pre-change per-point ``_collect_unique_cells_into``."""
    if points.shape[0] == 0:
        return
    if ex.symmetry is not None:
        points = ex.symmetry.apply(points).astype(np.int64)
    vals = points @ A.T + c
    signs = np.sign(vals).astype(np.int64)
    for sign_row, point in zip(signs, points):
        if np.any(sign_row == 0):
            continue
        key = tuple(sign_row.tolist())
        pt = point.astype(np.int64)
        existing = out.get(key)
        if existing is None or np.abs(pt).sum() < np.abs(existing).sum():
            out[key] = pt
        if counts is not None and stats is not None:
            old = counts.get(key, 0)
            if old == 0:
                stats.f1 += 1
            elif old == 1:
                stats.f1 -= 1
            counts[key] = old + 1
            stats.n += 1


def _same_map(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# 1. Vectorised collection == old per-point loop
# ---------------------------------------------------------------------------

class TestCollectEquivalence:
    @pytest.mark.parametrize("use_symmetry", [False, True])
    def test_matches_old_per_point_loop(self, arr5, use_symmetry):
        hps, sym = arr5
        ex = RayShootingExtractor(symmetry=sym if use_symmetry else None)
        A, c = ex.hyperplanes_to_matrix(hps)
        rng = np.random.default_rng(0)
        new_out, new_counts, new_stats = {}, {}, _GTStats()
        old_out, old_counts, old_stats = {}, {}, _GTStats()
        for _ in range(4):  # several batches: cross-batch merge semantics
            # Small coordinates ⇒ many ties in L1 and many repeated cells; some
            # points land on hyperplanes (sign 0) and must be dropped.
            pts = rng.integers(-3, 4, size=(3000, A.shape[1]), dtype=np.int64)
            ex._collect_unique_cells_into(pts, A, c, new_out, new_counts, new_stats)
            _old_collect(ex, pts, A, c, old_out, old_counts, old_stats)
        assert _same_map(new_out, old_out)
        assert new_counts == old_counts
        assert (new_stats.f1, new_stats.n) == (old_stats.f1, old_stats.n)

    def test_empty_and_all_on_hyperplane(self, arr3):
        hps, _ = arr3
        ex = RayShootingExtractor()
        A, c = ex.hyperplanes_to_matrix(hps)
        out, counts, stats = {}, {}, _GTStats()
        ex._collect_unique_cells_into(np.zeros((0, A.shape[1]), dtype=np.int64), A, c, out, counts, stats)
        ex._collect_unique_cells_into(np.zeros((5, A.shape[1]), dtype=np.int64), A, c, out, counts, stats)
        assert out == {} and stats.n == 0  # origin lies on the central hyperplanes


# ---------------------------------------------------------------------------
# 2. Worker-count invariance
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
class TestWorkerInvariance:
    @pytest.mark.parametrize("use_symmetry", [False, True])
    def test_one_vs_many_workers(self, arr5, use_symmetry):
        hps, sym = arr5
        kw = dict(seed=11, face_aligned=True, face_subsets=40, batch_size=4000,
                  symmetry=sym if use_symmetry else None)
        serial = RayShootingExtractor(workers=1, **kw).extract(hps)
        pooled = RayShootingExtractor(workers=3, parallel_after_seconds=0.0, **kw).extract(hps)
        assert len(serial) > 10
        assert _same_map(serial, pooled)

    def test_pool_starting_mid_phase(self, arr5, monkeypatch):
        """Start in-process, switch to the pool after the first few batches."""
        hps, sym = arr5
        import dreamer.extraction.v2.ray_extractor as rx
        clock = {"t": 0.0}
        monkeypatch.setattr(rx.time, "time", lambda: clock.setdefault("t", 0.0))
        real_local = rx._ShootRunner._local
        calls = {"n": 0}

        def local(self, phase, index):
            calls["n"] += 1
            clock["t"] += 10.0 if calls["n"] >= 3 else 0.0   # 3rd local unit → past threshold
            return real_local(self, phase, index)

        monkeypatch.setattr(rx._ShootRunner, "_local", local)
        kw = dict(seed=5, face_aligned=True, face_subsets=30, batch_size=3000, symmetry=sym)
        switched = RayShootingExtractor(workers=3, parallel_after_seconds=5.0, **kw).extract(hps)
        assert 0 < calls["n"] < 50, "pool never took over — test is vacuous"
        monkeypatch.setattr(rx._ShootRunner, "_local", real_local)
        serial = RayShootingExtractor(workers=1, **kw).extract(hps)
        assert _same_map(serial, switched)

    def test_ray_cap_respected_in_parallel(self, arr5):
        hps, _ = arr5
        kw = dict(seed=3, num_rays=9000, batch_size=2000, missing_mass=0.0)
        a = RayShootingExtractor(workers=1, **kw).extract(hps)
        b = RayShootingExtractor(workers=3, parallel_after_seconds=0.0, **kw).extract(hps)
        assert _same_map(a, b)


# ---------------------------------------------------------------------------
# 3. Seeding + worker resolution
# ---------------------------------------------------------------------------

class TestSeedAndWorkers:
    def test_same_seed_same_result_different_seed_different_rays(self, arr5):
        hps, _ = arr5
        kw = dict(num_rays=4000, batch_size=2000, missing_mass=0.0)
        a1 = RayShootingExtractor(seed=1, **kw).extract(hps)
        a2 = RayShootingExtractor(seed=1, **kw).extract(hps)
        b = RayShootingExtractor(seed=2, **kw).extract(hps)
        assert _same_map(a1, a2)
        assert not _same_map(a1, b)

    def test_none_seed_runs(self, arr3):
        """``seed=None`` (nondeterministic) still finds every 3D cell."""
        from dreamer.extraction.v2.lrs_extractor import LrslibExtractor
        hps, _ = arr3
        assert set(RayShootingExtractor(seed=None).extract(hps)) == set(LrslibExtractor().extract(hps))

    @pytest.mark.parametrize("value, expected", [(1, 1), (5, 5)])
    def test_resolve_explicit(self, value, expected):
        assert _resolve_workers(value) == expected

    @pytest.mark.parametrize("value", [None, 0])
    def test_resolve_all_cores(self, value, monkeypatch):
        monkeypatch.setattr(sys_config, "TOTAL_CORES", 6)
        assert _resolve_workers(value) == 6

    def test_negative_parallel_after_rejected(self):
        with pytest.raises(ValueError):
            RayShootingExtractor(parallel_after_seconds=-1)


# ---------------------------------------------------------------------------
# 4. Coverage sanity vs exact ground truth
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
class TestCoverage:
    def test_3d_complete(self, arr3):
        from dreamer.extraction.v2.lrs_extractor import LrslibExtractor
        hps, _ = arr3
        exact = LrslibExtractor().extract(hps)
        heur = RayShootingExtractor(seed=0).extract(hps)
        assert set(heur) == set(exact)

    def test_5d_coverage_vs_known_exact_count(self, arr5):
        """Symmetry off: the exact extractor finds 490 cells for 3F2(z=1)
        (``LrslibExtractor``, verified 2026-10-10; recomputing it costs ~2 min, so
        it is a known answer here).  Every heuristic witness lies strictly inside
        its cell (sign check), so a reported cell is never empty; the pre-change
        heuristic found 463/490, so ≥ 93 % guards against a coverage regression."""
        hps, _ = arr5
        exact_cells = 490
        for seed in (0, 1, 2):
            ex = RayShootingExtractor(seed=seed)
            heur = ex.extract(hps)
            A, c = ex.hyperplanes_to_matrix(hps)
            for key, pt in heur.items():
                assert tuple(np.sign(A @ pt + c).astype(int)) == key
            assert 0.93 * exact_cells <= len(heur) <= exact_cells, (seed, len(heur))
