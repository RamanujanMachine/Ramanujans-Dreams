"""
Tests for Small Angle Search (method + module).

Coverage:
  - reduce_to_primitive GCD vector helper
  - FlatlandGeometry conversions / membership
  - reservoir seed selection (ascending L2, first identifier, NoInitialIdentification)
  - hill-climb logic (climb, length-doubling, patience early-stop) with mocked _evaluate
  - SmallAngleSearchMod orchestration (per-constant run, error caught)
"""

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.extraction.utils.fast_gcd import reduce_to_primitive
from dreamer.configs import config
from dreamer.search.methods.small_angle import (
    SmallAngleSearch,
    NoInitialIdentification,
    FlatlandGeometry,
)

search_config = config.search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def simple_shard(simple_cmf, symbols, zero_shift):
    """Bounded shard: cone x>=0, y>=0 with interior point (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


# ---------------------------------------------------------------------------
# 1. reduce_to_primitive
# ---------------------------------------------------------------------------

class TestReduceToPrimitive:
    def test_reduces_common_factor(self):
        out = reduce_to_primitive(np.array([4, 6, 8], dtype=np.int64))
        assert list(out) == [2, 3, 4]

    def test_noop_when_already_primitive(self):
        out = reduce_to_primitive(np.array([3, 5], dtype=np.int64))
        assert list(out) == [3, 5]

    def test_noop_on_zero_vector(self):
        out = reduce_to_primitive(np.array([0, 0], dtype=np.int64))
        assert list(out) == [0, 0]

    def test_handles_negatives(self):
        out = reduce_to_primitive(np.array([-6, 9], dtype=np.int64))
        assert list(out) == [-2, 3]


# ---------------------------------------------------------------------------
# 2. FlatlandGeometry
# ---------------------------------------------------------------------------

class TestFlatlandGeometry:
    def test_whole_space_basis_is_identity(self, whole_space_shard):
        geom = FlatlandGeometry(whole_space_shard)
        assert geom.d_flat == len(whole_space_shard.symbols)
        assert np.array_equal(geom.Z_reduced, np.eye(geom.d_flat, dtype=np.int64))

    def test_to_real_to_flatland_round_trip(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        z = np.array([2, -3], dtype=np.int64)[: geom.d_flat]
        v = geom.to_real(z)
        z_back = geom.to_flatland(v)
        assert np.array_equal(z_back, z)

    def test_is_inside_matches_shard(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        for z in (np.array([1, 1]), np.array([-1, -1]), np.array([1, -1])):
            z = z[: geom.d_flat]
            assert geom.is_inside(z) == simple_shard.is_valid_trajectory(geom.to_real(z))

    def test_perturbations_are_primitive_and_nonzero(self, simple_shard):
        geom = FlatlandGeometry(simple_shard)
        z = np.array([2, 2], dtype=np.int64)[: geom.d_flat]
        perts = list(geom.perturbations(z))
        assert len(perts) == 2 * geom.d_flat
        for p in perts:
            assert np.any(p)
            from dreamer.extraction.utils.fast_gcd import get_gcd_of_array
            assert get_gcd_of_array(p) <= 1 or get_gcd_of_array(p) == 1


# ---------------------------------------------------------------------------
# 3. Seed selection
# ---------------------------------------------------------------------------

class TestSeedSelection:
    def _make_method(self, shard):
        return SmallAngleSearch(shard, e, use_LIReC=False)

    def test_picks_first_identifier_in_l2_order(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        # Reservoir returned out of L2 order; the shortest that identifies wins.
        far = Position({symbols[0]: sp.Integer(5), symbols[1]: sp.Integer(0)})
        near = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)})
        from dreamer.search.methods.small_angle import small_angle_scan as sas

        monkeypatch.setattr(
            sas.ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {far, near},
        )
        evaluated = []

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True  # everything identifies

        method._evaluate = fake_eval
        geom = FlatlandGeometry(whole_space_shard)
        seed = method._select_seed(geom, dict(geom=geom), "sid", e)
        # First evaluated must be the near (smaller L2) candidate → flatland [1,0].
        assert list(evaluated[0]) == [1, 0]
        assert list(seed) == [1, 0]

    def test_raises_when_none_identify(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        from dreamer.search.methods.small_angle import small_angle_scan as sas

        monkeypatch.setattr(
            sas.ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {t},
        )
        method._evaluate = lambda z, **kw: (-1.0, False)
        geom = FlatlandGeometry(whole_space_shard)
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, dict(geom=geom), "sid", e)


# ---------------------------------------------------------------------------
# 4. Hill-climb loop (mocked _evaluate, seed bypassed)
# ---------------------------------------------------------------------------

class TestHillClimb:
    def _make_method(self, shard, monkeypatch, seed):
        method = SmallAngleSearch(shard, e, use_LIReC=False)
        monkeypatch.setattr(method, "_select_seed", lambda *a, **k: np.array(seed, dtype=np.int64))
        return method

    def test_climbs_toward_target(self, whole_space_shard, monkeypatch):
        # Perturbations are GCD-reduced (the climb explores primitive *directions*,
        # not magnitude), so the target must be a reachable primitive direction.
        method = self._make_method(whole_space_shard, monkeypatch, seed=[1, 0])
        target = np.array([2, 1])  # reachable: [1,0] -> [1,1] -> [2,1]
        evaluated = []

        def fake_eval(z, **kw):
            z = np.asarray(z)
            evaluated.append(z.copy())
            return -float(np.linalg.norm(z - target)), True

        method._evaluate = fake_eval
        monkeypatch.setattr(search_config, "SA_MAX_DEPTH", 20)
        monkeypatch.setattr(search_config, "SA_PATIENCE", 100)
        monkeypatch.setattr(search_config, "SA_IMPROVE_THRESHOLD", 1e-6)

        method.run(
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={},
        )
        best = max(-float(np.linalg.norm(z - target)) for z in evaluated)
        # Should reach the exact target direction (delta == 0).
        assert best > -0.5

    def test_early_stops_on_patience(self, whole_space_shard, monkeypatch):
        method = self._make_method(whole_space_shard, monkeypatch, seed=[1, 0])
        calls = {"n": 0}

        def fake_eval(z, **kw):
            calls["n"] += 1
            return 0.5, True  # flat landscape: no improvement ever

        method._evaluate = fake_eval
        monkeypatch.setattr(search_config, "SA_MAX_DEPTH", 1000)
        monkeypatch.setattr(search_config, "SA_PATIENCE", 3)
        monkeypatch.setattr(search_config, "SA_IMPROVE_THRESHOLD", 1e-3)

        method.run(
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={},
        )
        # Bounded by patience, far below SA_MAX_DEPTH * perturbations.
        assert calls["n"] < 1000

    def test_doubles_when_no_perturbation_inside(self, simple_shard, monkeypatch):
        method = SmallAngleSearch(simple_shard, e, use_LIReC=False)
        monkeypatch.setattr(method, "_select_seed", lambda *a, **k: np.array([1, 1], dtype=np.int64))
        method._evaluate = lambda z, **kw: (1.0, True)
        # Force every perturbation to be rejected → only the doubling branch runs.
        seen_z = []
        from dreamer.search.methods.small_angle import flatland as fl

        monkeypatch.setattr(fl.FlatlandGeometry, "is_inside", lambda self, z: False)
        monkeypatch.setattr(search_config, "SA_MAX_DEPTH", 1000)
        monkeypatch.setattr(search_config, "SA_MAX_DOUBLINGS", 4)

        # Should terminate (no hang) after SA_MAX_DOUBLINGS.
        method.run(
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: seen_z.append(x), seen_trajectories={},
        )
        assert True  # reaching here means it terminated


# ---------------------------------------------------------------------------
# 5. Module orchestration
# ---------------------------------------------------------------------------

class TestWalkReuse:
    """Verify that already-computed walks and deltas are not recomputed."""

    def test_case_a_delta_cached_skips_evaluate(self, whole_space_shard, symbols):
        """Case A: delta already in seen_trajectories → return immediately, no handler built."""
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        from dreamer.search.methods.small_angle.flatland import FlatlandGeometry
        geom = FlatlandGeometry(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        z = geom.to_flatland(t)
        start = whole_space_shard.get_interior_point()
        from dreamer.utils.storage.trajectory_attributes import (
            _position_to_tuple, derive_trajectory_id
        )
        start_t = _position_to_tuple(start)
        dir_t = _position_to_tuple(geom.to_real(z))
        tid = derive_trajectory_id("sid", whole_space_shard.cmf_name, "", start_t, dir_t)

        from dreamer.utils.storage.trajectory_attributes import (
            tier1_config_fingerprint, walk_depth_for,
        )
        fp = tier1_config_fingerprint(walk_depth_for(whole_space_shard.cmf, geom.to_real_primitive(z)))
        seen = {tid: {e.name: {"trajectory_id": tid, "constant": e.name,
                               "delta": 2.5, "identified": True,
                               "config_fingerprint": fp}}}
        built = []

        from dreamer.search.methods.small_angle import small_angle_scan as sas
        orig_from_cmf = sas.TrajectoryAttributesHandler.from_cmf
        sas.TrajectoryAttributesHandler.from_cmf = staticmethod(lambda *a, **k: built.append(1) or orig_from_cmf(*a, **k))

        try:
            delta, ided = method._evaluate(
                z, geom=geom, start=start, constant=e,
                cmf_id="", shard_id="sid", shard_encoding_str="",
                sink=lambda x: None, seen_trajectories=seen, handler_cache={},
            )
        finally:
            sas.TrajectoryAttributesHandler.from_cmf = staticmethod(orig_from_cmf)

        assert delta == 2.5
        assert ided is True
        assert not built  # no handler was built

    def test_case_b_handler_cached_skips_walk(self, whole_space_shard, symbols):
        """Case B: handler in handler_cache → compute_for_constant only, no new from_cmf call."""
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        from dreamer.search.methods.small_angle.flatland import FlatlandGeometry
        geom = FlatlandGeometry(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        z = geom.to_flatland(t)
        start = whole_space_shard.get_interior_point()
        from dreamer.utils.storage.trajectory_attributes import (
            _position_to_tuple, derive_trajectory_id
        )
        start_t = _position_to_tuple(start)
        dir_t = _position_to_tuple(geom.to_real(z))
        tid = derive_trajectory_id("sid", whole_space_shard.cmf_name, "", start_t, dir_t)

        # Simulate: this trajectory's walk was already done (handler cached this run).
        from unittest.mock import MagicMock
        from dreamer.utils.storage.dtos import TrajectoryDTO
        cached_handler = MagicMock()
        cached_handler.trajectory_matrix = "MATRIX"

        seen = {}   # nothing on disk — reuse comes from the handler cache
        emitted = []

        from dreamer.search.methods.small_angle import small_angle_scan as sas
        from dreamer.search.methods.flatland import evaluator as ev

        # Stub the row builder so the cached (mock) handler isn't really walked; the
        # evaluator must reuse the cached handler rather than build a fresh one.
        fake_dto = TrajectoryDTO(
            trajectory_id=tid, cmf_id="", shard_id="sid", constant=e.name,
            start_point=(), direction=(), identified=True, delta=1.5,
        )
        built_fresh = []
        orig_from_cmf = sas.TrajectoryAttributesHandler.from_cmf
        sas.TrajectoryAttributesHandler.from_cmf = staticmethod(
            lambda *a, **k: built_fresh.append(1) or orig_from_cmf(*a, **k)
        )
        orig_build = ev.build_trajectory_dtos
        ev.build_trajectory_dtos = lambda *a, **k: [fake_dto]

        try:
            method._evaluate(
                z, geom=geom, start=start, constant=e,
                cmf_id="", shard_id="sid", shard_encoding_str="",
                sink=lambda x: emitted.append(x),
                seen_trajectories=seen,
                handler_cache={tid: cached_handler},
            )
        finally:
            sas.TrajectoryAttributesHandler.from_cmf = staticmethod(orig_from_cmf)
            ev.build_trajectory_dtos = orig_build

        assert not built_fresh, "Should not have built a new handler"
        assert len(emitted) == 1, "Should have emitted one item via sink"
        dto = emitted[0][2]
        assert dto.constant == e.name and dto.delta == 1.5

    def test_handler_stored_in_cache_after_case_c(self, whole_space_shard, symbols):
        """Case C: new trajectory → handler is stored in handler_cache for future reuse."""
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        from dreamer.search.methods.small_angle.flatland import FlatlandGeometry
        geom = FlatlandGeometry(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        z = geom.to_flatland(t)
        start = whole_space_shard.get_interior_point()
        cache: dict = {}
        method._evaluate(
            z, geom=geom, start=start, constant=e,
            cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache=cache,
        )
        assert len(cache) == 1

    def test_module_shares_handler_cache_across_constants(self, simple_shard, monkeypatch, tmp_path):
        """SmallAngleSearchMod passes the same handler_cache to each constant's run()."""
        from dreamer.search.searchers import small_angle_mod as mod
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(search_config, "TIER2_ATTRIBUTES", ())

        caches_seen = []

        def fake_run(self, *, constant, handler_cache, **kw):
            caches_seen.append(id(handler_cache))

        monkeypatch.setattr(mod.SmallAngleSearch, "run", fake_run)

        from dreamer import pi
        priorities = {e: [simple_shard], pi: [simple_shard]}
        mod.SmallAngleSearchMod(priorities, use_LIReC=False).execute()

        # Both constants for the same shard must share the same cache object.
        assert len(caches_seen) == 2
        assert caches_seen[0] == caches_seen[1], "handler_cache must be the same object for both constants"


class TestSmallAngleSearchMod:
    def test_runs_once_per_identified_constant_and_catches_error(
        self, simple_shard, monkeypatch, tmp_path
    ):
        from dreamer.search.searchers import small_angle_mod as mod
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(search_config, "TIER2_ATTRIBUTES", ())  # direct-write, no workers

        calls = []

        def fake_run(self, *, constant, **kw):
            calls.append(constant)
            if constant.name == "pi":
                raise NoInitialIdentification(kw["shard_id"], constant)

        monkeypatch.setattr(mod.SmallAngleSearch, "run", fake_run)

        from dreamer import pi
        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = mod.SmallAngleSearchMod(priorities, use_LIReC=False)
        searcher.execute()

        names = sorted(c.name for c in calls)
        assert names == ["e", "pi"]  # both constants attempted; pi's error swallowed

    def test_empty_searchables_is_noop(self, monkeypatch, tmp_path):
        from dreamer.search.searchers import small_angle_mod as mod
        searcher = mod.SmallAngleSearchMod({}, use_LIReC=False)
        assert searcher.execute() is None

    def test_geometry_built_once_per_shard(self, simple_shard, monkeypatch, tmp_path):
        """FlatlandGeometry (LLL/BKZ) is constructed once per shard, reused
        across the shard's identified constants."""
        from dreamer.search.searchers import small_angle_mod as mod
        from dreamer.configs.system import sys_config
        from dreamer import pi

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(search_config, "TIER2_ATTRIBUTES", ())
        monkeypatch.setattr(search_config, "SA_NUM_EVAL_WORKERS", 0)  # serial, no subprocesses

        construct_count = [0]
        real_geom = mod.FlatlandGeometry

        def counting_geom(shard):
            construct_count[0] += 1
            return real_geom(shard)

        monkeypatch.setattr(mod, "FlatlandGeometry", counting_geom)

        received = []

        def fake_run(self, *, constant, geom=None, **kw):
            received.append(geom)

        monkeypatch.setattr(mod.SmallAngleSearch, "run", fake_run)

        priorities = {e: [simple_shard], pi: [simple_shard]}
        mod.SmallAngleSearchMod(priorities, use_LIReC=False).execute()

        assert construct_count[0] == 1
        assert len(received) == 2 and received[0] is received[1] is not None


class TestParallelPerturbation:
    """_best_inside_perturbation parallel branch (dummy pool + stubbed walk)."""

    class _DummyPool:
        def map(self, fn, args):
            return [fn(a) for a in args]

        def imap_unordered(self, fn, args, chunksize=1):
            return [fn(a) for a in args]

    def _ctx(self, shard):
        geom = FlatlandGeometry(shard)
        return dict(
            geom=geom, start=shard.get_interior_point(), constant=e,
            cmf_id="c", shard_id="s", shard_encoding_str="",
            sink=lambda item: None, seen_trajectories={}, handler_cache={},
        )

    def test_parallel_picks_global_best(self, whole_space_shard, monkeypatch):
        from dreamer.search.methods.flatland import parallel_eval as pe
        from dreamer.utils.storage.dtos import TrajectoryDTO

        def fake_pool_walk(args):
            idx, direction, constant, cmf_id, shard_id, shard_encoding_str = args
            val = float(sum(int(v) for v in direction.values()))
            dto = TrajectoryDTO(
                trajectory_id="t", cmf_id=cmf_id, shard_id=shard_id,
                constant=constant.name, start_point=(), direction=(),
                identified=True, delta=val,
            )
            return (idx, ("M", constant.value_sympy, dto))

        monkeypatch.setattr(pe, "_pool_walk", fake_pool_walk)
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        ctx = self._ctx(whole_space_shard)
        d = ctx["geom"].d_flat
        z = np.array([5, 5], dtype=np.int64)[:d]
        best_z, best_score = method._best_inside_perturbation(z, ctx, self._DummyPool())
        # The best in-cone perturbation maximises the coord sum.
        assert best_z is not None
        assert best_score == max(
            float(sum(int(v) for v in ctx["geom"].to_real_primitive(c).values()))
            for c in ctx["geom"].perturbations(z, reduce=False)
            if ctx["geom"].is_inside(c)
        )

    def test_parallel_all_failed_returns_none(self, whole_space_shard, monkeypatch):
        from dreamer.search.methods.flatland import parallel_eval as pe
        monkeypatch.setattr(pe, "_pool_walk", lambda args: (args[0], pe.WalkError("boom")))
        method = SmallAngleSearch(whole_space_shard, e, use_LIReC=False)
        ctx = self._ctx(whole_space_shard)
        d = ctx["geom"].d_flat
        z = np.array([3, 0], dtype=np.int64)[:d]
        best_z, best_score = method._best_inside_perturbation(z, ctx, self._DummyPool())
        assert best_z is None and best_score == float("-inf")
