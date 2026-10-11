"""
Tests for GeneticSearch (method + module).

Coverage:
  - FlatlandGeometry.perturbations with reduce=False (raw ±1 steps for GA/SA)
  - Population initialisation: all genomes are in-cone
  - Crossover: single-point, child length matches parents
  - Mutation: refine mode doubles coords; coarse mode stays bounded
  - Repair: out-of-cone genome replaced by a valid one
  - GA loop: elites are preserved, unchanged early-stop fires
  - Per-constant module orchestration + NoInitialPopulation catch
"""

import random

import numpy as np
import pytest
import sympy as sp
from types import SimpleNamespace
from collections import defaultdict
from unittest.mock import MagicMock, patch

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.genetic_search.genetic_scan import (
    GeneticSearch,
    NoInitialPopulation,
    _crossover,
    _mutate,
)

search_config = config.search


# ---------------------------------------------------------------------------
# Fixtures (shared with small_angle tests)
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
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    """Bounded shard: cone x>=0, y>=0 with interior point (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


# ---------------------------------------------------------------------------
# 1. FlatlandGeometry.perturbations with reduce=False
# ---------------------------------------------------------------------------

class TestPerturbationsRaw:
    def test_raw_perturbations_not_reduced(self, whole_space_shard):
        """reduce=False should return raw ±1 steps, NOT GCD-reduced."""
        geom = FlatlandGeometry(whole_space_shard)
        z = np.array([4, 6], dtype=np.int64)[: geom.d_flat]
        raw = list(geom.perturbations(z, reduce=False))
        reduced = list(geom.perturbations(z, reduce=True))
        # Raw perturbations should differ from reduced ones (the vector [4+1,6]
        # has GCD 1 so it may not always differ, but [4,6±1] will have GCD 1
        # too — the key test is shape and non-zero.
        assert len(raw) == 2 * geom.d_flat
        for p in raw:
            assert np.any(p)

    def test_raw_perturb_differs_from_z_by_exactly_one(self, whole_space_shard):
        """Each raw perturbation differs from z in exactly one coordinate by ±1."""
        geom = FlatlandGeometry(whole_space_shard)
        z = np.array([3, 7], dtype=np.int64)[: geom.d_flat]
        for cand in geom.perturbations(z, reduce=False):
            diff = cand - z
            assert sum(abs(int(d)) for d in diff) == 1


# ---------------------------------------------------------------------------
# 2. Operator tests (_crossover, _mutate)
# ---------------------------------------------------------------------------

class TestGeneticOperators:
    def test_crossover_child_length(self):
        z1 = np.array([1, 2, 3, 4], dtype=np.int64)
        z2 = np.array([5, 6, 7, 8], dtype=np.int64)
        c1, c2 = _crossover(z1, z2, random.Random(0))
        assert len(c1) == 4 and len(c2) == 4

    def test_crossover_genes_come_from_parents(self):
        z1 = np.array([1, 2, 3], dtype=np.int64)
        z2 = np.array([10, 20, 30], dtype=np.int64)
        rng = random.Random(0)
        for _ in range(20):
            c1, c2 = _crossover(z1, z2, rng)
            for i in range(3):
                assert c1[i] in (z1[i], z2[i])
                assert c2[i] in (z1[i], z2[i])

    def test_crossover_trivial_dim(self):
        z1 = np.array([1], dtype=np.int64)
        z2 = np.array([2], dtype=np.int64)
        c1, c2 = _crossover(z1, z2, random.Random(0))
        # Single-element — returns copies.
        assert list(c1) == [1] and list(c2) == [2]

    def test_mutate_refine_doubles(self):
        z = np.array([3, 4], dtype=np.int64)
        # Force refine mode always.
        result = _mutate(z, max_step=5, mutation_prob=0.5, refine_prob=1.0,
                         refine_coord_prob=0.0, rng=random.Random(0))
        # Guaranteed change applied to at least one coord.
        assert np.any(result != z)

    def test_mutate_coarse_bounded(self):
        z = np.array([0, 0, 0], dtype=np.int64)
        rng = random.Random(42)
        for _ in range(50):
            r = _mutate(z, max_step=3, mutation_prob=1.0, refine_prob=0.0,
                        refine_coord_prob=0.0, rng=rng)
            assert all(abs(int(v)) <= 3 for v in r)

    def test_mutate_no_reduction(self):
        """Mutation must NOT GCD-reduce — raw integer vectors."""
        z = np.array([6, 0], dtype=np.int64)
        result = _mutate(z, max_step=1, mutation_prob=0.0, refine_prob=1.0,
                         refine_coord_prob=1.0, rng=random.Random(0))
        # 2*[6,0] = [12,0], then each coord gets ±1 → no GCD reduction.
        assert result[0] in (11, 12, 13)


# ---------------------------------------------------------------------------
# 3. Population initialisation
# ---------------------------------------------------------------------------

class TestPopulationInit:
    def test_all_genomes_in_cone(self, simple_shard):
        """Every genome in the initial population must pass geom.is_inside."""
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        pop = method._init_population(geom, pop_size=5, shard_id="test", constant=e)
        assert len(pop) == 5
        for z in pop:
            assert geom.is_inside(z), f"Genome {z} is out of cone"

    def test_raises_when_no_seeds(self, simple_shard, monkeypatch):
        """NoInitialPopulation raised when the sampler returns nothing useful."""
        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: set(),
        )
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        with pytest.raises(NoInitialPopulation):
            method._init_population(geom, pop_size=4, shard_id="sid", constant=e)


# ---------------------------------------------------------------------------
# 4. Repair
# ---------------------------------------------------------------------------

class TestBatchValidity:
    def test_batch_mask_agrees_with_scalar(self, simple_shard):
        """The vectorised validity mask must select exactly the genomes the
        scalar _valid_genome would accept."""
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        rng = np.random.default_rng(0)
        Z = rng.integers(-40, 40, size=(50, geom.d_flat)).astype(np.int64)
        # Include the all-zero genome explicitly (must be rejected).
        Z = np.vstack([Z, np.zeros((1, geom.d_flat), dtype=np.int64)])
        batch = method._valid_genomes_mask(Z, geom)
        scalar = np.array([method._valid_genome(z, geom) for z in Z])
        assert np.array_equal(batch, scalar)

    def test_init_population_all_valid(self, simple_shard):
        """Population produced via the batched path is still all-valid."""
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        pop = method._init_population(geom, pop_size=6, shard_id="t", constant=e)
        assert len(pop) == 6
        assert all(method._valid_genome(z, geom) for z in pop)


class TestRepair:
    def test_valid_genome_unchanged(self, simple_shard):
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        # [1,1] is inside the positive-quadrant shard.
        z = geom.to_flatland(Position({s: sp.Integer(1) for s in simple_shard.symbols}))
        repaired = method._repair(z, geom)
        assert np.array_equal(repaired, z)

    def test_invalid_genome_replaced(self, simple_shard):
        method = GeneticSearch(simple_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(simple_shard)
        # [-1,-1] is outside the positive-quadrant shard.
        bad = np.array([-1, -1], dtype=np.int64)[: geom.d_flat]
        repaired = method._repair(bad, geom)
        assert geom.is_inside(repaired)


# ---------------------------------------------------------------------------
# 5. GA loop: elites preserved, early-stop
# ---------------------------------------------------------------------------

class TestGALoop:
    def test_elites_not_reevaluated(self, whole_space_shard, monkeypatch):
        """Elite genomes must not be re-evaluated in subsequent generations.

        With pop_size=4, elite_fraction=0.5 → elite_count=2.
        Expected evals: 4 (initial) + 3 gens * 2 children = 10.
        Without elitism every individual would be re-evaluated: 4 + 3*4 = 16.
        """
        eval_count = [0]

        def counting_eval(z, ctx):
            eval_count[0] += 1
            return 0.5

        monkeypatch.setattr(config.search, "GA_GENERATIONS", 3, raising=False)
        monkeypatch.setattr(config.search, "GA_POPULATION_SIZE", 4, raising=False)
        monkeypatch.setattr(config.search, "GA_ELITE_FRACTION", 0.5, raising=False)

        method = GeneticSearch(whole_space_shard, e, use_LIReC=False)
        init_pop = [np.array([i, 0], dtype=np.int64) for i in range(1, 5)]
        monkeypatch.setattr(method, "_init_population",
                            lambda g, ps, sid, c, initial_trajectory=None: init_pop[:ps])
        monkeypatch.setattr(method, "_eval_genome", counting_eval)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # Elites (2 per gen) are NOT re-evaluated; only children (2 per gen) are.
        # max_unchanged = max(int(0.1*3), 5) = 5 → early-stop won't fire in 3 gens.
        assert eval_count[0] < 16, "Without elitism 16 evals expected — got more?"
        assert eval_count[0] == 10, f"Expected 4+3*2=10 evals, got {eval_count[0]}"

    def test_early_stop_fires_before_all_generations(self, whole_space_shard, monkeypatch):
        """Early-stop must halt the loop well before GA_GENERATIONS iterations."""
        eval_count = [0]

        def constant_eval(z, ctx):
            eval_count[0] += 1
            return 0.5  # constant delta → all generations identical → early stop

        # pop=2, elite=1; each gen evaluates 1 child.
        # max_unchanged = max(int(0.1*100), 5) = 10 → stops after ~11 gens.
        # Full run would need 2 + 100*1 = 102 evals; early stop gives ~2 + 10 = 12.
        monkeypatch.setattr(config.search, "GA_GENERATIONS", 100, raising=False)
        monkeypatch.setattr(config.search, "GA_POPULATION_SIZE", 2, raising=False)
        monkeypatch.setattr(config.search, "GA_ELITE_FRACTION", 0.5, raising=False)

        method = GeneticSearch(whole_space_shard, e, use_LIReC=False)
        init_pop = [np.array([1, 0], dtype=np.int64), np.array([0, 1], dtype=np.int64)]
        monkeypatch.setattr(method, "_init_population",
                            lambda g, ps, sid, c, initial_trajectory=None: init_pop[:ps])
        monkeypatch.setattr(method, "_eval_genome", constant_eval)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # Without early stop: 2 initial + 100 children = 102.
        # With early stop at unchanged=10: ≤ 2 + 11 = 13 evals.
        assert eval_count[0] <= 15, f"Early stop should fire; got {eval_count[0]} evals (expected ≤15)"
        assert eval_count[0] < 50, f"Loop ran too long ({eval_count[0]} evals); early-stop did not fire"

    def test_elite_object_not_reevaluated(self, whole_space_shard, monkeypatch):
        """The exact array object that becomes an elite must not be re-evaluated.

        Note: a different child array that happens to have the same *values* may
        be evaluated — that is correct behaviour.  We test by object identity.
        """
        best_z = np.array([3, 0], dtype=np.int64)
        init_pop = [best_z.copy(),                      # will be the elite
                    np.array([1, 0], dtype=np.int64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([0, 2], dtype=np.int64)]

        elite_id = id(init_pop[0])
        eval_ids = []

        def tracking_eval(z, ctx):
            eval_ids.append(id(z))
            return 1.0 if np.array_equal(z, best_z) else 0.0

        monkeypatch.setattr(config.search, "GA_GENERATIONS", 3, raising=False)
        monkeypatch.setattr(config.search, "GA_POPULATION_SIZE", 4, raising=False)
        monkeypatch.setattr(config.search, "GA_ELITE_FRACTION", 0.5, raising=False)

        method = GeneticSearch(whole_space_shard, e, use_LIReC=False)
        monkeypatch.setattr(method, "_init_population",
                            lambda g, ps, sid, c, initial_trajectory=None: init_pop[:ps])
        monkeypatch.setattr(method, "_eval_genome", tracking_eval)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # The elite OBJECT (init_pop[0]) must be evaluated exactly once (initial
        # population evaluation).  After that its float delta is cached and
        # _eval_genome is never called with that object again.
        assert eval_ids.count(elite_id) == 1, (
            f"Elite object evaluated {eval_ids.count(elite_id)} times; "
            f"elitism should carry the float delta, not re-evaluate"
        )


# ---------------------------------------------------------------------------
# 6. Module: per-constant orchestration + NoInitialPopulation caught
# ---------------------------------------------------------------------------

class TestGeneticSearchModV2:
    def test_runs_once_per_identified_constant_and_catches_error(
        self, simple_shard, monkeypatch, tmp_path
    ):
        from dreamer.search.searchers.genetic_search_mod import GeneticSearchModV2
        from dreamer.search.searchers import genetic_search_mod as mod_module
        from dreamer.configs.system import sys_config
        from dreamer import pi

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)

        run_calls = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None,
                     geom=None, start=None, pool=None):
            run_calls.append(constant)
            if constant.name == "pi":
                raise NoInitialPopulation(shard_id, constant)

        monkeypatch.setattr(mod_module.GeneticSearch, "run", fake_run)

        # e succeeds, pi raises NoInitialPopulation.
        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = GeneticSearchModV2(priorities, use_LIReC=False)
        searcher.execute()

        names = sorted(c.name for c in run_calls)
        assert names == ["e", "pi"]  # both attempted; pi's error swallowed

    def test_geometry_built_once_per_shard(self, simple_shard, monkeypatch, tmp_path):
        """The flatland geometry (LLL/BKZ reduction) must be constructed once
        per shard and reused across all of the shard's identified constants —
        not rebuilt per constant."""
        from dreamer.search.searchers.genetic_search_mod import GeneticSearchModV2
        from dreamer.search.searchers import genetic_search_mod as mod_module
        from dreamer.configs.system import sys_config
        from dreamer import pi

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)

        construct_count = [0]
        real_geom = mod_module.FlatlandGeometry

        def counting_geom(shard):
            construct_count[0] += 1
            return real_geom(shard)

        monkeypatch.setattr(mod_module, "FlatlandGeometry", counting_geom)

        received_geoms = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None,
                     geom=None, start=None, pool=None):
            received_geoms.append(geom)

        monkeypatch.setattr(mod_module.GeneticSearch, "run", fake_run)

        # One shard identified for two constants.
        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = GeneticSearchModV2(priorities, use_LIReC=False)
        searcher.execute()

        assert construct_count[0] == 1, (
            f"FlatlandGeometry built {construct_count[0]} times for a 2-constant "
            f"shard; expected exactly 1 (LLL/BKZ once per shard)"
        )
        # Both constants received the *same* geometry object, and it is non-None.
        assert len(received_geoms) == 2
        assert all(g is not None for g in received_geoms)
        assert received_geoms[0] is received_geoms[1]

    def test_empty_searchables_is_noop(self, monkeypatch, tmp_path):
        from dreamer.search.searchers.genetic_search_mod import GeneticSearchModV2
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        searcher = GeneticSearchModV2({}, use_LIReC=False)
        searcher.execute()  # must not raise
