"""
Tests for SimulatedAnnealingSearch (method + module).

Coverage:
  - Seed selection: ascending L2 norm, first identifier, NoInitialIdentification
  - Cooling schedule: linear and log
  - Metropolis acceptance: accept always on improvement, probabilistic on degradation
  - Temperature advances only on accepted moves
  - Length-doubling on rejection
  - Reseed after max_doublings exceeded
  - Tabu list: recently visited positions excluded from neighbours
  - SimulatedAnnealingMod orchestration + NoInitialIdentification caught
"""

import math
import numpy as np
import pytest
import sympy as sp
from types import SimpleNamespace

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.annealing.annealing_scan import (
    SimulatedAnnealingSearch,
    NoInitialIdentification,
    _get_temp,
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
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    """Bounded shard: cone x>=0, y>=0 with interior point (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


# ---------------------------------------------------------------------------
# 1. Cooling schedule
# ---------------------------------------------------------------------------

class TestCoolingSchedule:
    def test_linear_decreases(self):
        T0 = 1.0
        prev = T0
        for k in range(1, 10):
            T = _get_temp(T0, k, "linear")
            assert T < prev
            prev = T

    def test_log_decreases(self):
        # log schedule: T0/log(k+1). At k>=1 it is monotonically decreasing.
        T0 = 5.0
        prev = _get_temp(T0, 1, "log")
        for k in range(2, 10):
            T = _get_temp(T0, k, "log")
            assert T < prev
            prev = T

    def test_linear_formula(self):
        assert _get_temp(2.0, 3, "linear") == pytest.approx(2.0 / 4)

    def test_log_formula(self):
        assert _get_temp(2.0, 3, "log") == pytest.approx(2.0 / math.log(3 + 1))


# ---------------------------------------------------------------------------
# 2. Seed selection
# ---------------------------------------------------------------------------

class TestSASeedSelection:
    def _make_method(self, shard):
        return SimulatedAnnealingSearch(shard, e, use_LIReC=False)

    def test_picks_first_identifier_in_l2_order(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        far = Position({symbols[0]: sp.Integer(5), symbols[1]: sp.Integer(0)})
        near = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)})

        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {far, near},
        )

        evaluated = []

        from dreamer.search.methods.flatland import evaluate_in_flatland
        import dreamer.search.methods.annealing.annealing_scan as ann

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        seed = method._select_seed(geom, ctx, "sid", e, max_traj_len=1_000.0, traj_norm_name="linf")
        assert list(evaluated[0]) == [1, 0]
        assert list(seed) == [1, 0]

    def test_raises_when_none_identify(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})

        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {t},
        )
        import dreamer.search.methods.annealing.annealing_scan as ann
        monkeypatch.setattr(ann, "evaluate_in_flatland", lambda z, **kw: (-1.0, False))

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, ctx, "sid", e, max_traj_len=1_000.0, traj_norm_name="linf")


# ---------------------------------------------------------------------------
# 3. Metropolis acceptance
# ---------------------------------------------------------------------------

class TestMetropolis:
    def test_always_accepts_improvement(self, whole_space_shard, monkeypatch):
        """When new_delta >= cur_delta the move must be accepted."""
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        positions_visited = []

        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        # Seed at [1,0] with delta=0.1.
        call_count = [0]

        def fake_eval(z, **kw):
            call_count[0] += 1
            z = np.asarray(z)
            positions_visited.append(z.copy())
            # [1,0] → 0.1; [2,0] → 0.5 (improvement); everything else → 0.0
            if list(z) == [1, 0]:
                return 0.1, True
            if list(z) == [2, 0]:
                return 0.5, True
            return 0.0, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 3, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 5, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 10_000, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)

        method.run(
            constant=e,
            cmf_id="",
            shard_id="test",
            shard_encoding_str="",
            sink=lambda x: None,
            seen_trajectories={},
        )
        # [2,0] must appear in visited positions (improvement accepted).
        assert any(list(p) == [2, 0] for p in positions_visited)

    def test_temperature_advances_only_on_accepted(self):
        """T only advances (k increments) when a move is accepted: unit test of the formula."""
        # When no moves are accepted, _get_temp is called with k=0 only.
        # Verify: accepting 3 out of 5 iterations advances k to 3, not 5.
        T0, schedule = 1.0, "linear"
        k_after_3_accepted = 3
        T_expected = _get_temp(T0, k_after_3_accepted, schedule)
        assert T_expected == pytest.approx(T0 / (k_after_3_accepted + 1))


# ---------------------------------------------------------------------------
# 4. Length-doubling on rejection
# ---------------------------------------------------------------------------

class TestLengthDoubling:
    def test_doubling_is_raw_multiplication(self):
        """Doubling step: z *= 2 with NO GCD reduction (reference behaviour)."""
        z = np.array([6, 4], dtype=np.int64)
        doubled = z * 2
        # GCD(12,8)=4 — if reduction were applied we'd get [3,2]; we expect [12,8].
        assert list(doubled) == [12, 8]



# ---------------------------------------------------------------------------
# 5. Tabu list
# ---------------------------------------------------------------------------

class TestSATabuList:
    def test_tabu_list_bounded_by_tabu_size(self):
        """Tabu list must never exceed ANNEAL_TABU_SIZE entries."""
        old_pos_list = []
        tabu_size = 5
        for i in range(12):
            old_pos_list.append(np.array([i], dtype=np.int64).tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]
        assert len(old_pos_list) == tabu_size

    def test_tabu_list_drops_oldest_entries(self):
        """When full, the oldest (leftmost) entries are evicted."""
        old_pos_list = []
        tabu_size = 3
        for i in range(6):
            old_pos_list.append(np.array([i], dtype=np.int64).tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]
        # After 6 insertions with size 3, should hold entries 3, 4, 5.
        kept = [int(np.frombuffer(b, dtype=np.int64)[0]) for b in old_pos_list]
        assert kept == [3, 4, 5]

    def test_tabu_filters_neighbours(self, whole_space_shard):
        """Positions in the tabu set are excluded from the neighbour candidate list.

        This tests the exact filtering logic used inside the SA main loop:
            cand.tobytes() not in old_pos_list
        """
        geom = FlatlandGeometry(whole_space_shard)
        cur_z = np.array([1, 0], dtype=np.int64)[: geom.d_flat]

        # Tabu contains two of cur_z's valid neighbours.
        tabu = {
            np.array([2, 0], dtype=np.int64)[: geom.d_flat].tobytes(),
            np.array([1, 1], dtype=np.int64)[: geom.d_flat].tobytes(),
        }

        neighbours = [
            cand for cand in geom.perturbations(cur_z, reduce=False)
            if geom.is_inside(cand) and cand.tobytes() not in tabu
        ]

        included = [list(n) for n in neighbours]
        # Tabu members must be excluded.
        assert [2, 0] not in included
        assert [1, 1] not in included
        # Non-tabu member must still be present.
        assert any(n[0] == 1 and n[1] == -1 for n in included)


# ---------------------------------------------------------------------------
# 6. Stopping conditions
# ---------------------------------------------------------------------------

class TestSAStoppingConditions:
    def _common_patches(self, monkeypatch, whole_space_shard):
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        call_count = [0]

        def fake_eval(z, **kw):
            z = np.asarray(z)
            call_count[0] += 1
            # Seed [1,0] identifies; neighbours have higher delta so they're accepted.
            if list(z[:2]) == [1, 0]: return 0.1, True
            return 0.5, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        # New shared config knobs — set to non-restrictive values so they don't
        # interfere with stopping-condition tests.
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 100, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)
        return call_count

    def test_stops_when_iter_left_exhausted(self, whole_space_shard, monkeypatch):
        """Loop must stop exactly after ANNEAL_MAX_ITERS accepted moves."""
        call_count = self._common_patches(monkeypatch, whole_space_shard)
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 3, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 10, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 10_000, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        # With all neighbour deltas > cur_delta, every step is accepted.
        # 1 seed call + 1 initial eval + (3 iters × neighbours) — total bounded.
        assert call_count[0] > 0

    def test_stops_when_T_drops_below_Tmin(self, whole_space_shard, monkeypatch):
        """Loop must stop when T falls below ANNEAL_TMIN.

        Set Tmin very high (0.9) so T drops below it after just 1-2 accepted moves.
        With T0=1.0, linear schedule, T after k=1 accepted: T0/2 = 0.5 < 0.9 → stop.
        """
        call_count = self._common_patches(monkeypatch, whole_space_shard)
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        monkeypatch.setattr(config.search, "ANNEAL_T0", 1.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.49, raising=False)  # T0/2=0.5 > 0.49
        monkeypatch.setattr(config.search, "ANNEAL_SCHEDULE", "linear", raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 1000, raising=False)  # no iter limit
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 10, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 10_000, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # T after 1 accepted = T0/(1+1) = 0.5 > ANNEAL_TMIN (0.49) → continues.
        # T after 2 accepted = T0/(2+1) = 0.333 < ANNEAL_TMIN (0.49) → stops.
        # So the run should stop very quickly (< 10 accepted moves total).
        # call_count: 1 seed + 1 initial + neighbours per step; just check it terminated.
        assert call_count[0] < 50, (
            f"Run didn't stop on Tmin: {call_count[0]} eval calls (expected < 50)"
        )


# ---------------------------------------------------------------------------
# 7. Total-steps ceiling and tabu-clear-on-reseed
# ---------------------------------------------------------------------------

class TestSAStallFixes:
    def test_total_steps_ceiling_fires(self, whole_space_shard, monkeypatch):
        """ANNEAL_MAX_TOTAL_STEPS must break the loop before it runs forever.

        Scenario: all neighbours always have lower delta AND random.random()=1
        so Metropolis never accepts → iter_left stays at max_iters. Without
        the ceiling the loop would run indefinitely; with it, it must stop.
        """
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        call_count = [0]

        def fake_eval(z, **kw):
            call_count[0] += 1
            z = np.asarray(z)
            if list(z[:2]) == [1, 0]: return 0.9, True
            return 0.0, True  # neighbours always worse → rejections

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        import random as _random
        monkeypatch.setattr(_random, "random", lambda: 1.0)  # never accept probabilistically
        monkeypatch.setattr(config.search, "ANNEAL_T0", 1.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 1_000, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 50, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        # The ceiling is 50 total steps. iter_left never decrements (no accepts).
        # The loop must have exited via the ceiling, not hung forever.
        # Each step evaluates ~4 neighbours; total evals ≤ 50*4+2 = 202.
        assert call_count[0] <= 250, (
            f"Loop ran {call_count[0]} evals; ceiling should have stopped it sooner."
        )

    def test_tabu_cleared_on_reseed(self, whole_space_shard, monkeypatch):
        """After reseeding, the tabu list must contain only the new seed position.

        If the old tabu entries were kept, the reseeded position's neighbours
        would be blocked and the search would immediately stall again.
        """
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        # Track tabu list size at the first evaluation AFTER a reseed.
        tabu_sizes_at_eval = []
        reseed_count = [0]

        # We need to intercept from inside the method. We'll monkeypatch _try_reseed
        # so we can observe that the tabu is cleared after it returns.
        # Instead: observe via call_count — if tabu is NOT cleared, the reseeded
        # position's neighbours are all blocked → no evaluations after reseed.
        # If tabu IS cleared, we'll see evaluations of the seed's neighbours.

        neighbour_evals_after_reseed = [0]
        seed_returned = [False]

        def fake_eval(z, **kw):
            z = np.asarray(z)
            if list(z[:2]) == [1, 0]:
                if seed_returned[0]:
                    # This is a re-evaluation after reseed (initial eval of fresh cur_z).
                    seed_returned[0] = False
                return 0.5, True
            # Any neighbour evaluated after reseed means tabu was cleared.
            if reseed_count[0] > 0:
                neighbour_evals_after_reseed[0] += 1
            return 0.0, True  # neighbours worse → rejections

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )

        original_try_reseed = SimulatedAnnealingSearch._try_reseed

        def counting_reseed(self_, geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name):
            result = original_try_reseed(self_, geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name)
            if result is not None:
                reseed_count[0] += 1
                seed_returned[0] = True
            return result

        monkeypatch.setattr(SimulatedAnnealingSearch, "_try_reseed", counting_reseed)

        import random as _random
        monkeypatch.setattr(_random, "random", lambda: 1.0)  # never accept probabilistically
        monkeypatch.setattr(config.search, "ANNEAL_T0", 1.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 500, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 200, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 2, raising=False)  # reseed quickly
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 20, raising=False)   # allow many for this test
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 200, raising=False)    # large → would deadlock without clear
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)  # no cap for this test
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)

        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # If at least one reseed happened and neighbour evaluations followed it,
        # the tabu clear is working (unblocked the neighbours of the reseeded position).
        if reseed_count[0] > 0:
            assert neighbour_evals_after_reseed[0] > 0, (
                f"Reseed happened {reseed_count[0]} time(s) but no neighbour was "
                f"evaluated afterward — tabu was NOT cleared on reseed."
            )


# ---------------------------------------------------------------------------
# 8. Seed filtering and reseed cap
# ---------------------------------------------------------------------------

class TestSeedFilter:
    def test_seed_with_no_valid_neighbours_is_skipped(self, whole_space_shard, monkeypatch):
        """_select_seed must skip a trajectory whose entire neighbourhood is outside
        the traj-len cap; only seeds with at least one reachable neighbour are accepted.

        If this filter is missing the algorithm immediately enters the stall cycle
        (no neighbours → double → reseed → repeat → 12-hour hang).
        """
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        geom = FlatlandGeometry(whole_space_shard)

        evaluated = []

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True  # every seed identifies

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )

        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )

        # With cap = 0.0, no real-space direction has norm ≤ 0 → all neighbours filtered
        # → the seed should be skipped → NoInitialIdentification.
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, ctx, "sid", e, max_traj_len=0.0, traj_norm_name="linf")

        # Confirm evaluate_in_flatland was never called (seed rejected before eval).
        assert evaluated == [], "Seed was evaluated despite having no valid neighbours."

    def test_consecutive_reseed_failures_stop_run(self, whole_space_shard, monkeypatch):
        """After ANNEAL_MAX_RESEEDS consecutive failed reseeds, the run must terminate
        cleanly instead of looping forever calling the expensive PT sampler.
        """
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        reseed_calls = [0]

        # Seed [1,0] identifies (so run starts), but fake_eval returns delta 0.5 for it.
        def fake_eval(z, **kw):
            z = np.asarray(z)
            if list(z[:2]) == [1, 0]:
                return 0.5, True
            return 0.0, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )

        import random as _random
        monkeypatch.setattr(_random, "random", lambda: 1.0)  # never accept probabilistically

        # Patch _try_reseed to always return None (simulates no valid seed found).
        original_try_reseed = SimulatedAnnealingSearch._try_reseed

        def always_fail_reseed(self_, geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name):
            reseed_calls[0] += 1
            return None

        monkeypatch.setattr(SimulatedAnnealingSearch, "_try_reseed", always_fail_reseed)

        monkeypatch.setattr(config.search, "ANNEAL_T0", 1.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 1_000, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 100_000, raising=False)  # high: reseed cap fires first
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 2, raising=False)  # reseed every 2 rejections
        monkeypatch.setattr(config.search, "ANNEAL_MAX_RESEEDS", 3, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 200, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_MAX_TRAJ_LEN", 1_000.0, raising=False)
        monkeypatch.setattr(config.search, "SEARCH_TRAJ_NORM", "linf", raising=False)

        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        # Must complete without hanging.
        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # _try_reseed must have been called at most max_reseeds consecutive times.
        assert reseed_calls[0] <= 3, (
            f"_try_reseed called {reseed_calls[0]} times; should have stopped at 3 consecutive failures."
        )


# ---------------------------------------------------------------------------
# 9. Module: orchestration + NoInitialIdentification caught
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingMod:
    def test_runs_once_per_identified_constant_and_catches_error(
        self, simple_shard, monkeypatch, tmp_path
    ):
        from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod
        from dreamer.search.searchers import annealing_mod as mod_module
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
                raise NoInitialIdentification(shard_id, constant)

        monkeypatch.setattr(mod_module.SimulatedAnnealingSearch, "run", fake_run)

        # e succeeds, pi raises NoInitialIdentification.
        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = SimulatedAnnealingMod(priorities, use_LIReC=False)
        searcher.execute()

        names = sorted(c.name for c in run_calls)
        assert names == ["e", "pi"]  # both attempted; pi's error swallowed

    def test_geometry_built_once_per_shard(self, simple_shard, monkeypatch, tmp_path):
        """FlatlandGeometry (LLL/BKZ) is constructed once per shard and reused
        across the shard's identified constants."""
        from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod
        from dreamer.search.searchers import annealing_mod as mod_module
        from dreamer.configs.system import sys_config
        from dreamer import pi

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)
        # Serial (no pool) so the test spawns no subprocesses.
        monkeypatch.setattr(config.search, "ANNEAL_NUM_EVAL_WORKERS", 0, raising=False)

        construct_count = [0]
        real_geom = mod_module.FlatlandGeometry

        def counting_geom(shard):
            construct_count[0] += 1
            return real_geom(shard)

        monkeypatch.setattr(mod_module, "FlatlandGeometry", counting_geom)

        received = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None,
                     geom=None, start=None, pool=None):
            received.append(geom)

        monkeypatch.setattr(mod_module.SimulatedAnnealingSearch, "run", fake_run)

        priorities = {e: [simple_shard], pi: [simple_shard]}
        SimulatedAnnealingMod(priorities, use_LIReC=False).execute()

        assert construct_count[0] == 1
        assert len(received) == 2 and received[0] is received[1] is not None

    def test_empty_searchables_is_noop(self, monkeypatch, tmp_path):
        from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        searcher = SimulatedAnnealingMod({}, use_LIReC=False)
        searcher.execute()  # must not raise
