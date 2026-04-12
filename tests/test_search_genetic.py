"""Regression and behavior tests for genetic search method and module."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import sympy as sp
from ramanujantools import Position

import dreamer.search.methods.genetic as genetic_method_mod
import dreamer.search.searchers.genetic_mod as genetic_searcher_mod
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.search.searchers.genetic_mod import GeneticSearchMod
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector


class DummyCMF:
    """Minimal CMF-like object with two symbols for deterministic tests."""

    def __init__(self):
        self.x, self.y = sp.symbols("x y")
        self.symbols = [self.x, self.y]

    def dim(self):
        """Return dimension expected by sampler interfaces.
        :return: Integer dimension value.
        """
        return 2


class DummySpace:
    """Simple searchable stub returning deterministic deltas for trajectories."""

    def __init__(self):
        self.cmf = DummyCMF()
        self.symbols = list(self.cmf.symbols)
        self.const = SimpleNamespace(name="dummy-const")
        self.cmf_name = "dummy-cmf"

    def get_interior_point(self):
        """Provide a fixed interior start point.
        :return: Origin-like Position for deterministic testing.
        """
        x, y = self.cmf.symbols
        return Position({x: 0, y: 0})

    def in_space(self, _point):
        """Accept all points in this dummy implementation.
        :param _point: Candidate point.
        :return: True always.
        """
        return True

    def is_valid_trajectory(self, _trajectory):
        """Accept all trajectories in this dummy implementation.
        :param _trajectory: Candidate trajectory.
        :return: True always.
        """
        return True

    def compute_trajectory_data(self, traj, start, **_kwargs):
        """Evaluate trajectory with a known-answer style synthetic score.
        :param traj: Trajectory to evaluate.
        :param start: Starting position.
        :return: SearchData with deterministic numeric delta.
        """
        x, y = self.cmf.symbols
        score = float(100 - abs(int(traj[x]) - 2) - abs(int(traj[y]) + 1))
        sd = SearchData(SearchVector(start, traj))
        sd.delta = score
        return sd


class NoStartSpace(DummySpace):
    """Dummy space variant that cannot provide an interior start."""

    def get_interior_point(self):
        """Return no start to validate error path.
        :return: None.
        """
        return None


class ConstrainedDummySpace(DummySpace):
    """Dummy space variant enforcing Av <= 0-style non-positive coordinates."""

    def is_valid_trajectory(self, trajectory):
        """Check trajectory is inside the constrained region.
        :param trajectory: Candidate trajectory.
        :return: True when both coordinates are non-positive.
        """
        x, y = self.cmf.symbols
        return int(trajectory[x]) <= 0 and int(trajectory[y]) <= 0

    def compute_trajectory_data(self, traj, start, **_kwargs):
        """Evaluate only constrained trajectories, failing loudly otherwise.
        :param traj: Trajectory to evaluate.
        :param start: Starting position.
        :return: SearchData with deterministic score in constrained region.
        :raises AssertionError: If an out-of-space trajectory is evaluated.
        """
        assert self.is_valid_trajectory(traj), "GA evaluated a trajectory outside A v <= 0"
        x, y = self.cmf.symbols
        score = float(100 - abs(int(traj[x]) + 2) - abs(int(traj[y]) + 1))
        sd = SearchData(SearchVector(start, traj))
        sd.delta = score
        return sd


class ImpossibleConstrainedSpace(DummySpace):
    """Dummy space variant where no trajectory is considered valid."""

    def is_valid_trajectory(self, _trajectory):
        """Reject all trajectories.
        :param _trajectory: Candidate trajectory.
        :return: False always.
        """
        return False


class InvalidDeltaSpace(DummySpace):
    """Dummy space variant producing invalid deltas to exercise retry behavior."""

    def compute_trajectory_data(self, traj, start, **_kwargs):
        """Return SearchData with missing delta to trigger INVALID_DELTA path.
        :param traj: Trajectory to evaluate.
        :param start: Starting position.
        :return: SearchData instance with delta=None.
        """
        sd = SearchData(SearchVector(start, traj))
        sd.delta = None
        return sd


class _StaticSampler:
    """Sampler stub that always yields a predefined set of trajectories."""

    pool = []

    def __init__(self, _space):
        """Store space argument for compatibility.
        :param _space: Unused searchable argument.
        :return: None.
        """

    def sample_trajectories(self, _compute_n_samples, exact=False):
        """Return deterministic trajectory candidates.
        :param _compute_n_samples: Ignored sampling callback.
        :param exact: Sampling mode flag forwarded by caller.
        :return: Set of trajectories from the static pool.
        """
        assert exact is True
        return set(self.pool)


def _patch_static_sampler(monkeypatch, trajectories):
    """Patch GA sampler with deterministic candidates for non-Shard dummy spaces.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param trajectories: Iterable of Position objects to return from sampler.
    :return: None.
    """
    _StaticSampler.pool = list(trajectories)
    monkeypatch.setattr(genetic_method_mod, "ShardSamplingOrchestrator", _StaticSampler)


def test_genetic_search_known_answer_finds_expected_best(monkeypatch):
    """Known-answer test: GA should retain/evaluate the target trajectory as best.
    Assumption: deterministic sampler pool includes the optimum candidate.
    Failure mode caught: regression in scoring/sorting that loses the best trajectory.
    """
    space = DummySpace()
    x, y = space.cmf.symbols
    target = Position({x: 2, y: -1})
    pool = [
        target,
        Position({x: 1, y: -1}),
        Position({x: 3, y: -1}),
        Position({x: 2, y: -2}),
    ]
    _patch_static_sampler(monkeypatch, pool)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=2,
        pop_size=4,
        mutation_prob=0.0,
        crossover_prob=0.0,
        elite_fraction=0.5,
        parallel_eval=False,
        max_retries=0,
    )

    result = method.search(template_trajectory=Position({x: 1, y: -1}))
    best_delta, best_sv = result.best_delta

    assert best_delta == pytest.approx(100.0)
    assert best_sv is not None
    assert best_sv.trajectory == target


def test_genetic_search_uses_parallel_pool_when_enabled(monkeypatch):
    """Ensure parallel evaluation path uses multiprocessing pool when enabled.
    Assumption: sampler is patched to avoid real Shard dependency.
    Failure mode caught: accidental serialization to sequential path.
    """
    space = DummySpace()
    called = {"pool_used": False}
    _patch_static_sampler(monkeypatch, [Position({space.cmf.x: 1, space.cmf.y: -1})] * 4)

    class _DummyPool:
        def map(self, func, trajectories, starts, chunksize=1):
            return [func(t, s) for t, s in zip(trajectories, starts)]

    @contextmanager
    def _dummy_pool_context():
        called["pool_used"] = True
        yield _DummyPool()

    monkeypatch.setattr(genetic_method_mod, "create_pool", _dummy_pool_context)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=4,
        parallel_eval=True,
    )
    method.search(template_trajectory=Position({space.cmf.x: 1, space.cmf.y: -1}))

    assert called["pool_used"]


def test_genetic_search_requires_valid_start_point():
    """Validate start-point guardrail raises when no interior point exists.
    Failure mode caught: silent None start propagating into trajectory evaluation.
    """
    method = GeneticSearchMethod(cast(Searchable, cast(object, NoStartSpace())), constant=None, parallel_eval=False)
    with pytest.raises(ValueError, match="requires a valid start point"):
        method.search()


def test_genetic_search_rejects_invalid_population_size():
    """Validate constructor guardrail rejects too-small populations.
    Failure mode caught: degenerate GA setup accepted without explicit error.
    """
    with pytest.raises(ValueError, match="pop_size"):
        GeneticSearchMethod(cast(Searchable, cast(object, DummySpace())), constant=None, pop_size=1)


def test_genetic_module_execute_exports_data(monkeypatch, tmp_path):
    """Verify module execution exports one DataManager chunk per searchable.
    Assumption: GeneticSearchMethod.search is patched to isolate module orchestration.
    Failure mode caught: broken export_stream wiring or filename handling.
    """
    exported = []
    space = DummySpace()

    def _identity_tqdm(iterable, *args, **kwargs):
        return iterable

    @contextmanager
    def _fake_export_stream(root, **_kwargs):
        assert str(tmp_path) in root

        def _writer(chunk, filename):
            exported.append((chunk, filename))

        yield _writer

    monkeypatch.setattr(genetic_searcher_mod, "SmartTQDM", _identity_tqdm)
    monkeypatch.setattr(genetic_searcher_mod.sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
    monkeypatch.setattr(genetic_searcher_mod.Exporter, "export_stream", _fake_export_stream)

    def _fake_search(_self):
        return DataManager(use_LIReC=True)

    monkeypatch.setattr(genetic_searcher_mod.GeneticSearchMethod, "search", _fake_search)

    mod = GeneticSearchMod(
        [cast(Searchable, cast(object, space))],
        use_LIReC=True,
        generations=1,
        pop_size=4,
        parallel_eval=False,
    )
    mod.execute()

    assert len(exported) == 1
    assert isinstance(exported[0][0], DataManager)
    assert exported[0][1]


def test_genetic_search_repairs_invalid_mutations_with_constrained_sampling(monkeypatch):
    """Ensure mutation outputs violating Av <= 0 are repaired before evaluation.
    Assumption: sampler returns only valid constrained trajectories.
    Failure mode caught: evaluating out-of-space trajectories after mutation.
    """
    space = ConstrainedDummySpace()
    x, y = space.cmf.symbols
    valid_pool = [
        Position({x: -1, y: -1}),
        Position({x: -2, y: -1}),
        Position({x: -1, y: -2}),
    ]

    class _FakeSampler:
        def __init__(self, _space):
            pass

        def sample_trajectories(self, _compute_n_samples, exact=False):
            assert exact is True
            return set(valid_pool)

    def _always_invalid_mutate(_pos, **_kwargs):
        return Position({x: 4, y: 4})

    monkeypatch.setattr(genetic_method_mod, "ShardSamplingOrchestrator", _FakeSampler)
    monkeypatch.setattr(genetic_method_mod, "_mutate_position", _always_invalid_mutate)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=2,
        pop_size=4,
        mutation_prob=1.0,
        crossover_prob=0.0,
        parallel_eval=False,
    )

    result = method.search(template_trajectory=Position({x: -1, y: -1}))
    assert result
    assert result.best_delta[0] is not None
    assert all(space.is_valid_trajectory(sd.sv.trajectory) for sd in result.values())


def test_genetic_search_raises_when_constraints_have_no_valid_trajectories(monkeypatch):
    """Ensure search fails loudly when constraints admit no valid trajectories.
    Assumption: sampler yields only invalid trajectories for constrained space.
    Failure mode caught: infinite retries or misleading exception messages.
    """
    space = ImpossibleConstrainedSpace()
    x, y = space.cmf.symbols
    _patch_static_sampler(monkeypatch, [Position({x: 1, y: 1}), Position({x: 2, y: 2})])
    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=2,
        max_retries=1,
        parallel_eval=False,
    )

    with pytest.raises(ValueError, match="could not sample enough valid trajectories"):
        method.search(template_trajectory=Position({x: 1, y: 1}))


def test_evaluate_population_resamples_invalid_trajectories_in_batch(monkeypatch):
    """Verify invalid individuals are resampled in one batch per retry round.
    Assumption: initial cache lookup returns no SearchData for all individuals.
    Failure mode caught: per-individual resampling that scales poorly.
    """
    space = DummySpace()
    x, y = space.cmf.symbols
    start = space.get_interior_point()
    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=3,
        max_retries=1,
        parallel_eval=False,
    )

    population = [
        {"trajectory": Position({x: 1, y: 1}), "delta": None, "sd": None},
        {"trajectory": Position({x: 2, y: 2}), "delta": None, "sd": None},
        {"trajectory": Position({x: 3, y: 3}), "delta": None, "sd": None},
    ]

    monkeypatch.setattr(method, "_compute_missing_search_data", lambda _pairs: None)
    sampled_counts = []

    def _sample_batch(*, count, template_pos):
        sampled_counts.append(count)
        del template_pos
        return [
            Position({x: -1, y: -1}),
            Position({x: -2, y: -2}),
            Position({x: -3, y: -3}),
        ][:count]

    monkeypatch.setattr(method, "_sample_valid_trajectories", _sample_batch)

    evaluated = method._evaluate_population(population, start=start, template_pos=Position({x: 0, y: 0}))

    assert sampled_counts == [3]
    assert all(ind["delta"] != genetic_method_mod.INVALID_DELTA for ind in evaluated)


def test_evaluate_population_retries_unresolved_invalids_in_batch(monkeypatch):
    """Verify unresolved invalids are retried in full batches across retries.
    Assumption: evaluation always returns invalid deltas.
    Failure mode caught: shrinking or singleton retries that break retry policy.
    """
    space = InvalidDeltaSpace()
    x, y = space.cmf.symbols
    start = space.get_interior_point()
    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=2,
        max_retries=2,
        parallel_eval=False,
    )

    population = [
        {"trajectory": Position({x: 1, y: 1}), "delta": None, "sd": None},
        {"trajectory": Position({x: 2, y: 2}), "delta": None, "sd": None},
    ]

    sampled_counts = []

    def _sample_batch(*, count, template_pos):
        sampled_counts.append(count)
        del template_pos
        return [Position({x: -1, y: -1}), Position({x: -2, y: -2})][:count]

    monkeypatch.setattr(method, "_sample_valid_trajectories", _sample_batch)

    evaluated = method._evaluate_population(population, start=start, template_pos=Position({x: 0, y: 0}))

    assert sampled_counts == [2, 2]
    assert all(ind["delta"] == genetic_method_mod.INVALID_DELTA for ind in evaluated)
