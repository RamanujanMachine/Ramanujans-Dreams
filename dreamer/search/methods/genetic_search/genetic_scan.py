"""
Genetic Search — population-based trajectory optimisation in flatland space.

Algorithm is faithful to ``context/resources/code/algos/genetic.py``.
Key differences from the legacy ``GeneticSearchMethod`` (which predates the
DTO/JSONL refactor):

* Genomes are flatland integer vectors (via :class:`FlatlandGeometry`) so that
  in-cone membership is fast and exact.
* Output uses the modern ``worker_pool`` sink / Tier-1 DTO pipeline.
* No GCD reduction of genome vectors — raw integer coords so that ``2*z ± 1``
  refinement and magnitude growth are meaningful (reference behaviour).
* Single-point crossover (reference), ``random.choice(elites)`` parent
  selection, refine_prob 0.7/0.3 child asymmetry.
* Early-stop on unchanged-best generations retained (user decision).
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.seed import resolve_injected_seed
from dreamer.search.methods.flatland.parallel_eval import evaluate_batch
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.rand import derive_py_random
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
from dreamer.utils.ui.tqdm_config import SmartTQDM

search_config = config.search


class NoInitialPopulation(Exception):
    """Raised when no in-cone seed genome can be found for the shard."""

    def __init__(self, shard_id: str, constant: Constant):
        """
        :param shard_id: Identifier of the shard with no in-cone seed.
        :param constant: Constant whose population could not be seeded.
        """
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Genetic Search: could not build an initial in-cone population "
            f"for constant '{constant.name}' in shard {shard_id}."
        )


# ---------------------------------------------------------------------------
# Reference operators (translated from resources/code/algos/genetic.py)
# ---------------------------------------------------------------------------

def _crossover(z1: np.ndarray, z2: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover over the flatland coordinate vector.

    :param rng: seeded :class:`random.Random` (per-(shard, method, constant)) — see
        :func:`dreamer.utils.rand.derive_py_random` — so the operator is reproducible.
    """
    d = len(z1)
    if d < 2:
        return z1.copy(), z2.copy()
    point = rng.randint(1, d - 1)
    c1 = np.concatenate([z1[:point], z2[point:]])
    c2 = np.concatenate([z2[:point], z1[point:]])
    return c1, c2


def _mutate(
    z: np.ndarray,
    *,
    max_step: int,
    mutation_prob: float,
    refine_prob: float,
    refine_coord_prob: float,
    rng: random.Random,
) -> np.ndarray:
    """Mutate a flatland genome vector.

    Faithful to ``mutate_position`` in the reference: no GCD reduction.

    * Refine mode (prob ``refine_prob``): ``z' = 2*z``, then per-coord add ±1
      with ``refine_coord_prob``; guarantee ≥ 1 coord changes.
    * Coarse mode (otherwise): per-coord add ``randint(-max_step, max_step)``
      with ``mutation_prob``.

    :param rng: seeded :class:`random.Random` for reproducibility.
    """
    if rng.random() < refine_prob:
        new_z = 2 * z.copy()
        changed = False
        for i in range(len(new_z)):
            if rng.random() < refine_coord_prob:
                new_z[i] += rng.choice([-1, 1])
                changed = True
        if not changed and len(new_z) > 0:
            idx = rng.randrange(len(new_z))
            new_z[idx] += rng.choice([-1, 1])
        return new_z

    new_z = z.copy()
    for i in range(len(new_z)):
        if rng.random() < mutation_prob:
            new_z[i] += rng.randint(-max_step, max_step)
    return new_z


# ---------------------------------------------------------------------------
# GeneticSearch
# ---------------------------------------------------------------------------

class GeneticSearch(SearchMethod):
    """Population-based search over flatland trajectory directions, single constant."""

    def __init__(
        self,
        space: Shard,
        constant: Constant,
        use_LIReC: bool = True,
    ):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this search optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """Standalone entry point — collect emitted DTOs into a list."""
        collected: list = []
        self.run(
            constant=self.constant,
            cmf_id="",
            shard_id=getattr(self.space, "cmf_name", "shard"),
            shard_encoding_str=",".join(str(e) for e in self.space.encoding),
            sink=lambda item: collected.append(item),
            seen_trajectories={},
        )
        return collected

    def run(
        self,
        *,
        constant: Constant,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
        handler_cache: Optional[Dict[str, "TrajectoryAttributesHandler"]] = None,
        geom: Optional[FlatlandGeometry] = None,
        start=None,
        pool=None,
        initial_trajectory: Optional[Position] = None,
    ) -> None:
        """Run the GA for a single constant, emitting DTOs to *sink*.

        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`
            (see :func:`parallel_eval.make_eval_pool`).  When provided, each
            generation's batch of genuinely-new trajectory walks is dispatched
            to worker processes; the main process retains ownership of the
            ``sink`` and the ``seen_trajectories`` / ``handler_cache`` dicts.
            ``None`` evaluates serially (or via the legacy thread path).

        :param geom: Pre-built :class:`FlatlandGeometry` for the shard.  The
            flatland conditioning (integer nullspace + LLL/BKZ reduction) is
            shard-dependent but **constant-independent**, so when a shard is
            searched for several constants the caller builds the geometry once
            and passes it in to avoid repeating the reduction per constant.
            ``None`` builds it here (standalone / single-constant path).
        :param start: Pre-fetched interior start :class:`Position` for the
            shard (also constant-independent).  ``None`` fetches it here.
        :param initial_trajectory: Optional user-supplied seed direction.  Defaults
            to the shard's ``selected_trajectory``.  When valid (a non-zero recession
            direction) it is seeded into the initial population (the rest sampled as
            usual); an invalid one is ignored (see :func:`resolve_injected_seed`).
        :raises NoInitialPopulation: if no in-cone seed genome can be built.
        """
        if handler_cache is None:
            handler_cache = {}

        # Per-(shard, method, constant) reproducible RNG for every GA operator
        # (selection / crossover / mutation / population top-up).  Same GLOBAL_SEED
        # + shard + constant => identical evolution; distinct shards/constants get
        # independent streams (nondeterministic when GLOBAL_SEED is None).
        self._rng_py = derive_py_random(shard_id, "genetic", constant.name)

        shard: Shard = self.space
        if geom is None:
            geom = FlatlandGeometry(shard)
        if start is None:
            start = shard.get_interior_point()

        eval_ctx = dict(
            geom=geom,
            shard=shard,
            start=start,
            constant=constant,
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=shard_encoding_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
            handler_cache=handler_cache,
        )

        # Resolve GA schedule (callable or int).
        dim = geom.d_flat
        generations = (
            search_config.GA_GENERATIONS(dim)
            if callable(search_config.GA_GENERATIONS)
            else search_config.GA_GENERATIONS
        )
        pop_size = (
            search_config.GA_POPULATION_SIZE(dim)
            if callable(search_config.GA_POPULATION_SIZE)
            else search_config.GA_POPULATION_SIZE
        )
        pop_size = max(pop_size, 2)

        if initial_trajectory is None:
            initial_trajectory = getattr(shard, "selected_trajectory", None)
        population = self._init_population(
            geom, pop_size, shard_id, constant, initial_trajectory=initial_trajectory
        )

        # Evaluate initial population in parallel (Fix C).
        deltas = self._eval_population(population, eval_ctx, pool)

        last_best = None
        unchanged_count = 0
        max_unchanged = max(int(0.1 * generations), 5)

        for _ in SmartTQDM(range(generations), desc='Evolving ...', **config.system.TQDM_CONFIG):
            # Sort by fitness descending.
            ranked = sorted(zip(deltas, population), key=lambda p: p[0], reverse=True)
            deltas, population = zip(*ranked) if ranked else ([], [])
            deltas, population = list(deltas), list(population)

            best_delta = deltas[0]
            if last_best is not None and abs(best_delta - last_best) < 1e-7:
                unchanged_count += 1
            else:
                unchanged_count = 0
            last_best = best_delta

            if unchanged_count >= max_unchanged:
                Logger(
                    f"Genetic Search: early stop after {unchanged_count} "
                    f"unchanged generations (shard {shard_id})",
                    Logger.Levels.debug,
                ).log()
                break

            # Elitism.
            elite_count = max(1, int(search_config.GA_ELITE_FRACTION * pop_size))
            elites = population[:elite_count]
            elite_deltas = deltas[:elite_count]

            next_pop: List[np.ndarray] = list(elites)
            next_deltas: List[float] = list(elite_deltas)

            # Reproduction until full.
            while len(next_pop) < pop_size:
                p1 = self._rng_py.choice(elites)
                p2 = self._rng_py.choice(elites)

                # Single-point crossover (reference child asymmetry).
                if self._rng_py.random() < search_config.GA_CROSSOVER_PROB:
                    c1, c2 = _crossover(p1, p2, self._rng_py)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Child-1: higher refine_prob (0.7 per reference).
                c1 = _mutate(
                    c1,
                    max_step=search_config.GA_MUTATION_STEP,
                    mutation_prob=search_config.GA_MUTATION_PROB,
                    refine_prob=0.7,
                    refine_coord_prob=search_config.GA_REFINE_COORD_PROB,
                    rng=self._rng_py,
                )
                c1 = self._repair(c1, geom)
                next_pop.append(c1)
                next_deltas.append(None)

                if len(next_pop) < pop_size:
                    # Child-2: lower refine_prob (0.3 per reference).
                    c2 = _mutate(
                        c2,
                        max_step=search_config.GA_MUTATION_STEP,
                        mutation_prob=search_config.GA_MUTATION_PROB,
                        refine_prob=0.3,
                        refine_coord_prob=search_config.GA_REFINE_COORD_PROB,
                        rng=self._rng_py,
                    )
                    c2 = self._repair(c2, geom)
                    next_pop.append(c2)
                    next_deltas.append(None)

            population = next_pop
            # Evaluate new children in parallel (elites already have cached deltas).
            children = [population[i] for i in range(len(population)) if next_deltas[i] is None]
            child_indices = [i for i in range(len(population)) if next_deltas[i] is None]
            child_deltas = self._eval_population(children, eval_ctx, pool)
            for idx, delta in zip(child_indices, child_deltas):
                next_deltas[idx] = delta
            deltas = next_deltas

        # Final evaluation pass.
        unevaluated = [(i, z) for i, z in enumerate(population) if deltas[i] is None]
        if unevaluated:
            idxs, zs = zip(*unevaluated)
            for idx, delta in zip(idxs, self._eval_population(list(zs), eval_ctx, pool)):
                deltas[idx] = delta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _valid_genomes_mask(self, Z: np.ndarray, geom: FlatlandGeometry) -> np.ndarray:
        """
        Vectorised validity test for a stack of candidate genomes.

        A genome is valid iff it is non-zero, in-cone, and within the
        trajectory-length cap — identical to :meth:`_valid_genome`, but
        evaluated for every row of *Z* in one batched NumPy pass (the cone
        membership and the shard-space norm are single matmuls over the whole
        stack via ``geom.is_inside_many`` / ``geom.traj_norm_many``).

        :param Z: Integer array of shape ``(k, d_flat)`` — one genome per row.
        :param geom: Flatland geometry for the current shard.
        :return: Boolean array of length ``k``; ``True`` where the row is valid.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.int64))
        if Z.size == 0:
            return np.zeros(Z.shape[0], dtype=bool)
        nonzero = np.any(Z != 0, axis=1)
        inside = geom.is_inside_many(Z)
        within = geom.traj_norm_many(Z, search_config.SEARCH_TRAJ_NORM) <= search_config.SEARCH_MAX_TRAJ_LEN
        return nonzero & inside & within

    def _valid_genome(self, z: np.ndarray, geom: FlatlandGeometry) -> bool:
        """
        Return True iff *z* is a valid (in-cone, within traj-length cap) genome.

        Thin scalar wrapper over :meth:`_valid_genomes_mask` so the single- and
        batch-genome paths can never diverge.

        :param z: Flatland integer vector candidate.
        :param geom: Flatland geometry for the current shard.
        :return: True if the genome is acceptable for evaluation.
        """
        return bool(self._valid_genomes_mask(np.asarray(z)[None, :], geom)[0])

    def _init_population(
        self,
        geom: FlatlandGeometry,
        pop_size: int,
        shard_id: str,
        constant: Constant,
        initial_trajectory: Optional[Position] = None,
    ) -> List[np.ndarray]:
        """
        Build an initial population of valid flatland genomes.

        :param geom: Flatland geometry for the current shard.
        :param pop_size: Target population size.
        :param shard_id: Shard identifier (used in exception message).
        :param constant: Constant being optimised (used in exception message).
        :param initial_trajectory: Optional user-supplied seed direction; when valid
            it is inserted as the first genome (the rest are sampled), so the search
            starts from the user's known-good direction.  Invalid/``None`` ⇒ ignored.
        :raises NoInitialPopulation: if no valid in-cone genome can be found.
        :return: List of ``pop_size`` valid flatland genomes.
        """
        population: List[np.ndarray] = []

        # Seed the user-supplied trajectory first (if it is a valid in-cone genome).
        seed_z = resolve_injected_seed(geom, initial_trajectory, shard_id, constant)
        if seed_z is not None:
            population.append(np.array(seed_z))

        orchestrator = ShardSamplingOrchestrator(self.space)
        n_sample = max(pop_size * 3, 10)
        trajectories = orchestrator.sample_trajectories(n_sample)

        traj_list = list(trajectories)
        if traj_list:
            # Convert all sampled trajectories to flatland and validate them in
            # a single batched NumPy pass (one matmul for cone membership, one
            # for the shard-space norm) instead of a per-genome scalar check.
            Z = np.stack([geom.to_flatland(t) for t in traj_list])
            mask = self._valid_genomes_mask(Z, geom)
            for z, ok in zip(Z, mask):
                if ok:
                    population.append(np.array(z))
                    if len(population) >= pop_size:
                        break

        # Top up with small random perturbations of found seeds when short.
        attempts = 0
        while len(population) < pop_size and attempts < pop_size * 10:
            attempts += 1
            if not population:
                break
            base = self._rng_py.choice(population).copy()
            cand = _mutate(
                base,
                max_step=1,
                mutation_prob=0.5,
                refine_prob=0.0,
                refine_coord_prob=0.0,
                rng=self._rng_py,
            )
            if self._valid_genome(cand, geom):
                population.append(cand)

        if not population:
            raise NoInitialPopulation(shard_id, constant)

        # Pad with copies if still short (will be repaired/mutated next gen).
        while len(population) < pop_size:
            population.append(self._rng_py.choice(population).copy())

        return population[:pop_size]

    def _repair(self, z: np.ndarray, geom: FlatlandGeometry) -> np.ndarray:
        """
        Return *z* if valid (in-cone and within traj-length cap), else resample.

        :param z: Candidate genome to validate.
        :param geom: Flatland geometry for the current shard.
        :return: A valid genome (original or resampled replacement).
        """
        if self._valid_genome(z, geom):
            return z
        orchestrator = ShardSamplingOrchestrator(self.space)
        for _ in range(search_config.GA_MAX_RETRIES * 3):
            trajectories = list(orchestrator.sample_trajectories(5))
            if not trajectories:
                continue
            Z = np.stack([geom.to_flatland(t) for t in trajectories])
            mask = self._valid_genomes_mask(Z, geom)
            valid_idx = np.nonzero(mask)[0]
            if valid_idx.size:
                return np.array(Z[valid_idx[0]])
        return z  # Fall back; _eval_genome returns −∞ for invalid genomes

    def _eval_genome(self, z: np.ndarray, eval_ctx: dict) -> float:
        """
        Evaluate a single genome, returning δ (−∞ for invalid genomes).

        :param z: Flatland genome vector.
        :param eval_ctx: Evaluation context dict (forwarded to evaluate_in_flatland).
        :return: δ value.
        """
        geom: FlatlandGeometry = eval_ctx["geom"]
        if not self._valid_genome(z, geom):
            return float("-inf")
        delta, _ = evaluate_in_flatland(z, **eval_ctx)
        return delta

    def _eval_population(
        self, population: List[np.ndarray], eval_ctx: dict, pool=None
    ) -> List[float]:
        """
        Evaluate a batch of genomes, optionally in parallel via a process pool.

        :param population: List of flatland genome vectors to evaluate.
        :param eval_ctx: Evaluation context dict (forwarded to evaluate_in_flatland).
        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`.
            When provided and the batch has more than one genome, genuinely-new
            walks are dispatched to worker processes (see
            :func:`flatland.parallel_eval.evaluate_batch`); otherwise the batch
            is evaluated serially in this process.
        :return: List of δ values in the same order as *population*.
        """
        if not population:
            return []
        if pool is not None and len(population) > 1:
            # Parallel path: the shared evaluator dispatches genuinely-new walks
            # to the per-shard process pool; the main process keeps ownership of
            # the sink / seen_trajectories / handler_cache.  GA only needs δ.
            geom = eval_ctx["geom"]
            results = evaluate_batch(
                population, eval_ctx=eval_ctx, pool=pool,
                valid_fn=lambda z: self._valid_genome(z, geom),
            )
            return [delta for delta, _ in results]
        return [self._eval_genome(z, eval_ctx) for z in population]
