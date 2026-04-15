import random
import numpy as np
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast, Callable
from numba import njit

from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs import sys_config
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import create_pool
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector
from dreamer.configs import search_config
from ramanujantools import Position


INVALID_DELTA = -2.0


def _delta_from_search_data(sd: Optional[SearchData]) -> float:
    """
    Normalize SearchData delta values into a comparable float score.
    :param sd: Optional trajectory evaluation payload from the data manager.
    :return: A float delta value or INVALID_DELTA when delta is missing or malformed.
    """
    if sd is None or sd.delta is None or isinstance(sd.delta, str):
        return INVALID_DELTA
    try:
        return float(sd.delta)
    except (TypeError, ValueError):
        return INVALID_DELTA


def _to_position(data: Dict[Any, Any]) -> Position:
    """
    Convert a coordinate mapping into a ramanujantools Position object.
    :param data: Mapping from symbols to integer-like coordinate values.
    :return: Position built from the mapping items.
    """
    return Position(list(data.items()))

def _crossover_positions(parent1: Position, parent2: Position, canonical_keys: List[Any]) -> Tuple[Position, Position]:
    """
    Perform one-point crossover over unified coordinate keys.
    :param parent1: First parent trajectory.
    :param parent2: Second parent trajectory.
    :param canonical_keys: List of keys that are shared between trajectories.
    :return: Two child trajectories after crossover (or parents if crossover is degenerate).
    """
    if len(canonical_keys) < 2:
        return parent1, parent2

    point = random.randint(1, len(canonical_keys) - 1)
    child1_dict: Dict[Any, Any] = {}
    child2_dict: Dict[Any, Any] = {}
    for i, key in enumerate(canonical_keys):
        if i < point:
            child1_dict[key] = parent1.get(key, 0)
            child2_dict[key] = parent2.get(key, 0)
        else:
            child1_dict[key] = parent2.get(key, 0)
            child2_dict[key] = parent1.get(key, 0)

    return _to_position(child1_dict), _to_position(child2_dict)


def _mutate_position(
    pos: Position,
    *,
    max_step: int,
    mutation_prob: float,
    refine_prob: float,
    refine_coord_prob: float,
) -> Position:
    """
    Apply either coarse refinement or per-coordinate mutation to a trajectory.
    :param pos: Input trajectory to mutate.
    :param max_step: Max absolute integer mutation step for non-refine mode.
    :param mutation_prob: Per-coordinate mutation probability.
    :param refine_prob: Probability to enter refine mode (scale then perturb).
    :param refine_coord_prob: Per-coordinate perturbation probability in refine mode.
    :return: A mutated trajectory.
    """
    if random.random() < refine_prob:
        new_pos = 2 * pos
        changed = False
        for key in list(new_pos.keys()):
            if random.random() < refine_coord_prob:
                new_pos[key] = new_pos[key] + random.choice([-1, 1])
                changed = True
        if not changed and new_pos:
            key = random.choice(list(new_pos.keys()))
            new_pos[key] = new_pos[key] + random.choice([-1, 1])
        return new_pos

    new_pos = _to_position(dict(pos))
    for key in list(new_pos.keys()):
        if random.random() < mutation_prob:
            new_pos[key] = new_pos[key] + random.randint(-max_step, max_step)
    return new_pos


@njit
def _batch_mutate_population(
        pop_matrix: np.ndarray,
        mutation_prob: float,
        max_step: int,
        refine_prob: float,
        refine_coord_prob: float
) -> np.ndarray:
    """
    Compiled Numba function to mutate an entire population matrix instantly.
    """
    pop_size, dim = pop_matrix.shape
    new_matrix = np.empty_like(pop_matrix)

    for i in range(pop_size):
        # Refine Mode
        if np.random.random() < refine_prob:
            scaled_pos = 2 * pop_matrix[i]
            changed = False
            for j in range(dim):
                if np.random.random() < refine_coord_prob:
                    scaled_pos[j] += (np.random.randint(0, 2) * 2 - 1)
                    changed = True

            if not changed and dim > 0:
                idx = np.random.randint(0, dim)
                scaled_pos[idx] += (np.random.randint(0, 2) * 2 - 1)

            new_matrix[i] = scaled_pos

        # Standard Mutation Mode
        else:
            mutated_pos = np.copy(pop_matrix[i])
            for j in range(dim):
                if np.random.random() < mutation_prob:
                    mutated_pos[j] += np.random.randint(-max_step, max_step + 1)
            new_matrix[i] = mutated_pos
    return new_matrix


def _positions_to_matrix(positions: List[Position], keys: List[Any]) -> np.ndarray:
    """Bridge: Convert a list of Position dictionaries to a 2D NumPy array."""
    matrix = np.zeros((len(positions), len(keys)), dtype=np.int64)
    for i, pos in enumerate(positions):
        for j, key in enumerate(keys):
            matrix[i, j] = pos.get(key, 0)
    return matrix


def _matrix_to_positions(matrix: np.ndarray, keys: List[Any]) -> List[Position]:
    """Bridge: Convert a 2D NumPy array back to a list of Position dictionaries."""
    return [Position([(keys[j], int(val)) for j, val in enumerate(row)]) for row in matrix]


class GeneticSearchMethod(SearchMethod):
    """
    Genetic trajectory search over a shard-constrained search space.
    """

    def __init__(
        self,
        space: Shard,
        constant,
        data_manager: DataManager = None,
        share_data: bool = True,
        use_LIReC: bool = True,
    ):
        """
        Initialize GA hyperparameters and storage dependencies.
        :param space: Search space (Shard) that defines validity and trajectory evaluation.
        :param constant: Target constant metadata associated with this search.
        :param data_manager: Optional pre-existing DataManager for result sharing/caching.
        :param share_data: Whether to share storage with sibling search methods.
        :param use_LIReC: Forwarded flag for trajectory evaluation backend.
        :raises ValueError: If key GA hyperparameters are outside valid ranges.
        :return: None.
        """
        super().__init__(space, constant, use_LIReC, data_manager, share_data)

        self.elite_fraction = search_config.GA_ELITE_FRACTION
        self.mutation_prob = search_config.GA_MUTATION_PROB
        self.mutation_step = search_config.GA_MUTATION_STEP
        self.crossover_prob = search_config.GA_CROSSOVER_PROB
        self.max_retries = search_config.GA_MAX_RETRIES
        self.refine_prob = search_config.GA_REFINE_PROB
        self.refine_coord_prob = search_config.GA_REFINE_COORD_PROB
        self.space = cast(Shard, self.space)

        if self.data_manager is None:
            self.data_manager = DataManager(use_LIReC=self.use_LIReC)

        self.sampling_orchestrator = ShardSamplingOrchestrator(self.space)
        self.canonical_keys = sorted(self.space.symbols, key=str)

        generations = search_config.GA_GENERATIONS
        if isinstance(generations, int):
            self.generations = generations
        else:
            self.generations = generations(self.sampling_orchestrator.search_space_dim)

        pop_size = search_config.GA_POPULATION_SIZE
        if isinstance(pop_size, int):
            self.pop_size = pop_size
        else:
            self.pop_size = pop_size(self.sampling_orchestrator.search_space_dim)

        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")

        Logger(
            f'Initiating GA with population of {self.pop_size} running for {self.generations} generations',
            Logger.Levels.debug
        ).log()

        self._valid_trajectory_buffer = []  # The dynamic pool
        self._buffer_chunk_size = 25 * self.sampling_orchestrator.search_space_dim   # Oversample heavily

    def _resolve_start(self, starts: Optional[Position | List[Position]]) -> Position:
        """
        Resolve the starting point for evaluating trajectories.
        :param starts: Optional single start Position or list of candidate starts.
        :raises ValueError: If no usable start point can be resolved.
        :return: A concrete start Position.
        """
        if starts is None:
            start_point = self.space.get_interior_point()
        elif isinstance(starts, list):
            start_point = starts[0] if starts else None
        else:
            start_point = starts

        if start_point is None:
            raise ValueError("Genetic search requires a valid start point or left as None")
        return start_point

    def _resolve_template(self, provided: Optional[Position]) -> Position:
        """
        Resolve a template trajectory used for initialization/mutation structure.
        :param provided: Optional explicit template trajectory.
        :return: Provided template or a sampled trajectory from the shard sampler.
        """
        if provided is not None:
            return provided

        sample_set = self.sampling_orchestrator.sample_trajectories(lambda dim: max(10, 2 * dim))
        return next(iter(sample_set))

    def _sample_valid_trajectories(self, *, count: int, template_pos: Position) -> List[Position]:
        """
        Sample valid trajectories that satisfy the shard's linear constraints.
        :param count: Number of valid trajectories to sample.
        :param template_pos: Template trajectory signature used by the sampler API.
        :raises ValueError: If not enough valid trajectories can be sampled.
        :return: List of valid trajectories with requested length.
        """
        if count <= 0:
            return []

        sampled: List[Position] = []
        max_sampling_rounds = max(3, self.max_retries + 1)

        for _ in range(max_sampling_rounds):
            if len(sampled) >= count:
                break

            needed = count - len(sampled)
            candidates = list(
                self.sampling_orchestrator.sample_trajectories(lambda dim: max(needed, 2 * dim), exact=True)
            )

            random.shuffle(candidates)
            for traj in candidates:
                if self.space.is_valid_trajectory(traj):
                    sampled.append(traj)
                    if len(sampled) >= count:
                        break

        if len(sampled) < count:
            Logger(
                f"Genetic search could not sample enough valid trajectories. Sampled {len(sampled)}/{count}",
                Logger.Levels.warning
            ).log()
        return sampled[:count]

    def _get_valid_repair_trajectory(self, template_pos: Position) -> Position:
        """Pops a fresh valid trajectory from the buffer, refilling if empty."""
        if not self._valid_trajectory_buffer:
            # Batch sample a large chunk at once to minimize orchestrator overhead
            self._valid_trajectory_buffer = self._sample_valid_trajectories(
                count=self._buffer_chunk_size,
                template_pos=template_pos
            )
            if self._valid_trajectory_buffer:
                random.shuffle(self._valid_trajectory_buffer)
        return self._valid_trajectory_buffer.pop()

    def _repair_trajectory(self, trajectory: Position, template_pos: Position) -> Position:
        """
        Ensure a trajectory is valid by replacing invalid candidates via sampling.
        :param trajectory: Candidate trajectory.
        :param template_pos: Template trajectory used for fallback sampling.
        :return: Original trajectory when valid, otherwise a sampled valid replacement.
        """
        if self.space.is_valid_trajectory(trajectory):
            return trajectory
        return self._get_valid_repair_trajectory(template_pos)

    def _compute_missing_search_data(self, pairs: List[Tuple[Position, Position]]) -> None:
        """
        Compute and cache missing SearchData entries for (trajectory, start) pairs.
        :param pairs: List of trajectory/start pairs to evaluate.
        :return: None.
        """
        missing_pairs = [pair for pair in pairs if SearchVector(pair[1], pair[0]) not in self.data_manager]
        if not missing_pairs:
            return

        traj_list = [traj for traj, _ in missing_pairs]
        start_list = [start for _, start in missing_pairs]

        if search_config.PARALLEL_SEARCH and len(missing_pairs) > 1:
            with create_pool() as pool:
                results = list(
                    pool.map(
                        partial(self.space.compute_trajectory_data, use_LIReC=self.use_LIReC),
                        traj_list,
                        start_list,
                        chunksize=search_config.SEARCH_VECTOR_CHUNK,
                    )
                )
        else:
            results = [
                self.space.compute_trajectory_data(traj, start, use_LIReC=self.use_LIReC)
                for traj, start in missing_pairs
            ]

        for sd in results:
            if sd is not None:
                self.data_manager[sd.sv] = sd

    def _vectorized_crossover(
            self, elite_count: int, children_needed: int, elite_matrix: np.ndarray, next_pop_matrix: np.ndarray
    ) -> np.ndarray:
        # Vectorized Crossover: Randomly pick parents from the elite pool
        parent1_indices = np.random.randint(0, elite_count, size=children_needed)
        parent2_indices = np.random.randint(0, elite_count, size=children_needed)

        parents1 = elite_matrix[parent1_indices]
        parents2 = elite_matrix[parent2_indices]

        # Create a uniform crossover mask across the entire matrix instantly
        crossover_mask = np.random.random(size=(children_needed, self.space.dim)) < self.crossover_prob
        children = np.where(crossover_mask, parents1, parents2)

        # Bridge Phase 2: Execute High-Speed Numba Mutation
        mutated_children = _batch_mutate_population(
            children,
            mutation_prob=self.mutation_prob,
            max_step=self.mutation_step,
            refine_prob=self.refine_prob,
            refine_coord_prob=self.refine_coord_prob
        )

        next_pop_matrix[elite_count:] = mutated_children
        return next_pop_matrix

    def _evaluate_population(
        self,
        population: List[Dict[str, Any]],
        *,
        start: Position,
        template_pos: Position,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate pending individuals and batch-resample invalid deltas.
        :param population: Population entries with trajectory, delta, and SearchData payload.
        :param start: Shared start point for this population evaluation.
        :param template_pos: Template trajectory used when repairs/resampling are required.
        :return: Population with updated delta/sd fields and repaired trajectories.
        """
        for ind in population:
            ind["trajectory"] = self._repair_trajectory(ind["trajectory"], template_pos)

        to_eval_indices = [i for i, ind in enumerate(population) if ind["delta"] is None]
        if not to_eval_indices:
            return population

        pairs = [(population[i]["trajectory"], start) for i in to_eval_indices]
        self._compute_missing_search_data(pairs)

        invalid_indices: List[int] = []
        for i in to_eval_indices:
            traj = population[i]["trajectory"]
            sv = SearchVector(start, traj)
            sd = self.data_manager.get(sv)
            delta = _delta_from_search_data(sd)

            if delta == INVALID_DELTA:
                population[i]["delta"] = INVALID_DELTA
                population[i]["sd"] = sd
                invalid_indices.append(i)
            else:
                population[i]["delta"] = delta
                population[i]["sd"] = sd

        unresolved = invalid_indices
        unchanged_count = 0
        last_found_amount = -1

        for _ in range(self.max_retries):
            if not unresolved:
                break

            if unchanged_count >= search_config.GA_MAX_NO_IMPROVEMENT_COUNT_RETRY:
                Logger(
                    "Genetic algorithm solving unresolved trajectories - giving up resampling.", Logger.Levels.debug
                ).log()
                break

            retry_trajectories = self._sample_valid_trajectories(count=len(unresolved), template_pos=template_pos)
            if len(retry_trajectories) == 0:
                Logger("No valid trajectories could be sampled")

            if last_found_amount == -1:
                last_found_amount = len(retry_trajectories)
            elif last_found_amount == len(retry_trajectories):
                unchanged_count += 1

            retry_pairs = [(traj, start) for traj in retry_trajectories]
            self._compute_missing_search_data(retry_pairs)

            next_unresolved: List[int] = []
            for i, new_traj in zip(unresolved, retry_trajectories):
                new_sd = self.data_manager.get(SearchVector(start, new_traj))
                if new_sd is None:
                    new_sd = self.space.compute_trajectory_data(new_traj, start, use_LIReC=self.use_LIReC)
                    if new_sd is not None:
                        self.data_manager[new_sd.sv] = new_sd

                new_delta = _delta_from_search_data(new_sd)
                if new_delta != INVALID_DELTA:
                    population[i]["trajectory"] = new_traj
                    population[i]["delta"] = new_delta
                    population[i]["sd"] = new_sd
                else:
                    next_unresolved.append(i)

            unresolved = next_unresolved
        return population

    def search(
        self,
        starts: Optional[Position | List[Position]] = None,
        *,
        template_trajectory: Optional[Position] = None,
    ) -> DataManager:
        """
        Run the GA loop and return all evaluated vectors in the data manager.
        :param starts: Optional start point(s) for trajectory evaluation.
        :param template_trajectory: Optional explicit template trajectory for initialization.
        :return: DataManager populated with trajectory evaluation results.
        """
        start_point = self._resolve_start(starts)
        template = self._resolve_template(template_trajectory)

        initial_trajectories = self._sample_valid_trajectories(count=self.pop_size, template_pos=template)
        if len(initial_trajectories) == 0:
            Logger("No valid trajectories could be sampled. Continue...", Logger.Levels.warning).log()
            return DataManager(self.use_LIReC)

        population: List[Dict[str, Any]] = [
            {"trajectory": traj, "delta": None, "sd": None} for traj in initial_trajectories
        ]

        last_delta = None
        unchanged_count = 0
        unchanged_threshold = 1e-7
        max_unchanged_count = max(0.1 * self.generations, 5)

        for _ in SmartTQDM(range(self.generations), desc="Evolving...", **sys_config.TQDM_CONFIG):
            population = self._evaluate_population(population, start=start_point, template_pos=template)
            population.sort(key=lambda ind: ind["delta"], reverse=True)

            if last_delta is not None and abs(population[0]["delta"] - last_delta) < unchanged_threshold:
                unchanged_count += 1
            last_delta = population[0]["delta"]

            elite_count = max(1, int(self.elite_fraction * self.pop_size))
            elites = population[:elite_count]

            elite_positions = [ind["trajectory"] for ind in elites]
            elite_matrix = _positions_to_matrix(elite_positions, self.canonical_keys)

            next_pop_matrix = np.zeros((self.pop_size, self.space.dim), dtype=np.int64)
            next_pop_matrix[:elite_count] = elite_matrix  # Carry over elites

            if (children_needed := self.pop_size - elite_count) > 0:
                next_pop_matrix = self._vectorized_crossover(
                    elite_count, children_needed, elite_matrix, next_pop_matrix
                )

            next_positions = _matrix_to_positions(next_pop_matrix, self.canonical_keys)

            # Rebuild the population dictionary and apply constraint repairs
            next_population: List[Dict[str, Any]] = []
            for i in range(self.pop_size):
                if i < elite_count:
                    # Elites bypass repair and retain their cached evaluation
                    next_population.append({
                        "trajectory": next_positions[i],
                        "delta": elites[i]["delta"],
                        "sd": elites[i]["sd"]
                    })
                else:
                    # Mutated children must be repaired against the shard boundaries
                    repaired_pos = self._repair_trajectory(next_positions[i], template)
                    next_population.append({"trajectory": repaired_pos, "delta": None, "sd": None})

            population = next_population
            if max_unchanged_count <= unchanged_count:
                Logger(
                    f'Stopping search after {unchanged_count} unchanged generations ...', Logger.Levels.debug
                ).log()
                break

        # Final evaluation of the last generation
        population = self._evaluate_population(population, start=start_point, template_pos=template)
        population.sort(key=lambda ind: ind["delta"], reverse=True)
        return self.data_manager
