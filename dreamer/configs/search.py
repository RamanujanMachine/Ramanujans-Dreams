from dataclasses import dataclass
from typing import Callable
from .configurable import Configurable
from typing import Tuple
import math


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


def depth_from_len(traj_len, dim) -> int:
    return min(round(1500 / max(traj_len / math.sqrt(dim), 1)), 1500)

def ga_generations(dim: int) -> int:
    return 15 + 4 * dim

def ga_population(dim: int) -> int:
    return 20 + 2 * dim ** 2


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = True
    SEARCH_VECTOR_CHUNK: int = 4                # number of search vectors per chunk for parallel search
    NUM_TRAJECTORIES_FROM_DIM: Callable = traj_from_dim
    DEPTH_FROM_TRAJECTORY_LEN: Callable = depth_from_len
    DEPTH_CONVERGENCE_THRESHOLD: Tuple[float, ...] = (0.9, 0.95, 1.0)
    DEFAULT_USES_INV_T: bool = True

    # ============================== Delta calculation and validation settings ==============================
    LIMIT_DIFF_ERROR_BOUND: float = 1e-10           # convergence limit difference thresholds
    MIN_ESTIMATE_DENOMINATOR: int = 1e6             # estimated = a / b (if b is too small, probably didn't converge)
    CACHE_ACCEPTANCE_THRESHOLD: float = 1e-12       # p,q vector cache acceptance threshold
    IDENTIFY_CHECK_THRESHOLD: float = 1e-10

    COMPUTE_EIGEN_VALUES: bool = False
    COMPUTE_GCD_SLOPE: bool = False
    COMPUTE_LIMIT: bool = False

    # ============================== Genetic search settings ==============================
    # Number of evolutionary generations to run.
    GA_GENERATIONS: Callable[[int], int] | int = ga_generations     # for 3D: 27, for 15D: 75
    # Number of individuals in each generation.
    GA_POPULATION_SIZE: Callable[[int], int] | int = ga_population  # for 3D: 38, for 15D: 470
    GA_ELITE_FRACTION: float = 0.2  # Fraction of top individuals kept unchanged each generation.
    GA_MUTATION_PROB: float = 0.3   # Probability to mutate each child.
    GA_MUTATION_STEP: int = 1       # Max mutation step for coordinate updates.
    GA_CROSSOVER_PROB: float = 0.5  # Probability to use crossover instead of cloning.
    GA_MAX_RETRIES: int = 3         # Retry rounds for invalid/failed trajectory evaluations.
    GA_REFINE_PROB: float = 0.5     # Probability of entering refine mutation mode.
    GA_REFINE_COORD_PROB: float = 0.5   # Per-coordinate refine perturbation probability.
    GA_MAX_NO_IMPROVEMENT_COUNT_RETRY: int = 5  # Max retries when no improvement is observed before giving up.

    MAX_TRAJECTORY_COORD: int = 50  # Max coordinate value for a trajectory.


search_config: SearchConfig = SearchConfig()
