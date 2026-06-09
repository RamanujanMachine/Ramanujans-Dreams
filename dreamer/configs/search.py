from dataclasses import dataclass, field
from typing import Callable
from .configurable import Configurable
from typing import Tuple
import math


def traj_from_dim(dim: int) -> int:
    """Default number of trajectories to sample for a CMF of dimension ``dim``.

    :param dim: CMF dimensionality.
    :return: Trajectory count (``10 ** dim``).
    """
    return 10 ** dim


def depth_from_len(traj_len, dim) -> int:
    """Default walk depth as a function of trajectory length and dimension.

    :param traj_len: Trajectory length in real shard space.
    :param dim: CMF dimensionality.
    :return: Walk depth, capped at 1500.
    """
    return min(round(1500 / max(traj_len / math.sqrt(dim), 1)), 1500)

# def ga_generations(dim: int) -> int:
#     return 15 + 4 * dim
#
# def ga_population(dim: int) -> int:
#     return 20 + 2 * dim ** 2

def ga_generations(dim: int) -> int:
    """Default genetic-algorithm generation count for dimension ``dim``.

    :param dim: Flatland dimensionality.
    :return: Number of generations (``15 + 3 * dim``).
    """
    return 15 + 3 * dim

def ga_population(dim: int) -> int:
    """Default genetic-algorithm population size for dimension ``dim``.

    :param dim: Flatland dimensionality.
    :return: Population size (``20 + 2 * dim``).
    """
    return 20 + 2 * dim


@dataclass
class SearchConfig(Configurable):
    """Configuration knobs for all search methods (GA, SA, Gradient Ascent,
    Small Angle) and the shared δ-evaluation / trajectory-sampling pipeline."""

    PARALLEL_SEARCH: bool = field(default=True, metadata={"description": "Enable parallel trajectory evaluation where available."})
    SEARCH_VECTOR_CHUNK: int = field(
        default=1,
        metadata={"description": "Number of search vectors batched per parallel chunk."},
    )
    NUM_TRAJECTORIES_FROM_DIM: Callable = field(
        default=traj_from_dim,
        metadata={"description": "Callable mapping CMF dimension to target number of trajectories."},
    )
    DEPTH_FROM_TRAJECTORY_LEN: Callable = field(
        default=depth_from_len,
        metadata={"description": "Callable mapping trajectory length/dimension to maximum walk depth."},
    )
    DEPTH_CONVERGENCE_THRESHOLD: Tuple[float, ...] = field(
        default=(0.9, 0.95, 1.0),
        metadata={"description": "Convergence quality checkpoints used while selecting candidate depths."},
    )
    DEFAULT_USES_INV_T: bool = field(
        default=True,
        metadata={"description": "Default toggle for using inverse-transformed trajectories in search."},
    )

    # ============================== Delta calculation and validation settings ==============================
    LIMIT_DIFF_ERROR_BOUND: float = field(
        default=1e-10,
        metadata={"description": "Maximum absolute limit mismatch accepted when validating convergents."},
    )
    MIN_ESTIMATE_DENOMINATOR: int = field(
        default=1e6,
        metadata={"description": "Minimum denominator magnitude required for a reliable rational estimate."},
    )
    CACHE_ACCEPTANCE_THRESHOLD: float = field(
        default=1e-12,
        metadata={"description": "Tolerance for accepting cached p/q vectors as equivalent."},
    )
    IDENTIFY_CHECK_THRESHOLD: float = field(
        default=1e-10,
        metadata={"description": "Tolerance used when deciding whether a searched trajectory identifies the constant."},
    )

    COMPUTE_EIGEN_VALUES: bool = field(
        default=False,
        metadata={"description": "Compute eigenvalue diagnostics for trajectory matrices in search results."},
    )
    COMPUTE_GCD_SLOPE: bool = field(
        default=False,
        metadata={"description": "Compute gcd-slope diagnostics for search trajectories."},
    )
    COMPUTE_LIMIT: bool = field(
        default=False,
        metadata={"description": "Compute explicit limit approximations during search evaluation."},
    )

    # ============================== Attribute selection (new DTO pipeline) ==============================
    # Names listed here are resolved through
    # ``dreamer.utils.storage.attribute_registry.ATTRIBUTE_REGISTRY``.
    # Misspelled entries raise KeyError loudly.
    #
    # Tier model:
    #   Tier-1 — core DTO fields (delta, identified, limit, order, recurrence_relation,
    #            p/q vectors).  Always computed in the main thread.
    #   Tier-2 — async extras computed in background worker processes during search.
    #            Default is empty so a vanilla run does no extra work beyond Tier-1.
    #   Tier-3 — expensive post-process attributes (asymptotics, kamidelta).  Not
    #            yet implemented; will run as a separate pipeline pass.
    TIER2_ATTRIBUTES: Tuple[str, ...] = field(
        default=(
            ("eigenvalues", "if_identified"), ("eigenvalue_errors", "if_identified"), ("spectral_gap", "if_identified"),
            ("companion_coboundary_rank", "if_identified"), #("asymptotics", "if_identified"),
            ("convergence_class", "if_identified"), ("kamidelta", "if_identified"), ("gcd_slope", "if_identified")
        ),
        metadata={"description": "Background-worker attributes computed asynchronously during search. Empty disables the worker/writer subprocesses entirely."},
    )

    # ============================== Genetic search settings ==============================
    # Number of evolutionary generations to run.
    GA_GENERATIONS: Callable[[int], int] | int = field(
        default=ga_generations,
        metadata={"description": "Genetic algorithm generation schedule as callable or fixed integer."},
    )
    # Number of individuals in each generation.
    GA_POPULATION_SIZE: Callable[[int], int] | int = field(
        default=ga_population,
        metadata={"description": "Genetic algorithm population-size schedule as callable or fixed integer."},
    )
    GA_ELITE_FRACTION: float = field(
        default=0.2,
        metadata={"description": "Fraction of top individuals carried unchanged between generations."},
    )
    GA_MUTATION_PROB: float = field(
        default=0.3,
        metadata={"description": "Per-child probability of applying mutation in genetic search."},
    )
    GA_MUTATION_STEP: int = field(
        default=1,
        metadata={"description": "Maximum coordinate perturbation magnitude for mutation steps."},
    )
    GA_CROSSOVER_PROB: float = field(
        default=0.5,
        metadata={"description": "Probability of crossover versus cloning during offspring creation."},
    )
    GA_MAX_RETRIES: int = field(
        default=3,
        metadata={"description": "Maximum retries when trajectory evaluation fails or produces invalid states."},
    )
    GA_REFINE_PROB: float = field(
        default=0.5,
        metadata={"description": "Probability of entering local-refinement mutation mode (used by legacy GeneticSearchMethod only; GeneticSearch uses the reference 0.7/0.3 asymmetry)."},
    )
    GA_REFINE_COORD_PROB: float = field(
        default=0.5,
        metadata={"description": "Per-coordinate probability for refinement perturbations."},
    )
    GA_MAX_NO_IMPROVEMENT_COUNT_RETRY: int = field(
        default=5,
        metadata={"description": "Retry budget before stopping when no GA improvement is observed."},
    )

    # ============================== Small Angle Search settings ==============================
    SA_MAX_DEPTH: int = field(
        default=50,
        metadata={"description": "Maximum number of small-angle hill-climb iterations (the search depth)."},
    )
    SA_IMPROVE_THRESHOLD: float = field(
        default=1e-3,
        metadata={"description": "Minimum delta gain counted as an improvement during the hill-climb."},
    )
    SA_PATIENCE: int = field(
        default=5,
        metadata={"description": "Consecutive non-improving iterations tolerated before early-stopping the climb."},
    )
    SA_MAX_DOUBLINGS: int = field(
        default=10,
        metadata={"description": "Cap on consecutive trajectory length-doublings when no perturbation stays inside the shard."},
    )
    SA_RESERVOIR_SIZE: int = field(
        default=10,
        metadata={"description": "Number of initial candidate trajectories sampled for small-angle seed selection."},
    )
    SA_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating each hill-climb step's in-cone perturbation batch. 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Simulated Annealing settings ==============================
    ANNEAL_T0: float = field(
        default=1.0,
        metadata={"description": "Initial temperature for simulated annealing cooling schedule."},
    )
    ANNEAL_TMIN: float = field(
        default=1e-4,
        metadata={"description": "Minimum temperature threshold; annealing stops when T drops below this. Lower than T0/(MAX_ITERS+1) means Tmin acts as a safety net rather than the primary stop condition."},
    )
    ANNEAL_SCHEDULE: str = field(
        default="linear",
        metadata={"description": "Cooling schedule type: 'linear' (T0/(k+1)) or 'log' (T0/log(k+1))."},
    )
    ANNEAL_MAX_ITERS: int = field(
        default=500,
        metadata={"description": "Maximum number of accepted moves (primary stop condition). Primary termination criterion; Tmin is a secondary safety net."},
    )
    ANNEAL_MAX_DOUBLINGS: int = field(
        default=10,
        metadata={"description": "Cap on consecutive trajectory length-doublings on rejection before reseeding."},
    )
    ANNEAL_MAX_TOTAL_STEPS: int = field(
        default=50_000,
        metadata={"description": "Hard ceiling on total while-loop iterations (accepted + rejected) to prevent infinite stalls. Should be large enough never to trigger in normal operation; exists as a safety net for tabu-deadlock edge cases."},
    )
    ANNEAL_TABU_SIZE: int = field(
        default=70,
        metadata={"description": "Maximum number of recent positions kept in the tabu list (reference: 14*5)."},
    )
    ANNEAL_RESERVOIR_SIZE: int = field(
        default=10,
        metadata={"description": "Number of initial candidate trajectories sampled for the SA seed selection."},
    )
    ANNEAL_MAX_TRAJ_LEN: int = field(
        default=35,
        metadata={"description": "Maximum allowed trajectory length in real shard space. Neighbours exceeding this bound are skipped, preventing expensive trajectory_matrix() calls for large-coordinate directions. Interpretation controlled by ANNEAL_TRAJ_NORM."},
    )
    ANNEAL_TRAJ_NORM: str = field(
        default="linf",
        metadata={"description": "Norm used to measure trajectory length for the ANNEAL_MAX_TRAJ_LEN cap. 'linf' = max absolute coordinate (default; directly bounds trajectory_matrix cost), 'l1' = sum of absolute coords (= exact trajectory_matrix symbolic mult count), 'l2' = Euclidean norm."},
    )
    ANNEAL_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating the neighbour batch in each SA step. 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Genetic search — trajectory cap + parallelism ==============================
    GA_MAX_TRAJ_LEN: int = field(
        default=35,
        metadata={"description": "Maximum allowed trajectory length for GA genomes in real shard space. Genomes and neighbours exceeding this bound are rejected/resampled. Interpretation controlled by GA_TRAJ_NORM."},
    )
    GA_TRAJ_NORM: str = field(
        default="linf",
        metadata={"description": "Norm used to measure trajectory length for the GA_MAX_TRAJ_LEN cap. Same options as ANNEAL_TRAJ_NORM: 'linf', 'l1', 'l2'."},
    )
    GA_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating GA population batches (initial population and per-generation children). 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Gradient Ascent settings ==============================
    # Gradient *Ascent* over the continuous trajectory-direction angle (larger delta is
    # better).  delta is continuous and generally smooth in the angle, so the optimizer
    # works in a real-valued direction space; each updated direction is realized as the
    # angle-best integer trajectory whose L2 norm does not exceed GRAD_MAX_NORM.
    GRAD_VARIANT: str = field(
        default="adam",
        metadata={"description": "Gradient-ascent optimizer variant: 'vanilla' | 'momentum' | 'rmsprop' | 'adam'."},
    )
    GRAD_LR: float = field(
        default=1.0,
        metadata={"description": "Learning rate (step scale) applied to the optimizer update before snapping to the lattice."},
    )
    GRAD_MOMENTUM: float = field(
        default=0.9,
        metadata={"description": "Momentum coefficient (beta) for the 'momentum' variant."},
    )
    GRAD_BETA1: float = field(
        default=0.9,
        metadata={"description": "First-moment decay (beta1) for the Adam variant."},
    )
    GRAD_BETA2: float = field(
        default=0.999,
        metadata={"description": "Second-moment decay (beta2) for the RMSprop / Adam variants."},
    )
    GRAD_EPSILON: float = field(
        default=1e-8,
        metadata={"description": "Numerical-stability epsilon in the RMSprop / Adam denominator."},
    )
    GRAD_MAX_STEPS: int = field(
        default=50,
        metadata={"description": "Maximum number of gradient-ascent steps per constant (the manual step budget)."},
    )
    GRAD_PATIENCE: int = field(
        default=3,
        metadata={"description": "Consecutive non-improving steps tolerated before early-stopping the ascent."},
    )
    GRAD_IMPROVE_THRESHOLD: float = field(
        default=1e-3,
        metadata={"description": "Minimum delta gain counted as an improvement during the ascent."},
    )
    GRAD_GRAD_TOL: float = field(
        default=1e-4,
        metadata={"description": "Convergence stop: terminate when the estimated gradient L2 norm falls below this (no better step to take)."},
    )
    GRAD_MAX_NORM: float = field(
        default=60.0,
        metadata={"description": "Maximum trajectory length (in real shard space) of a realized integer trajectory direction when snapping a real direction onto the lattice. Interpretation controlled by GRAD_TRAJ_NORM."},
    )
    GRAD_TRAJ_NORM: str = field(
        default="l2",
        metadata={"description": "Norm used to measure trajectory length for the GRAD_MAX_NORM cap, in real shard space. Same options as ANNEAL_TRAJ_NORM: 'linf', 'l1', 'l2' (default 'l2' — gradient ascent reasons about Euclidean direction length)."},
    )
    GRAD_FD_ANGLE: float = field(
        default=0.1,
        metadata={"description": "Finite-difference rotation angle (radians) used to estimate the gradient by forward differences in angle space."},
    )
    GRAD_SKIP_LIMIT: int = field(
        default=3,
        metadata={"description": "Consecutive unproductive (non-identified) steps tolerated by 'skip' before the length-doubling fallback fires."},
    )
    GRAD_MAX_DOUBLINGS: int = field(
        default=2,
        metadata={"description": "Cap on consecutive length-doublings before falling back to diffraction off the unidentified wall."},
    )
    GRAD_DIFFRACT_TRIES: int = field(
        default=5,
        metadata={"description": "Number of random in-cone 'diffraction' directions tried from the last identified trajectory before the shard search is abandoned (SearchStalled)."},
    )
    GRAD_RESERVOIR_SIZE: int = field(
        default=10,
        metadata={"description": "Number of initial candidate trajectories sampled for gradient-ascent seed selection."},
    )
    GRAD_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating the per-step forward-difference gradient probe batch. 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Raycaster settings ==============================
    MAX_TRAJECTORY_LENGTH: int = field(
        default=70,
        metadata={"description": "Upper bound for absolute trajectory coordinate values during search."},
    )

    MAX_SEARCH_RADIUS: int = field(
        default=10_000,
        metadata={"description": "Upper bound for search radius used to sample trajectories."},
    )

    SAMPLING_METHOD: str = field(
        default="raycast",
        metadata={
            "description": (
                "Trajectory-sampling engine used by ShardSamplingOrchestrator: "
                "'raycast' (default; continuous guide-ray + raycast pipeline, "
                "RaycastPipelineSampler), 'discrete' (DiscreteMCMCSampler: a "
                "repulsive / PID-annealed discrete lattice walk), or 'pt' "
                "(ParallelTemperingSampler: replica-exchange lattice walk that "
                "beats the single chain in tightly constrained cones).  The "
                "'discrete' / 'pt' engines harvest primitive integer directions "
                "with original-space norm <= MAX_TRAJECTORY_LENGTH."
            )
        },
    )

    CONSTANT_NO_DIGITS_HIGH_RES: int = field(
        default=50_000,
        metadata={"description": "Number of digits to use for high-resolution constant values."},
    )

    CONSTANT_NO_DIGITS_LOW_RES: int = field(
        default=1000, #600,
        metadata={"description": "Number of digits to use for low-resolution constant values."},
    )


search_config: SearchConfig = SearchConfig()
