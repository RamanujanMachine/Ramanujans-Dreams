from dataclasses import dataclass, field
from typing import Callable, Optional
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
    return max(min(round(1500 / max(traj_len / math.sqrt(dim), 1)), 1500), 1000)

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

    GLOBAL_SEED: Optional[int] = field(
        default=42,
        metadata={"description": (
            "Master RNG seed for ALL samplers and search methods (see "
            "dreamer.utils.rand).  Every stochastic unit of work derives an "
            "independent, reproducible stream from (GLOBAL_SEED, shard_id, "
            "method[, constant]) via numpy.random.SeedSequence.  Set to None for "
            "nondeterministic (OS-entropy) runs.  Propagated to worker processes "
            "via config export, but randomness only ever runs in the main process."
        )},
    )

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
    IDENTIFY_DEPTH: int = field(
        default=1000,
        metadata={"description": "Walk depth at which the p/q integer relation is identified via "
                  "LIReC. The relation is depth-independent, so it is found once at this (cheap) "
                  "depth and reused for the deeper delta / spectral computations, avoiding a "
                  "redundant deep identification walk (LIReC and the walk feeding it get expensive "
                  "at large depth). Capped by the actual walk depth; falls back to the full walk "
                  "depth if identification fails here (slow-converging trajectory), so results are "
                  "unchanged -- only faster. Not part of the Tier-1 config fingerprint for that "
                  "reason."},
    )
    COMPUTE_EIGEN_VALUES: bool = field( # deprecated
        default=False,
        metadata={"description": "Compute eigenvalue diagnostics for trajectory matrices in search results."},
    )
    COMPUTE_GCD_SLOPE: bool = field( # deprecated
        default=False,
        metadata={"description": "Compute gcd-slope diagnostics for search trajectories."},
    )
    COMPUTE_LIMIT: bool = field( # deprecated
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
            #("companion_coboundary_rank", "if_identified"), #("asymptotics", "if_identified"),
            ("delta_prediction", "if_identified"),
            ("gcd_slope", "if_identified"), ("error_formula_ratio", "if_identified"),
            ("approximated_digits_per_step", "if_identified"), ("digits_approximation", "if_identified"),
            ("convergence_rate", "if_identified"),
            ("digits_computed", "if_identified"), ("avg_computed_digits_per_step", "if_identified"),
        ),
        metadata={"description": "Background-worker attributes computed asynchronously during search. Empty disables the worker/writer subprocesses entirely."},
    )

    USE_DELTA_PREDICTION: bool = field(
        default=False,
        metadata={"description": (
            "When True, use delta_prediction (eigenvalue-based) as the primary ranking "
            "metric stored in the `delta` column for analysis and search, instead of the "
            "regular walk-based delta.  The regular delta is always computed first because "
            "it is needed to select the best eigenvalue pair for delta_prediction; both "
            "values are computed, only the ranking metric changes."
        )},
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
        default="log",
        metadata={"description": "Cooling schedule type: 'linear' (T0/(k+1)) or 'log' (T0/log(k+1))."},
    )
    ANNEAL_MAX_ITERS: int = field(
        default=500,
        metadata={"description": "Maximum number of accepted moves (primary stop condition). Primary termination criterion; Tmin is a secondary safety net."},
    )
    ANNEAL_MAX_DOUBLINGS: int = field(
        default=50,
        metadata={"description": "Cap on consecutive trajectory length-doublings on rejection before reseeding. Reference effectively uses infinity; 50 keeps reseeding rare while acting as a safety net."},
    )
    ANNEAL_MAX_TOTAL_STEPS: int = field(
        default=10_000,
        metadata={"description": "Hard ceiling on total while-loop iterations (accepted + rejected). Exists as a safety net for stalls; fires in seconds/minutes rather than hours so the shard is abandoned quickly when stuck."},
    )
    ANNEAL_MAX_RESEEDS: int = field(
        default=5,
        metadata={"description": "Hard cap on consecutive failed _try_reseed calls (returning None) per SA run. After this many consecutive failures the run terminates, preventing the PT sampler being called indefinitely."},
    )
    ANNEAL_TABU_SIZE: int = field(
        default=70,
        metadata={"description": "Maximum number of recent positions kept in the tabu list (reference: 14*5)."},
    )
    ANNEAL_RESERVOIR_SIZE: int = field(
        default=10,
        metadata={"description": "Number of initial candidate trajectories sampled for the SA seed selection."},
    )
    ANNEAL_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating the neighbour batch in each SA step. 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Shared trajectory-length cap (all search methods) ==============================
    SEARCH_MAX_TRAJ_LEN: float = field(
        default=60.0,
        metadata={"description": "Maximum real-space trajectory norm applied by all search methods (SA, GA, Gradient Ascent). Trajectories/neighbours/genomes exceeding this bound are skipped or resampled. Bounding trajectory length directly bounds trajectory_matrix() symbolic cost. Interpretation controlled by SEARCH_TRAJ_NORM."},
    )
    SEARCH_TRAJ_NORM: str = field(
        default="l2",
        metadata={"description": "Norm used to measure trajectory length for the SEARCH_MAX_TRAJ_LEN cap, shared by all search methods. 'linf' = max absolute coordinate (tightest bound on trajectory_matrix cost), 'l1' = sum of abs coords (exact symbolic mult count), 'l2' = Euclidean norm."},
    )

    # ============================== Discrete Micro-Hill-Climb finalization (all search methods) ==============================
    # Optional post-search assurance endgame applied to the best-delta trajectory
    # (or trajectories, on a tie up to 2 decimal places) of EVERY search method
    # (Gradient Ascent, Hybrid SPSA, Simulated Annealing, Genetic, Small Angle).
    # Phase A is the existing 2*d_flat orthogonal +-1 lattice hill-climb (the
    # discrete local-maximum certificate); Phase B subdivides the angular
    # resolution by treating ``2^j z +- e_i`` (j = 1..K) as continuous directional
    # probes, re-snapping each into a primitive in-cone ray via snap_to_trajectory,
    # and re-climbing around any superior interstitial ray.  Doubling continues
    # purely until the max-length (SEARCH_MAX_TRAJ_LEN) resolution is reached
    # (K = ceil(log2(SEARCH_MAX_TRAJ_LEN / |primitive ray|))); the final level's
    # probe is projected back down to the nearest in-cone max-length ray by
    # snap_to_trajectory.  No round-count knob — the resolution IS the bound.
    ENABLE_MICRO_HILL_CLIMB: bool = field(
        default=True,
        metadata={"description": "Enable the discrete micro-hill-climb finalization (resolution-doubling endgame) after every search method completes, run on the best-delta trajectory(ies) of each shard/constant. False => no finalization, byte-identical legacy behaviour."},
    )

    # ============================== Genetic search — parallelism ==============================
    GA_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating GA population batches (initial population and per-generation children). 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Gradient Ascent settings ==============================
    # Gradient *Ascent* over the continuous trajectory-direction angle (larger delta is
    # better).  delta is continuous and generally smooth in the angle, so the optimizer
    # works in a real-valued direction space; each updated direction is realized as the
    # angle-best integer trajectory whose norm does not exceed SEARCH_MAX_TRAJ_LEN.
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
        default=1000,
        metadata={"description": "Safety ceiling on gradient-ascent steps per constant. The ascent normally stops earlier when no improving lattice move exists (the snapped step cannot reach a new in-cone trajectory) or delta plateaus for GRAD_PATIENCE steps; this bound only guards against a pathologically long productive climb."},
    )
    GRAD_PATIENCE: int = field(
        default=3,
        metadata={"description": "Consecutive non-improving steps tolerated before early-stopping the ascent."},
    )
    GRAD_IMPROVE_THRESHOLD: float = field(
        default=1e-3,
        metadata={"description": "Minimum delta gain counted as an improvement during the ascent."},
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

    # ===================== Hybrid SPSA + Adam Ascent settings =====================
    # SPSA (Simultaneous Perturbation Stochastic Approximation) macro-navigation over
    # the continuous flatland direction, fed into the same Adam optimizer used by
    # Gradient Ascent, with a discrete 2D-orthogonal-neighbour micro-navigation
    # fallback when the continuous search stalls on a lattice plateau / loops.
    # Only TWO δ-evaluations per macro step (d ± c_k·Δ) regardless of dimension,
    # versus D / 2D for forward / central differences.  See
    # dreamer/search/methods/gradient_ascent/spsa_adam_ascent.py.
    SPSA_C0: float = field(
        default=0.2,
        metadata={"description": "Initial SPSA perturbation magnitude c_0 (radians). Decays as c_k = c_0 / (k+1)^SPSA_GAMMA and is floored at the lattice min-angle (sin θ ≈ 1/L²) so the two probes never snap to the same integer trajectory (which would give a zero gradient)."},
    )
    SPSA_GAMMA: float = field(
        default=0.101,
        metadata={"description": "SPSA perturbation decay exponent γ in c_k = c_0 / (k+1)^γ (classic SPSA default 0.101). 0.0 keeps c_k constant at c_0 (still floored)."},
    )
    SPSA_LR: float = field(
        default=0.5,
        metadata={"description": "Learning rate (step scale) applied to the Adam update before it is added to the continuous direction and snapped to the lattice."},
    )
    SPSA_BETA1: float = field(
        default=0.9,
        metadata={"description": "Adam first-moment decay (beta1) for the SPSA macro-navigation. Adam momentum acts as a low-pass filter on the noisy SPSA gradient."},
    )
    SPSA_BETA2: float = field(
        default=0.999,
        metadata={"description": "Adam second-moment decay (beta2) for the SPSA macro-navigation."},
    )
    SPSA_EPSILON: float = field(
        default=1e-8,
        metadata={"description": "Numerical-stability epsilon in the Adam denominator for the SPSA macro-navigation."},
    )
    SPSA_MAX_STEPS: int = field(
        default=1000,
        metadata={"description": "Safety ceiling on SPSA macro-navigation steps per constant. The macro phase normally ends earlier on a resolution-derived stall (Adam step below the lattice min-angle, loop, out-of-cone, or unidentified probe); this bound only guards against a pathologically long productive climb. Either way the run always finishes with the discrete ±1-neighbour local-maximum certificate."},
    )
    SPSA_LOOP_WINDOW: int = field(
        default=10,
        metadata={"description": "Length of the visited-trajectory history window for loop detection. If Adam revisits an integer trajectory already in this window (momentum oscillating at a peak), a stall is forced and the discrete fallback fires."},
    )
    SPSA_PROBE_RETRIES: int = field(
        default=4,
        metadata={"description": "Number of fresh Rademacher Δ vectors tried when an SPSA probe (d ± c_k·Δ) cannot be realised in-cone or is not identified, before the macro step is treated as a stall and the discrete fallback fires."},
    )
    SPSA_IMPROVE_FALLBACK: float = field(
        default=1e-5,
        metadata={"description": "Minimum δ gain for an orthogonal neighbour to count as a *strict* improvement in the discrete fallback. A neighbour must beat the current δ by more than this to be accepted; otherwise the current point is declared the discrete local maximum."},
    )
    SPSA_RESERVOIR_SIZE: int = field(
        default=10,
        metadata={"description": "Number of initial candidate trajectories sampled for SPSA seed selection (shortest identified trajectory wins)."},
    )
    SPSA_NUM_EVAL_WORKERS: int = field(
        default=0,
        metadata={"description": "Worker cap for evaluating the 2D-orthogonal-neighbour batch in the discrete fallback. 0/None = use the full core budget (search_worker_budget); a positive value caps at min(value, budget). A resolved count <= 1 runs serial."},
    )

    # ============================== Raycaster settings ==============================
    MAX_TRAJECTORY_LENGTH: int = field(
        default=60,
        metadata={"description": "Upper bound for absolute trajectory coordinate values during search."},
    )

    MAX_SEARCH_RADIUS: int = field(
        default=10_000,
        metadata={"description": "Upper bound for search radius used to sample trajectories."},
    )

    SAMPLING_METHOD: str = field(
        default="pt",
        metadata={
            "description": (
                "Trajectory-sampling engine used by ShardSamplingOrchestrator: "
                "'raycast' (continuous guide-ray + raycast pipeline, "
                "RaycastPipelineSampler), 'discrete' (DiscreteMCMCSampler: a "
                "repulsive / PID-annealed discrete lattice walk), or 'pt' "
                "(default; ParallelTemperingSampler: replica-exchange lattice walk that "
                "beats the single chain in tightly constrained cones).  The "
                "'discrete' / 'pt' engines harvest primitive integer directions "
                "with original-space norm <= MAX_TRAJECTORY_LENGTH."
            )
        },
    )

    CONSTANT_NO_DIGITS_HIGH_RES: int = field(
        default=10_000,
        metadata={"description": "Number of digits to use for high-resolution constant values."},
    )

    CONSTANT_NO_DIGITS_LOW_RES: int = field(
        default=1000,
        metadata={"description": "Number of digits to use for low-resolution constant values."},
    )

    MAX_CONSTANT_RESOLUTION: int = field(
        default=200_000,
        metadata={"description": "Maximum number of digits to use for constant values in delta computation."},
    )


search_config: SearchConfig = SearchConfig()
