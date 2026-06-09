"""Parallel-Tempering (Replica-Exchange) MCMC sampler for primitive integer cone vectors.

An **architecture branch** of :mod:`dreamer.extraction.samplers.discrete_raycaster`:
same conditioning, mixture proposal, adaptive scale-jump, gravity funnel, two-phase
log-ratio PID, and useful-band harvest filter — but instead of a single chain it runs
``N_replicas`` chains at different temperatures and periodically **swaps** adjacent
states (replica exchange).  This is designed to beat the single-chain failure mode where
the walker is trapped behind topological "energy mountains" in highly constrained 15D
cones that LLL/BKZ (``beta=25``) cannot flatten.

Why tempering helps (see ``context/sampling_trajectories/SAMPLING_MATH.md`` §12):

* A **temperature ladder** ``beta_ladder`` scales each replica's energy.  Replica 0
  (``beta=1``) is the cold **Harvester** feeling full gravity; the last replica
  (``beta=0``) is the **Free Explorer** feeling *zero* gravity — it wanders the cone
  bounded only by the flatland cage and walls, sailing over barriers the harvester
  cannot cross.
* Every ``swap_interval`` steps, adjacent replicas attempt a Metropolis **swap**, so a
  low-norm state discovered by a hot explorer can "trickle down" and teleport the cold
  harvester past a barrier.
* **Cold-chain harvesting:** ONLY the cold harvester (replica 0, ``beta=1``) banks
  points, so hot explorers cannot pollute the quota with high-norm vectors before the
  inner shell is mined.  The explorers are strict auxiliaries — they still propose,
  evaluate, and swap (a low-norm state they find trickles down to replica 0 via a swap,
  and is harvested only once replica 0 holds it).  The PID tracks this cold-chain yield.

The helper njit kernels (``_gcd_abs``, ``_scale_jump``) and the standardized error types
are imported from :mod:`discrete_raycaster` — that module is intentionally left
untouched so the single-chain sampler cannot regress.
"""

import numpy as np
import scipy.optimize as opt
from numba import njit

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.sampler import Sampler
from dreamer.extraction.samplers.discrete_raycaster import (
    _gcd_abs,
    _scale_jump,
    NarrowConeError,
    NoUsefulPointsError,
)
from dreamer.utils.logger import Logger
from dreamer.utils.rand import GLOBAL_SEED


@njit(cache=True)
def _pt_mcmc_walk(
    Z, B, z0, v0, beta_ladder,
    quota, max_steps, swap_interval,
    initial_lambda, gamma,
    target_yield_ratio, learning_rate, min_gravity_floor,
    monitor_window, repulsion_subset,
    max_useful_norm, flatland_box,
    burn_in_steps, initial_harvest_limit, harvest_expansion_factor,
    tol, rng_seed,
):
    """Run a parallel-tempering replica-exchange walk; return the cold-chain harvest.

    Maintains ``n_rep = len(beta_ladder)`` independent states.  Each step proposes one
    move per replica (mixture proposal + per-replica adaptive scale-jump) and accepts it
    under the **tempered** energy ``beta_i * (lam * ||Z z|| + gamma * maxcos)``; only the
    cold harvester (replica 0) banks useful primitives, so hot explorers act purely as
    auxiliaries.  Every ``swap_interval`` steps adjacent replicas attempt a Metropolis
    state swap.  ``lam`` is the *base* gravity, retuned by a two-phase log-ratio PID on
    the cold-chain yield (using the cold harvester's
    norm for the phase decision), then scaled per replica by ``beta_ladder``.

    Two harvest-quality refinements gate replica 0's banking:

    * **Burn-in:** once replica 0 enters the useful band it must spend ``burn_in_steps``
      steps thermalising (gravity locked high so it settles at the bottom) before it may
      bank *any* point — avoids recording the first transient high-norm hits.
    * **Adaptive radius:** banking starts at a tight ``initial_harvest_limit`` and only
      relaxes (``*= harvest_expansion_factor``, capped at ``max_useful_norm``) when a
      monitor window banks nothing — so a fat cone stays greedy at the low limit while a
      needle auto-expands until it reaches its (lowest available) integer points.

    :param Z: ``(d_orig, d_flat)`` integer basis of the equality solution lattice.
    :param B: ``(m, d_flat)`` facet normals; strict interior iff ``B z < 0``.
    :param z0: ``(d_flat,)`` strict-interior integer seed (shared start for all replicas).
    :param v0: ``Z @ z0`` original-space seed.
    :param beta_ladder: ``(n_rep,)`` temperature coefficients, descending from 1.0 to 0.0.
    :param quota: target number of useful primitive vectors harvested by the cold chain.
    :param max_steps: hard cap on outer chain steps (each does ``n_rep`` proposals).
    :param swap_interval: attempt adjacent replica swaps every this many steps.
    :param initial_lambda: starting/ceiling base gravity weight.
    :param gamma: repulsion weight.
    :param target_yield_ratio: desired cold-chain useful-yield-per-step the PID targets.
    :param learning_rate: PID aggressiveness in ``lam *= exp(lr * log_error)``.
    :param min_gravity_floor: floor on ``lam``.
    :param monitor_window: steps per PID update window.
    :param repulsion_subset: max past harvests sampled for the cosine penalty.
    :param max_useful_norm: the Phase-1/Phase-2 boundary (on replica 0) and the hard
        ceiling the adaptive harvest limit may expand to.
    :param flatland_box: hard ``max|z_i|`` bound on lateral flatland wandering.
    :param burn_in_steps: in-band steps replica 0 must thermalise before banking begins.
    :param initial_harvest_limit: starting (tight) norm limit for banking; expands on stall.
    :param harvest_expansion_factor: multiplicative relaxation of the harvest limit per
        stalled monitor window (>= 1).
    :param tol: feasibility tolerance; a move is rejected unless ``B z' < -tol``.
    :param rng_seed: if ``>= 0``, seeds numba's RNG for reproducibility.
    :return: ``(harvest_buffer, harvest_count, accept_rate)`` — buffer is
        ``(quota, d_orig)`` int64; ``accept_rate`` is accepted/proposed over all replicas.
    """
    if rng_seed >= 0:
        np.random.seed(rng_seed)

    d_flat = Z.shape[1]
    d_orig = Z.shape[0]
    m = B.shape[0]
    n_rep = beta_ladder.shape[0]

    harvest = np.zeros((quota, d_orig), dtype=np.int64)
    harvest_unit = np.zeros((quota, d_orig), dtype=np.float64)
    harvest_count = 0
    total_proposed = 0
    total_accepted = 0

    # Per-replica state (each row is one replica's current point).
    z_curr = np.zeros((n_rep, d_flat), dtype=np.int64)
    v_curr = np.zeros((n_rep, d_orig), dtype=np.float64)
    norm_curr = np.zeros(n_rep, dtype=np.float64)
    stride = np.zeros(n_rep, dtype=np.int64)          # per-replica adaptive scale-jump cap

    seed_norm = 0.0
    for k in range(d_orig):
        seed_norm += v0[k] * v0[k]
    seed_norm = np.sqrt(seed_norm)
    for i in range(n_rep):
        for k in range(d_flat):
            z_curr[i, k] = z0[k]
        for k in range(d_orig):
            v_curr[i, k] = v0[k]
        norm_curr[i] = seed_norm
        stride[i] = 10

    lam = initial_lambda
    window_yield = 0
    burn_in_counter = 0                       # in-band steps replica 0 has thermalised
    # never bank above the useful ceiling, even if the configured start limit exceeds it
    current_harvest_limit = min(initial_harvest_limit, max_useful_norm)

    # Scratch buffers (reused every proposal / swap to avoid per-step allocation).
    z_prop = np.zeros(d_flat, dtype=np.int64)
    v_prop = np.zeros(d_orig, dtype=np.float64)
    swap_z = np.zeros(d_flat, dtype=np.int64)
    swap_v = np.zeros(d_orig, dtype=np.float64)

    done = False
    for step in range(max_steps):
        # Burn-in: count steps the cold harvester spends inside the useful band; banking
        # is withheld until it has thermalised for burn_in_steps (settles at the bottom).
        if norm_curr[0] <= max_useful_norm:
            burn_in_counter += 1

        # ---------------- Independent per-replica steps ----------------
        for i in range(n_rep):
            total_proposed += 1
            for k in range(d_flat):
                z_prop[k] = z_curr[i, k]

            # ---- Mixture proposal (symmetric -> no Hastings term) ----
            is_scale_jump = False
            r = np.random.rand()
            if r < 0.60:                       # axis-aligned +-e_i
                a = np.random.randint(d_flat)
                z_prop[a] += 1 if np.random.rand() < 0.5 else -1
            elif r < 0.85:                     # diagonal +-e_i +-e_j
                a = np.random.randint(d_flat)
                b = np.random.randint(d_flat)
                z_prop[a] += 1 if np.random.rand() < 0.5 else -1
                z_prop[b] += 1 if np.random.rand() < 0.5 else -1
            elif r < 0.95:                     # discrete scale jump (adaptive stride)
                dim = np.random.randint(d_flat)
                z_prop[dim] += _scale_jump(stride[i])
                is_scale_jump = True
            else:                              # local box jump U{-2..2}^d
                for k in range(d_flat):
                    z_prop[k] += np.random.randint(-2, 3)

            # ---- Flatland cage ----
            maxabs = 0
            for k in range(d_flat):
                a = z_prop[k]
                if a < 0:
                    a = -a
                if a > maxabs:
                    maxabs = a
            if maxabs > flatland_box:
                if is_scale_jump and stride[i] > 2:
                    stride[i] -= 1
                continue

            # ---- Strict interior (B z' < -tol) ----
            inside = True
            for row in range(m):
                acc = 0.0
                for k in range(d_flat):
                    acc += B[row, k] * z_prop[k]
                if acc >= -tol:
                    inside = False
                    break
            if not inside:
                if is_scale_jump and stride[i] > 2:
                    stride[i] -= 1
                continue

            # ---- Original-space image + norm ----
            for ii in range(d_orig):
                acc = 0.0
                for k in range(d_flat):
                    acc += Z[ii, k] * z_prop[k]
                v_prop[ii] = acc
            norm_prop = 0.0
            for ii in range(d_orig):
                norm_prop += v_prop[ii] * v_prop[ii]
            norm_prop = np.sqrt(norm_prop)
            if norm_prop < 1e-12:              # skip the origin
                if is_scale_jump and stride[i] > 2:
                    stride[i] -= 1
                continue

            # ---- Repulsion vs a random subset of the (cold-chain) harvest ----
            s_prop = 0.0
            s_cur = 0.0
            if harvest_count > 0:
                n_sub = repulsion_subset
                if n_sub > harvest_count:
                    n_sub = harvest_count
                norm_i = norm_curr[i]
                for _ in range(n_sub):
                    idx = np.random.randint(harvest_count)
                    dp = 0.0
                    dc = 0.0
                    for ii in range(d_orig):
                        u = harvest_unit[idx, ii]
                        dp += v_prop[ii] * u
                        dc += v_curr[i, ii] * u
                    dp /= norm_prop
                    if norm_i > 1e-12:
                        dc /= norm_i
                    if dp > s_prop:
                        s_prop = dp
                    if dc > s_cur:
                        s_cur = dc

            # ---- Tempered Metropolis acceptance ----
            beta_i = beta_ladder[i]
            score_cur = beta_i * (lam * norm_curr[i] + gamma * s_cur)
            score_prop = beta_i * (lam * norm_prop + gamma * s_prop)
            diff = score_cur - score_prop
            accept = diff >= 0.0
            if not accept:
                if np.random.rand() < np.exp(diff):
                    accept = True

            # ---- Adapt this replica's scale-jump stride ----
            if is_scale_jump:
                if accept:
                    if stride[i] < 50:
                        stride[i] += 1
                elif stride[i] > 2:
                    stride[i] -= 1

            if accept:
                total_accepted += 1
                for k in range(d_flat):
                    z_curr[i, k] = z_prop[k]
                for ii in range(d_orig):
                    v_curr[i, ii] = v_prop[ii]
                norm_curr[i] = norm_prop

                # ---- Cold-chain harvest filter: ONLY the cold harvester (replica 0,
                # beta=1.0) banks points, after burn-in, and only inside the *adaptive*
                # radius (which starts tight and expands on stall).  Hot replicas (i > 0)
                # still propose, evaluate, and swap — they are strictly auxiliaries. ----
                if i == 0 and burn_in_counter > burn_in_steps and norm_prop <= current_harvest_limit:
                    v_int = np.zeros(d_orig, dtype=np.int64)
                    for ii in range(d_orig):
                        v_int[ii] = np.int64(np.round(v_prop[ii]))
                    g = _gcd_abs(v_int)
                    if g == 1:
                        is_new = True
                        if harvest_count > 0:
                            same = True
                            for ii in range(d_orig):
                                if harvest[harvest_count - 1, ii] != v_int[ii]:
                                    same = False
                                    break
                            if same:
                                is_new = False
                        if is_new:
                            for ii in range(d_orig):
                                harvest[harvest_count, ii] = v_int[ii]
                                harvest_unit[harvest_count, ii] = v_prop[ii] / norm_prop
                            harvest_count += 1
                            window_yield += 1
                            if harvest_count >= quota:
                                done = True
                                break
        if done:
            break

        # ---------------- Replica-exchange swap move ----------------
        if (step + 1) % swap_interval == 0:
            for i in range(n_rep - 1):
                j = i + 1
                # Energies without the beta multiplier (base gravity only).
                e_i = lam * norm_curr[i]
                e_j = lam * norm_curr[j]
                a_swap = (beta_ladder[i] - beta_ladder[j]) * (e_i - e_j)
                do_swap = a_swap >= 0.0
                if not do_swap:
                    if np.random.rand() < np.exp(a_swap):
                        do_swap = True
                if do_swap:
                    for k in range(d_flat):
                        swap_z[k] = z_curr[i, k]
                        z_curr[i, k] = z_curr[j, k]
                        z_curr[j, k] = swap_z[k]
                    for k in range(d_orig):
                        swap_v[k] = v_curr[i, k]
                        v_curr[i, k] = v_curr[j, k]
                        v_curr[j, k] = swap_v[k]
                    tmp = norm_curr[i]
                    norm_curr[i] = norm_curr[j]
                    norm_curr[j] = tmp

        # ---------------- Two-phase log-ratio PID (cold-chain yield) ----------------
        if (step + 1) % monitor_window == 0:
            if norm_curr[0] > max_useful_norm or burn_in_counter <= burn_in_steps:
                # PHASE 1 (funnel) or still thermalising: lock max gravity so the cold
                # harvester descends and settles at the bottom of the cone.
                lam = initial_lambda
            else:
                # PHASE 2 (harvest).  Adaptive radius: if the window banked nothing the
                # cone is too sparse at the current limit, so relax it (capped at the
                # useful-norm ceiling); a fat cone keeps yielding and never expands.
                if window_yield == 0 and current_harvest_limit < max_useful_norm:
                    current_harvest_limit = min(max_useful_norm,
                                                current_harvest_limit * harvest_expansion_factor)
                # log-ratio PID on the cold-chain yield.
                actual_yield_ratio = window_yield / monitor_window
                epsilon = 1e-5
                yield_ratio = (actual_yield_ratio + epsilon) / target_yield_ratio
                error = np.log(yield_ratio)
                lam = lam * np.exp(learning_rate * error)
            if lam < min_gravity_floor:
                lam = min_gravity_floor
            elif lam > initial_lambda:
                lam = initial_lambda
            window_yield = 0

    accept_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
    return harvest, harvest_count, accept_rate


class ParallelTemperingSampler(Sampler):
    """Replica-exchange MCMC sampler over a shard's conditioned lattice.

    Drop-in alternative to :class:`DiscreteMCMCSampler` for benchmarking: same
    construction (raw constraint matrix ``A_prime``, conditioned once) and same
    diagnostics surface (``d_flat``, ``A_prime``, ``B``, ``last_accept_rate``), but runs
    a temperature ladder of replicas with periodic swaps so hot explorers can carry the
    cold harvester past barriers a single chain cannot cross.
    """

    def __init__(
        self,
        A_prime,
        *,
        beta_ladder=(1.0, 0.1, 0.01, 0.0),
        initial_lambda: float = 0.5,
        gamma: float = 1.0,
        target_yield_ratio: float = 0.01,
        learning_rate: float = 0.5,
        min_gravity_floor: float = 0.05,
        monitor_window: int = 500,
        repulsion_subset: int = 50,
        max_useful_norm: float = 1000.0,
        flatland_box: int = 10000,
        burn_in_steps: int = 2000,
        initial_harvest_limit: float = 50.0,
        harvest_expansion_factor: float = 1.5,
        seed_bounds=(20, 100, 500, 2000, 5000),
        seed_eps: float = 1e-6,
        max_steps_per_quota: int = 200,
        tol: float = 1e-6,
        rng_seed: int = GLOBAL_SEED,
    ):
        """
        :param A_prime: ``(rows, d_orig)`` constraint matrix for the shard.
        :param beta_ladder: descending temperature coefficients (1.0 cold .. 0.0 explorer).
        :param initial_lambda: starting/ceiling base gravity weight.
        :param gamma: repulsion weight.
        :param target_yield_ratio: desired cold-chain useful-primitives-per-step the PID targets.
        :param learning_rate: PID aggressiveness in ``lam *= exp(lr * log_error)``.
        :param min_gravity_floor: hard floor on base gravity.
        :param monitor_window: steps per PID update window.
        :param repulsion_subset: max past harvests sampled for the cosine penalty.
        :param max_useful_norm: useful-norm ceiling — the Phase boundary and the cap the
            adaptive harvest limit may expand to.
        :param flatland_box: hard ``max|z_i|`` bound on lateral flatland wandering.
        :param burn_in_steps: in-band thermalisation steps before replica 0 starts banking.
        :param initial_harvest_limit: tight starting norm limit for banking (expands on stall).
        :param harvest_expansion_factor: per-stalled-window relaxation of the harvest limit.
        :param seed_bounds: expanding box half-widths for the Chebyshev seed search.
        :param seed_eps: minimum inscribed radius required of the seed.
        :param max_steps_per_quota: outer-step budget = this times the quota.
        :param tol: feasibility tolerance for the strict in-cone test.
        :param rng_seed: seed for numba's RNG (and NumPy) for reproducibility; ``<0`` disables.
        """
        self.A_prime = np.asarray(A_prime, dtype=np.float64)
        self.d_orig = int(self.A_prime.shape[1])

        Logger("Initializing ParallelTemperingSampler: Conditioning...", Logger.Levels.debug).log()
        conditioner = HyperSpaceConditioner(self.A_prime, max_beta=25, defect_tolerance=5.0)
        Z_reduced, B_reduced, _ = conditioner.process()

        self.Z = np.asarray(Z_reduced, dtype=np.int64)
        self.B = np.asarray(B_reduced, dtype=np.float64)
        self.d_flat = int(self.Z.shape[1])

        # Cone-volume fraction (Gaussian dart-throw) — reused verbatim from the raycaster
        # so the requested quota scales with the cone's solid angle, exactly as before.
        self.fraction = float(RaycastPipelineSampler._estimate_cone_fraction(
            self.B, self.d_flat,
            samples=min(500_000, max(10_000, 10 ** self.d_flat)),
        ))

        self.beta_ladder = np.asarray(beta_ladder, dtype=np.float64)
        self.initial_lambda = initial_lambda
        self.gamma = gamma
        self.target_yield_ratio = target_yield_ratio
        self.learning_rate = learning_rate
        self.min_gravity_floor = min_gravity_floor
        self.monitor_window = monitor_window
        self.repulsion_subset = repulsion_subset
        self.max_useful_norm = max_useful_norm
        self.flatland_box = flatland_box
        self.burn_in_steps = burn_in_steps
        self.initial_harvest_limit = initial_harvest_limit
        self.harvest_expansion_factor = harvest_expansion_factor
        self.seed_bounds = tuple(seed_bounds)
        self.seed_eps = seed_eps
        self.max_steps_per_quota = max_steps_per_quota
        self.tol = tol
        self.rng_seed = rng_seed
        self.last_accept_rate = 0.0

        super().__init__(self.d_flat)

    def _compute_chebyshev_center(self):
        """Find the fattest strict-interior integer seed via an expanding Chebyshev MILP.

        Identical seed strategy to :meth:`DiscreteMCMCSampler._compute_chebyshev_center`:
        maximise the inscribed radius ``r`` s.t. ``B_i z + ||B_i|| r <= 0`` with ``z``
        integer in an expanding box, so the seed is the fattest lattice point (never a
        vertex) and a far-out needle interior is still reached.

        :return: ``(d_flat,)`` integer seed strictly inside the cone.
        :raises NarrowConeError: if no box admits an integer point with ``r > seed_eps``.
        """
        m, d = self.B.shape
        if m == 0:
            return np.zeros(d, dtype=np.int64)

        norms = np.linalg.norm(self.B, axis=1)
        A_milp = np.hstack([self.B, norms[:, None]])
        constraints = opt.LinearConstraint(A_milp, -np.inf, 0.0)
        c = np.zeros(d + 1)
        c[-1] = -1.0  # maximise the inscribed radius r
        integrality = np.concatenate([np.ones(d), np.zeros(1)])

        for L in self.seed_bounds:
            lb = np.concatenate([np.full(d, -float(L)), np.zeros(1)])
            ub = np.concatenate([np.full(d, float(L)), np.array([np.inf])])
            try:
                res = opt.milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=opt.Bounds(lb, ub),
                )
            except Exception as exc:  # pragma: no cover - solver availability
                Logger(f"Chebyshev MILP error at box {L} ({exc}).", Logger.Levels.debug).log()
                continue
            if res.success and res.x is not None and res.x[-1] > self.seed_eps:
                z0 = np.round(res.x[:d]).astype(np.int64)
                if np.any(z0 != 0) and np.max(self.B @ z0) < -self.tol:
                    Logger(
                        f"Seed found in box ±{L} with inscribed radius {res.x[-1]:.4f}.",
                        Logger.Levels.debug,
                    ).log()
                    return z0

        raise NarrowConeError(max(self.seed_bounds), self.seed_eps)

    def harvest(self, compute_n_samples, exact: bool = False) -> np.ndarray:
        """Harvest useful primitive integer directions via parallel-tempering MCMC.

        :param compute_n_samples: quota as an int (taken literally), or a callable
            ``d_flat -> int``.  For a callable the requested count is scaled by the
            cone-volume fraction (``requested * fraction * 1.05``, floor 5) — the same
            volume-dependent quota the raycaster uses — unless ``exact`` is set.
        :param exact: if ``True`` with a callable, the requested count is used as-is
            (no volume scaling); the walk targets the quota exactly and stops on reaching it.
        :return: ``(n, d_orig)`` array of unique primitive integer vectors, sorted by
            ascending norm; empty array on a handled failure (logged to the log file).
        """
        if callable(compute_n_samples):
            requested = int(compute_n_samples(self.d_flat))
            # volume-scaled quota, floored at 5 so a near-zero-volume cone still asks for >=5
            quota = requested if exact else max(int(requested * self.fraction * 1.05), 5)
        else:
            quota = int(compute_n_samples)
        if quota <= 0 or self.d_flat == 0:
            Logger(
                f"ParallelTemperingSampler: nothing to sample (quota={quota}, d_flat={self.d_flat}); "
                f"returning no trajectories.",
                Logger.Levels.debug,
            ).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        try:
            z0 = self._compute_chebyshev_center()
        except NarrowConeError as err:
            Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        v0 = self.Z @ z0
        seed_norm = float(np.linalg.norm(v0))
        Logger(
            f"PT gravity funnel: {len(self.beta_ladder)} replicas, seed norm {seed_norm:.1f}, "
            f"useful band <= {self.max_useful_norm:.0f}.",
            Logger.Levels.debug,
        ).log()

        # Budget must cover the burn-in *plus* the harvesting walk, or a small (volume-
        # scaled) quota could exhaust max_steps before burn-in even completes.
        max_steps = max(self.monitor_window, self.burn_in_steps + quota * self.max_steps_per_quota)
        # Dynamic swap interval: ~100 swap attempts over the walk regardless of budget
        # (a fixed interval starves swaps when the quota fills in few steps).
        dynamic_swap_interval = max(20, max_steps // 100)

        # Lowest-norm guarantee now comes from the burn-in + adaptive-radius harvesting
        # inside the kernel (replaces the old Top-K over-sample/sort filter).
        harvest, count, accept_rate = _pt_mcmc_walk(
            self.Z, self.B, z0.astype(np.int64), v0.astype(np.int64), self.beta_ladder,
            quota, max_steps, dynamic_swap_interval,
            self.initial_lambda, self.gamma,
            self.target_yield_ratio, self.learning_rate, self.min_gravity_floor,
            self.monitor_window, self.repulsion_subset,
            self.max_useful_norm, self.flatland_box,
            self.burn_in_steps, self.initial_harvest_limit, self.harvest_expansion_factor,
            self.tol, self.rng_seed,
        )
        self.last_accept_rate = float(accept_rate)
        Logger(
            f"PT walk acceptance rate: {accept_rate * 100:.2f}% ({count} useful harvested).",
            Logger.Levels.debug,
        ).log()

        if count == 0:
            try:
                raise NoUsefulPointsError(self.max_useful_norm, max_steps, seed_norm)
            except NoUsefulPointsError as err:
                Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        # Return unique harvested vectors sorted by ascending L2 norm (lowest first).
        unique = np.unique(harvest[:count], axis=0)
        order = np.argsort(np.linalg.norm(unique, axis=1))
        return unique[order]
