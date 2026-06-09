"""Discrete, repulsive, PID-annealed MCMC sampler for primitive integer vectors in a cone.

This is the discrete counterpart to :mod:`dreamer.extraction.samplers.raycaster`
(continuous guide-ray + raycast).  Instead of rounding continuous rays, it runs a
random walk directly on the conditioned flatland integer lattice
``Z^{d_flat}`` and harvests primitive integer directions ``v = Z z``.

Key architecture (see ``context/sampling_trajectories/SAMPLING_MATH.md``):

* **Gravity funnel** — the Chebyshev seed can be far out (cones widen outward; a
  badly-conditioned basis can put it at ``10^8``).  We do **not** cage the walker
  there.  A *flatland* box bounds lateral wandering; raw (un-normalised) gravity
  ``lambda * ||Z z||`` pulls the walker downhill toward the origin via a per-step
  drift; and a point is only *harvested* once its original-space norm enters the
  useful band ``<= max_useful_norm``.
* **Two-phase log-ratio PID** — above the useful band gravity is locked at
  ``initial_lambda`` (funnel); inside it, ``lambda`` is retuned every window by the
  log-ratio of observed useful-yield to target, so it acts at tiny yield scales.
* **Mixture proposal** — axis / diagonal / discrete scale-jump / box, all symmetric;
  the scale-jump stride adapts online to the local acceptance rate.

It is intentionally a *copy* of the raycaster pipeline surface (its own file, its own
class) so the existing production sampler is untouched and cannot regress.
"""

import numpy as np
import scipy.optimize as opt
from numba import njit

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.samplers.sampler import Sampler
from dreamer.utils.logger import Logger
from dreamer.utils.rand import GLOBAL_SEED


class SamplingError(Exception):
    """Base class for recoverable failures of the discrete MCMC sampler."""


class NarrowConeError(SamplingError):
    """Raised when no box admits a strict-interior integer seed (cone too narrow)."""

    def __init__(self, max_box: int, seed_eps: float):
        """
        :param max_box: the largest box half-width that was tried.
        :param seed_eps: the minimum inscribed radius that could not be met.
        """
        self.max_box = max_box
        self.seed_eps = seed_eps
        super().__init__(
            f"DiscreteMCMCSampler: cone interior is too narrow for integer isolation "
            f"(no point with inscribed radius > {seed_eps:g} within box +-{max_box})."
        )


class NoUsefulPointsError(SamplingError):
    """Raised when the walk completes without harvesting any useful primitive point."""

    def __init__(self, max_useful_norm: float, steps: int, seed_norm: float):
        """
        :param max_useful_norm: the original-space norm band the harvest was filtered to.
        :param steps: number of chain steps spent.
        :param seed_norm: the seed's original-space norm (how far the funnel had to drop).
        """
        self.max_useful_norm = max_useful_norm
        self.steps = steps
        self.seed_norm = seed_norm
        super().__init__(
            f"DiscreteMCMCSampler: no primitive point with norm <= {max_useful_norm:g} "
            f"harvested in {steps} steps (seed norm {seed_norm:.1f}). The conditioned "
            f"lattice may have no usable short strictly-interior vectors in this cone."
        )


@njit(cache=True)
def _gcd_abs(vec):
    """Greatest common divisor of the absolute values of an integer vector.

    :param vec: Integer vector.
    :return: gcd of ``|vec[i]|`` (0 only if every entry is 0).
    """
    a = 0
    for i in range(vec.shape[0]):
        b = vec[i]
        if b < 0:
            b = -b
        while b:
            a, b = b, a % b
    return a


@njit(cache=True)
def _scale_jump(current_max_stride):
    """Sample a symmetric scale-jump stride in ``[2, current_max_stride]``.

    The cap ``current_max_stride`` is **adapted online** by the walk from its recent
    accept/reject history (see ``_mcmc_walk``): it grows when scale jumps land inside
    the cone and shrinks when they hit a wall, so the walker "feels" the geometry — long
    strides in open space, short strides in a tight needle — keeping a healthy
    acceptance rate.  The stride is signed symmetrically so the proposal stays reversible.

    :param current_max_stride: current adaptive upper bound on the stride magnitude.
    :return: a signed integer stride with magnitude in ``[2, current_max_stride]``.
    """
    k_val = np.random.randint(2, current_max_stride + 1)
    return k_val if np.random.rand() > 0.5 else -k_val


@njit(cache=True)
def _mcmc_walk(
    Z, B, z0, v0,
    quota, max_steps,
    initial_lambda, gamma,
    target_yield_ratio, learning_rate, min_gravity_floor,
    monitor_window, repulsion_subset,
    max_useful_norm, flatland_box,
    tol, rng_seed,
):
    """Run the discrete repulsive walk with a gravity funnel + two-phase PID controller.

    Acceptance follows ``alpha = min(1, exp(Score(z) - Score(z')))`` with the **raw**
    (un-normalised) energy ``Score(x) = lam * ||Z x|| + gamma * maxcos(Z x, subset)``.
    The raw absolute gravity difference is what gives a real per-step downhill drift
    that funnels the walker from a distant seed into the useful band; ``float64``
    handles the large magnitudes fine.  A state is *harvested* only when
    ``||Z x|| <= max_useful_norm``.

    The controller is **phase-shifted** to resolve the funnel/PID paradox: while the
    walker is still above the useful band (Phase 1, funnel) gravity is *locked* at
    ``initial_lambda`` (maximum tractor beam — never decayed by a zero yield); once
    inside the band (Phase 2, harvest) the log-ratio PID manages exploration vs caging.

    :param Z: ``(d_orig, d_flat)`` integer basis of the equality solution lattice.
    :param B: ``(m, d_flat)`` facet normals; strict interior iff ``B z < 0``.
    :param z0: ``(d_flat,)`` strict-interior integer seed (may be far from origin).
    :param v0: ``Z @ z0`` (original-space seed), passed in to avoid a recompute.
    :param quota: target number of useful primitive harvested vectors.
    :param max_steps: hard cap on chain length.
    :param initial_lambda: starting gravity weight; also the controller ceiling.
    :param gamma: repulsion weight.
    :param target_yield_ratio: desired useful-primitives-per-step the controller targets.
    :param learning_rate: controller aggressiveness (multiplier ``exp(lr * log_error)``).
    :param min_gravity_floor: controller floor on ``lam`` (keeps the downhill pull alive).
    :param monitor_window: number of steps per controller update window.
    :param repulsion_subset: max past harvests sampled for the cosine penalty.
    :param max_useful_norm: only states with ``||Z z|| <= this`` are harvested/counted,
        and the Phase-1/Phase-2 controller boundary.
    :param flatland_box: hard ``max|z_i|`` bound (caps lateral flatland wandering).
    :param tol: feasibility tolerance; a move is rejected unless ``B z' < -tol``.
    :param rng_seed: if ``>= 0``, seeds numba's RNG for reproducibility.
    :return: ``(harvest_buffer, harvest_count, accept_rate)`` — buffer is
        ``(quota, d_orig)`` int64; ``accept_rate`` is accepted/proposed over the run.
    """
    if rng_seed >= 0:
        np.random.seed(rng_seed)

    d_flat = Z.shape[1]
    d_orig = Z.shape[0]
    m = B.shape[0]

    harvest = np.zeros((quota, d_orig), dtype=np.int64)
    harvest_unit = np.zeros((quota, d_orig), dtype=np.float64)
    harvest_count = 0
    total_proposed = 0
    total_accepted = 0

    z = z0.copy()
    v = v0.astype(np.float64)
    norm_v = 0.0
    for i in range(d_orig):
        norm_v += v[i] * v[i]
    norm_v = np.sqrt(norm_v)

    lam = initial_lambda
    window_yield = 0
    current_max_stride = 10            # adaptive scale-jump cap (in [2, 50])
    z_prop = np.zeros(d_flat, dtype=np.int64)
    v_prop = np.zeros(d_orig, dtype=np.float64)

    for step in range(max_steps):
        total_proposed += 1
        # ---- Mixture proposal (all families symmetric -> no Hastings term) ----
        for k in range(d_flat):
            z_prop[k] = z[k]
        is_scale_jump = False
        r = np.random.rand()
        if r < 0.60:                       # axis-aligned +-e_i
            i = np.random.randint(d_flat)
            z_prop[i] += 1 if np.random.rand() < 0.5 else -1
        elif r < 0.85:                     # diagonal +-e_i +-e_j
            i = np.random.randint(d_flat)
            j = np.random.randint(d_flat)
            z_prop[i] += 1 if np.random.rand() < 0.5 else -1
            z_prop[j] += 1 if np.random.rand() < 0.5 else -1
        elif r < 0.95:                     # discrete scale jump (adaptive stride)
            dim = np.random.randint(d_flat)
            z_prop[dim] += _scale_jump(current_max_stride)
            is_scale_jump = True
        else:                              # local box jump U{-2..2}^d
            for k in range(d_flat):
                z_prop[k] += np.random.randint(-2, 3)

        # ---- Flatland cage: bound lateral wandering in z-space ----
        maxabs = 0
        for k in range(d_flat):
            a = z_prop[k]
            if a < 0:
                a = -a
            if a > maxabs:
                maxabs = a
        if maxabs > flatland_box:
            if is_scale_jump and current_max_stride > 2:   # hit a wall -> shrink stride
                current_max_stride -= 1
            continue

        # ---- Hard boundary: strict interior (B z' < -tol on every facet) ----
        inside = True
        for row in range(m):
            acc = 0.0
            for k in range(d_flat):
                acc += B[row, k] * z_prop[k]
            if acc >= -tol:
                inside = False
                break
        if not inside:
            if is_scale_jump and current_max_stride > 2:   # hit a wall -> shrink stride
                current_max_stride -= 1
            continue

        # original-space image + its norm
        for i in range(d_orig):
            acc = 0.0
            for k in range(d_flat):
                acc += Z[i, k] * z_prop[k]
            v_prop[i] = acc
        norm_prop = 0.0
        for i in range(d_orig):
            norm_prop += v_prop[i] * v_prop[i]
        norm_prop = np.sqrt(norm_prop)

        if norm_prop < 1e-12:              # skip the origin
            if is_scale_jump and current_max_stride > 2:
                current_max_stride -= 1
            continue

        # ---- Repulsion: max cosine vs a random subset of the harvest ----
        s_prop = 0.0
        s_cur = 0.0
        if harvest_count > 0:
            n_sub = repulsion_subset
            if n_sub > harvest_count:
                n_sub = harvest_count
            for _ in range(n_sub):
                idx = np.random.randint(harvest_count)
                dp = 0.0
                dc = 0.0
                for i in range(d_orig):
                    u = harvest_unit[idx, i]
                    dp += v_prop[i] * u
                    dc += v[i] * u
                dp /= norm_prop
                if norm_v > 1e-12:
                    dc /= norm_v
                if dp > s_prop:
                    s_prop = dp
                if dc > s_cur:
                    s_cur = dc

        # ---- Metropolis acceptance: raw gravity + repulsion (downhill drift) ----
        score_cur = lam * norm_v + gamma * s_cur
        score_prop = lam * norm_prop + gamma * s_prop
        diff = score_cur - score_prop
        accept = diff >= 0.0
        if not accept:
            if np.random.rand() < np.exp(diff):
                accept = True

        # ---- Adapt the scale-jump stride: grow on accept, shrink on reject ----
        if is_scale_jump:
            if accept:
                if current_max_stride < 50:
                    current_max_stride += 1
            elif current_max_stride > 2:
                current_max_stride -= 1

        if accept:
            total_accepted += 1
            for k in range(d_flat):
                z[k] = z_prop[k]
            for i in range(d_orig):
                v[i] = v_prop[i]
            norm_v = norm_prop

            # ---- Harvest filter: only useful (short) primitive points count ----
            if norm_prop <= max_useful_norm:
                v_int = np.zeros(d_orig, dtype=np.int64)
                for i in range(d_orig):
                    v_int[i] = np.int64(np.round(v_prop[i]))
                g = _gcd_abs(v_int)
                if g == 1:
                    # cheap dedup against the most recent harvest (repulsion covers the rest)
                    is_new = True
                    if harvest_count > 0:
                        same = True
                        for i in range(d_orig):
                            if harvest[harvest_count - 1, i] != v_int[i]:
                                same = False
                                break
                        if same:
                            is_new = False
                    if is_new:
                        for i in range(d_orig):
                            harvest[harvest_count, i] = v_int[i]
                            harvest_unit[harvest_count, i] = v_prop[i] / norm_prop
                        harvest_count += 1
                        window_yield += 1
                        if harvest_count >= quota:
                            break

        # ---- Two-phase gravity controller (every monitor_window steps) ----
        if (step + 1) % monitor_window == 0:
            if norm_v > max_useful_norm:
                # PHASE 1 (funnel): still above the band -> lock max tractor-beam
                # gravity.  Do NOT let a zero yield decay lambda toward the floor.
                lam = initial_lambda
            else:
                # PHASE 2 (harvest): inside the band -> log-ratio PID balances
                # exploration vs caging on the observed useful yield.
                actual_yield_ratio = window_yield / monitor_window
                epsilon = 1e-5
                yield_ratio = (actual_yield_ratio + epsilon) / target_yield_ratio
                error = np.log(yield_ratio)
                lam = lam * np.exp(learning_rate * error)
            # strict bounds always apply
            if lam < min_gravity_floor:
                lam = min_gravity_floor
            elif lam > initial_lambda:
                lam = initial_lambda
            window_yield = 0

    accept_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
    return harvest, harvest_count, accept_rate


class DiscreteMCMCSampler(Sampler):
    """Discrete repulsive/PID-annealed MCMC sampler over a shard's conditioned lattice.

    Drop-in alternative to :class:`RaycastPipelineSampler`: constructed from the raw
    constraint matrix ``A_prime`` (equalities stacked as ``E, -E`` plus inequalities),
    conditions it once, then harvests primitive integer directions via a discrete walk
    that funnels down from a (possibly distant) Chebyshev seed into the useful norm band.
    """

    def __init__(
        self,
        A_prime,
        *,
        initial_lambda: float = 0.5,
        gamma: float = 1.0,
        target_yield_ratio: float = 0.01,
        learning_rate: float = 0.5,
        min_gravity_floor: float = 0.05,
        monitor_window: int = 500,
        repulsion_subset: int = 50,
        max_useful_norm: float = 1000.0,
        flatland_box: int = 10000,
        seed_bounds=(20, 100, 500, 2000, 5000),
        seed_eps: float = 1e-6,
        max_steps_per_quota: int = 200,
        tol: float = 1e-6,
        rng_seed: int = GLOBAL_SEED,
    ):
        """
        :param A_prime: ``(rows, d_orig)`` constraint matrix for the shard.
        :param initial_lambda: starting gravity weight and the controller's ceiling.
        :param gamma: repulsion weight.
        :param target_yield_ratio: desired useful-primitives-per-step the controller targets.
        :param learning_rate: controller aggressiveness in ``lam *= exp(lr * log_error)``.
        :param min_gravity_floor: hard floor on gravity (keeps the downhill funnel pull).
        :param monitor_window: steps per controller update window.
        :param repulsion_subset: max past harvests sampled for the cosine penalty.
        :param max_useful_norm: only states with original-space norm ``<= this`` are harvested.
        :param flatland_box: hard ``max|z_i|`` bound on lateral flatland wandering.
        :param seed_bounds: expanding box half-widths for the Chebyshev seed search.
        :param seed_eps: minimum inscribed radius required of the seed.
        :param max_steps_per_quota: chain-length budget = this times the quota.
        :param tol: feasibility tolerance for the strict in-cone test (tightened to 1e-6).
        :param rng_seed: seed for numba's RNG (and NumPy) for reproducibility; ``<0`` disables.
        """
        self.A_prime = np.asarray(A_prime, dtype=np.float64)
        self.d_orig = int(self.A_prime.shape[1])

        Logger("Initializing DiscreteMCMCSampler: Conditioning...", Logger.Levels.debug).log()
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

        self.initial_lambda = initial_lambda
        self.gamma = gamma
        self.target_yield_ratio = target_yield_ratio
        self.learning_rate = learning_rate
        self.min_gravity_floor = min_gravity_floor
        self.monitor_window = monitor_window
        self.repulsion_subset = repulsion_subset
        self.max_useful_norm = max_useful_norm
        self.flatland_box = flatland_box
        self.seed_bounds = tuple(seed_bounds)
        self.seed_eps = seed_eps
        self.max_steps_per_quota = max_steps_per_quota
        self.tol = tol
        self.rng_seed = rng_seed
        self.last_accept_rate = 0.0

        super().__init__(self.d_flat)

    def _compute_chebyshev_center(self):
        """Find the fattest strict-interior integer seed via an expanding Chebyshev MILP.

        Solves ``max r  s.t.  B_i z + ||B_i|| r <= 0``, ``z`` integer in a box, over an
        expanding ladder of box half-widths.  Maximising the inscribed radius ``r``
        pushes ``z`` to the *fattest* lattice point (never a boundary vertex), and the
        ladder reaches needles whose interior is far from the origin.  The walk's
        gravity funnel later drags that (possibly distant) seed down into the useful band.

        :return: ``(d_flat,)`` integer seed strictly inside the cone.
        :raises NarrowConeError: if no box admits an integer point with ``r > seed_eps``
            (the cone interior is too narrow to isolate an integer).
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
        """Harvest useful primitive integer trajectory directions via the discrete walk.

        :param compute_n_samples: quota as an int (taken literally), or a callable
            ``d_flat -> int``.  For a callable the requested count is scaled by the
            cone-volume fraction (``requested * fraction * 1.05``, floor 5) — the same
            volume-dependent quota the raycaster uses — unless ``exact`` is set.
        :param exact: if ``True`` with a callable, the requested count is used as-is
            (no volume scaling); the walk targets the quota exactly and stops on reaching it.
        :return: ``(n, d_orig)`` array of unique primitive integer vectors, all with
            original-space norm ``<= max_useful_norm``; empty array on a handled failure.
        """
        if callable(compute_n_samples):
            requested = int(compute_n_samples(self.d_flat))
            quota = requested if exact else max(int(requested * self.fraction * 1.05), 5)
        else:
            quota = int(compute_n_samples)
        if quota <= 0 or self.d_flat == 0:
            Logger(
                f"DiscreteMCMCSampler: nothing to sample (quota={quota}, d_flat={self.d_flat}); "
                f"returning no trajectories.",
                Logger.Levels.debug,
            ).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        # Seed search: a too-narrow cone raises NarrowConeError -> log + give up cleanly.
        try:
            z0 = self._compute_chebyshev_center()
        except NarrowConeError as err:
            Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        v0 = self.Z @ z0
        seed_norm = float(np.linalg.norm(v0))
        Logger(
            f"Gravity funnel: seed norm {seed_norm:.1f}, useful band <= {self.max_useful_norm:.0f}.",
            Logger.Levels.debug,
        ).log()

        max_steps = max(self.monitor_window, quota * self.max_steps_per_quota)

        harvest, count, accept_rate = _mcmc_walk(
            self.Z, self.B, z0.astype(np.int64), v0.astype(np.int64),
            quota, max_steps,
            self.initial_lambda, self.gamma,
            self.target_yield_ratio, self.learning_rate, self.min_gravity_floor,
            self.monitor_window, self.repulsion_subset,
            self.max_useful_norm, self.flatland_box,
            self.tol, self.rng_seed,
        )
        # expose the mechanical acceptance rate for diagnostics
        self.last_accept_rate = float(accept_rate)
        Logger(
            f"Walk acceptance rate: {accept_rate * 100:.2f}% ({count} useful harvested).",
            Logger.Levels.debug,
        ).log()

        # Empty harvest: raise the standardized error, catch it, surface the reason.
        if count == 0:
            try:
                raise NoUsefulPointsError(self.max_useful_norm, max_steps, seed_norm)
            except NoUsefulPointsError as err:
                Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)
        return np.unique(harvest[:count], axis=0)
