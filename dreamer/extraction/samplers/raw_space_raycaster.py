"""Raw-space (un-conditioned) MCMC baseline sampler — a deliberate negative control.

This is a benchmarking *baseline* for :mod:`dreamer.extraction.samplers.discrete_raycaster`:
it runs the **same** discrete repulsive / PID-annealed walk, but **without** the
:class:`HyperSpaceConditioner`.  Concretely it sets ``Z = I`` (identity) so the walk
happens directly in the original lattice ``Z^{d_orig}``, and the equality constraints
``E v = 0`` are checked *manually* inside the kernel alongside the inequalities ``B v < 0``.

The point is empirical: on any shard with equality constraints, a ``+-1`` (or any small)
integer step in raw space almost never preserves ``E v = 0`` exactly, so essentially
every proposal is rejected → an acceptance rate near **0.00%** and (near-)zero harvest.
That is the proof of *why* conditioning (mapping into the integer null-space of ``E`` via
``Z``) is mandatory: it makes every lattice move automatically satisfy the equalities.

The njit helpers (``_gcd_abs``, ``_scale_jump``) and the standardized error types are
imported from :mod:`discrete_raycaster`; that module is left untouched.
"""

import numpy as np
import scipy.optimize as opt
from numba import njit

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


def _separate_constraints(A_prime):
    """Split ``A_prime`` rows into equality (``E``) and inequality (``B``) matrices.

    Mirrors ``HyperSpaceConditioner._extract_constraints`` (standalone, so the baseline
    truly avoids the conditioner): a row is an equality iff it cannot be made strictly
    positive while every row stays ``<= 0`` (i.e. it is pinned to ``= 0`` on the cone).

    :param A_prime: ``(rows, d_orig)`` constraint matrix.
    :return: ``(E, B)`` — the equality and inequality row matrices.
    """
    A = np.asarray(A_prime, dtype=np.float64)
    m, d = A.shape
    eq_rows, ineq_rows = [], []
    for i in range(m):
        res = opt.linprog(-A[i], A_ub=-A, b_ub=np.zeros(m), bounds=(-1, 1), method="highs")
        if res.success and -res.fun < 1e-7:
            eq_rows.append(A[i])
        else:
            ineq_rows.append(A[i])
    E = np.array(eq_rows, dtype=np.float64) if eq_rows else np.empty((0, d))
    B = np.array(ineq_rows, dtype=np.float64) if ineq_rows else np.empty((0, d))
    return E, B


@njit(cache=True)
def _raw_mcmc_walk(
    E, B, v0,
    quota, max_steps,
    initial_lambda, gamma,
    target_yield_ratio, learning_rate, min_gravity_floor,
    monitor_window, repulsion_subset,
    max_useful_norm, flatland_box,
    tol, rng_seed,
):
    """Single-chain discrete walk in **raw** space ``Z^{d_orig}`` (no conditioning).

    Identical dynamics to :func:`discrete_raycaster._mcmc_walk` (mixture proposal,
    adaptive scale-jump, raw gravity + repulsion energy, two-phase log-ratio PID,
    useful-band harvest) except: there is no ``Z`` (the lattice *is* the original space),
    so a proposal must satisfy **both** ``|E v'| <= tol`` (equalities, checked manually)
    and ``B v' < -tol`` (strict interior).

    :param E: ``(p, d_orig)`` equality rows; a move is rejected unless ``|E v'| <= tol``.
    :param B: ``(m, d_orig)`` inequality rows; a move is rejected unless ``B v' < -tol``.
    :param v0: ``(d_orig,)`` strict-interior integer seed (``E v0 = 0``, ``B v0 < 0``).
    :param quota: target number of useful primitive harvested vectors.
    :param max_steps: hard cap on chain steps.
    :param initial_lambda: starting/ceiling gravity weight.
    :param gamma: repulsion weight.
    :param target_yield_ratio: desired useful-yield-per-step the PID targets.
    :param learning_rate: PID aggressiveness in ``lam *= exp(lr * log_error)``.
    :param min_gravity_floor: floor on ``lam``.
    :param monitor_window: steps per PID update window.
    :param repulsion_subset: max past harvests sampled for the cosine penalty.
    :param max_useful_norm: only states with ``||v|| <= this`` are harvested.
    :param flatland_box: hard ``max|v_i|`` bound on lateral wandering.
    :param tol: tolerance for the equality (``|E v|``) and strict-interior (``B v``) tests.
    :param rng_seed: if ``>= 0``, seeds numba's RNG for reproducibility.
    :return: ``(harvest_buffer, harvest_count, accept_rate)`` — buffer is
        ``(quota, d_orig)`` int64; ``accept_rate`` is accepted/proposed over the run.
    """
    if rng_seed >= 0:
        np.random.seed(rng_seed)

    d = E.shape[1] if E.shape[0] > 0 else B.shape[1]
    p = E.shape[0]
    m = B.shape[0]

    harvest = np.zeros((quota, d), dtype=np.int64)
    harvest_unit = np.zeros((quota, d), dtype=np.float64)
    harvest_count = 0
    total_proposed = 0
    total_accepted = 0

    v = v0.astype(np.int64).copy()
    norm_v = 0.0
    for i in range(d):
        norm_v += v[i] * v[i]
    norm_v = np.sqrt(norm_v)

    lam = initial_lambda
    window_yield = 0
    current_max_stride = 10
    v_prop = np.zeros(d, dtype=np.int64)

    for step in range(max_steps):
        total_proposed += 1
        for k in range(d):
            v_prop[k] = v[k]
        is_scale_jump = False
        r = np.random.rand()
        if r < 0.60:                       # axis-aligned +-e_i
            a = np.random.randint(d)
            v_prop[a] += 1 if np.random.rand() < 0.5 else -1
        elif r < 0.85:                     # diagonal +-e_i +-e_j
            a = np.random.randint(d)
            b = np.random.randint(d)
            v_prop[a] += 1 if np.random.rand() < 0.5 else -1
            v_prop[b] += 1 if np.random.rand() < 0.5 else -1
        elif r < 0.95:                     # discrete scale jump (adaptive stride)
            dim = np.random.randint(d)
            v_prop[dim] += _scale_jump(current_max_stride)
            is_scale_jump = True
        else:                              # local box jump U{-2..2}^d
            for k in range(d):
                v_prop[k] += np.random.randint(-2, 3)

        # ---- Flatland cage ----
        maxabs = 0
        for k in range(d):
            a = v_prop[k]
            if a < 0:
                a = -a
            if a > maxabs:
                maxabs = a
        if maxabs > flatland_box:
            if is_scale_jump and current_max_stride > 2:
                current_max_stride -= 1
            continue

        # ---- Equality constraints E v' = 0 (checked manually; this is the wall) ----
        eq_ok = True
        for row in range(p):
            acc = 0.0
            for k in range(d):
                acc += E[row, k] * v_prop[k]
            if acc > tol or acc < -tol:
                eq_ok = False
                break
        if not eq_ok:
            if is_scale_jump and current_max_stride > 2:
                current_max_stride -= 1
            continue

        # ---- Strict interior B v' < -tol ----
        inside = True
        for row in range(m):
            acc = 0.0
            for k in range(d):
                acc += B[row, k] * v_prop[k]
            if acc >= -tol:
                inside = False
                break
        if not inside:
            if is_scale_jump and current_max_stride > 2:
                current_max_stride -= 1
            continue

        norm_prop = 0.0
        for k in range(d):
            norm_prop += v_prop[k] * v_prop[k]
        norm_prop = np.sqrt(norm_prop)
        if norm_prop < 1e-12:
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
                for k in range(d):
                    u = harvest_unit[idx, k]
                    dp += v_prop[k] * u
                    dc += v[k] * u
                dp /= norm_prop
                if norm_v > 1e-12:
                    dc /= norm_v
                if dp > s_prop:
                    s_prop = dp
                if dc > s_cur:
                    s_cur = dc

        # ---- Metropolis acceptance: raw gravity + repulsion ----
        score_cur = lam * norm_v + gamma * s_cur
        score_prop = lam * norm_prop + gamma * s_prop
        diff = score_cur - score_prop
        accept = diff >= 0.0
        if not accept:
            if np.random.rand() < np.exp(diff):
                accept = True

        if is_scale_jump:
            if accept:
                if current_max_stride < 50:
                    current_max_stride += 1
            elif current_max_stride > 2:
                current_max_stride -= 1

        if accept:
            total_accepted += 1
            for k in range(d):
                v[k] = v_prop[k]
            norm_v = norm_prop

            if norm_prop <= max_useful_norm:
                g = _gcd_abs(v_prop)
                if g == 1:
                    is_new = True
                    if harvest_count > 0:
                        same = True
                        for k in range(d):
                            if harvest[harvest_count - 1, k] != v_prop[k]:
                                same = False
                                break
                        if same:
                            is_new = False
                    if is_new:
                        for k in range(d):
                            harvest[harvest_count, k] = v_prop[k]
                            harvest_unit[harvest_count, k] = v_prop[k] / norm_prop
                        harvest_count += 1
                        window_yield += 1
                        if harvest_count >= quota:
                            break

        # ---- Two-phase log-ratio PID gravity controller ----
        if (step + 1) % monitor_window == 0:
            if norm_v > max_useful_norm:
                lam = initial_lambda
            else:
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


class RawSpaceMCMCSampler(Sampler):
    """Un-conditioned single-chain MCMC baseline (``Z = I``, manual ``E v = 0`` check).

    A negative-control counterpart to :class:`DiscreteMCMCSampler`: same walk, but it
    samples directly in ``Z^{d_orig}`` with no conditioning, checking the equalities
    manually.  Exposes the same diagnostic surface (``d_flat`` == ``d_orig``, ``A_prime``,
    ``B``, ``last_accept_rate``) so it slots straight into the diagnostic harness.
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
        :param initial_lambda: starting/ceiling gravity weight.
        :param gamma: repulsion weight.
        :param target_yield_ratio: desired useful-primitives-per-step the PID targets.
        :param learning_rate: PID aggressiveness in ``lam *= exp(lr * log_error)``.
        :param min_gravity_floor: hard floor on gravity.
        :param monitor_window: steps per PID update window.
        :param repulsion_subset: max past harvests sampled for the cosine penalty.
        :param max_useful_norm: only states with original-space norm ``<= this`` are harvested.
        :param flatland_box: hard ``max|v_i|`` bound on lateral wandering.
        :param seed_bounds: expanding box half-widths for the Chebyshev seed search.
        :param seed_eps: minimum inscribed radius required of the seed.
        :param max_steps_per_quota: chain-length budget = this times the quota.
        :param tol: tolerance for the equality / strict-interior tests.
        :param rng_seed: seed for numba's RNG (and NumPy) for reproducibility; ``<0`` disables.
        """
        self.A_prime = np.asarray(A_prime, dtype=np.float64)
        self.d_orig = int(self.A_prime.shape[1])

        Logger("Initializing RawSpaceMCMCSampler (NO conditioning): separating constraints...",
               Logger.Levels.debug).log()
        # No HyperSpaceConditioner: Z is the identity, the walk is in raw Z^{d_orig}.
        self.E, self.B = _separate_constraints(self.A_prime)
        self.Z = np.eye(self.d_orig, dtype=np.int64)
        self.d_flat = self.d_orig

        # Cone-volume fraction (inequality-only; equalities are measure-zero) — reused
        # from the raycaster for parity with the conditioned samplers' quota scaling.
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
        """Find a strict-interior integer seed in raw space via an expanding Chebyshev MILP.

        Maximises the inscribed radius ``r`` over ``B_i v + ||B_i|| r <= 0`` (inequalities)
        subject to ``E v = 0`` (equalities as hard MILP equality constraints), ``v`` integer
        in an expanding box.  This finds a valid raw-space start even though the walk that
        follows will be unable to *move* off it once equalities are present.

        :return: ``(d_orig,)`` integer seed with ``E v = 0`` and ``B v < 0``.
        :raises NarrowConeError: if no box admits such a point with ``r > seed_eps``.
        """
        d = self.d_orig
        if self.B.shape[0] == 0 and self.E.shape[0] == 0:
            return np.zeros(d, dtype=np.int64)

        constraints = []
        if self.B.shape[0] > 0:
            norms = np.linalg.norm(self.B, axis=1)
            A_ineq = np.hstack([self.B, norms[:, None]])
            constraints.append(opt.LinearConstraint(A_ineq, -np.inf, 0.0))
        if self.E.shape[0] > 0:
            A_eq = np.hstack([self.E, np.zeros((self.E.shape[0], 1))])
            constraints.append(opt.LinearConstraint(A_eq, 0.0, 0.0))

        c = np.zeros(d + 1)
        c[-1] = -1.0  # maximise inscribed radius r
        integrality = np.concatenate([np.ones(d), np.zeros(1)])

        for L in self.seed_bounds:
            lb = np.concatenate([np.full(d, -float(L)), np.zeros(1)])
            ub = np.concatenate([np.full(d, float(L)), np.array([np.inf])])
            try:
                res = opt.milp(c=c, constraints=constraints, integrality=integrality,
                               bounds=opt.Bounds(lb, ub))
            except Exception as exc:  # pragma: no cover - solver availability
                Logger(f"Raw Chebyshev MILP error at box {L} ({exc}).", Logger.Levels.debug).log()
                continue
            if res.success and res.x is not None and res.x[-1] > self.seed_eps:
                v0 = np.round(res.x[:d]).astype(np.int64)
                ineq_ok = self.B.shape[0] == 0 or np.max(self.B @ v0) < -self.tol
                eq_ok = self.E.shape[0] == 0 or np.max(np.abs(self.E @ v0)) <= self.tol
                if np.any(v0 != 0) and ineq_ok and eq_ok:
                    Logger(f"Raw seed found in box ±{L} with inscribed radius {res.x[-1]:.4f}.",
                           Logger.Levels.debug).log()
                    return v0

        raise NarrowConeError(max(self.seed_bounds), self.seed_eps)

    def harvest(self, compute_n_samples, exact: bool = False) -> np.ndarray:
        """Harvest useful primitive integer directions via the raw-space discrete walk.

        :param compute_n_samples: quota as an int (literal), or a callable ``d_flat -> int``
            (scaled by the cone-volume fraction unless ``exact``), matching the other engines.
        :param exact: if ``True`` with a callable, use the requested count as-is.
        :return: ``(n, d_orig)`` array of unique primitive integer vectors; empty array on a
            handled failure (and, for an equality-constrained shard, expected to be empty).
        """
        if callable(compute_n_samples):
            requested = int(compute_n_samples(self.d_flat))
            quota = requested if exact else max(int(requested * self.fraction * 1.05), 5)
        else:
            quota = int(compute_n_samples)
        if quota <= 0 or self.d_flat == 0:
            Logger(
                f"RawSpaceMCMCSampler: nothing to sample (quota={quota}, d_flat={self.d_flat}); "
                f"returning no trajectories.",
                Logger.Levels.debug,
            ).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        try:
            v0 = self._compute_chebyshev_center()
        except NarrowConeError as err:
            Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)

        seed_norm = float(np.linalg.norm(v0))
        Logger(
            f"Raw-space walk: {self.E.shape[0]} equality + {self.B.shape[0]} inequality rows, "
            f"seed norm {seed_norm:.1f}.",
            Logger.Levels.debug,
        ).log()

        max_steps = max(self.monitor_window, quota * self.max_steps_per_quota)

        harvest, count, accept_rate = _raw_mcmc_walk(
            self.E, self.B, v0.astype(np.int64),
            quota, max_steps,
            self.initial_lambda, self.gamma,
            self.target_yield_ratio, self.learning_rate, self.min_gravity_floor,
            self.monitor_window, self.repulsion_subset,
            self.max_useful_norm, self.flatland_box,
            self.tol, self.rng_seed,
        )
        self.last_accept_rate = float(accept_rate)
        Logger(
            f"Raw-space walk acceptance rate: {accept_rate * 100:.2f}% ({count} useful harvested).",
            Logger.Levels.debug,
        ).log()

        if count == 0:
            try:
                raise NoUsefulPointsError(self.max_useful_norm, max_steps, seed_norm)
            except NoUsefulPointsError as err:
                Logger(str(err), Logger.Levels.debug).log()
            return np.empty((0, self.d_orig), dtype=np.int64)
        return np.unique(harvest[:count], axis=0)
