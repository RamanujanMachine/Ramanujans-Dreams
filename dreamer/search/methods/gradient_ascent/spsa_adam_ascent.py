"""
Hybrid SPSA + Adam Ascent — trajectory optimisation by ascending δ over the
continuous direction *angle*, using Simultaneous Perturbation Stochastic
Approximation for macro-navigation and a discrete orthogonal-neighbour
hill-climb for micro-navigation.

This is a **separate** search method from :class:`GradientAscentSearch`
(``grad_ascent_scan.py``); that module is left untouched.  Both ascend δ over the
real-valued flatland direction and realise each direction as the angle-best,
length-capped, in-cone integer trajectory (:func:`snap_to_trajectory`).  They
differ in *how the gradient is obtained and how the discrete endgame is played*:

Macro-navigation (SPSA + Adam)
------------------------------
Forward / central finite differences cost ``D`` / ``2D`` δ-evaluations per step
(one symbolic walk each — the dominant cost).  SPSA instead estimates the whole
gradient from **two** evaluations regardless of dimension:

* draw a Rademacher vector ``Δ ∈ {-1, +1}^D``;
* evaluate δ at the realised directions of ``u + c_k·Δ`` and ``u − c_k·Δ``
  (``u`` is the current *unit* direction, ``c_k`` a small angular perturbation);
* form the noisy gradient ``g = (δ⁺ − δ⁻) / (2·c_k) · Δ``  (since every entry of
  ``Δ`` is ``±1``, the element-wise inverse ``Δ⁻¹`` *is* ``Δ``);
* feed ``g`` into the shared :class:`Adam` optimizer — its momentum (β₁) acts as
  a low-pass filter, smoothing the high-variance SPSA gradient across steps.

The Flatland snapping problem
-----------------------------
Continuous angle directions are snapped to an integer lattice, so there is a
minimum angular resolution ("pixel size") below which a perturbation snaps back
to the *same* integer trajectory ``z`` and the measured gradient is exactly
zero.  For a trajectory whose real-space norm is bounded by ``L`` the angle to
flip to a *distinct* lattice direction is roughly ``sin θ ≈ 1/L²``.  We enforce:

* **Constraint 1 — SPSA perturbation floor:** ``c_k ← max(c_k, θ_min)`` so the
  two probes always land on distinct integer trajectories.
* **Constraint 2 — stall detection:** if the L2 norm of the *applied* Adam step
  (``SPSA_LR · update``) drops below ``θ_min``, the continuous iterate can no
  longer move to a new lattice cell → plateau stall.
* **Constraint 3 — loop detection:** a finite history (``SPSA_LOOP_WINDOW``) of
  visited integer trajectories; revisiting one (Adam oscillating at a peak under
  accumulated momentum) forces a stall.

Micro-navigation (discrete local-maximum certificate)
-----------------------------------------------------
The macro phase **always** finishes with the shared discrete ±1-neighbour
hill-climb (:func:`flatland.discrete_local_max.discrete_hill_climb`): Adam state
is dropped (``reset()``) and we evaluate the ``2D`` minimal integer neighbours of
the current trajectory (one coordinate ``±1``), routed through the shard-cone
boundary filter (``A·v ≤ 0``) so invalid shards are never walked, greedily moving
to the strictly-best neighbour until none improves δ — the **true discrete local
maximum** at the lattice resolution.  When the macro phase stalled (plateau /
loop / unidentified) this is the *fallback* that escapes the plateau; when it
merely exhausted its (safety) step budget while still moving, this *confirms* the
result is a genuine ±1 local max rather than wherever the budget cut off.  The
same routine is the local-maximum certificate used by Gradient Ascent.

Output uses the shared :func:`evaluate_in_flatland` Tier-1 DTO pipeline (sink /
walk-reuse cache), identical to the other flatland search methods.
"""

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.discrete_local_max import discrete_hill_climb
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.seed import resolve_injected_seed
from dreamer.search.methods.gradient_ascent.lattice import snap_to_trajectory
from dreamer.search.methods.gradient_ascent.optimizers import Adam
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.rand import derive_rng
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
from dreamer.utils.ui.tqdm_config import SmartTQDM

search_config = config.search


class NoInitialIdentification(Exception):
    """Raised when no reservoir trajectory identifies the constant in a shard."""

    def __init__(self, shard_id: str, constant: Constant):
        """
        :param shard_id: Id of the shard whose reservoir produced no identification.
        :param constant: The constant that could not be seeded.
        """
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Hybrid SPSA Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


class HybridSPSASearch(SearchMethod):
    """Hybrid SPSA + Adam macro-navigation with a discrete-neighbour fallback.

    Single constant per instance, mirroring :class:`GradientAscentSearch`.  Unlike
    that method, an unidentified / stalled region is never fatal: the search
    *always* has a well-defined terminal state — either an SPSA convergence stop
    or the discrete local maximum reached by the orthogonal-neighbour fallback —
    so no ``SearchStalled`` analogue is raised.
    """

    def __init__(self, space: Shard, constant: Constant, use_LIReC: bool = True):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this search optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant
        self._rng = np.random.default_rng()
        #: Diagnostics (populated by :meth:`run`): whether the discrete fallback
        #: fired, and how many SPSA macro steps ran before it did.
        self.used_discrete_fallback: bool = False
        self.macro_steps: int = 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """Standalone entry point — collect emitted DTOs into a list.

        :param starts: Unused; present for the :class:`SearchMethod` interface.
        :return: The list of ``(traj_matrix, const_sympy, dto)`` items emitted.
        """
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
        """Run the hybrid SPSA + Adam ascent for a single constant.

        :param constant: The constant whose δ is maximised.
        :param cmf_id: Structural id of the parent CMF.
        :param shard_id: Structural id of the shard being searched.
        :param shard_encoding_str: ±1 sign-encoding string of the shard.
        :param sink: Callable receiving ``(traj_matrix, const_sympy, dto)`` items.
        :param seen_trajectories: On-disk/in-memory trajectory cache (walk reuse).
        :param handler_cache: Per-shard handler cache for cross-constant walk reuse.
        :param geom: Pre-built :class:`FlatlandGeometry` for the shard (built once
            per shard by the module, shared across constants).  ``None`` builds it
            here (standalone path).
        :param start: Pre-fetched interior start :class:`Position`.  ``None``
            fetches it here.
        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`;
            the discrete-fallback ``2D``-neighbour batch is walked across worker
            processes.  ``None`` evaluates serially.
        :param initial_trajectory: Optional user-supplied seed direction.  Defaults
            to the shard's ``selected_trajectory``.  When valid (a non-zero recession
            direction) it seeds the optimiser instead of the reservoir; an invalid
            one falls back to reservoir seeding (see :func:`resolve_injected_seed`).
        :raises NoInitialIdentification: If no reservoir seed identifies *constant*.
        """
        if handler_cache is None:
            handler_cache = {}

        # Per-(shard, method, constant) reproducible RNG for the Rademacher draws.
        # Same GLOBAL_SEED + shard + constant => identical run; distinct shards/
        # constants get independent streams (nondeterministic when GLOBAL_SEED is
        # None).  Overrides the unseeded generator created in __init__.
        self._rng = derive_rng(shard_id, "spsa_adam", constant.name)

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

        cfg = search_config
        max_norm = cfg.SEARCH_MAX_TRAJ_LEN

        # --- Lattice "pixel size": the smallest angle that can flip a length-≤L
        # trajectory to a distinct integer direction (sin θ ≈ 1/L²).  Both the
        # SPSA perturbation floor (Constraint 1) and the stall threshold
        # (Constraints 2 & 3) are measured against this.
        min_angle = self._min_lattice_angle(max_norm)

        # --- Seed: user-supplied trajectory if valid, else shortest identified
        # reservoir trajectory --------------------------------------------------
        if initial_trajectory is None:
            initial_trajectory = getattr(shard, "selected_trajectory", None)
        cur_z = resolve_injected_seed(
            geom, initial_trajectory, shard_id, constant,
            identify_fn=lambda z: evaluate_in_flatland(z, **eval_ctx)[1],
        )
        if cur_z is None:
            cur_z = self._select_seed(geom, eval_ctx, shard_id, constant)
        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
        best_delta = cur_delta
        d = cur_z.astype(np.float64)  # continuous SPSA iterate (direction)

        optimizer = Adam(
            geom.d_flat,
            beta1=cfg.SPSA_BETA1,
            beta2=cfg.SPSA_BETA2,
            epsilon=cfg.SPSA_EPSILON,
        )

        # Loop-detection history of recently visited integer trajectories.
        history: Deque[Tuple[int, ...]] = deque(maxlen=cfg.SPSA_LOOP_WINDOW)
        history.append(tuple(int(v) for v in cur_z))

        Logger(
            f"SPSA macro-navigation start — shard {shard_id}, constant "
            f"{constant.name}: c0={cfg.SPSA_C0:.4g}, min_angle={min_angle:.4g}, "
            f"L={max_norm}.",
            Logger.Levels.info,
        ).log()

        stalled = False
        self.used_discrete_fallback = False
        self.macro_steps = 0
        for k in SmartTQDM(
            range(cfg.SPSA_MAX_STEPS), desc="SPSA ascending ... ", **config.system.TQDM_CONFIG
        ):
            self.macro_steps = k + 1
            # Constraint 1 — perturbation floor.  Decay c_k, then clamp so the two
            # probes can never collapse onto the same integer trajectory.
            c_k = max(cfg.SPSA_C0 / (k + 1) ** cfg.SPSA_GAMMA, min_angle)

            grad = self._spsa_gradient(d, c_k, eval_ctx, geom, max_norm)
            if grad is None:
                # All probe retries failed to land on identified, in-cone pairs:
                # the continuous search has no usable gradient here → stall.
                Logger(
                    f"SPSA probe stall — shard {shard_id}, constant {constant.name}: "
                    f"no identified ± perturbation pair after {cfg.SPSA_PROBE_RETRIES} "
                    f"retries at step {k}.",
                    Logger.Levels.debug,
                ).log()
                stalled = True
                break

            # Adam update (momentum low-pass filters the SPSA noise).
            update = optimizer.step(grad)
            step_vec = cfg.SPSA_LR * update
            step_norm = float(np.linalg.norm(step_vec))

            # Constraint 2 — stall detection: the applied step is below the lattice
            # resolution, so the iterate can no longer reach a new integer cell.
            if step_norm < min_angle:
                Logger(
                    f"SPSA plateau stall — shard {shard_id}, constant {constant.name}: "
                    f"Adam step norm {step_norm:.3g} < min_angle {min_angle:.3g} at "
                    f"step {k}.",
                    Logger.Levels.debug,
                ).log()
                stalled = True
                break

            d_new = d + step_vec
            z_new = snap_to_trajectory(d_new, geom, max_norm, cfg.SEARCH_TRAJ_NORM)
            if z_new is None:
                # The continuous step left the cone at every realisable length.
                stalled = True
                break

            d = d_new
            z_key = tuple(int(v) for v in z_new)

            # Constraint 3 — loop detection: Adam is oscillating over already-seen
            # integer trajectories (momentum carrying it around the peak).
            if z_key in history:
                Logger(
                    f"SPSA loop detected — shard {shard_id}, constant {constant.name}: "
                    f"revisited trajectory at step {k}; forcing discrete fallback.",
                    Logger.Levels.debug,
                ).log()
                stalled = True
                break
            history.append(z_key)

            delta_new, identified_new = evaluate_in_flatland(z_new, **eval_ctx)
            if not identified_new:
                # Stepped onto a non-identified cell — momentum will keep us in the
                # continuous loop; hand off to the discrete fallback from the last
                # known-good trajectory instead of spinning.
                stalled = True
                break

            cur_z, cur_delta = z_new, delta_new
            if delta_new > best_delta:
                best_delta = delta_new

        # --- Micro-navigation: discrete local-maximum certificate -----------
        # The discrete ±1-neighbour hill-climb ALWAYS runs as the final step,
        # regardless of how the macro phase ended.  When the macro phase *stalled*
        # (plateau / loop / unidentified) it is the fallback that escapes the
        # lattice plateau; when the macro phase merely ran out its SAFETY step
        # budget while still moving, it confirms the result is a genuine ±1 local
        # maximum (the lattice resolution is exhausted) instead of wherever the
        # budget happened to cut off.  Adam state is dropped first either way.
        self.used_discrete_fallback = stalled
        Logger(
            (f"Transition: Adam/SPSA macro-navigation -> 2D-neighbour discrete "
             f"fallback — shard {shard_id}, constant {constant.name} "
             f"(D={geom.d_flat}, {2 * geom.d_flat} neighbours).")
            if stalled else
            (f"SPSA macro budget reached — shard {shard_id}, constant "
             f"{constant.name}: confirming discrete local maximum via the "
             f"{2 * geom.d_flat} ±1 neighbours."),
            Logger.Levels.info,
        ).log()
        optimizer.reset()  # completely drop Adam state before the discrete phase.
        cur_z, cur_delta = discrete_hill_climb(
            cur_z, cur_delta,
            geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
            traj_norm=cfg.SEARCH_TRAJ_NORM, improve_threshold=cfg.SPSA_IMPROVE_FALLBACK,
            pool=pool,
            on_local_max=lambda z, dlt: Logger(
                f"Discrete local maximum reached — shard {shard_id}, constant "
                f"{constant.name}: δ={dlt:.6g} (no improving ±1 neighbour).",
                Logger.Levels.info,
            ).log(),
        )
        best_delta = max(best_delta, cur_delta)

        self.best_delta = best_delta

    # ------------------------------------------------------------------
    # Macro-navigation internals (SPSA)
    # ------------------------------------------------------------------

    @staticmethod
    def _min_lattice_angle(max_norm: float) -> float:
        """Smallest angle that flips a length-≤``L`` trajectory to a distinct lattice
        direction: ``θ_min = arcsin(1/L²)`` (``sin θ ≈ 1/L²``).

        :param max_norm: Trajectory norm cap ``L`` (``SEARCH_MAX_TRAJ_LEN``).
        :return: ``θ_min`` in radians (always finite and > 0).
        """
        L = max(float(max_norm), 1.0)
        return float(np.arcsin(min(1.0, 1.0 / (L * L))))

    def _spsa_gradient(
        self,
        d: np.ndarray,
        c_k: float,
        eval_ctx: dict,
        geom: FlatlandGeometry,
        max_norm: float,
    ) -> Optional[np.ndarray]:
        """Estimate ∇δ from a single simultaneous perturbation (two δ-evaluations).

        Draws a Rademacher ``Δ``, realises ``u ± c_k·Δ`` (``u`` the unit current
        direction) as integer trajectories, evaluates both, and returns
        ``g = (δ⁺ − δ⁻) / (2·c_k) · Δ``.  A draw is *rejected* (and re-tried up to
        ``SPSA_PROBE_RETRIES`` times) if either probe cannot be realised in-cone,
        the two probes collapse onto the same trajectory, or either probe is not
        identified — in those cases the finite difference carries no usable signal.

        :param d: Current continuous direction (any non-zero norm; only its angle
            is used).
        :param c_k: Floored SPSA perturbation magnitude for this step.
        :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
        :param geom: Flatland geometry.
        :param max_norm: Trajectory norm cap for snapping.
        :return: The noisy gradient estimate, or ``None`` if every retry failed.
        """
        norm = float(np.linalg.norm(d))
        u = d / norm if norm > 0.0 else d
        traj_norm = search_config.SEARCH_TRAJ_NORM

        for _ in range(max(1, search_config.SPSA_PROBE_RETRIES)):
            # Rademacher Δ ∈ {-1, +1}^D.  Its element-wise inverse is itself.
            delta_vec = self._rng.choice(np.array([-1.0, 1.0]), size=geom.d_flat)

            z_plus = snap_to_trajectory(u + c_k * delta_vec, geom, max_norm, traj_norm)
            z_minus = snap_to_trajectory(u - c_k * delta_vec, geom, max_norm, traj_norm)
            if z_plus is None or z_minus is None or np.array_equal(z_plus, z_minus):
                continue  # collapsed / out-of-cone probe — try a fresh Δ.

            delta_p, ident_p = evaluate_in_flatland(z_plus, **eval_ctx)
            delta_m, ident_m = evaluate_in_flatland(z_minus, **eval_ctx)
            if not (ident_p and ident_m):
                continue  # an unidentified probe gives no usable δ difference.

            g = (delta_p - delta_m) / (2.0 * c_k) * delta_vec
            return g.astype(np.float64)

        return None

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _select_seed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
    ) -> np.ndarray:
        """Pick the shortest reservoir trajectory (ascending L2 norm) that identifies.

        :param geom: Flatland geometry.
        :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
        :param shard_id: Structural shard id.
        :param constant: The constant to seed.
        :return: The flatland direction of the first identifying reservoir trajectory.
        :raises NoInitialIdentification: If no sampled trajectory identifies the constant.
        """
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.SPSA_RESERVOIR_SIZE
        )
        candidates: List[Tuple[float, Position]] = []
        for t in trajectories:
            norm = float(np.linalg.norm([float(t[s]) for s in geom.symbols]))
            candidates.append((norm, t))
        candidates.sort(key=lambda pair: pair[0])

        for _, t in candidates:
            z = geom.to_flatland(t)
            if not np.any(z):
                continue
            _, identified = evaluate_in_flatland(z, **eval_ctx)
            if identified:
                return z

        raise NoInitialIdentification(shard_id, constant)
