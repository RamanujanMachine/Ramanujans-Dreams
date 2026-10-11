"""
Gradient Ascent Search — trajectory optimisation by ascending delta over the
continuous direction *angle*.

delta is continuous and generally smooth in the trajectory direction's angle
(non-differentiable only at a finite set of points), so a gradient method over
the real-valued direction space is well-posed.  The optimizer maintains a
real-valued direction ``d``; each updated direction is *realized* as the
angle-best integer trajectory with bounded real-space norm (the shared
``SEARCH_MAX_TRAJ_LEN`` / ``SEARCH_TRAJ_NORM`` cap, via
:func:`snap_to_trajectory`), which is then walked / evaluated.

Design choices:

* The gradient is estimated by **forward differences in angle space**: rotate
  ``d`` by a small angle toward each coordinate axis (:func:`rotate_toward`),
  realize + evaluate, and form ``g_i = (delta_i - base_delta) / angle``.
* Optimizer variants (vanilla / momentum / RMSprop / Adam) are selected via the
  :mod:`optimizers` strategy and ``GRAD_VARIANT``.
* **Convergence stop (resolution-driven):** ``delta`` realized through
  :func:`snap_to_trajectory` is *piecewise-constant* on the integer lattice, so a
  small estimated gradient norm is not a reliable "near optimum" signal (and Adam
  rescales the step regardless of gradient magnitude).  The ascent therefore runs
  until it can no longer make an **improving lattice move** at the current
  resolution — the snapped step cannot reach a new in-cone trajectory
  (``z_new == cur_z`` / ``None``) or ``delta`` plateaus for ``GRAD_PATIENCE``
  steps.  ``GRAD_MAX_STEPS`` is only a safety ceiling.  It never spins on a local
  optimum.  The lattice resolution itself is set by ``SEARCH_MAX_TRAJ_LEN``.
* **Local-maximum certificate:** the ascent always finishes with the shared
  discrete ±1-neighbour hill-climb
  (:func:`flatland.discrete_local_max.discrete_hill_climb`).  Adam combines the
  per-axis probes into a single step, so the continuous loop can stop where a
  coordinate-axis neighbour still improves δ; the certificate climbs any such
  neighbour and guarantees the returned trajectory is a genuine discrete local
  maximum at the resolution.
* **Non-identified handling (three-stage):** skip the offending probe; after
  ``GRAD_SKIP_LIMIT`` unproductive steps, length-double; after
  ``GRAD_MAX_DOUBLINGS`` doublings, *diffract* off the wall by drawing a random
  in-cone direction from the last identified trajectory; if that also fails
  ``GRAD_DIFFRACT_TRIES`` times, ``SearchStalled`` is raised.
* Output uses the modern ``worker_pool`` sink / Tier-1 DTO pipeline.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.discrete_local_max import discrete_hill_climb
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.seed import resolve_injected_seed
from dreamer.search.methods.flatland.parallel_eval import evaluate_batch
from dreamer.search.methods.gradient_ascent.lattice import rotate_toward, snap_to_trajectory
from dreamer.search.methods.gradient_ascent.optimizers import optimizer_for
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
            f"Gradient Ascent Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


class SearchStalled(Exception):
    """Raised when the ascent cannot escape an unidentified region of a shard.

    After exhausting the skip / length-doubling / diffraction recovery ladder,
    no identified trajectory can be reached, so the shard search is abandoned.
    """

    def __init__(self, shard_id: str, constant: Constant, tries: int):
        """
        :param shard_id: Id of the shard whose search stalled.
        :param constant: The constant being searched when the stall occurred.
        :param tries: Number of diffraction attempts that failed before giving up.
        """
        self.shard_id = shard_id
        self.constant = constant
        self.tries = tries
        super().__init__(
            f"Gradient Ascent Search: stalled on an unidentified region of shard "
            f"{shard_id} for '{constant.name}' — {tries} diffraction attempts from the "
            f"last identified trajectory all failed to land inside an identified cell."
        )


class GradientAscentSearch(SearchMethod):
    """Gradient ascent over flatland trajectory directions, single constant."""

    def __init__(self, space: Shard, constant: Constant, use_LIReC: bool = True):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this search optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant
        self._rng = np.random.default_rng()

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
        """Run gradient ascent for a single constant, emitting DTOs to *sink*.

        :param constant: The constant whose δ is maximised.
        :param cmf_id: Structural id of the parent CMF.
        :param shard_id: Structural id of the shard being searched.
        :param shard_encoding_str: ±1 sign-encoding string of the shard.
        :param sink: Callable receiving ``(traj_matrix, const_sympy, dto)`` items.
        :param seen_trajectories: On-disk/in-memory trajectory cache (walk reuse).
        :param handler_cache: Per-shard handler cache for cross-constant walk reuse.
        :raises NoInitialIdentification: If no reservoir seed identifies *constant*.
        :param geom: Pre-built :class:`FlatlandGeometry` for the shard (built once
            per shard by the module, shared across constants).  ``None`` builds it
            here (standalone path).
        :param start: Pre-fetched interior start :class:`Position`.  ``None``
            fetches it here.
        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`;
            the per-step forward-difference probe batch is walked across worker
            processes.  ``None`` evaluates serially.
        :param initial_trajectory: Optional user-supplied seed direction.  Defaults
            to the shard's ``selected_trajectory``.  When valid (a non-zero recession
            direction) it seeds the optimiser instead of the reservoir; an invalid
            one falls back to reservoir seeding (see :func:`resolve_injected_seed`).
        :raises SearchStalled: If the recovery ladder cannot reach an identified
            trajectory after diffraction.
        """
        if handler_cache is None:
            handler_cache = {}

        # Per-(shard, method, constant) reproducible RNG for the diffraction kicks.
        # Same GLOBAL_SEED + shard + constant => identical run; distinct shards/
        # constants get independent streams (nondeterministic when GLOBAL_SEED is
        # None).  Overrides the unseeded generator created in __init__.
        self._rng = derive_rng(shard_id, "gradient", constant.name)

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

        # --- Seed: user-supplied trajectory if valid, else reservoir ----------
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
        last_identified_z = cur_z.copy()
        d = cur_z.astype(np.float64)

        optimizer = optimizer_for(cfg.GRAD_VARIANT, geom.d_flat, cfg)

        skip_count = 0
        doubling_count = 0
        no_improve = 0

        for _ in SmartTQDM(range(cfg.GRAD_MAX_STEPS), desc='Ascending ... ', **config.system.TQDM_CONFIG):
            # --- 1. Estimate the gradient (forward differences in angle) ---
            grad, usable = self._estimate_gradient(d, cur_delta, eval_ctx, geom, pool)

            if usable == 0:
                # Unproductive step (no probe identified) — escalate the ladder.
                cur_z, cur_delta, d, last_identified_z, skip_count, doubling_count = (
                    self._recover(
                        geom, eval_ctx, shard_id, constant,
                        last_identified_z, skip_count, doubling_count, max_norm,
                    )
                )
                optimizer.reset()
                continue

            # --- 2. Optimizer update + lattice realization (backtrack into cone) ---
            update = optimizer.step(grad)
            z_new, d_new = self._stepped_direction(d, update, geom, cfg.GRAD_LR, max_norm)

            if z_new is None or np.array_equal(z_new, cur_z):
                # No better step at lattice resolution / pushed outside the cone:
                # no improving move exists -> stop (convergence), do not spin.
                break

            delta_new, identified_new = evaluate_in_flatland(z_new, **eval_ctx)

            if not identified_new:
                # Landed on a non-identified trajectory — escalate the ladder.
                skip_count += 1
                cur_z, cur_delta, d, last_identified_z, skip_count, doubling_count = (
                    self._recover(
                        geom, eval_ctx, shard_id, constant,
                        last_identified_z, skip_count, doubling_count, max_norm,
                    )
                )
                optimizer.reset()
                continue

            # --- 3. Accept the move ---
            improved = delta_new > cur_delta + cfg.GRAD_IMPROVE_THRESHOLD
            cur_z, cur_delta, d = z_new, delta_new, d_new
            last_identified_z = z_new.copy()
            skip_count = 0
            doubling_count = 0

            if delta_new > best_delta:
                best_delta = delta_new
            no_improve = 0 if improved else no_improve + 1
            if no_improve >= cfg.GRAD_PATIENCE:
                break

        # --- Discrete local-maximum certificate -----------------------------
        # The continuous ascent stops on a resolution-derived signal (no snapped
        # move / δ plateau), but Adam combines the per-axis probes into a single
        # step direction, so it can stop at a point where a coordinate-axis ±1
        # neighbour — not aligned with that step — still improves δ.  Finish with
        # the shared discrete hill-climb so GA returns a genuine ±1 local maximum
        # at the lattice resolution (the honest "resolution exhausted" stop),
        # climbing any improving neighbour the continuous phase left on the table.
        cur_z, cur_delta = discrete_hill_climb(
            cur_z, cur_delta,
            geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
            traj_norm=cfg.SEARCH_TRAJ_NORM, improve_threshold=cfg.GRAD_IMPROVE_THRESHOLD,
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
    # Internals
    # ------------------------------------------------------------------

    def _estimate_gradient(
        self,
        d: np.ndarray,
        base_delta: float,
        eval_ctx: dict,
        geom: FlatlandGeometry,
        pool=None,
    ) -> Tuple[np.ndarray, int]:
        """Estimate ∇δ by forward differences in angle space.

        For each coordinate axis the direction ``d`` is rotated by
        ``GRAD_FD_ANGLE`` toward that axis, realized as an integer trajectory and
        evaluated.  Probes that cannot be realized in-cone or whose trajectory is
        not identified are *skipped* (their gradient component is left at 0).

        The per-axis probes are independent, so with a *pool* the whole probe
        batch is walked across worker processes in one shot; otherwise each probe
        is walked serially.

        :param d: Current real direction.
        :param base_delta: δ at the current (unrotated) direction.
        :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
        :param geom: Flatland geometry.
        :param pool: Optional persistent per-shard process pool for the batch.
        :return: ``(gradient, usable)`` — the estimate and the number of axes that
            yielded a usable (identified, in-cone) probe.
        """
        h = search_config.GRAD_FD_ANGLE
        max_norm = search_config.SEARCH_MAX_TRAJ_LEN
        grad = np.zeros(geom.d_flat, dtype=np.float64)
        usable = 0

        # Realize the in-cone probe for each axis (cheap, vectorised lattice ops).
        probes: List[Tuple[int, np.ndarray]] = []
        for i in range(geom.d_flat):
            d_rot = rotate_toward(d, i, h)
            z_probe = snap_to_trajectory(d_rot, geom, max_norm, search_config.SEARCH_TRAJ_NORM)
            if z_probe is not None:
                probes.append((i, z_probe))

        if not probes:
            return grad, 0

        if pool is not None and len(probes) > 1:
            results = evaluate_batch(
                [z for _, z in probes], eval_ctx=eval_ctx, pool=pool,
            )
        else:
            results = [evaluate_in_flatland(z, **eval_ctx) for _, z in probes]

        for (i, _), (delta_i, identified_i) in zip(probes, results):
            if not identified_i:
                continue
            grad[i] = (delta_i - base_delta) / h
            usable += 1

        return grad, usable

    def _stepped_direction(
        self,
        d: np.ndarray,
        update: np.ndarray,
        geom: FlatlandGeometry,
        lr: float,
        max_norm: float,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Apply the optimizer update with backtracking line-search into the cone.

        Halves the step a few times until the realized integer direction lies
        inside the shard cone.

        :param d: Current real direction.
        :param update: Optimizer update vector.
        :param geom: Flatland geometry.
        :param lr: Base learning rate (step scale).
        :param max_norm: Trajectory norm cap for snapping (``SEARCH_MAX_TRAJ_LEN``).
        :return: ``(z_new, d_new)`` — the realized integer direction (or ``None``
            if no in-cone realization was found) and the real direction used.
        """
        scale = lr
        d_new = d + scale * update
        for _ in range(5):
            z_new = snap_to_trajectory(d_new, geom, max_norm, search_config.SEARCH_TRAJ_NORM)
            if z_new is not None:
                return z_new, d_new
            scale *= 0.5
            d_new = d + scale * update
        return None, d_new

    def _recover(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
        last_identified_z: np.ndarray,
        skip_count: int,
        doubling_count: int,
        max_norm: float,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, int, int]:
        """Advance the non-identified recovery ladder by one stage.

        Stage 1 (skip) is handled by the caller (it simply keeps the previous
        direction).  This method is invoked once the skip budget is spent:

        * while ``doubling_count < GRAD_MAX_DOUBLINGS``: length-double the last
          identified direction (``z*2``, capped to ``max_norm``);
        * otherwise: *diffract* — draw random in-cone directions from the last
          identified trajectory until one is identified.

        :return: ``(cur_z, cur_delta, d, last_identified_z, skip_count,
            doubling_count)`` for the recovered state.
        :raises SearchStalled: If diffraction exhausts ``GRAD_DIFFRACT_TRIES``.
        """
        cfg = search_config

        if skip_count < cfg.GRAD_SKIP_LIMIT:
            # Stage 1 — keep moving in the last identified direction (skip).
            delta, _ = evaluate_in_flatland(last_identified_z, **eval_ctx)
            return (
                last_identified_z, delta, last_identified_z.astype(np.float64),
                last_identified_z, skip_count, doubling_count,
            )

        if doubling_count < cfg.GRAD_MAX_DOUBLINGS:
            # Stage 2 — length-doubling fallback (escape the dead region).
            doubled = last_identified_z * 2
            if geom.traj_norm(doubled, search_config.SEARCH_TRAJ_NORM) <= max_norm and geom.is_inside(doubled):
                delta, identified = evaluate_in_flatland(doubled, **eval_ctx)
                if identified:
                    return (
                        doubled, delta, doubled.astype(np.float64),
                        doubled.copy(), 0, doubling_count + 1,
                    )
            # Doubling left the cone / cap or de-identified: count the doubling and
            # fall back to the last identified direction; the bumped skip_count keeps
            # us on the doubling/diffract path on the next unproductive step.
            delta, _ = evaluate_in_flatland(last_identified_z, **eval_ctx)
            return (
                last_identified_z, delta, last_identified_z.astype(np.float64),
                last_identified_z, cfg.GRAD_SKIP_LIMIT, doubling_count + 1,
            )

        # Stage 3 — diffract off the wall: random in-cone direction from last identified.
        z, delta = self._diffract(geom, eval_ctx, last_identified_z, shard_id, constant, max_norm)
        return z, delta, z.astype(np.float64), z.copy(), 0, 0

    def _diffract(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        last_identified_z: np.ndarray,
        shard_id: str,
        constant: Constant,
        max_norm: float,
    ) -> Tuple[np.ndarray, float]:
        """Draw random in-cone directions near the last identified trajectory.

        Each attempt rotates the last identified direction by a random angle into
        a random plane; the result is realized, checked for cone membership and
        identification.  The first identified hit is returned.

        :raises SearchStalled: If all ``GRAD_DIFFRACT_TRIES`` attempts fail.
        """
        cfg = search_config
        base = last_identified_z.astype(np.float64)
        for _ in range(cfg.GRAD_DIFFRACT_TRIES):
            # Random angular kick: add a random vector scaled to the base length.
            kick = self._rng.standard_normal(geom.d_flat)
            kick /= (np.linalg.norm(kick) or 1.0)
            angle = self._rng.uniform(cfg.GRAD_FD_ANGLE, np.pi / 3.0)
            d_rand = np.cos(angle) * base + np.sin(angle) * np.linalg.norm(base) * kick
            z = snap_to_trajectory(d_rand, geom, max_norm, search_config.SEARCH_TRAJ_NORM)
            if z is None:
                continue
            delta, identified = evaluate_in_flatland(z, **eval_ctx)
            if identified:
                return z, delta
        raise SearchStalled(shard_id, constant, cfg.GRAD_DIFFRACT_TRIES)

    def _select_seed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
    ) -> np.ndarray:
        """Pick the first reservoir trajectory (ascending L2 norm) that identifies.

        :raises NoInitialIdentification: If no sampled trajectory identifies the constant.
        """
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.GRAD_RESERVOIR_SIZE
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
