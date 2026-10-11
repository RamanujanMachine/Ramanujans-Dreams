"""
Small Angle Search — per-shard local hill-climb on the trajectory *direction*.

See ``context/SEARCH_ALGORITHMS.md``.  The method:

0. Conditions the shard into flatland space (:class:`FlatlandGeometry`).
1. Draws a small reservoir of candidate trajectories from the shard sampler,
   sorted by ascending L2 norm (start close to the origin), and uses the first
   one that **identifies** the constant as the climb seed.  If none identifies,
   :class:`NoInitialIdentification` is raised (caught by the module).
2. Computes δ for the current direction (attributes are computed in the *real*
   shard space).
3. Probes the ±1 perturbations of each flatland coordinate; keeps those that
   stay inside the shard cone and re-centers on the best δ.
4. If no perturbation stays inside, doubles the trajectory length (no GCD
   reduction) and retries.  Stops at ``SA_MAX_DEPTH`` iterations or after
   ``SA_PATIENCE`` iterations without δ improvement above ``SA_IMPROVE_THRESHOLD``.

This method is iterative/stateful (δ at one step decides the next), so unlike the
hedgehog searcher it computes δ inline via :class:`TrajectoryAttributesHandler`
and pushes one DTO per evaluated trajectory to an injected ``sink`` callable.

Walk-reuse
----------
The expensive step is building ``TrajectoryAttributesHandler`` (the matrix walk).
Once a handler is built its walk matrices are cached internally; calling
``compute_for_constant(B)`` on the same handler is cheap.  :meth:`_evaluate`
therefore accepts an optional *handler_cache* dict (keyed by ``trajectory_id``)
shared across all per-constant climbs for the same shard.  This means:

* Within a single run, if constant-A's climb evaluates trajectory T and constant-
  B's climb later encounters T, the walk is **not** repeated — ``compute_for_constant``
  is called on the already-built handler, and a merged DTO (containing both
  constants' Tier-1 data) is emitted so the JSONL record is complete.

* Across runs (from JSONL): if the flat ``(trajectory, constant)`` row for this
  constant already carries its ``delta`` it is returned immediately, no handler built.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.seed import resolve_injected_seed
from dreamer.search.methods.flatland.parallel_eval import evaluate_batch
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
from dreamer.utils.ui.tqdm_config import SmartTQDM

search_config = config.search


class NoInitialIdentification(Exception):
    """Raised when no reservoir trajectory identifies the constant in a shard."""

    def __init__(self, shard_id: str, constant: Constant):
        """
        :param shard_id: Identifier of the shard with no identifying seed.
        :param constant: Constant that could not be identified.
        """
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Small Angle Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


class SmallAngleSearch(SearchMethod):
    """Local hill-climb search over trajectory directions, single constant."""

    def __init__(
        self,
        space: Shard,
        constant: Constant,
        use_LIReC: bool = True,
    ):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this climb optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """Standalone entry point: collect emitted DTOs into a list and return them.

        The module drives :meth:`run` directly with the worker-pool sink; this
        wrapper exists for the abstract-method contract and for testing.
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
        """Run the hill-climb for a single constant, emitting DTOs to *sink*.

        :param handler_cache: Optional dict shared across all per-constant climbs
            for the same shard.  Maps ``trajectory_id → handler`` so that
            trajectories already evaluated (for another constant or in a previous
            step) skip the expensive walk and reuse ``compute_for_constant`` only.
            Callers (``SmallAngleSearchMod._run_shard``) should create one dict per
            shard and pass it to every constant's ``run()`` call.
        :param geom: Pre-built :class:`FlatlandGeometry` for the shard (built once
            per shard by the module, shared across constants).  ``None`` builds it
            here (standalone path).
        :param start: Pre-fetched interior start :class:`Position`.  ``None``
            fetches it here.
        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`;
            each step's in-cone perturbation batch is walked across worker
            processes.  ``None`` evaluates serially.
        :param initial_trajectory: Optional user-supplied seed direction.  Defaults
            to the shard's ``selected_trajectory``.  When valid (a non-zero recession
            direction) it seeds the climb instead of the reservoir; an invalid one
            falls back to reservoir seeding (see :func:`resolve_injected_seed`).

        :raises NoInitialIdentification: if no reservoir seed identifies *constant*.
        """
        if handler_cache is None:
            handler_cache = {}

        shard: Shard = self.space
        if geom is None:
            geom = FlatlandGeometry(shard)
        if start is None:
            start = shard.get_interior_point()

        # Context shared by every _evaluate call.
        ctx = dict(
            geom=geom,
            start=start,
            constant=constant,
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=shard_encoding_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
            handler_cache=handler_cache,
        )

        if initial_trajectory is None:
            initial_trajectory = getattr(shard, "selected_trajectory", None)
        z = resolve_injected_seed(
            geom, initial_trajectory, shard_id, constant,
            identify_fn=lambda zz: self._evaluate(zz, **ctx)[1],
        )
        if z is None:
            z = self._select_seed(geom, ctx, shard_id, constant)

        best_delta, _ = self._evaluate(z, **ctx)
        no_improve = 0
        doublings = 0

        for _ in SmartTQDM(
                range(search_config.SA_MAX_DEPTH),
                desc='Searching small angle filters', **config.system.TQDM_CONFIG
        ):
            best_z, best_score = self._best_inside_perturbation(z, ctx, pool)

            if best_z is None:
                # No perturbation stays inside: lengthen the trajectory and retry.
                if doublings >= search_config.SA_MAX_DOUBLINGS:
                    break
                z = z * 2  # no GCD reduction (spec)
                doublings += 1
                continue

            doublings = 0
            z = best_z
            if best_score - best_delta >= search_config.SA_IMPROVE_THRESHOLD:
                best_delta = max(best_delta, best_score)
                no_improve = 0
            else:
                best_delta = max(best_delta, best_score)
                no_improve += 1
                if no_improve >= search_config.SA_PATIENCE:
                    break

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_seed(
        self, geom: FlatlandGeometry, ctx: dict, shard_id: str, constant: Constant
    ) -> np.ndarray:
        """Pick the first reservoir trajectory (ascending L2 norm) that identifies."""
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.SA_RESERVOIR_SIZE
        )
        # Sort by ascending L2 norm so we start as close to the origin as possible.
        candidates: List[Tuple[float, Position]] = []
        for t in trajectories:
            norm = float(np.linalg.norm([float(t[s]) for s in geom.symbols]))
            candidates.append((norm, t))
        candidates.sort(key=lambda pair: pair[0])

        for _, t in candidates:
            z = geom.to_flatland(t)
            if not np.any(z):
                continue
            _, identified = self._evaluate(z, **ctx)
            if identified:
                return z

        raise NoInitialIdentification(shard_id, constant)

    def _best_inside_perturbation(
        self, z: np.ndarray, ctx: dict, pool=None
    ) -> Tuple[Optional[np.ndarray], float]:
        """Evaluate all in-shard ±1 perturbations; return the best (z, δ).

        Every in-cone perturbation is evaluated and the global-best δ is kept
        (no first-improvement break).  With a *pool* the whole in-cone batch is
        walked across worker processes in one shot; otherwise each candidate is
        evaluated serially via :meth:`_evaluate`.
        """
        geom: FlatlandGeometry = ctx["geom"]
        cands = [
            cand
            for cand in geom.perturbations(z, reduce=False)  # TODO: reduce=True (default, SmallAngle style)
            if geom.is_inside(cand)
        ]
        if not cands:
            return None, float("-inf")

        if pool is not None and len(cands) > 1:
            eval_ctx = {**ctx, "shard": self.space}
            scores = [d for d, _ in evaluate_batch(cands, eval_ctx=eval_ctx, pool=pool)]
        else:
            scores = [self._evaluate(cand, **ctx)[0] for cand in cands]

        best_i = int(np.argmax(scores))
        best_score = scores[best_i]
        if best_score == float("-inf"):
            # No candidate beat −∞ (all walks failed/non-identified): signal the
            # caller to length-double, matching the original first-improvement logic.
            return None, float("-inf")
        return cands[best_i], best_score

    def _evaluate(
        self,
        z: np.ndarray,
        *,
        geom: FlatlandGeometry,
        start: Position,
        constant: Constant,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
        handler_cache: Dict[str, "TrajectoryAttributesHandler"],
    ) -> Tuple[float, bool]:
        """Compute δ/identified for *constant* and direction *z*, emitting a DTO.

        Delegates to the shared :func:`evaluate_in_flatland` helper (Cases A/B/C
        walk-reuse).  See that function's docstring for full details.
        """
        return evaluate_in_flatland(
            z,
            geom=geom,
            shard=self.space,
            start=start,
            constant=constant,
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=shard_encoding_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
            handler_cache=handler_cache,
        )
