"""
Simulated Annealing Search — trajectory optimisation with Metropolis acceptance.

Algorithm is faithful to ``context/resources/code/algos/annealing.py`` and
``positions.py``.  Key design choices:

* Genomes are flatland integer vectors (via :class:`FlatlandGeometry`).
* Neighbours = raw (non-reduced) ±1 unit steps in flatland, filtered by
  ``geom.is_inside`` and excluding the tabu list.
* Temperature decreases only on accepted moves (reference semantics).
* On a rejected step the current genome is doubled (length-doubling, no GCD
  reduce); a doubling counter is incremented; on exceeding
  ``ANNEAL_MAX_DOUBLINGS`` the counter is reset and a fresh seed direction is
  drawn (reference's dead "give up" branch fixed to its evident intent).
* On reseed the tabu list is fully cleared — the old region's history is
  irrelevant to the new starting point.
* A hard ``ANNEAL_MAX_TOTAL_STEPS`` ceiling prevents infinite stalls from
  tabu-deadlock (all neighbours blocked) combined with low-temperature
  near-zero Metropolis acceptance.
* Output uses the modern ``worker_pool`` sink / Tier-1 DTO pipeline.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.seed import resolve_injected_seed
from dreamer.search.methods.flatland.parallel_eval import evaluate_batch
from dreamer.utils.rand import derive_py_random
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

search_config = config.search


class NoInitialIdentification(Exception):
    """Raised when no reservoir trajectory identifies the constant in a shard."""

    def __init__(self, shard_id: str, constant: Constant):
        """
        :param shard_id: Identifier of the shard in which no seed was found.
        :param constant: The constant for which identification failed.
        """
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Simulated Annealing Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


def _get_temp(T0: float, k: int, schedule: str) -> float:
    """
    Cooling schedule (reference ``get_temp``).

    :param T0: Initial temperature.
    :param k: Number of accepted moves so far.
    :param schedule: ``'linear'`` (T0/(k+1)) or ``'log'`` (T0/log(k+1)).
    :return: Current temperature.
    """
    if schedule == "log":
        return T0 / math.log(k + 1) if k > 0 else T0
    return T0 / (k + 1)


class SimulatedAnnealingSearch(SearchMethod):
    """Simulated annealing search over flatland trajectory directions, single constant."""

    def __init__(
        self,
        space: Shard,
        constant: Constant,
        use_LIReC: bool = True,
    ):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this search optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """
        Standalone entry point — collect emitted DTOs into a list.

        :param starts: Unused; present for :class:`SearchMethod` interface compliance.
        :return: List of emitted ``(traj_matrix, sympy_const, TrajectoryDTO)`` tuples.
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
        """
        Run SA for a single constant, emitting Tier-1 DTOs to *sink*.

        Termination conditions (whichever fires first):
        1. ``iter_left == 0`` — ``ANNEAL_MAX_ITERS`` accepted moves completed.
        2. ``T <= Tmin`` — temperature dropped below the minimum threshold.
        3. ``total_steps >= ANNEAL_MAX_TOTAL_STEPS`` — hard ceiling on total
           loop iterations, preventing infinite stalls from tabu-deadlock at
           low temperature.

        :param constant: The constant to optimise δ for.
        :param cmf_id: Parent CMF identifier (forwarded to DTOs).
        :param shard_id: Shard identifier (forwarded to DTOs).
        :param shard_encoding_str: Shard encoding string (forwarded to DTOs).
        :param sink: Callable receiving ``(traj_matrix, sympy_const, dto)`` tuples.
        :param seen_trajectories: Mutable dict of already-evaluated trajectory records.
        :param handler_cache: Optional shared walk-cache (keyed by trajectory_id).
        :param geom: Pre-built :class:`FlatlandGeometry` for the shard (built once
            per shard by the module and shared across constants).  ``None`` builds
            it here (standalone path).
        :param start: Pre-fetched interior start :class:`Position`.  ``None``
            fetches it here.
        :param pool: Optional persistent per-shard :class:`multiprocessing.Pool`;
            each step's neighbour batch is walked across worker processes.  ``None``
            evaluates serially.
        :param initial_trajectory: Optional user-supplied seed direction.  Defaults
            to the shard's ``selected_trajectory``.  When valid (a non-zero recession
            direction) it seeds SA instead of the reservoir (bypassing the
            valid-neighbour guard — the user chose it explicitly; the reseed/stop
            logic handles an immediate stall).  An invalid one falls back to
            reservoir seeding (see :func:`resolve_injected_seed`).
        :raises NoInitialIdentification: if no reservoir seed identifies *constant*.
        """
        if handler_cache is None:
            handler_cache = {}

        # Per-(shard, method, constant) reproducible RNG for the Metropolis accept.
        # Same GLOBAL_SEED + shard + constant => identical SA trajectory; different
        # shards/constants get independent streams.  Nondeterministic when
        # GLOBAL_SEED is None.  (Neighbour generation is deterministic; the seed
        # reservoir comes from the already-seeded sampler.)
        self._rng_py = derive_py_random(shard_id, "annealing", constant.name)

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

        T0 = search_config.ANNEAL_T0
        Tmin = search_config.ANNEAL_TMIN
        schedule = search_config.ANNEAL_SCHEDULE
        max_iters = search_config.ANNEAL_MAX_ITERS
        max_doublings = search_config.ANNEAL_MAX_DOUBLINGS
        max_total_steps = search_config.ANNEAL_MAX_TOTAL_STEPS
        max_reseeds = search_config.ANNEAL_MAX_RESEEDS
        tabu_size = search_config.ANNEAL_TABU_SIZE
        max_traj_len = search_config.SEARCH_MAX_TRAJ_LEN
        traj_norm_name = search_config.SEARCH_TRAJ_NORM

        if initial_trajectory is None:
            initial_trajectory = getattr(shard, "selected_trajectory", None)
        cur_z = resolve_injected_seed(
            geom, initial_trajectory, shard_id, constant,
            identify_fn=lambda z: evaluate_in_flatland(z, **eval_ctx)[1],
        )
        if cur_z is None:
            cur_z = self._select_seed(geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name)
        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
        best_delta = cur_delta

        T = T0
        iter_left = max_iters
        doubling_count = 0
        total_steps = 0
        consecutive_failed_reseeds = 0

        # Tabu: bounded recent-position list (reference update_old_list_neighs).
        old_pos_list: List[bytes] = [cur_z.tobytes()]

        while iter_left > 0 and T > Tmin:
            total_steps += 1
            if total_steps > max_total_steps:
                Logger(
                    f"Simulated Annealing: total-step ceiling ({max_total_steps}) reached "
                    f"in shard {shard_id} — terminating early.",
                    Logger.Levels.warning,
                ).log()
                break

            # Generate in-cone, non-tabu, within-traj-length neighbours.
            neighbours: List[np.ndarray] = []
            for cand in geom.perturbations(cur_z, reduce=False):
                if not geom.is_inside(cand):
                    continue
                if cand.tobytes() in old_pos_list:
                    continue
                if geom.traj_norm(cand, traj_norm_name) > max_traj_len:
                    continue
                neighbours.append(cand)

            if not neighbours:
                # No valid neighbour: double and continue (reference adaptive scaling).
                if doubling_count >= max_doublings:
                    doubling_count = 0
                    fresh = self._try_reseed(geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name)
                    if fresh is not None:
                        consecutive_failed_reseeds = 0
                        cur_z = fresh
                        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
                        # Full tabu clear on reseed: old region's history is
                        # irrelevant to the new starting point and would otherwise
                        # block all of its neighbours (tabu deadlock).
                        old_pos_list = [cur_z.tobytes()]
                    else:
                        consecutive_failed_reseeds += 1
                        if consecutive_failed_reseeds >= max_reseeds:
                            Logger(
                                f"Simulated Annealing: {max_reseeds} consecutive failed reseeds "
                                f"in shard {shard_id} — terminating early.",
                                Logger.Levels.debug,
                            ).log()
                            break
                else:
                    cur_z = cur_z * 2  # no GCD reduce
                    doubling_count += 1
                continue

            # Evaluate all neighbours; pick best (reference semantics).
            # Neighbours are already filtered in-cone/non-tabu/within-cap above.
            # With a pool the batch is walked across worker processes; otherwise
            # each neighbour is walked serially in this process.
            if pool is not None:
                results = evaluate_batch(neighbours, eval_ctx=eval_ctx, pool=pool)
                neighbour_deltas = [(d, nb) for (d, _), nb in zip(results, neighbours)]
            else:
                neighbour_deltas = [
                    (evaluate_in_flatland(nb, **eval_ctx)[0], nb) for nb in neighbours
                ]

            neighbour_deltas.sort(key=lambda x: x[0], reverse=True)
            new_delta, new_z = neighbour_deltas[0]

            # Update tabu list with neighbours + current (reference update_old_list_neighs).
            for _, nb in neighbour_deltas:
                old_pos_list.append(nb.tobytes())
            old_pos_list.append(cur_z.tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]

            accepted = False
            if new_delta >= cur_delta:
                cur_z = new_z
                cur_delta = new_delta
                accepted = True
                iter_left -= 1
            else:
                diff = new_delta - cur_delta
                if self._rng_py.random() < math.exp(diff / T):
                    cur_z = new_z
                    cur_delta = new_delta
                    accepted = True
                    iter_left -= 1

            if accepted:
                doubling_count = 0
                if cur_delta > best_delta:
                    best_delta = cur_delta
                # Temperature decreases only on accepted moves (reference).
                T = _get_temp(T0, max_iters - iter_left, schedule)
            else:
                # Adaptive scaling: double on rejection (reference).
                if doubling_count >= max_doublings:
                    doubling_count = 0
                    fresh = self._try_reseed(geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name)
                    if fresh is not None:
                        consecutive_failed_reseeds = 0
                        cur_z = fresh
                        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
                        # Full tabu clear on reseed (same reason as above).
                        old_pos_list = [cur_z.tobytes()]
                    else:
                        consecutive_failed_reseeds += 1
                        if consecutive_failed_reseeds >= max_reseeds:
                            Logger(
                                f"Simulated Annealing: {max_reseeds} consecutive failed reseeds "
                                f"in shard {shard_id} — terminating early.",
                                Logger.Levels.debug,
                            ).log()
                            break
                else:
                    cur_z = cur_z * 2  # no GCD reduce
                    doubling_count += 1
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_seed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
        max_traj_len: float,
        traj_norm_name: str,
    ) -> np.ndarray:
        """
        Pick the first reservoir trajectory (ascending L2 norm) that identifies
        *and* has at least one valid neighbour within the trajectory-length cap.

        A seed whose entire neighbourhood is filtered by the traj-len cap would
        immediately enter the no-neighbours stall loop (doubling → reseed → repeat),
        causing the 12-hour stall.  Skipping such seeds here ensures every accepted
        seed can actually make a step.

        :param geom: Flatland geometry for the current shard.
        :param eval_ctx: Evaluation context dict (forwarded to evaluate_in_flatland).
        :param shard_id: Shard identifier used in the exception message.
        :param constant: The constant to identify.
        :param max_traj_len: Maximum allowed trajectory norm (``SEARCH_MAX_TRAJ_LEN``).
        :param traj_norm_name: Which norm to use (``SEARCH_TRAJ_NORM``).
        :raises NoInitialIdentification: if no reservoir trajectory identifies *constant*
            while also having a reachable neighbourhood.
        :return: Flatland coordinate vector of the seed trajectory.
        """
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.ANNEAL_RESERVOIR_SIZE
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
            # Skip seeds whose entire neighbourhood is outside the traj-len cap —
            # they would immediately enter the no-neighbours stall cycle.
            has_valid_neighbour = any(
                geom.is_inside(nb) and geom.traj_norm(nb, traj_norm_name) <= max_traj_len
                for nb in geom.perturbations(z, reduce=False)
            )
            if not has_valid_neighbour:
                continue
            _, identified = evaluate_in_flatland(z, **eval_ctx)
            if identified:
                return z

        raise NoInitialIdentification(shard_id, constant)

    def _try_reseed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
        max_traj_len: float,
        traj_norm_name: str,
    ) -> Optional[np.ndarray]:
        """
        Attempt to find a fresh seed when the doubling budget is exhausted.

        :param geom: Flatland geometry for the current shard.
        :param eval_ctx: Evaluation context dict.
        :param shard_id: Shard identifier.
        :param constant: The constant to identify.
        :param max_traj_len: Maximum allowed trajectory norm.
        :param traj_norm_name: Which norm to use.
        :return: New flatland seed vector, or ``None`` if no identifying trajectory found.
        """
        try:
            return self._select_seed(geom, eval_ctx, shard_id, constant, max_traj_len, traj_norm_name)
        except NoInitialIdentification:
            return None
