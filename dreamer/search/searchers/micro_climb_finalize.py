"""
Universal discrete micro-hill-climb finalization for the search stage.

Every search method (Gradient Ascent, Hybrid SPSA, Simulated Annealing, Genetic,
Small Angle) writes its evaluated trajectories to a per-shard JSONL.  When
``search_config.ENABLE_MICRO_HILL_CLIMB`` is on, each method's per-shard runner
calls :func:`finalize_best_trajectories` once the search has finished and the
JSONL is flushed.  This is a *method-agnostic assurance pass*: it picks the
best-δ trajectory of the shard for each identified constant — and any trajectory
tied with it up to two decimal places — and runs the orthogonal ±1 certificate +
angular-resolution-doubling endgame around each, so the reported maximum is a
genuine refined lattice local maximum rather than wherever the macro search
happened to stop.

The pass operates on the *recorded* best trajectories (read back from the flushed
JSONL, which stores each trajectory's ``direction``), maps each back to its
flatland integer vector via :meth:`FlatlandGeometry.to_flatland`, and writes any
refined trajectory it discovers to the same JSONL through its own ``worker_pool``.

Each tied best trajectory is refined by an **independent** micro-climb (its own
Phase-B exploration set) so the finalized result is *complete* — every tied
trajectory's basin is explored regardless of the order the ties are visited — and
*reproducible*: since each climb is deterministic and the climbs are order-
independent, the union of refined trajectories does not depend on the
(nondeterministic, parallel) JSONL append order.  A per-constant set of already-
climbed start rays merely avoids re-running an *identical* climb.

The ties are climbed **concurrently** via
:func:`~dreamer.search.methods.flatland.discrete_local_max.parallel_micro_climb`:
each round unions the candidate rays of *all* active climbs into one wide batch, so
the process pool stays saturated across ties (a shard with ``N`` tied bests would
otherwise dispatch ``N`` thin ``2·d_flat``-wide batches, idling most cores).  This
is a pure scheduling change — the per-climb result is identical to a sequential
``discrete_micro_climb`` per anchor — and the shared walk cache
(``seen_trajectories`` + ``handler_cache``) still walks each ray once.

Off by default ⇒ this module is never entered and the search is byte-identical
to its pre-finalization behaviour.
"""

from __future__ import annotations

import math
from typing import List, Set, Tuple

from dreamer.configs import config
from dreamer.search.methods.flatland.discrete_local_max import (
    evaluate_neighbours,
    parallel_micro_climb,
    primitive_ray_key,
)
from dreamer.search.methods.flatland.evaluator import active_objective
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.storage.optimization_objectives import score_record
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import (
    compute_tier2_for_item,
    load_seen_trajectories,
    worker_pool,
    write_jsonl_line,
)
from dreamer.utils.storage.handler_reconstruction import reconstruct_positions

search_config = config.search

#: Minimum δ gain that counts as a strict improvement during finalization.  Tiny
#: (well below any real δ resolution) so the assurance pass captures any genuine
#: lattice improvement while ignoring floating-point noise.
_IMPROVE_EPS = 1e-9

#: Decimal places at which two best trajectories count as "the same score" (tie).
_TIE_DECIMALS = 2


def _best_records_for_constant(
    seen: dict, constant_name: str, objective_name: str,
) -> Tuple[float, List[dict]]:
    """Best objective score and the records tied with it (to ``_TIE_DECIMALS``).

    Scans the flushed per-shard records for identified trajectories of
    *constant_name* with a finite **signed** objective score (oriented so larger
    is better — see
    :func:`dreamer.utils.storage.optimization_objectives.signed_score`), returning
    the maximum score and every record whose score rounds to the same two decimals
    as that maximum (so a plurality of equally-good trajectories are all refined,
    not just one).  Ranking follows the active objective, but identification stays
    a hard prerequisite regardless of the objective.

    :return: ``(max_score, best_records)``; ``(-inf, [])`` when none qualify.
    """
    candidates: List[Tuple[float, dict]] = []
    for by_const in seen.values():           # seen: {trajectory_id: {constant: record}}
        rec = by_const.get(constant_name)
        if rec is None:
            continue
        # Rank by the active objective via the shared record scorer; identification
        # stays a hard prerequisite (a record only qualifies when identified).
        scored = score_record(rec, objective_name)
        if scored is None:
            continue
        score, identified = scored
        if not identified or not math.isfinite(score):
            continue
        candidates.append((score, rec))

    if not candidates:
        return float("-inf"), []

    max_score = max(s for s, _ in candidates)
    threshold = round(max_score, _TIE_DECIMALS)
    best = [rec for s, rec in candidates if round(s, _TIE_DECIMALS) == threshold]
    # Return in a deterministic order: the JSONL append order is nondeterministic
    # (parallel Tier-2 workers write on completion), so sorting by the stable
    # trajectory_id makes the finalization's per-tie climb sequence reproducible.
    best.sort(key=lambda r: r.get("trajectory_id", ""))
    return max_score, best


def finalize_best_trajectories(
    *,
    shard,
    identified_consts: List[Constant],
    geom: FlatlandGeometry,
    start,
    eval_pool,
    cmf_id: str,
    shard_id: str,
    shard_encoding_str: str,
    output_path: str,
    num_workers: int,
    config_overrides: dict,
) -> None:
    """Run the micro-hill-climb assurance endgame on a shard's best trajectories.

    No-op unless ``search_config.ENABLE_MICRO_HILL_CLIMB`` is set.  Must be called
    *after* the search's own ``worker_pool`` has closed (so the JSONL is flushed)
    and while ``eval_pool`` / ``geom`` / ``start`` are still alive.

    :param shard: The searched shard (provides ``shard.cmf`` for direction rebuild).
    :param identified_consts: Constants searched on this shard.
    :param geom: Flatland geometry built once for the shard.
    :param start: Interior start ``Position`` for the shard.
    :param eval_pool: Persistent per-shard evaluation pool (or ``None``).
    :param cmf_id: Structural CMF id.
    :param shard_id: Structural shard id (names the JSONL).
    :param shard_encoding_str: ±1 encoding string for the shard.
    :param output_path: ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl`` to read + append.
    :param num_workers: Worker count for the finalization writer pool.
    :param config_overrides: Exported config propagated to writer subprocesses.
    """
    cfg = search_config
    if not cfg.ENABLE_MICRO_HILL_CLIMB:
        return

    seen = load_seen_trajectories(output_path)
    if not seen:
        return

    max_norm = cfg.SEARCH_MAX_TRAJ_LEN
    traj_norm = cfg.SEARCH_TRAJ_NORM

    # Shared *walk* cache (built once, reused by every climb): a ray walked by one
    # climb is a cache hit for the next, so nothing is re-walked — but each climb
    # still explores independently (see below), so the result stays complete and
    # order-independent.
    handler_cache: dict = {}
    objective_name = active_objective()

    with worker_pool(
        num_workers=num_workers,
        worker_fn=compute_tier2_for_item,
        writer_fn=write_jsonl_line,
        output_path=output_path,
        config_overrides=config_overrides,
        parallel=bool(cfg.TIER2_ATTRIBUTES),
    ) as push:
        for const in identified_consts:
            max_delta, best_recs = _best_records_for_constant(
                seen, const.name, objective_name
            )
            if not best_recs:
                continue

            # Log the tie multiplicity: the finalization cost scales with the number
            # of trajectories tied (to _TIE_DECIMALS) with the best score, since each
            # is refined by its own micro-climb.  Recorded so a run can be sized /
            # optimised against the real tie count on production shards.
            Logger(
                f"Micro-climb finalization: '{const.name}' on shard {shard_id} — "
                f"{len(best_recs)} trajectory(ies) tied within {_TIE_DECIMALS} dp of "
                f"best {objective_name}={max_delta:.6g}; refining each.",
                Logger.Levels.info,
            ).log()

            eval_ctx = dict(
                geom=geom,
                shard=shard,
                start=start,
                constant=const,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=push,
                seen_trajectories=seen,
                handler_cache=handler_cache,
            )

            # Baseline walk-cache size for this constant, so the log below can report
            # how many *new* trajectories the finalization actually walks (the real
            # compute cost, distinct from the tie count and from cache-hit re-checks).
            walked_before = sum(1 for by_const in seen.values() if const.name in by_const)

            # Collect the distinct start rays of the tied best trajectories.  Dedup is
            # per-constant (the same ray identifies different constants differently, so
            # a ray climbed for one must still be climbed for another).
            climbed_centers: Set[Tuple[int, ...]] = set()
            anchor_zs: list = []
            for rec in best_recs:
                try:
                    _, direction = reconstruct_positions(shard.cmf, rec)
                except Exception as exc:  # malformed record — skip, don't abort.
                    Logger(
                        f"Micro-climb finalization could not rebuild a best "
                        f"trajectory on shard {shard_id} for '{const.name}': {exc}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                z = geom.to_flatland(direction)
                if not geom.is_inside(z):
                    continue  # outside the cone after the round-trip — skip.

                key = primitive_ray_key(z, geom)
                if key in climbed_centers:
                    continue  # identical start ray already collected for this constant.
                climbed_centers.add(key)
                anchor_zs.append(z)

            # Anchor every start under the current config (batched cache hits), then
            # micro-climb all identified anchors *concurrently*: each tied trajectory's
            # basin is still explored fully (completeness) — the result is identical to
            # climbing them one by one — but every round unions all active climbs'
            # candidate rays into one wide batch so the process pool stays saturated
            # across ties instead of dispatching 2·d_flat neighbours at a time.
            anchor_scores = (
                evaluate_neighbours(anchor_zs, eval_ctx, eval_pool) if anchor_zs else []
            )
            anchors = [
                (z, score)
                for z, (score, identified) in zip(anchor_zs, anchor_scores)
                if identified and math.isfinite(score)
            ]
            climbed = len(anchors)
            if climbed == 0:
                continue

            refined = parallel_micro_climb(
                anchors,
                geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
                traj_norm=traj_norm, improve_threshold=_IMPROVE_EPS, pool=eval_pool,
            )
            improved_to = max([max_delta] + [d for _, d in refined])
            new_walks = sum(1 for by_const in seen.values() if const.name in by_const) - walked_before
            if improved_to > max_delta + _IMPROVE_EPS:
                Logger(
                    f"Micro-hill-climb finalization improved best {objective_name} for "
                    f"'{const.name}' on shard {shard_id}: "
                    f"{max_delta:.6g} -> {improved_to:.6g} "
                    f"({climbed} best trajectory(ies) refined, {new_walks} new trajectories walked).",
                    Logger.Levels.info,
                ).log()
            else:
                Logger(
                    f"Micro-hill-climb finalization confirmed best {objective_name} for "
                    f"'{const.name}' on shard {shard_id}: {objective_name}={max_delta:.6g} "
                    f"({climbed} tied trajectory(ies) certified, {new_walks} new trajectories walked).",
                    Logger.Levels.debug,
                ).log()
