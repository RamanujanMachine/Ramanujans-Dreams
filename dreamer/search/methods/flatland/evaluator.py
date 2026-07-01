"""
Shared flatland trajectory evaluator.

Provides :func:`evaluate_in_flatland` — the three-case (A/B/C) walk-reuse
logic shared by SmallAngleSearch, GeneticSearch, and SimulatedAnnealingSearch.
All three methods evaluate a flatland integer vector ``z``, emit a Tier-1 DTO
to a ``sink`` callable, and return ``(delta, identified)`` for one constant.
"""

import dataclasses
from typing import Callable, Dict, Tuple

from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.storage.attribute_registry import attribute_name
from dreamer.utils.storage.optimization_objectives import score_record
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.configs import config

search_config = config.search


def active_objective() -> str:
    """The system-wide optimisation objective the search stage scores against."""
    return config.system.OPTIMIZATION_OBJECTIVE


#: Turn a cached ``seen_trajectories`` record into ``(signed_score, identified)``
#: under the active objective.  Thin alias over the shared
#: :func:`dreamer.utils.storage.optimization_objectives.score_record` so the
#: evaluator, the analysis-stage ranking, and the finalization all score records
#: identically.  ``None`` ⇒ the record cannot supply the score (recompute).
_score_from_record = score_record


def flatland_trajectory_key(
    z,
    *,
    geom: FlatlandGeometry,
    shard: Shard,
    start,
    shard_id: str,
    shard_encoding_str: str,
) -> Tuple[object, str, str]:
    """Compute the cache key for a flatland direction *z* without walking it.

    Centralises the (primitive direction → ``trajectory_id`` + Tier-1
    fingerprint) derivation so the serial :func:`evaluate_in_flatland` and the
    batched parallel evaluators (e.g. the GA process pool) agree exactly on how
    a trajectory is identified and when a cached record is stale.

    :param z: Integer flatland direction.
    :param geom: Flatland geometry for the shard.
    :param shard: The shard being searched.
    :param start: Interior start :class:`Position`.
    :param shard_id: Structural shard id.
    :param shard_encoding_str: Comma-joined ±1 encoding string.
    :return: ``(direction, trajectory_id, current_fp)`` — the primitive
        real-space direction, its trajectory id, and the current Tier-1 config
        fingerprint.
    """
    direction = geom.to_real_primitive(z)
    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)
    trajectory_id = derive_trajectory_id(
        shard_id, shard.cmf_name, shard_encoding_str, start_t, dir_t
    )
    current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, direction))
    return direction, trajectory_id, current_fp


def evaluate_in_flatland(
    z,
    *,
    geom: FlatlandGeometry,
    shard: Shard,
    start,
    constant: Constant,
    cmf_id: str,
    shard_id: str,
    shard_encoding_str: str,
    sink: Callable,
    seen_trajectories: dict,
    handler_cache: Dict[str, "TrajectoryAttributesHandler"],
) -> Tuple[float, bool]:
    """Compute the objective score / identified for *constant* at flatland *z*.

    Returns ``(score, identified)`` for *constant*, where ``score`` is the
    **signed** value of the system-wide ``OPTIMIZATION_OBJECTIVE`` oriented so
    that *larger is always better* (see
    :func:`dreamer.utils.storage.optimization_objectives.signed_score`).  For the
    default ``"delta"`` objective ``score`` is exactly δ, so the returned value is
    identical to the historical behaviour.  Three cases, each cheaper than the
    next:

    **Case A — delta already cached (on-disk or in-memory):**
    ``seen_trajectories`` contains a record whose ``delta_estimate`` already
    includes this constant → return immediately, no handler built, no walk.

    **Case B — handler cached (another constant evaluated this trajectory this
    run):**
    A :class:`TrajectoryAttributesHandler` for this trajectory_id is in
    *handler_cache* → call ``compute_for_constant`` only; build a merged DTO
    and emit it.

    **Case C — new trajectory:**
    Build handler from scratch, full walk, emit Tier-1 DTO.

    In all cases the handler (if available) is stored in *handler_cache* for
    future same-shard cross-constant reuse.

    This runs single-threaded in the main process — batch parallelism is
    process-based (see :func:`flatland.parallel_eval.evaluate_batch`), with the
    main process the sole owner of ``seen_trajectories`` / ``handler_cache`` /
    the sink — so no locking is needed.
    """
    # Always walk the GCD-reduced (primitive) ray: δ depends on the direction's
    # angle, not its length, so scaled/doubled copies of ``z`` map to the same
    # ray — same ``trajectory_id`` — and reuse the cached walk (Case A/B).
    # The fingerprint guards staleness: a cached record is only reusable when
    # its stored fingerprint matches the current config (walk depth / walk type
    # / identification tolerances), else the stored δ / identification are stale.
    direction, trajectory_id, current_fp = flatland_trajectory_key(
        z, geom=geom, shard=shard, start=start,
        shard_id=shard_id, shard_encoding_str=shard_encoding_str,
    )

    desired = {attribute_name(s) for s in search_config.TIER2_ATTRIBUTES}
    objective_name = active_objective()
    seen_record = seen_trajectories.get(trajectory_id)
    cached_handler = handler_cache.get(trajectory_id)

    # --- Case A: objective score already known for this constant (same config) ---
    if seen_record is not None and seen_record.get("config_fingerprint") == current_fp:
        cached = _score_from_record(seen_record, constant.name, objective_name)
        if cached is not None:
            return cached

    # --- Case B: handler cached — reuse walk, only compute new constant ---
    if cached_handler is not None:
        try:
            new_dto = build_trajectory_dto(
                cached_handler,
                cmf_id=cmf_id,
                shard_id=shard_id,
                cmf_name=shard.cmf_name,
                shard_encoding_str=shard_encoding_str,
                start=start,
                direction=direction,
                constants=[constant],
            )
            # Only fold in previously-stored per-constant data when it was
            # computed under the *same* config — a stale record's δ/identified
            # must not leak into the freshly-recomputed merged DTO.
            fresh = seen_record if (seen_record and seen_record.get("config_fingerprint") == current_fp) else {}
            existing_delta = dict(fresh.get("delta_estimate") or {})
            existing_ided = dict(fresh.get("identified") or {})
            existing_p = dict(fresh.get("p_vector") or {})
            existing_q = dict(fresh.get("q_vector") or {})
            # Fold in stored objective values only when they were recorded under
            # the *same* objective (else they are stale w.r.t. the active one).
            existing_obj = dict(fresh.get("objective_value") or {}) \
                if fresh.get("objective_name") == objective_name else {}
            merged_dto = dataclasses.replace(
                new_dto,
                delta_estimate={**existing_delta, **new_dto.delta_estimate},
                identified={**existing_ided, **new_dto.identified},
                p_vector={**existing_p, **(new_dto.p_vector or {})},
                q_vector={**existing_q, **(new_dto.q_vector or {})},
                objective_name=objective_name,
                objective_value={**existing_obj, **(new_dto.objective_value or {})},
            )
        except Exception as exc:
            Logger(
                f"Flatland evaluator handler-cache error — shard {shard_id}, "
                f"constant={constant.name}: {exc}",
                Logger.Levels.warning,
            ).log()
            return float("-inf"), False

        sink((cached_handler.trajectory_matrix, constant.value_sympy, merged_dto))
        seen_trajectories[trajectory_id] = {
            "extended_metrics": dict.fromkeys(desired),
            "delta_estimate": dict(merged_dto.delta_estimate),
            "identified": dict(merged_dto.identified),
            "objective_name": objective_name,
            "objective_value": dict(merged_dto.objective_value or {}),
            "config_fingerprint": current_fp,
        }
        score, identified = _score_from_record(
            seen_trajectories[trajectory_id], constant.name, objective_name
        )
        return score, identified

    # --- Case C: new trajectory — full walk ---
    try:
        handler = TrajectoryAttributesHandler.from_cmf(
            shard.cmf, direction, start, constant=None, searchable=shard
        )
        dto = build_trajectory_dto(
            handler,
            cmf_id=cmf_id,
            shard_id=shard_id,
            cmf_name=shard.cmf_name,
            shard_encoding_str=shard_encoding_str,
            start=start,
            direction=direction,
            constants=[constant],
        )
    except Exception as exc:
        Logger(
            f"Flatland evaluator handler error — shard {shard_id}, "
            f"direction={direction}: {exc}",
            Logger.Levels.warning,
        ).log()
        return float("-inf"), False

    sink((handler.trajectory_matrix, constant.value_sympy, dto))
    handler_cache[trajectory_id] = handler
    seen_trajectories[trajectory_id] = {
        "extended_metrics": dict.fromkeys(desired),
        "delta_estimate": dict(dto.delta_estimate),
        "identified": dict(dto.identified),
        "objective_name": dto.objective_name,
        "objective_value": dict(dto.objective_value or {}),
        "config_fingerprint": current_fp,
    }

    score, identified = _score_from_record(
        seen_trajectories[trajectory_id], constant.name, objective_name
    )
    return score, identified
