"""
Shared flatland trajectory evaluator.

Provides :func:`evaluate_in_flatland` — the walk-reuse logic shared by
SmallAngleSearch, GeneticSearch, SimulatedAnnealingSearch, and the gradient
methods.  Each evaluates a flatland integer vector ``z`` for **one constant**,
emits the flat per-``(trajectory, constant)`` :class:`TrajectoryDTO` to a
``sink``, and returns ``(score, identified)`` — where ``score`` is the signed
"higher is better" value of the active optimisation objective.
"""

import json
from typing import Callable, Dict, Tuple

from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.storage.optimization_objectives import score_record
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dtos,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.configs import config

search_config = config.search


def active_objective() -> str:
    """The system-wide optimisation objective the search stage scores against."""
    return config.system.OPTIMIZATION_OBJECTIVE


#: Turn a cached ``(trajectory, constant)`` record into ``(signed_score,
#: identified)`` under the active objective.  Thin alias over the shared
#: :func:`dreamer.utils.storage.optimization_objectives.score_record` so the
#: evaluator, the analysis-stage ranking, and the finalization all score records
#: identically.  ``None`` ⇒ the objective column is absent (recompute).
_score_from_record = score_record


def _dto_record(dto) -> dict:
    """The flat dict form of *dto* (JSON-native types) for the in-memory cache."""
    return json.loads(dto.to_json_line())


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
    batched parallel evaluators agree exactly on how a trajectory is identified
    and when a cached record is stale.

    :return: ``(direction, trajectory_id, current_fp)`` — the primitive
        real-space direction, its (constant-independent) trajectory id, and the
        current Tier-1 config fingerprint.
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

    Returns ``(score, identified)``, where ``score`` is the **signed** value of
    ``system.OPTIMIZATION_OBJECTIVE`` oriented so *larger is always better*.  For
    the default ``"delta"`` objective ``score`` is exactly δ.

    ``seen_trajectories`` is the nested ``{trajectory_id: {constant: record}}``
    cache; the reuse decision is **per constant**:

    * **Cached** — a fresh (matching-fingerprint) row for this constant already
      carries the objective column → score it, no handler, no walk.
    * **Reuse walk** — a :class:`TrajectoryAttributesHandler` for this
      ``trajectory_id`` is in *handler_cache* (another constant walked it this
      run) → reuse it, compute only this constant.
    * **New** — build the handler, full walk.

    Runs single-threaded in the main process (batch parallelism is process-based;
    see :func:`flatland.parallel_eval.evaluate_batch`), so no locking is needed.
    """
    # Always walk the GCD-reduced (primitive) ray: the objective depends on the
    # direction's angle, not its length, so scaled/doubled copies of ``z`` share a
    # ``trajectory_id`` and reuse the cached walk.  The fingerprint guards Tier-1
    # staleness (walk depth / walk type / identification tolerances).
    direction, trajectory_id, current_fp = flatland_trajectory_key(
        z, geom=geom, shard=shard, start=start,
        shard_id=shard_id, shard_encoding_str=shard_encoding_str,
    )

    objective_name = active_objective()
    bucket = seen_trajectories.get(trajectory_id, {})
    seen_record = bucket.get(constant.name)

    # --- Cached: this constant's row is fresh and already scored ---
    if seen_record is not None and seen_record.get("config_fingerprint") == current_fp:
        cached = score_record(seen_record, objective_name)
        if cached is not None:
            return cached

    # --- Compute (reusing the cached walk when available) ---
    cached_handler = handler_cache.get(trajectory_id)
    try:
        if cached_handler is not None:
            handler = cached_handler
        else:
            handler = TrajectoryAttributesHandler.from_cmf(
                shard.cmf, direction, start, constant=None, searchable=shard
            )
        dtos = build_trajectory_dtos(
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
            f"direction={direction}, constant={constant.name}: {exc}",
            Logger.Levels.warning,
        ).log()
        return float("-inf"), False

    if not dtos:
        return float("-inf"), False
    dto = dtos[0]

    sink((handler.trajectory_matrix, constant.value_sympy, dto))
    handler_cache[trajectory_id] = handler
    record = _dto_record(dto)
    seen_trajectories.setdefault(trajectory_id, {})[constant.name] = record

    scored = score_record(record, objective_name)
    if scored is None:
        return float("-inf"), bool(dto.identified)
    return scored
