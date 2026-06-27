"""
Universal discrete micro-hill-climb finalization for the search stage.

Every search method (Gradient Ascent, Hybrid SPSA, Simulated Annealing, Genetic,
Small Angle) writes its evaluated trajectories to a per-shard JSONL.  When
``search_config.ENABLE_MICRO_HILL_CLIMB`` is on, each method's per-shard runner
calls :func:`finalize_best_trajectories` once the search has finished and the
JSONL is flushed.  This is a *method-agnostic assurance pass*: it picks the
best-δ trajectory of the shard for each identified constant — and any trajectory
tied with it up to two decimal places — and runs the
:func:`~dreamer.search.methods.flatland.discrete_local_max.discrete_micro_climb`
endgame (orthogonal ±1 certificate + angular-resolution-doubling) around each, so
the reported maximum is a genuine refined lattice local maximum rather than
wherever the macro search happened to stop.

The pass operates on the *recorded* best trajectories (read back from the flushed
JSONL, which stores each trajectory's ``direction``), maps each back to its
flatland integer vector via :meth:`FlatlandGeometry.to_flatland`, and writes any
refined trajectory it discovers to the same JSONL through its own ``worker_pool``.
A single ``visited`` set of primitive-ray identities is shared across all tied
trajectories (and all constants) so no trajectory is walked or re-checked twice.

Off by default ⇒ this module is never entered and the search is byte-identical
to its pre-finalization behaviour.
"""

from __future__ import annotations

import math
from typing import List, Set, Tuple

from dreamer.configs import config
from dreamer.search.methods.flatland.discrete_local_max import (
    discrete_micro_climb,
    primitive_ray_key,
)
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
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

#: Decimal places at which two best trajectories count as "the same δ" (tie).
_TIE_DECIMALS = 2


def _best_records_for_constant(seen: dict, constant_name: str) -> Tuple[float, List[dict]]:
    """Best δ and the records tied with it (to ``_TIE_DECIMALS`` places).

    Scans the flushed per-shard records for identified, finite-δ trajectories of
    *constant_name*, returning the maximum δ and every record whose δ rounds to
    the same two decimals as that maximum (so a plurality of equally-good
    trajectories are all refined, not just one).

    :return: ``(max_delta, best_records)``; ``(-inf, [])`` when none qualify.
    """
    candidates: List[Tuple[float, dict]] = []
    for rec in seen.values():
        d_map = rec.get("delta_estimate") or {}
        id_map = rec.get("identified") or {}
        if constant_name not in d_map or not id_map.get(constant_name):
            continue
        try:
            delta = float(d_map[constant_name])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(delta):
            continue
        candidates.append((delta, rec))

    if not candidates:
        return float("-inf"), []

    max_delta = max(d for d, _ in candidates)
    threshold = round(max_delta, _TIE_DECIMALS)
    best = [rec for d, rec in candidates if round(d, _TIE_DECIMALS) == threshold]
    return max_delta, best


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

    # Shared across every constant + every tied trajectory: primitive-ray identities
    # already explored, so the endgame never walks or re-checks the same trajectory.
    visited: Set[Tuple[int, ...]] = set()
    handler_cache: dict = {}

    with worker_pool(
        num_workers=num_workers,
        worker_fn=compute_tier2_for_item,
        writer_fn=write_jsonl_line,
        output_path=output_path,
        config_overrides=config_overrides,
        parallel=bool(cfg.TIER2_ATTRIBUTES),
    ) as push:
        for const in identified_consts:
            max_delta, best_recs = _best_records_for_constant(seen, const.name)
            if not best_recs:
                continue

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

            improved_to = max_delta
            climbed = 0
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
                if key in visited:
                    continue  # this best ray was already covered by an earlier climb.

                # Re-evaluate the start under the current config (cache hit when the
                # stored fingerprint matches — no extra walk) to anchor the climb.
                cur_delta, identified = evaluate_in_flatland(z, **eval_ctx)
                visited.add(key)
                if not identified or not math.isfinite(cur_delta):
                    continue

                _, refined_delta = discrete_micro_climb(
                    z, cur_delta,
                    geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
                    traj_norm=traj_norm, improve_threshold=_IMPROVE_EPS,
                    pool=eval_pool, visited=visited,
                )
                improved_to = max(improved_to, refined_delta)
                climbed += 1

            if climbed == 0:
                continue
            if improved_to > max_delta + _IMPROVE_EPS:
                Logger(
                    f"Micro-hill-climb finalization improved best δ for "
                    f"'{const.name}' on shard {shard_id}: "
                    f"{max_delta:.6g} -> {improved_to:.6g} "
                    f"({climbed} best trajectory(ies) refined).",
                    Logger.Levels.info,
                ).log()
            else:
                Logger(
                    f"Micro-hill-climb finalization confirmed best δ for "
                    f"'{const.name}' on shard {shard_id}: δ={max_delta:.6g} "
                    f"({climbed} tied trajectory(ies) certified).",
                    Logger.Levels.debug,
                ).log()
