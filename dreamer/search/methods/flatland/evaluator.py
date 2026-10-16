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
import math
import time
from typing import Callable, Dict, Optional, Tuple

from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils import monitoring
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
from dreamer.utils.storage.attribute_fingerprints import tier1_record_fresh
from dreamer.utils.task_memory import release_task_memory

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


class EvalStats:
    """How a search method's evaluations of one ``(shard, constant)`` were served.

    Every evaluation of a direction lands in exactly one bucket:

    * ``reused`` — a fresh stored row from an **earlier run** (or the analysis
      stage) already carries the objective: scored without a walk;
    * ``revisited`` — the row was computed **earlier in this run** (the method
      came back to the same ray, or two directions in one batch share a ray);
    * ``cached_walk`` — no row for this constant yet, but the trajectory's walk
      is still in the handler cache (another constant walked it): only this
      constant's values are computed;
    * ``new`` — a new walk (``stale`` of them replace a stored row whose
      fingerprint no longer matches the configuration);
    * ``failed`` — the walk or its evaluation raised;
    * ``outside`` — the direction left the cone (never walked).

    :param seen_trajectories: The method's ``{trajectory_id: {constant: record}}``
        cache **as loaded from the store**, before the method runs — rows present
        now count as ``reused`` later, rows added during the run as ``revisited``.
    :param constant_name: The constant being searched.
    """

    _FIELDS = ("reused", "revisited", "cached_walk", "new", "stale", "failed", "outside")

    #: A ``new_best`` event needs at least this much gain, or this many seconds
    #: since the last one (so a creeping optimiser does not flood the log).
    BEST_MIN_GAIN = 1e-3
    BEST_MIN_SECONDS = 10.0

    def __init__(self, seen_trajectories: dict, constant_name: str):
        self._stored = {tid for tid, by_const in seen_trajectories.items() if constant_name in by_const}
        self.counts = dict.fromkeys(self._FIELDS, 0)
        self.constant_name = constant_name
        self.best: Optional[float] = None          # best (signed) objective score seen
        self._best_announced: Optional[float] = None
        self._best_announced_at = 0.0

    def note_score(self, score: float) -> None:
        """Track the best finite score; announce improvements as ``new_best`` events."""
        if score is None or not math.isfinite(score) or (self.best is not None and score <= self.best):
            return
        self.best = score
        now = time.time()
        first = self._best_announced is None
        if (first or score - self._best_announced >= self.BEST_MIN_GAIN
                or now - self._best_announced_at >= self.BEST_MIN_SECONDS):
            self._best_announced, self._best_announced_at = score, now
            monitoring.event("new_best", f"{self.constant_name}: new best {active_objective()} = {score:.6g}",
                             constant=self.constant_name, objective=active_objective(), score=score,
                             evaluations=self.total)

    def scored(self, trajectory_id: str) -> None:
        """A fresh row answered the evaluation (no walk)."""
        self.counts["reused" if trajectory_id in self._stored else "revisited"] += 1

    def walked(self, trajectory_id: str, *, cached_walk: bool, ok: bool) -> None:
        """A walk (new, or on a cached handler) was evaluated."""
        if not ok:
            self.counts["failed"] += 1
        elif cached_walk:
            self.counts["cached_walk"] += 1
        else:
            self.counts["new"] += 1
            if trajectory_id in self._stored:
                self.counts["stale"] += 1

    def add(self, field: str, n: int = 1) -> None:
        """Count *n* evaluations in *field* (``revisited`` / ``failed`` / ``outside``)."""
        self.counts[field] += n

    @property
    def total(self) -> int:
        """Number of evaluations counted (``stale`` is a subset of ``new``)."""
        return sum(v for k, v in self.counts.items() if k != "stale")

    def summary(self) -> str:
        """One line: ``N evaluations → a reused · b revisited · … · f new``."""
        c = self.counts
        parts = [f"{c['reused']} reused", f"{c['revisited']} revisited"]
        if c["cached_walk"]:
            parts.append(f"{c['cached_walk']} on a cached walk")
        parts.append(f"{c['new']} new" + (f" ({c['stale']} replacing stale rows)" if c["stale"] else ""))
        if c["failed"]:
            parts.append(f"{c['failed']} failed")
        if c["outside"]:
            parts.append(f"{c['outside']} outside the cone")
        return f"{self.total} evaluations → " + " · ".join(parts)


def log_eval_stats(method: str, shard_id: str, constant_name: str, stats: "EvalStats",
                   lookahead: Optional[dict] = None) -> None:
    """Log the per-``(shard, constant)`` reuse summary of a search method.

    Same vocabulary as the analysis / hedgehog plan lines (``reused`` · ``new``).

    :param method: Human-readable method name (e.g. ``"Simulated Annealing"``).
    :param shard_id: The shard searched.
    :param constant_name: The constant searched.
    :param stats: The counts collected during the run.
    :param lookahead: Optional speculative look-ahead counters
        (``submitted`` / ``taken`` / ``dropped``); reported with the hit rate
        ``taken / submitted`` in the ``eval_stats`` event.
    """
    Logger(
        f"{method} — shard {shard_id[:24]}…, {constant_name}: {stats.summary()}",
        Logger.Levels.info,
    ).log()
    fields = dict(stats.counts, method=method, shard=shard_id, constant=constant_name,
                  evaluations=stats.total, best=stats.best, objective=active_objective())
    if lookahead:
        fields["lookahead"] = dict(lookahead, hit_rate=(lookahead["taken"] / lookahead["submitted"]
                                                        if lookahead.get("submitted") else None))
    monitoring.event("eval_stats", f"{method}: {stats.summary()}", **fields)


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
    stats: Optional[EvalStats] = None,
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
    if tier1_record_fresh(seen_record, current_fp):
        cached = score_record(seen_record, objective_name)
        if cached is not None:
            if stats is not None:
                stats.scored(trajectory_id)
                stats.note_score(cached[0])
            return cached

    # --- Compute (reusing the cached handler when available) ---
    cached_handler = handler_cache.get(trajectory_id)
    started = time.perf_counter()
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
            # Tier-2 in the same scope as the walk (no second walk / LIReC).
            extra_metrics=tuple(search_config.TIER2_ATTRIBUTES),
        )
    except Exception as exc:
        Logger(
            f"Flatland evaluator handler error — shard {shard_id}, "
            f"direction={direction}, constant={constant.name}: {exc}",
            Logger.Levels.warning,
        ).log()
        if stats is not None:
            stats.walked(trajectory_id, cached_walk=cached_handler is not None, ok=False)
        return float("-inf"), False

    if not dtos:
        if stats is not None:
            stats.walked(trajectory_id, cached_walk=cached_handler is not None, ok=False)
        return float("-inf"), False
    dto = dtos[0]
    if stats is not None:
        stats.walked(trajectory_id, cached_walk=cached_handler is not None, ok=True)

    sink((None, constant.value_sympy, dto))
    # Keep the handler's computed values, not its walk products (megabytes per
    # long trajectory, x HANDLER_CACHE_SIZE); a later constant re-walks.
    handler.release_walks()
    handler_cache[trajectory_id] = handler
    release_task_memory(time.perf_counter() - started)
    record = _dto_record(dto)
    seen_trajectories.setdefault(trajectory_id, {})[constant.name] = record

    scored = score_record(record, objective_name)
    if scored is None:
        return float("-inf"), bool(dto.identified)
    if stats is not None:
        stats.note_score(scored[0])
    return scored
