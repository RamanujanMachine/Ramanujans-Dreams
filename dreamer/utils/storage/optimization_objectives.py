"""
Optimization-objective registry — the property the pipeline optimises for.

Historically the analysis and search stages hard-coded **δ** (the irrationality
measure) as the single thing to maximise.  This module generalises that: an
*objective* is any **numeric** trajectory attribute for which the optimal
direction (``"max"`` / ``"min"``) is known *in advance*.  The active objective is
chosen system-wide via ``system.OPTIMIZATION_OBJECTIVE`` and steers both the
analysis-stage shard ranking and the search-stage optimisers.

Two design rules keep the rest of the system simple:

* **Membership is the validity gate.**  Only objectives registered in
  :data:`OBJECTIVES` may be selected.  This structurally rejects non-numeric /
  binary attributes (the constant name, p/q vectors, ``identified``, …): they are
  simply never registered here.

* **Direction is normalised away at a single boundary.**  Every search method in
  the codebase *ascends* a scalar ("higher is better").  Rather than teach each
  of them about ``"min"`` objectives, :func:`signed_score` flips the sign of a
  ``"min"`` objective's raw value, so the optimisers keep maximising unchanged.
  The *raw* (unsigned) value is what gets stored and reported; only the
  optimisation loop sees the signed score.

The objective's ``extract`` reads its value straight off a
``TrajectoryAttributesHandler`` whose target constant is already set (the caller
sets it, e.g. via ``handler.compute_for_constant`` in
``build_trajectory_dto``).  Because the handler's methods are lazily cached and
call each other, requesting the objective transparently resolves its full
dependency chain (``convergence_rate`` → ``approximated_digits_per_step`` →
``delta_prediction`` → ``gcd_slope`` + ``sorted_eigenvalues`` + ``delta``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:  # avoid a runtime import cycle with trajectory_attributes
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

#: Optimal-value direction for an objective, known ahead of time.
Direction = Literal["max", "min"]

#: Extract the objective's raw (unsigned) value from a handler whose target
#: constant is already set.  Returns ``None`` when the value is unavailable
#: (e.g. a non-identified trajectory with no eigenvalue pair).
ObjectiveExtractor = Callable[["TrajectoryAttributesHandler"], Optional[float]]


@dataclass(frozen=True)
class Objective:
    """A numeric attribute the pipeline can optimise, with its optimal direction.

    :param name: Public, config-facing objective name (also the attribute name in
        :data:`dreamer.utils.storage.attribute_registry.ATTRIBUTE_REGISTRY`).
    :param direction: ``"max"`` if larger is better, ``"min"`` if smaller is
        better.  Stored per objective so a "best convergence rate is the *lowest*
        eigenvalue ratio" attribute can be optimised without inverting the search.
    :param extract: Reads the raw value off a handler (constant already set).
    """
    name: str
    direction: Direction
    extract: ObjectiveExtractor


def _delta_value(handler: "TrajectoryAttributesHandler") -> Optional[float]:
    """δ for the handler's current constant.

    ``delta`` returns the ``-inf`` non-convergence sentinel rather than ``None``;
    that is left intact because ``-inf`` is already the worst possible value for a
    ``"max"`` objective (the optimisers reject it naturally).
    """
    return float(handler.delta())


#: The registry of selectable objectives.  Add an entry here (plus, if needed, a
#: handler method + an ``ATTRIBUTE_REGISTRY`` entry) to make a new attribute
#: optimisable.  Only ``"max"`` objectives exist today; when the first ``"min"``
#: objective is added, ensure its ``extract`` returns ``None`` (not a signed
#: sentinel) for the unavailable case so :func:`signed_score` stays well-defined.
OBJECTIVES: Dict[str, Objective] = {
    "delta": Objective("delta", "max", _delta_value),
    "convergence_rate": Objective(
        "convergence_rate", "max", lambda h: h.convergence_rate()
    ),
}


def is_valid_objective(name: str) -> bool:
    """:return: Whether *name* is a registered, selectable objective."""
    return name in OBJECTIVES


def get_objective(name: str) -> Objective:
    """Look up an objective by name, failing loudly on a misspelled config.

    :raises KeyError: If *name* is not a registered objective (so an invalid
        ``OPTIMIZATION_OBJECTIVE`` surfaces immediately instead of silently
        falling back to δ).
    """
    try:
        return OBJECTIVES[name]
    except KeyError:
        raise KeyError(
            f"Unknown optimization objective {name!r}. "
            f"Registered objectives: {sorted(OBJECTIVES)}."
        )


def objective_raw_value(
    name: str, handler: "TrajectoryAttributesHandler"
) -> Optional[float]:
    """Raw (unsigned) objective value for *handler* under objective *name*.

    This is the value stored and reported (e.g. in ``objective_value`` on the
    DTO), independent of the optimisation direction.
    """
    return get_objective(name).extract(handler)


def signed_score(name: str, raw: Optional[float]) -> Optional[float]:
    """Orient *raw* so that **larger is always better** for the search loop.

    ``"max"`` objectives pass through; ``"min"`` objectives are negated so the
    optimisers (which universally ascend) drive the raw value *down*.  ``None``
    (value unavailable) propagates unchanged so callers can treat it as "skip".

    :param name: Registered objective name.
    :param raw: The raw value from :func:`objective_raw_value`, or ``None``.
    :return: The signed score, or ``None`` when *raw* is ``None``.
    """
    if raw is None:
        return None
    return raw if get_objective(name).direction == "max" else -raw


def objective_score(
    name: str, handler: "TrajectoryAttributesHandler"
) -> Optional[float]:
    """Convenience: the signed "higher is better" score for *handler*.

    Equivalent to ``signed_score(name, objective_raw_value(name, handler))``.
    """
    return signed_score(name, objective_raw_value(name, handler))


#: Sentinel: the record carries no value for the objective (distinct from a
#: stored ``None``, which means "objective unavailable for this trajectory").
_MISSING = object()


def _record_raw(record: Dict[str, Any], constant_name: str, objective_name: str):
    """The stored raw objective value, or :data:`_MISSING`.

    Single source of truth for *where* a record's objective value comes from,
    shared by :func:`record_raw_value` (display) and :func:`score_record`
    (ranking):

    * the record's ``objective_value[constant]`` when its stored ``objective_name``
      matches *objective_name*; otherwise
    * for the default ``"delta"`` objective only, a **backward-compatible
      fallback** to ``delta_estimate[constant]`` — so records written before
      ``objective_value`` existed (and any plain δ cache) still resolve with zero
      recomputation.  No fallback exists for other objectives (their value cannot
      be reconstructed from δ).

    A stored ``None`` (objective unavailable for the trajectory) is returned as-is;
    only a wholly absent value yields :data:`_MISSING`.
    """
    obj_map = record.get("objective_value") or {}
    if record.get("objective_name") == objective_name and constant_name in obj_map:
        return obj_map[constant_name]
    if objective_name == "delta":
        d_map = record.get("delta_estimate") or {}
        if constant_name in d_map:
            return d_map[constant_name]
    return _MISSING


def record_raw_value(
    record: Dict[str, Any], constant_name: str, objective_name: str,
) -> Optional[float]:
    """Raw (unsigned) objective value from a stored record — for *display*.

    ``None`` when the record has no value for the objective (or the objective is
    unavailable for that trajectory).  Reporting code shows this value under the
    objective's own name; ranking should use :func:`score_record` instead so that
    ``"min"`` objectives are oriented correctly.
    """
    raw = _record_raw(record, constant_name, objective_name)
    return None if raw is _MISSING else raw


def score_record(
    record: Dict[str, Any], constant_name: str, objective_name: str,
) -> Optional[Tuple[float, bool]]:
    """``(signed_score, identified)`` for *constant_name* from a **stored** record.

    The single place that turns a persisted trajectory record (a JSONL dict, or the
    in-memory ``seen_trajectories`` cache entry) into a comparable, "higher is
    better" score under the active objective.  Shared by the flatland evaluator's
    cache reuse, the analysis-stage shard ranking, the summary/reporting layer, and
    the micro-climb finalization so they all agree on how a record is scored.

    Returns ``None`` when the record cannot supply a score under *objective_name*
    (see :func:`_record_raw`), so the caller falls through to a (re)computation or
    skips the record.  A stored raw value of ``None`` (objective unavailable for the
    trajectory) maps to the ``-inf`` worst-score sentinel; since every consumer
    *maximises* the signed score, ``-inf`` is correctly the worst regardless of the
    optimal direction.
    """
    raw = _record_raw(record, constant_name, objective_name)
    if raw is _MISSING:
        return None
    score = signed_score(objective_name, raw)
    if score is None:
        score = float("-inf")
    ided_map = record.get("identified") or {}
    return score, bool(ided_map.get(constant_name, False))


def objective_display_label(objective_name: str) -> str:
    """Short human label for *objective_name* used in logs / summaries / plots.

    ``"delta"`` renders as the familiar ``"δ"`` (so default runs read exactly as
    before); every other objective uses its registered name.
    """
    return "δ" if objective_name == "delta" else objective_name
