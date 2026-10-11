"""
Optimization-objective registry — the property the pipeline optimises for.

Historically the analysis and search stages hard-coded **δ** (the irrationality
measure) as the single thing to maximise.  This module generalises that: an
*objective* is any **numeric** trajectory attribute for which the optimal
direction (``"max"`` / ``"min"``) is known *in advance*.  The active objective is
chosen system-wide via ``system.OPTIMIZATION_OBJECTIVE`` and steers both the
analysis-stage shard ranking and the search-stage optimisers.

Because a stored result is one flat row per ``(trajectory, constant)`` (see
:class:`dreamer.utils.storage.dtos.TrajectoryDTO`), the objective is simply a
**column** on that row: ``delta`` is the core ``delta`` field, every other
objective is its own top-level metric key (``convergence_rate``, ...).  So
scoring a stored record needs no constant argument — the record already *is* a
single constant's result.

Two design rules keep the rest of the system simple:

* **Membership is the validity gate.**  Only objectives registered in
  :data:`OBJECTIVES` may be selected — this structurally rejects non-numeric /
  binary attributes (they are simply never registered).

* **Direction is normalised away at a single boundary.**  Every search method
  *ascends* a scalar ("higher is better").  :func:`signed_score` flips the sign
  of a ``"min"`` objective's value, so the optimisers keep maximising unchanged;
  the *raw* value is what gets stored and reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

#: Optimal-value direction for an objective, known ahead of time.
Direction = Literal["max", "min"]


@dataclass(frozen=True)
class Objective:
    """A numeric attribute the pipeline can optimise, with its optimal direction.

    :param name: Public, config-facing objective name — also the attribute name in
        :data:`dreamer.utils.storage.attribute_registry.ATTRIBUTE_REGISTRY` (how it
        is computed) and the record column it is read from.
    :param direction: ``"max"`` if larger is better, ``"min"`` if smaller is better.
    """
    name: str
    direction: Direction


#: Objective attributes whose value is a **core** ``TrajectoryDTO`` field (always
#: computed as Tier-1) rather than an ``extra`` metric — so they need no extra
#: synchronous computation.  Currently just δ.
_CORE_FIELD_OBJECTIVES = frozenset({"delta"})


#: The registry of selectable objectives.  Add an entry here (plus, if needed, a
#: handler method + an ``ATTRIBUTE_REGISTRY`` entry) to make a new attribute
#: optimisable.  Only ``"max"`` objectives exist today; when the first ``"min"``
#: objective is added, its record value simply gets sign-flipped by
#: :func:`signed_score` — no search-loop change needed.
OBJECTIVES: Dict[str, Objective] = {
    "delta": Objective("delta", "max"),
    "convergence_rate": Objective("convergence_rate", "max"),
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


def objective_metric_attribute(name: str) -> Optional[str]:
    """The registry attribute to compute+store for objective *name*, or ``None``.

    Returns ``None`` when the objective's value is a **core** DTO field (δ), which
    is always computed as Tier-1 and needs no extra synchronous work; otherwise
    returns the attribute name so ``build_trajectory_dtos`` computes it into the
    row's flat ``extra`` under that key.
    """
    get_objective(name)  # validate
    return None if name in _CORE_FIELD_OBJECTIVES else name


def signed_score(name: str, raw: Optional[float]) -> Optional[float]:
    """Orient *raw* so that **larger is always better** for the search loop.

    ``"max"`` objectives pass through; ``"min"`` objectives are negated so the
    optimisers (which universally ascend) drive the raw value *down*.  ``None``
    (value unavailable) propagates unchanged so callers can treat it as "skip".
    """
    if raw is None:
        return None
    return raw if get_objective(name).direction == "max" else -raw


def objective_display_label(objective_name: str) -> str:
    """Short human label for *objective_name* used in logs / summaries / plots.

    ``"delta"`` renders as the familiar ``"δ"`` (so default runs read exactly as
    before); every other objective uses its registered name.
    """
    return "δ" if objective_name == "delta" else objective_name


# ---------------------------------------------------------------------------
# Reading / scoring a stored (trajectory, constant) record
# ---------------------------------------------------------------------------

#: Sentinel: the record has no column for the objective at all (distinct from a
#: stored ``None``/non-finite value, which means "computed but unavailable").
_MISSING = object()


def _record_raw(record: Dict[str, Any], objective_name: str):
    """The objective's raw value from a flat per-constant *record*, or ``_MISSING``.

    Every objective is a top-level column on the row (``delta`` is a core field;
    other objectives are their own metric key), so this is a single lookup.  The
    ``_MISSING`` sentinel (column absent) is what tells a caller "not computed yet
    — recompute", whereas a present-but-``None``/non-finite value means "computed,
    but unavailable for this trajectory" (worst score, no recompute).
    """
    return record[objective_name] if objective_name in record else _MISSING


def record_raw_value(record: Dict[str, Any], objective_name: str) -> Optional[float]:
    """Raw (unsigned) objective value from a stored record — for *display*.

    ``None`` when the column is absent or its value is not a finite number.
    Reporting shows this under the objective's own label; ranking should use
    :func:`score_record` so ``"min"`` objectives are oriented correctly.
    """
    raw = _record_raw(record, objective_name)
    if raw is _MISSING:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def score_record(
    record: Dict[str, Any], objective_name: str,
) -> Optional[Tuple[float, bool]]:
    """``(signed_score, identified)`` for a stored per-constant *record*.

    The single place that turns a persisted ``(trajectory, constant)`` row into a
    comparable, "higher is better" score under the active objective.  Shared by the
    search evaluator's cache reuse, the analysis-stage ranking, the reporting
    layer, and the micro-climb finalization so they all agree.

    * Returns ``None`` when the objective **column is absent** from the row — the
      caller should (re)compute it (this is the "compute the attribute if we
      haven't yet" path; note the objective is *not* part of the config hash, so
      switching objective never invalidates δ, it just requires filling the new
      column).
    * Returns ``(-inf, identified)`` when the column is present but non-finite
      (objective unavailable for this trajectory) — worst score, **no** recompute.
      Because every consumer *maximises* the signed score, ``-inf`` is correctly
      the worst regardless of the objective's optimal direction.
    """
    raw = _record_raw(record, objective_name)
    if raw is _MISSING:
        return None
    identified = bool(record.get("identified", False))
    value = record_raw_value(record, objective_name)  # None when non-finite
    score = signed_score(objective_name, value)
    if score is None:
        score = float("-inf")
    return score, identified
