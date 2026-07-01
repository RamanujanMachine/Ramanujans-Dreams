"""
Read-only *metric extractors* over stored trajectory records.

A metric extractor maps a plain trajectory record (a ``dict`` exactly as stored
in the per-shard JSONL — see :class:`dreamer.utils.storage.dtos.TrajectoryDTO`)
to a single ``float`` ranking value, or ``None`` when the metric is missing /
unparseable for that record.

These are used by the Tier-3 post-process *top-N selectors* to rank trajectories
**without re-walking** them: the metric must already live in the JSONL (a Tier-1
field, a per-constant ``delta_estimate`` entry, or a Tier-2/Tier-3
``extended_metrics`` entry).  To rank on a metric that is not yet stored (e.g.
``approximated_digits_per_step``), add it to the relevant ``TIER2`` / ``TIER3``
attribute list first so the search / post-process stage computes it for every
trajectory.

The signature is uniform — ``f(record, constant_name) -> float | None`` — so the
ranking caller can stay agnostic about whether a given metric is per-constant
(``delta``) or constant-independent (everything spectral).  Constant-independent
extractors simply ignore ``constant_name``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

#: A metric extractor: ``(record, constant_name) -> float | None``.
MetricExtractor = Callable[[Dict[str, Any], Optional[str]], Optional[float]]


def _extended(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("extended_metrics") or {}


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def delta_metric(record: Dict[str, Any], constant_name: Optional[str]) -> Optional[float]:
    """Per-constant irrationality measure δ from ``delta_estimate[constant_name]``."""
    if constant_name is None:
        return None
    d = record.get("delta_estimate") or {}
    v = d.get(constant_name)
    if v is None:
        return None
    try:
        val = float(v)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _extended_float(key: str) -> MetricExtractor:
    """Build an extractor reading a numeric scalar from ``extended_metrics[key]``."""
    def _fn(record: Dict[str, Any], _constant_name: Optional[str]) -> Optional[float]:
        raw = _extended(record).get(key)
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None
    return _fn


#: Public registry of stored-record metric extractors keyed by the name used in
#: the top-N selector grammar (``"top N highest <metric> in <scope>"``).
METRIC_EXTRACTORS: Dict[str, MetricExtractor] = {
    "delta": delta_metric,
    # Length-normalised spectral convergence rate — the single system-wide
    # definition, computed by ``TrajectoryAttributesHandler.convergence_rate``
    # (``approximated_digits_per_step / ||direction||₂``) and stored in
    # ``extended_metrics`` when ``convergence_rate`` is in the Tier-2 list.
    # **Larger = faster convergence.**  Read here rather than recomputed so the
    # ranking metric never diverges from the handler's definition.
    "convergence_rate": _extended_float("convergence_rate"),
    "approximated_digits_per_step": _extended_float("approximated_digits_per_step"),
    "digits_approximation": _extended_float("digits_approximation"),
    "digits_computed": _extended_float("digits_computed"),
    "avg_computed_digits_per_step": _extended_float("avg_computed_digits_per_step"),
    "spectral_gap": _extended_float("spectral_gap"),
    "gcd_slope": _extended_float("gcd_slope"),
    "precision_at": _extended_float("precision_at"),
}


def register_metric(name: str, fn: MetricExtractor) -> None:
    """Register a custom stored-record metric extractor (e.g. for experiments)."""
    METRIC_EXTRACTORS[name] = fn
