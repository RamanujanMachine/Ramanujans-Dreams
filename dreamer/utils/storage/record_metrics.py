"""
Read-only *metric extractors* over stored trajectory records.

A metric extractor maps a plain trajectory record (a ``dict`` exactly as stored
in the per-shard JSONL — see :class:`dreamer.utils.storage.dtos.TrajectoryDTO`)
to a single ``float`` ranking value, or ``None`` when the metric is missing /
unparseable for that record.

These are used by the Tier-3 post-process *top-N selectors* to rank trajectories
**without re-walking** them.  Because a stored record is one flat row per
``(trajectory, constant)`` (see :class:`TrajectoryDTO`), every metric — δ and all
spectral attributes — is a **top-level column**, so an extractor is a single
lookup.  To rank on a metric that is not yet stored (e.g.
``approximated_digits_per_step``), add it to the relevant ``TIER2`` / ``TIER3``
attribute list first so the search / post-process stage computes it.

The signature stays ``f(record, constant_name) -> float | None`` for API
stability, but *constant_name* is now vestigial: the record already **is** a
single constant's result, so extractors ignore it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

#: A metric extractor: ``(record, constant_name) -> float | None``.
MetricExtractor = Callable[[Dict[str, Any], Optional[str]], Optional[float]]


def _flat_float(key: str) -> MetricExtractor:
    """Build an extractor reading a finite numeric scalar from ``record[key]``.

    The record is flat, so every metric — the core ``delta`` field and every
    spectral column alike — is read the same way.
    """
    def _fn(record: Dict[str, Any], _constant_name: Optional[str] = None) -> Optional[float]:
        raw = record.get(key)
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None
    return _fn


#: Irrationality measure δ (the core ``delta`` column).
delta_metric: MetricExtractor = _flat_float("delta")


#: Public registry of stored-record metric extractors keyed by the name used in
#: the top-N selector grammar (``"top N highest <metric> in <scope>"``).
#: ``convergence_rate`` is the single system-wide definition
#: (``approximated_digits_per_step / ||direction||₂``, computed by
#: ``TrajectoryAttributesHandler.convergence_rate``); it is **read** from the
#: stored column, never recomputed, so the ranking metric never diverges.
METRIC_EXTRACTORS: Dict[str, MetricExtractor] = {
    "delta": delta_metric,
    "convergence_rate": _flat_float("convergence_rate"),
    "approximated_digits_per_step": _flat_float("approximated_digits_per_step"),
    "digits_approximation": _flat_float("digits_approximation"),
    "digits_computed": _flat_float("digits_computed"),
    "avg_computed_digits_per_step": _flat_float("avg_computed_digits_per_step"),
    "spectral_gap": _flat_float("spectral_gap"),
    "gcd_slope": _flat_float("gcd_slope"),
    "precision_at": _flat_float("precision_at"),
}


def register_metric(name: str, fn: MetricExtractor) -> None:
    """Register a custom stored-record metric extractor (e.g. for experiments)."""
    METRIC_EXTRACTORS[name] = fn
