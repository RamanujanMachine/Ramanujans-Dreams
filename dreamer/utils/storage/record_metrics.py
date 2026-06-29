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

import math
from typing import Any, Callable, Dict, Optional

import numpy as np

#: A metric extractor: ``(record, constant_name) -> float | None``.
MetricExtractor = Callable[[Dict[str, Any], Optional[str]], Optional[float]]


def _extended(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("extended_metrics") or {}


def _direction_norm(record: Dict[str, Any]) -> Optional[float]:
    """L2 norm of the trajectory ``direction`` vector, or ``None`` when absent/zero."""
    direction = record.get("direction")
    if not direction:
        return None
    try:
        norm = float(np.linalg.norm([float(x) for x in direction]))
    except (TypeError, ValueError):
        return None
    return norm if norm > 0 else None


def _eig_lognorm(record: Dict[str, Any], index: int) -> Optional[float]:
    """``log|λ_index| / ||direction||`` from ``extended_metrics["eigenvalues"]``.

    Mirrors ``graphs.shard_delta_sphere_jsonl.eigenvalue_lognorm_value`` — the
    eigenvalues are stored sorted by magnitude as serialised sympy strings, so
    ``index=0`` is λ₁ and ``index=1`` is λ₂.
    """
    eigs = _extended(record).get("eigenvalues")
    if not isinstance(eigs, (list, tuple)) or index >= len(eigs):
        return None
    norm = _direction_norm(record)
    if norm is None:
        return None
    import sympy as sp
    try:
        lam = complex(sp.sympify(str(eigs[index])).evalf())
    except (sp.SympifyError, TypeError, ValueError):
        return None
    mag = abs(lam)
    if mag == 0 or not np.isfinite(mag):
        return None
    return math.log(mag) / norm


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


def convergence_rate_metric(
    record: Dict[str, Any], _constant_name: Optional[str]
) -> Optional[float]:
    r"""Normalised spectral convergence rate ``(log|λ₁| − log|λ₂|) / ||direction||``.

    This is the *normalised eigenvalue error* gap: ``log|λ₁/λ₂|`` per unit
    trajectory length.  **Larger = faster convergence** (a bigger dominant /
    sub-dominant eigenvalue gap).  It is the sign-flipped twin of
    ``graphs.shard_delta_sphere_jsonl.convergence_rate`` (which returns
    ``lognorm(λ₂) − lognorm(λ₁) ≤ 0``); we return the positive gap so that
    "top N highest convergence_rate" selects the fastest-converging trajectories.

    Requires ``eigenvalues`` in ``extended_metrics`` (i.e. ``eigenvalues`` must be
    in the Tier-2 attribute list).
    """
    l1 = _eig_lognorm(record, 0)
    l2 = _eig_lognorm(record, 1)
    if l1 is None or l2 is None:
        return None
    return l1 - l2


#: Public registry of stored-record metric extractors keyed by the name used in
#: the top-N selector grammar (``"top N highest <metric> in <scope>"``).
METRIC_EXTRACTORS: Dict[str, MetricExtractor] = {
    "delta": delta_metric,
    "convergence_rate": convergence_rate_metric,
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
