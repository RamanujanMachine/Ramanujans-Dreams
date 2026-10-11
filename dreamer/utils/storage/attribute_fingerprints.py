"""
Per-attribute validity fingerprints and the reuse policy.

Every stored trajectory attribute carries a short fingerprint describing **what
its value depends on**: the config knobs it reads, the per-trajectory walk depth
(when it uses it), the fingerprints of the attributes it is built from, and a
code ``version``.  A stored value is reused only while its fingerprint still
matches the current one, so a config change invalidates **exactly** the
attributes that depend on it — and, transitively, whatever is derived from
them — and nothing else.

Dependency graph
----------------
``ATTRIBUTE_META`` declares, per attribute:

* ``config`` — config knobs read by the attribute's own computation
  (``"<section>.<KNOB>"``), not counting what its dependencies read;
* ``deps`` — attributes (or the ``"tier1"`` core group) it is built from;
* ``uses_depth`` — whether it reads the trajectory's walk depth directly;
* ``version`` — bump when the formula changes in code.

The Tier-1 core (``delta``, ``identified``, ``p_vector``, ``q_vector``,
``projection_column``) is computed together and shares one fingerprint — the
``"tier1"`` group, stored in a row as ``config_fingerprint``.

The declarations are checked against the code by
``tests/test_attribute_fingerprints.py``: a static scan of
``TrajectoryAttributesHandler`` collects every knob (and walk-depth / walk-type
state) each registry attribute reads, transitively, and fails if one is not
covered by the attribute's declaration or its dependencies.  Attributes that are
not declared (e.g. registered at runtime) get a conservative fingerprint over
**every** search/system knob, so they are never reused across a config change.

Reuse policy (``system.REUSE_POLICY``)
--------------------------------------
* ``"strict"`` (default) — reuse a stored value only when its fingerprint
  matches;
* ``"keep"`` — reuse any stored value, ignoring fingerprints (fast exploratory
  reruns; values may be stale);
* ``"recompute"`` — ignore values from **earlier runs**.  Implemented by
  folding the run id (``system.RUN_ID``, set by ``System.run``) into every
  fingerprint: rows from previous runs mismatch everywhere, while rows written
  during this run (e.g. a trajectory an optimiser revisits) are reused as usual.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from dreamer.configs import config

#: Name of the Tier-1 core group (its fingerprint is a row's ``config_fingerprint``).
TIER1 = "tier1"

#: Attributes computed together as the Tier-1 core (one shared fingerprint).
TIER1_ATTRIBUTES: Tuple[str, ...] = (
    "delta", "identified", "p_vector", "q_vector", "projection_column",
)

#: Knobs the code reads but which provably do not change any stored value.
#: Each entry needs a justification; the dependency test skips them.
EXEMPT_KNOBS: Dict[str, str] = {
    "search.IDENTIFY_DEPTH": (
        "identification first runs at a cheaper depth and falls back to the full "
        "walk depth; verified result-preserving (same p/q and δ), roadmap 2026-07-04"
    ),
}

#: Walk type is chosen from this knob when the handler is built (``__init__``).
WALK_TYPE_KNOB = "search.DEFAULT_USES_INV_T"


@dataclass(frozen=True)
class AttributeMeta:
    """Dependency declaration of one stored attribute (see module docstring)."""

    config: Tuple[str, ...] = ()
    deps: Tuple[str, ...] = ()
    uses_depth: bool = False
    version: int = 1


_SPECTRAL = AttributeMeta(config=(WALK_TYPE_KNOB,))
_RECURRENCE = AttributeMeta(config=(WALK_TYPE_KNOB,))

#: Declared dependencies of every built-in attribute (checked against the code).
ATTRIBUTE_META: Dict[str, AttributeMeta] = {
    # ----- Tier-1 core group -----
    TIER1: AttributeMeta(
        config=(
            WALK_TYPE_KNOB,
            "search.DEPTH_CONVERGENCE_THRESHOLD",
            "search.LIMIT_DIFF_ERROR_BOUND",
            "search.MIN_ESTIMATE_DENOMINATOR",
            "search.CACHE_ACCEPTANCE_THRESHOLD",
            "search.IDENTIFY_CHECK_THRESHOLD",
            "search.CONSTANT_NO_DIGITS_HIGH_RES",
            "search.CONSTANT_NO_DIGITS_LOW_RES",
            "search.MAX_CONSTANT_RESOLUTION",
        ),
        deps=("traj_size",),
        uses_depth=True,
        # v2 (2026-10-11): δ trusts |estimate − constant| only beyond the
        # constant's rounding (was: any nonzero error) — spurious δ on long
        # trajectories before.
        version=2,
    ),
    **{name: AttributeMeta(deps=(TIER1,)) for name in TIER1_ATTRIBUTES},
    "traj_size": AttributeMeta(),
    # ----- recurrence (symbolic, constant-independent) -----
    "order": _RECURRENCE,
    "formula": _RECURRENCE,
    "coeff_degrees": _RECURRENCE,
    "relation": _RECURRENCE,
    "recurrence_coeffs": AttributeMeta(config=(WALK_TYPE_KNOB,), deps=("order",)),
    "companion_coboundary_rank": _RECURRENCE,
    # ----- spectral (trajectory matrix only) -----
    "eigenvalues": _SPECTRAL,
    "eigenvalue_errors": AttributeMeta(deps=("eigenvalues",)),
    "spectral_gap": AttributeMeta(deps=("eigenvalues",)),
    # ----- built on the identified p/q and the walk -----
    "gcd_slope": AttributeMeta(deps=(TIER1,)),
    "delta_prediction": AttributeMeta(deps=(TIER1, "eigenvalues", "gcd_slope")),
    "error_formula_ratio": AttributeMeta(deps=("delta_prediction",)),
    "approximated_digits_per_step": AttributeMeta(deps=("delta_prediction",)),
    "digits_approximation": AttributeMeta(deps=("approximated_digits_per_step",), uses_depth=True),
    "convergence_rate": AttributeMeta(deps=("approximated_digits_per_step",)),
    "precision_at": AttributeMeta(deps=(TIER1,), uses_depth=True),
    "digits_computed": AttributeMeta(deps=(TIER1,), uses_depth=True),
    "avg_computed_digits_per_step": AttributeMeta(deps=("digits_computed",), uses_depth=True),
    "delta_sequence": AttributeMeta(deps=(TIER1,), uses_depth=True),
    "digits_per_step": AttributeMeta(deps=(TIER1,), uses_depth=True),
}


# ---------------------------------------------------------------------------
# Knob access
# ---------------------------------------------------------------------------

def knob_value(knob: str):
    """Return the current value of a ``"<section>.<KNOB>"`` config knob."""
    section, name = knob.split(".", 1)
    return getattr(getattr(config, section), name)


def _canonical(value) -> object:
    """JSON-stable representation of a knob value (callables by qualified name)."""
    if callable(value):
        return f"callable:{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}"
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return repr(value)
    return value if isinstance(value, (int, str, bool, type(None))) else repr(value)


def _all_knobs() -> Tuple[str, ...]:
    """Every search/system knob — the conservative dependency set of an undeclared attribute."""
    out = []
    for section in ("search", "system"):
        sec = getattr(config, section)
        out.extend(f"{section}.{k}" for k in vars(sec) if k.isupper())
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _meta(name: str) -> Optional[AttributeMeta]:
    return ATTRIBUTE_META.get(name)


def attribute_fingerprint(name: str, walk_depth: Optional[int]) -> str:
    """Current validity fingerprint of attribute *name* for a trajectory.

    ``hash(version, own knob values, walk depth if used, fingerprints of deps)``.
    Undeclared attributes depend on every search/system knob (and the depth).

    :param name: Registry attribute name, or ``"tier1"`` for the core group.
    :param walk_depth: The trajectory's walk depth (``None`` when unknown).
    :return: A 16-hex-char fingerprint.
    """
    meta = _meta(name)
    if meta is None:
        knobs, deps, uses_depth, version = _all_knobs(), (TIER1,), True, 0
    else:
        knobs, deps, uses_depth, version = meta.config, meta.deps, meta.uses_depth, meta.version
    values = tuple((k, json.dumps(_canonical(knob_value(k)), sort_keys=True)) for k in sorted(knobs))
    if reuse_policy() == "recompute":
        values += (("@run", str(getattr(config.system, "RUN_ID", None))),)
    return _fingerprint_for(
        name, version, values,
        int(walk_depth) if (uses_depth and walk_depth is not None) else None,
        tuple(attribute_fingerprint(d, walk_depth) for d in deps),
    )


@lru_cache(maxsize=16384)
def _fingerprint_for(name, version, values, depth, dep_fps) -> str:
    payload = json.dumps([name, version, values, depth, dep_fps])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tier1_fingerprint(walk_depth: Optional[int]) -> str:
    """Fingerprint of the Tier-1 core group (a row's ``config_fingerprint``)."""
    return attribute_fingerprint(TIER1, walk_depth)


def fingerprints_for(names: Iterable[str], walk_depth: Optional[int]) -> Dict[str, str]:
    """``{name: fingerprint}`` for several attributes of one trajectory."""
    return {n: attribute_fingerprint(n, walk_depth) for n in names}


# ---------------------------------------------------------------------------
# Reuse decision
# ---------------------------------------------------------------------------

REUSE_POLICIES = ("strict", "keep", "recompute")


def reuse_policy() -> str:
    """The active ``system.REUSE_POLICY`` (validated)."""
    policy = str(getattr(config.system, "REUSE_POLICY", "strict")).lower()
    if policy not in REUSE_POLICIES:
        raise ValueError(f"REUSE_POLICY must be one of {REUSE_POLICIES}, got {policy!r}")
    return policy


def tier1_record_fresh(record: Optional[dict], current_fp: str) -> bool:
    """Can *record*'s Tier-1 core be reused, given the current Tier-1 fingerprint?

    The single freshness rule for Tier-1 rows (analysis, search, optimisers):
    ``"keep"`` accepts any row that carries a δ; otherwise the stored
    ``config_fingerprint`` must equal *current_fp* (which, under
    ``"recompute"``, embeds the run id — so only this run's rows match).
    """
    if record is None:
        return False
    if reuse_policy() == "keep":
        return "delta" in record
    return record.get("config_fingerprint") == current_fp


def tier1_is_fresh(record: Optional[dict], walk_depth: Optional[int]) -> bool:
    """:func:`tier1_record_fresh` for a trajectory of the given walk depth."""
    return tier1_record_fresh(record, tier1_fingerprint(walk_depth))


def stale_attributes(record: Optional[dict], names: Iterable[str], walk_depth: Optional[int]) -> List[str]:
    """The attributes of *names* that must be (re)computed for *record*.

    A value is reusable when present and — under ``"strict"`` — stored with the
    current fingerprint (``record["fingerprints"][name]``).  An error entry
    (``<name>_error``) counts as computed only if its fingerprint matches too.

    :param record: The stored ``(trajectory, constant)`` row, or ``None``.
    :param names: Attribute names wanted.
    :param walk_depth: The trajectory's walk depth.
    :return: Names to compute, in the given order.
    """
    names = list(names)
    if record is None:
        return names
    policy = reuse_policy()
    stored = record.get("fingerprints") or {}
    out = []
    for n in names:
        present = n in record or f"{n}_error" in record
        if not present:
            out.append(n)
        elif policy != "keep" and stored.get(n) != attribute_fingerprint(n, walk_depth):
            out.append(n)
    return out
