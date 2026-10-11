"""
Data Transfer Objects (DTOs) for the CMF Atlas pipeline.

DTOs are immutable snapshots of pipeline entities (CMF families, CMFs, shards,
trajectories) intended for incremental storage in JSONL files and eventual
migration into a relational database.

Design rules:
  - All fields are JSON-serializable primitives or collections of primitives.
  - Tuple-typed fields are serialized as JSON arrays; ``from_dict`` converts them
    back to tuples so round-trips are lossless.
  - ``extended_metrics`` on TrajectoryDTO is an intentionally open dict for
    asynchronous workers to populate without schema changes.
  - ``frozen=True`` prevents accidental field reassignment; mutable dict fields
    (``extended_metrics``) can still be updated in place by background workers.
"""

import json
import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CMF family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CmfFamilyDTO:
    """Top-level CMF family record (e.g. the whole pFq family)."""
    family_id: str                          # e.g. "4F3"
    global_family_id: str                   # e.g. "pFq"
    matrix_definitions: Dict[str, str]      # symbol name → symbolic matrix str
    dimensions: int

    def to_json_line(self) -> str:
        """Serialize this record to a single JSON line for JSONL storage."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "CmfFamilyDTO":
        """Reconstruct a ``CmfFamilyDTO`` from a JSON-parsed dict."""
        return cls(
            family_id=d["family_id"],
            global_family_id=d["global_family_id"],
            matrix_definitions=d["matrix_definitions"],
            dimensions=d["dimensions"],
        )


# ---------------------------------------------------------------------------
# CMF instance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CmfDTO:
    """A single CMF instance within a family."""
    cmf_id: str
    family_id: str
    cmf_hyperplanes: List[str]
    coordinate_shift: Tuple[int | str, ...]
    found_constants: List[str]

    def to_json_line(self) -> str:
        """Serialize this record to a single JSON line for JSONL storage."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "CmfDTO":
        """Reconstruct a ``CmfDTO`` from a JSON-parsed dict."""
        return cls(
            cmf_id=d["cmf_id"],
            family_id=d["family_id"],
            cmf_hyperplanes=d["cmf_hyperplanes"],
            coordinate_shift=tuple(d["coordinate_shift"]),
            found_constants=d["found_constants"],
        )


# ---------------------------------------------------------------------------
# Shard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardDTO:
    """A bounded convex region of a CMF's integer lattice (Ax < b)."""
    shard_id: str
    cmf_id: str
    shard_encoding: Tuple[int, ...]         # sign-vector encoding of the shard
    dimensionality: int                     # number of CMF variables (ambient dim)
    dimension: int                          # number of free (non-redundant) variables
    found_constants: List[str]
    # --- optional fields (computed lazily or not yet available) ---
    interior_point: Optional[Tuple[int | str, ...]] = None  # str for rational coords (e.g. "7/2")
    orthogonality_defect: Optional[float] = None  # LLL-based; None when fpylll unavailable

    def to_json_line(self) -> str:
        """Serialize this record to a single JSON line for JSONL storage."""
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ShardDTO":
        """Reconstruct a ``ShardDTO`` from a JSON-parsed dict."""
        return cls(
            shard_id=d["shard_id"],
            cmf_id=d["cmf_id"],
            shard_encoding=tuple(d["shard_encoding"]),
            dimensionality=d["dimensionality"],
            dimension=d.get("dimension", d["dimensionality"]),  # backward compat
            found_constants=d["found_constants"],
            interior_point=tuple(d["interior_point"]) if d.get("interior_point") is not None else None,
            orthogonality_defect=d.get("orthogonality_defect"),
        )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryDTO:
    """A single **(trajectory, constant)** result — one flat JSONL line.

    The unit of a stored result is a (trajectory, constant) pair, because almost
    every attribute is constant-dependent: identification (LIReC → p/q) is coupled
    to the target constant, and it is identification that selects which eigenvalue
    pair the spectral metrics use.  So each constant a trajectory was evaluated
    against gets its **own** row.  The underlying walk (the trajectory matrix /
    limit) is constant-independent and computed **once** per trajectory — the rows
    for a trajectory's several constants share that walk at compute time and merely
    duplicate the constant-independent columns (``start_point``, ``direction``,
    walk metadata) on disk.

    Layout is a **flat dict** on disk (no nested ``extended_metrics``): the fixed
    *core* fields below plus any number of registry-computed metrics carried in
    ``extra`` and serialised at top level.  ``extra`` keeps the schema open — Tier-2
    workers and user-registered attributes add keys without a schema change — while
    every metric still lands as its own top-level column (DB-migration friendly).

    Composite key: ``(trajectory_id, constant)``.  ``trajectory_id`` is
    constant-independent (identifies the walk) and is shared across a trajectory's
    per-constant rows.  ``extra`` is mutable so background workers can add metrics
    to a frozen DTO without breaking the frozen contract.
    """
    trajectory_id: str
    cmf_id: str
    shard_id: str
    constant: str                             # the constant this row is a result for

    # Raw parameters (tuples instead of Position objects for JSON compatibility)
    start_point: Tuple[int | str, ...]
    direction: Tuple[int | str, ...]

    # Per-constant Tier-1 scalars.
    # ``delta``: irrationality measure δ (``-inf`` = non-converged sentinel).
    # ``identified``: LIReC found a convergent p/q for this constant.
    # ``p_vector`` / ``q_vector``: LIReC projection vectors (``None`` = unidentified).
    identified: bool = False
    delta: float = float("-inf")
    p_vector: Optional[Tuple[int | str, ...]] = None
    q_vector: Optional[Tuple[int | str, ...]] = None

    # Walk metadata (constant-independent; duplicated across a trajectory's rows).
    # ``walk_type``: 1 → ``inv().T`` applied after the walk; 2 → walked directly.
    # ``config_fingerprint``: hash of the config knobs that influenced the Tier-1
    #   values, so a later run under a changed config recomputes instead of reusing
    #   a stale row.  ``projection_column``: the walk-matrix column p/q was
    #   projected onto (needed to reproduce the exact p_n/q_n sequence).
    walk_type: int = 1
    walk_depth: Optional[int] = None
    config_fingerprint: Optional[str] = None
    projection_column: Optional[int] = None

    # Recurrence (symbolic ``LinearRecurrence``) — optional/heavy; ``None`` unless
    # ``build_trajectory_dto(..., compute_recurrence=True)`` or requested as a
    # Tier-2 attribute.  Constant-independent, duplicated across rows.
    recurrence_relation: Optional[str] = None
    recurrence_order: Optional[int] = None

    # Open, flat extension: registry-computed metrics (gcd_slope, delta_prediction,
    # convergence_rate, eigenvalues, ...) keyed by attribute name.  Serialised at
    # top level (not nested) so each is its own column; mutated in place by workers.
    extra: Dict[str, Any] = field(default_factory=dict, hash=False)

    #: The fixed (non-``extra``) fields, in declaration order — used to partition a
    #: flat JSON dict back into core vs. extension keys on read.
    CORE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "trajectory_id", "cmf_id", "shard_id", "constant",
        "start_point", "direction",
        "identified", "delta", "p_vector", "q_vector",
        "walk_type", "walk_depth", "config_fingerprint", "projection_column",
        "recurrence_relation", "recurrence_order",
    )

    def to_json_line(self) -> str:
        """Serialize this record to a single **flat** JSON line for JSONL storage."""
        flat = {k: getattr(self, k) for k in self.CORE_FIELDS}
        flat.update(self.extra)          # metrics at top level, not nested
        return json.dumps(flat)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryDTO":
        """Reconstruct from a flat JSON-parsed dict.

        Known ``CORE_FIELDS`` are pulled into the dataclass; every other key is
        treated as an extension metric and collected into ``extra``.
        """
        def _restore_vec(raw) -> Optional[tuple]:
            return tuple(raw) if isinstance(raw, (list, tuple)) else None

        extra = {k: v for k, v in d.items() if k not in cls.CORE_FIELDS}
        return cls(
            trajectory_id=d["trajectory_id"],
            cmf_id=d.get("cmf_id", ""),
            shard_id=d.get("shard_id", ""),
            constant=d.get("constant", ""),
            start_point=tuple(d.get("start_point", ())),
            direction=tuple(d.get("direction", ())),
            identified=bool(d.get("identified", False)),
            delta=float(d["delta"]) if d.get("delta") is not None else float("-inf"),
            p_vector=_restore_vec(d.get("p_vector")),
            q_vector=_restore_vec(d.get("q_vector")),
            walk_type=int(d.get("walk_type", 1)),
            walk_depth=d.get("walk_depth"),
            config_fingerprint=d.get("config_fingerprint"),
            projection_column=d.get("projection_column"),
            recurrence_relation=d.get("recurrence_relation"),
            recurrence_order=d.get("recurrence_order"),
            extra=extra,
        )
