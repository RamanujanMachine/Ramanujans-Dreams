"""
Atlas writer — emits DB-ready DTOs (CmfFamilyDTO, CmfDTO, ShardDTO) as
JSONL files.  This is the single source of truth for the pipeline — no
pickle files are used.

Used by:
  * Loading stage  — writes ``cmfs.jsonl`` + ``cmf_families.jsonl`` + per-
                     constant formatter JSON (``<const>/<cmf_name>.json``).
  * Extraction stage — writes ``<cmf>__shards.jsonl`` per CMF.

Writes are idempotent: each file is keyed by the appropriate id field
(``cmf_id`` / ``family_id`` / ``shard_id``) and existing ids are skipped
on rerun.  This matches the trajectory-stage dedup pattern and means we
can safely re-invoke the pipeline without growing the files.

Shard reconstruction (no pickle needed):
  ``load_shards_from_export(export_root, constants)`` reads the formatter
  JSON + shards JSONL and reconstructs live ``Shard`` objects entirely from
  JSON-serialisable data.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import sympy as sp
from ramanujantools.cmf import CMF, pFq
from ramanujantools import Position

from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.dtos import CmfDTO, CmfFamilyDTO, ShardDTO
from dreamer.utils.storage.trajectory_attributes import (
    _pq_to_jsonsafe,
    _stable_id,
    derive_cmf_and_shard_ids,
)
from dreamer.utils.types import CMFData


# ---------------------------------------------------------------------------
# Orthogonality defect helper
# ---------------------------------------------------------------------------

def _compute_orthogonality_defect(A: np.ndarray) -> Optional[float]:
    """Compute the orthogonality defect of the shard constraint matrix *A*.

    Treats rows of *A* (the hyperplane normal vectors) as lattice vectors,
    applies LLL reduction (via fpylll), then computes:
        defect = ∏_i ‖a_i‖ / sqrt(det(A A^T))

    A defect of 1.0 means perfectly orthogonal hyperplane normals.  Higher
    values indicate a more skewed / ill-conditioned shard geometry.

    Falls back to the defect of the *unreduced* rows when fpylll is
    unavailable (Windows / no fpylll install).  Returns ``None`` on any
    unexpected failure.
    """
    if A is None or A.size == 0:
        return None

    try:
        A_f = np.asarray(A, dtype=np.float64)

        def _defect(M: np.ndarray) -> float:
            norms = np.linalg.norm(M, axis=1)
            prod_norms = float(np.prod(norms))
            gram = M @ M.T
            det_val = float(np.sqrt(max(0.0, np.linalg.det(gram))))
            if det_val < 1e-9:
                return float("inf")
            return prod_norms / det_val

        try:
            from fpylll import IntegerMatrix, LLL
            A_int = np.round(A_f).astype(np.int64)
            M_fp = IntegerMatrix.from_matrix(A_int.tolist())
            LLL.reduction(M_fp)
            A_reduced = np.array([list(row) for row in M_fp], dtype=np.float64)
            return _defect(A_reduced)
        except ImportError:
            # fpylll not available — return raw (unreduced) defect
            return _defect(A_f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DTO builders
# ---------------------------------------------------------------------------

def _family_id_for(cmf: CMF) -> str:
    """Return a stable family identifier for *cmf*.

    For ``pFq`` instances we use ``"{p}F{q}"``; for everything else we fall
    back to the class name.  Keeps the family record compact while still
    distinguishing e.g. ``3F2`` from ``4F3``.
    """
    if isinstance(cmf, pFq):
        return f"{cmf.p}F{cmf.q}"
    return cmf.__class__.__name__


def _global_family_id_for(cmf: CMF) -> str:
    """Top-level family bucket — the class name (e.g. ``pFq``)."""
    return cmf.__class__.__name__


def build_cmf_family_dto(cmf: CMF) -> CmfFamilyDTO:
    """Construct a ``CmfFamilyDTO`` from a live CMF object.

    Matrix definitions are serialised via ``str(matrix)`` — a lossless,
    portable representation that round-trips through SymPy.
    """
    matrix_definitions = {
        str(sym): str(matrix) for sym, matrix in cmf.matrices.items()
    }
    return CmfFamilyDTO(
        family_id=_family_id_for(cmf),
        global_family_id=_global_family_id_for(cmf),
        matrix_definitions=matrix_definitions,
        dimensions=len(cmf.matrices),
    )


def build_cmf_dto(
    cmf_data: CMFData,
    constants: Iterable[Constant],
) -> CmfDTO:
    """Construct a ``CmfDTO`` from loaded ``CMFData``.

    The ``cmf_id`` is ``cmf_name`` to stay consistent with the rest of the
    pipeline (see :func:`derive_cmf_and_shard_ids`).  ``cmf_hyperplanes``
    is left empty here — hyperplanes are only known after extraction; a
    later patch (or a future extraction-side update) can fill them in.
    """
    shift = tuple(
        int(v) if isinstance(v, int) or (hasattr(v, "is_Integer") and v.is_Integer)
        else str(v)
        for v in cmf_data.shift.values()
    )
    return CmfDTO(
        cmf_id=cmf_data.cmf_name,
        family_id=_family_id_for(cmf_data.cmf),
        cmf_hyperplanes=[],
        coordinate_shift=shift,
        found_constants=[c.name for c in constants],
    )


def build_shard_dto(
    shard: Shard, found_constants: Optional[List[str]] = None
) -> ShardDTO:
    """Construct a ``ShardDTO`` from a live ``Shard`` instance.

    :param found_constants: The constant names to record as *found* in this shard.
        ``None`` (default) uses every constant the shard is searched for
        (``shard.consts``) — appropriate for a self-contained DTO.  The extraction
        stage instead passes ``[]`` (nothing confirmed yet); the analysis stage
        later fills in only the constants an identified trajectory converged to,
        via :func:`update_shard_found_constants`.

    ``shard_id`` is the stable SHA-256 hash of the canonical inequality
    system (see :func:`derive_cmf_and_shard_ids`) — this stays
    deterministic across runs regardless of hyperplane enumeration order.

    ``shard_encoding`` is the **±1 sign vector** the shard was constructed
    with: ``encoding[i] = +1`` means the shard lies above hyperplane *i*,
    ``-1`` means below.  Paired one-to-one with the CMF's hyperplane list
    (stored in ``cmf_hyperplanes`` on the corresponding CmfDTO), which is
    the natural combinatorial label of a shard.

    ``dimension`` is the number of free (non-redundant) CMF variables in this
    shard — equal to ``dimensionality`` unless the shard lives in a strict
    affine subspace.

    ``orthogonality_defect`` is computed via LLL reduction of the constraint
    matrix rows (hyperplane normals).  A value of 1.0 means perfectly
    orthogonal normals; higher means more skewed geometry.  ``None`` when
    fpylll is unavailable or the matrix is degenerate.
    """
    cmf_id, shard_id, _ = derive_cmf_and_shard_ids(shard)

    encoding = tuple(getattr(shard, "encoding", ()) or ())
    dimensionality = len(shard.symbols)
    # Effective dimension: number of independent constraint rows (rank of A).
    dimension = dimensionality
    if shard.A is not None and shard.A.size > 0:
        try:
            dimension = int(np.linalg.matrix_rank(shard.A))
        except Exception:
            pass

    interior_point = None
    if shard.start_coord is not None:
        # Use ``_pq_to_jsonsafe`` (int when integer, else str): a rational shift
        # makes start coordinates like ``7/2``, and ``int(Rational(7, 2))``
        # returns ``3`` *without raising*, so a plain ``int()`` would silently
        # truncate the fraction.  Strings round-trip through ``sympify``.
        interior_point = tuple(_pq_to_jsonsafe(v) for v in shard.start_coord.values())

    orthogonality_defect = _compute_orthogonality_defect(
        shard.A if shard.A is not None else None
    )

    return ShardDTO(
        shard_id=shard_id,
        cmf_id=cmf_id,
        shard_encoding=encoding,
        dimensionality=dimensionality,
        dimension=dimension,
        found_constants=(
            [c.name for c in shard.consts]
            if found_constants is None
            else list(found_constants)
        ),
        interior_point=interior_point,
        orthogonality_defect=orthogonality_defect,
    )


# ---------------------------------------------------------------------------
# Idempotent JSONL append
# ---------------------------------------------------------------------------

def _load_existing_ids(path: str, id_field: str) -> Set[str]:
    """Return the set of *id_field* values already in the JSONL file.

    Returns an empty set when the file is absent or unreadable.  Malformed
    lines are skipped silently — consistent with ``load_seen_trajectories``.
    """
    seen: Set[str] = set()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = record.get(id_field)
                if value is not None:
                    seen.add(value)
    except FileNotFoundError:
        pass
    return seen


def append_dtos_jsonl(
    path: str,
    dtos: Iterable,
    id_field: str,
) -> int:
    """Append each DTO's JSON line to *path*, skipping ids already present.

    Parameters
    ----------
    path:
        Destination JSONL file.  Parent directory must exist.
    dtos:
        Iterable of dataclass DTOs that expose ``to_json_line()``.
    id_field:
        Field name to dedup on (e.g. ``"cmf_id"``, ``"shard_id"``).

    Returns
    -------
    Number of records actually appended (newcomers only).
    """
    existing = _load_existing_ids(path, id_field)
    written = 0
    with open(path, "a") as f:
        for dto in dtos:
            value = getattr(dto, id_field, None)
            if value is None or value in existing:
                continue
            f.write(dto.to_json_line() + "\n")
            existing.add(value)
            written += 1
    return written


# ---------------------------------------------------------------------------
# High-level helpers used by the stages
# ---------------------------------------------------------------------------

def write_cmf_records(
    root: str,
    cmf_data_list: List[CMFData],
) -> None:
    """Write CmfDTOs + CmfFamilyDTOs into flat ``<root>/cmfs.jsonl``.

    Two files are produced (or extended) at the root level — one file for
    all CMFs across all constants, one for all families:
      * ``cmfs.jsonl``         — one ``CmfDTO`` per CMF (``found_constants``
                                 starts empty; call :func:`update_found_constants`
                                 after analysis to populate it).
      * ``cmf_families.jsonl`` — one ``CmfFamilyDTO`` per distinct family.
    """
    os.makedirs(root, exist_ok=True)

    # Write with found_constants=[] — updated after analysis via
    # update_found_constants() once we know which constants were actually found.
    cmf_dtos = [build_cmf_dto(data, []) for data in cmf_data_list]
    family_dtos = [build_cmf_family_dto(data.cmf) for data in cmf_data_list]

    append_dtos_jsonl(os.path.join(root, "cmfs.jsonl"), cmf_dtos, "cmf_id")
    append_dtos_jsonl(os.path.join(root, "cmf_families.jsonl"), family_dtos, "family_id")


def update_found_constants(
    root: str,
    cmf_name: str,
    const_names: List[str],
) -> bool:
    """Add *const_names* to the ``found_constants`` list of an existing CmfDTO.

    Called after the analysis stage confirms which constants were identified
    in a CMF.  Only adds names not already present (idempotent).

    Returns ``True`` if the record was found and (possibly) updated.
    """
    path = os.path.join(root, "cmfs.jsonl")
    if not os.path.exists(path):
        return False

    with open(path, "r") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    updated = False
    out_lines: List[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if record.get("cmf_id") == cmf_name:
            existing = set(record.get("found_constants") or [])
            new_consts = [c for c in const_names if c not in existing]
            if new_consts:
                record["found_constants"] = sorted(existing | set(new_consts))
                updated = True
        out_lines.append(json.dumps(record))

    if updated:
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    return updated


def update_shard_found_constants(
    root: str,
    cmf_name: str,
    found_by_shard_id: Dict[str, Iterable[str]],
) -> bool:
    """Add identified constant names to per-shard records in ``<cmf>__shards.jsonl``.

    The extraction stage writes each shard with ``found_constants=[]``; this is
    called after analysis with ``{shard_id: [identified const names]}`` to record
    only the constants an identified trajectory actually converged to (LIReC).

    Additive and idempotent (union with any existing names), so repeated runs and
    multi-constant runs accumulate rather than clobber.  Returns ``True`` if any
    record was updated.
    """
    path = shard_records_path(root, cmf_name)
    if not os.path.exists(path):
        return False

    with open(path, "r") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    updated = False
    out_lines: List[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        sid = record.get("shard_id")
        if sid in found_by_shard_id:
            existing = set(record.get("found_constants") or [])
            new_consts = {c for c in found_by_shard_id[sid] if c not in existing}
            if new_consts:
                record["found_constants"] = sorted(existing | new_consts)
                updated = True
        out_lines.append(json.dumps(record))

    if updated:
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    return updated


def update_cmf_hyperplanes(
    root: str,
    cmf_name: str,
    hyperplanes: Iterable,
) -> bool:
    """Fill in ``cmf_hyperplanes`` for an existing CmfDTO record.

    The loading stage writes the CMF record with an empty hyperplane list
    because hyperplanes are only known after extraction.  This helper
    rewrites the existing ``<root>/cmfs.jsonl`` line for ``cmf_name`` so
    the record is complete.

    Implementation: read all records, update the matching one in place,
    rewrite the file.  ``cmfs.jsonl`` is typically small so this is cheap.

    Parameters
    ----------
    root:
        Same root used by :func:`write_cmf_records` (``sys_config.EXPORT_CMFS``).
    cmf_name:
        Identifies the CmfDTO row to update (matches ``cmf_id``).
    hyperplanes:
        Iterable of ``Hyperplane`` objects; each is serialised via
        ``str(hp.expr)``.

    Returns
    -------
    ``True`` if a record was updated, ``False`` if no matching cmf_id was
    found or the file does not yet exist.
    """
    path = os.path.join(root, "cmfs.jsonl")
    if not os.path.exists(path):
        return False

    serialized = [str(getattr(hp, "expr", hp)) for hp in hyperplanes]

    with open(path, "r") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    updated = False
    out_lines: List[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if record.get("cmf_id") == cmf_name:
            record["cmf_hyperplanes"] = serialized
            updated = True
        out_lines.append(json.dumps(record))

    if updated:
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    return updated


def shard_records_path(root: str, cmf_name: str) -> str:
    """Return the flat ``<root>/<cmf>__shards.jsonl`` path for a CMF.

    Centralises the safe-name munging shared by the writer and reader so
    they can never drift apart.
    """
    safe_cmf = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in str(cmf_name)
    ).strip("_") or "unknown"
    return os.path.join(root, f"{safe_cmf}__shards.jsonl")


def write_shard_records(
    root: str,
    cmf_name: str,
    shards: Iterable[Shard],
    found_constants: Optional[List[str]] = None,
) -> int:
    """Write ShardDTOs for one CMF into ``<root>/<cmf>__shards.jsonl``.

    Returns the number of new records appended (existing ids are skipped).

    :param found_constants: Forwarded to :func:`build_shard_dto` for every shard.
        The extraction stage passes ``[]`` (no constant confirmed found until the
        analysis stage identifies one); ``None`` keeps the shard's candidate
        constants.
    """
    path = shard_records_path(root, cmf_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shard_dtos = [build_shard_dto(s, found_constants) for s in shards]
    return append_dtos_jsonl(path, shard_dtos, "shard_id")


def read_shard_records(
    root: str,
    cmf_name: str,
) -> List[ShardDTO]:
    """Read the cached ShardDTOs for one CMF, or ``[]`` if none on disk.

    The inverse of :func:`write_shard_records`: lets the extraction stage
    reload previously-computed shards (encoding + interior point) and
    skip re-running the expensive enumeration.  Malformed lines are
    skipped rather than aborting the load.
    """
    path = shard_records_path(root, cmf_name)
    if not os.path.isfile(path):
        return []
    out: List[ShardDTO] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ShardDTO.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Shard reconstruction from JSONL (no pickle required)
# ---------------------------------------------------------------------------

def reconstruct_shard_from_dto(
    dto: "ShardDTO",
    cmf_data,
    constants: List[Constant],
) -> Optional[Shard]:
    """Reconstruct a live ``Shard`` object from a ``ShardDTO`` + ``CMFData``.

    Hyperplanes are recomputed deterministically from the CMF (same logic
    as the extraction stage) so no separate hyperplane serialisation is
    needed.  Returns ``None`` when the encoding length does not match the
    current hyperplane count (stale cache from a changed CMF).

    Parameters
    ----------
    dto:
        Cached shard record.
    cmf_data:
        The CMFData (CMF object + shift) produced by the corresponding
        formatter's ``.to_cmf()``.
    constants:
        Constants to associate with the reconstructed shard.
    """
    from dreamer.extraction.extractor import extract_cmf_hyperplanes

    hps = extract_cmf_hyperplanes(cmf_data)
    if len(hps) != len(dto.shard_encoding):
        return None  # stale — encoding doesn't match current hyperplane set

    encoding = list(dto.shard_encoding)
    interior_point: Optional[Position] = None
    if dto.interior_point is not None:
        symbols = list(cmf_data.cmf.matrices.keys())
        # ``sympify`` (not ``int``): a rational shift makes coordinates like ``"3/2"``
        # (stored as a string by ``_pq_to_jsonsafe``), which ``int()`` cannot parse.
        # Matches the extractor's own DTO reconstruction and ``reconstruct_positions``.
        interior_point = Position(
            {s: sp.sympify(v) for s, v in zip(symbols, dto.interior_point)}
        )

    return Shard.from_cmf_data(
        cmf_data,
        constants,
        hps,
        encoding,
        interior_point,
        hyperplanes_already_shifted=False,
    )


def load_shards_from_export(
    export_root: str,
    constants: List[Constant],
    relevant_cmf_names: Optional[Dict[str, Optional[set]]] = None,
) -> Dict[Constant, List[Shard]]:
    """Load previously-extracted shards from ``export_root`` without pickle.

    For each constant, scans ``<export_root>/<const>/`` for formatter JSON
    files (written by the loading stage as ``<cmf_name>.json``).  For each
    formatter, reconstructs a ``CMFData`` via ``.to_cmf()``, then loads the
    matching ``<cmf>__shards.jsonl`` and reconstructs ``Shard`` objects.

    A shard is included for a constant only when the constant's name appears
    in ``shard_dto.found_constants`` (populated after analysis).

    Parameters
    ----------
    export_root:
        ``sys_config.EXPORT_CMFS`` — contains per-constant formatter JSONs
        and flat ``<cmf>__shards.jsonl`` files.
    constants:
        Constants to load shards for.
    relevant_cmf_names:
        Optional per-constant filter.  ``None`` value for a constant = load
        all CMFs under that constant.  Mirrors ``__derive_relevant_cmf_names``
        in ``System``.
    """
    from dreamer.loading.funcs.formatter import Formatter
    from dreamer.utils.logger import Logger

    result: Dict[Constant, List[Shard]] = {}

    for const in constants:
        safe_key = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
        const_path = os.path.join(export_root, safe_key)
        if not os.path.isdir(const_path):
            continue

        allowed = (relevant_cmf_names or {}).get(const.name)  # None → all CMFs

        shards_for_const: List[Shard] = []

        for fname in sorted(os.listdir(const_path)):
            if not fname.endswith(".json"):
                continue
            cmf_name = os.path.splitext(fname)[0]
            if allowed is not None and cmf_name not in allowed:
                continue

            formatter_path = os.path.join(const_path, fname)
            try:
                with open(formatter_path, "r") as fh:
                    raw = json.load(fh)
                formatter = Formatter.from_json_obj(raw)
                cmf_data = formatter.to_cmf()
            except Exception as exc:
                Logger(
                    f"load_shards_from_export: could not load formatter "
                    f"{formatter_path}: {exc}",
                    Logger.Levels.warning,
                ).log()
                continue

            dtos = read_shard_records(export_root, cmf_data.cmf_name)
            if not dtos:
                continue

            for dto in dtos:
                if const.name not in (dto.found_constants or []):
                    continue  # not identified for this constant yet
                shard = reconstruct_shard_from_dto(dto, cmf_data, [const])
                if shard is not None:
                    shards_for_const.append(shard)

        if shards_for_const:
            result[const] = shards_for_const

    return result
