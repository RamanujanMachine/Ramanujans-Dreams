r"""
Aesthetic δ-on-a-sphere projections from the **current JSONL pipeline**.

This is the pickle-free successor to ``graphs/shard delta sphere.py``.  The old
script unpickled ``Shard`` and ``DataManager`` objects and matched data to shards
by comparing start coordinates.  The system has since moved to a self-sustaining
JSONL layout (no pickle files):

  * Shards          - reconstructed from ``<EXPORT_CMFS>/<const>/<cmf>.json`` +
                      ``<EXPORT_CMFS>/<cmf>__shards.jsonl`` via
                      :func:`dreamer.utils.storage.atlas_writer.load_shards_from_export`.
  * Trajectories    - one file per shard at
                      ``<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl``; each line is a
                      ``TrajectoryDTO`` with a ``direction`` tuple and a
                      **per-constant** ``delta_estimate`` dict.

Matching data to a shard is therefore trivial now: the JSONL filename *is*
``<shard_id>.jsonl`` and ``derive_cmf_and_shard_ids(shard)`` yields that id - no
start-coordinate matching needed.

The sphere-rendering maths (rotate the best direction to the equator for a clean
local interpolation, ``griddata`` the δ field, trim to the shard's cone ``A\cdot v ≤ 0``,
draw the bounding hyperplane circles) carries straight over from the old script.

Three options, exactly as requested:

  1. ``one_sphere_per_shard=True``  → an atlas (one sphere per shard, like the old
     ``generate_shard_atlas2``).  ``False`` → every shard drawn on a single shared
     sphere (they live in the same CMF coordinate frame).

  2. For a ``D > 3`` CMF, a :class:`ProjectionSpec` selects which 3 coordinates
     become the sphere's ``(x, y, z)`` and constrains the remaining coordinates to
     a *linear* function of those three - only trajectories lying in that 3-D
     subspace (within ``subspace_tol``) are drawn.  This realises the
     ``(x, y, z, f(x,y,z)) / (f(x,y,z), x, y, z) / ...`` notation from the request.

  3. Overlay paths (e.g. gradient-ascent steps) are optional.  ``None`` →
     δ-projection only (like the old script's example output). Otherwise a path
     is either pulled automatically from a second search-results directory or
     passed explicitly as a list of direction vectors per shard.

Run with the WSL conda env ``rama`` (matplotlib + scipy + the dreamer package).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from typing import Any, Callable

from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.atlas_writer import load_shards_from_export
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


# ===========================================================================
# Attribute extraction (what scalar to colour the sphere with)
# ===========================================================================
#
# A *value function* maps a merged trajectory record (a plain dict, exactly as
# stored in the JSONL) to the scalar to plot - or ``None`` to skip that
# trajectory.  δ is the default, but any Tier-1/Tier-2/Tier-3 attribute can be
# plotted instead, including string-formatted sympy expressions (e.g. the
# trajectory asymptotics) reduced to a number via substitution.

ValueFn = Callable[[Dict[str, Any]], Optional[float]]


def delta_value(constant: Constant) -> ValueFn:
    """Value function returning δ for *constant* (the default, classic behaviour).

    :param constant: the constant whose entry to read from ``delta_estimate``.
    :return: a :data:`ValueFn` extracting ``record["delta_estimate"][name]``.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        d = rec.get("delta_estimate") or {}
        v = d.get(constant.name)
        return None if v is None else float(v)
    return _fn


def field_value(key: str, transform: Optional[Callable[[Any], float]] = None) -> ValueFn:
    """Value function reading a top-level numeric field (e.g. ``limit_value``).

    :param key: the record key (e.g. ``"limit_value"``, ``"recurrence_order"``).
    :param transform: optional post-processor applied to the raw field value.
    :return: a :data:`ValueFn`.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        v = rec.get(key)
        if v is None:
            return None
        return float(transform(v)) if transform is not None else float(v)
    return _fn


def extended_metric_value(
    key: str,
    transform: Optional[Callable[[Any], float]] = None,
) -> ValueFn:
    """Value function reading a Tier-2/Tier-3 attribute from ``extended_metrics``.

    Background workers (and the post-process stage) populate the open
    ``extended_metrics`` dict - e.g. ``digits_per_step``, ``spectral_gap``,
    ``asymptotics``.  Use *transform* for non-numeric attributes (see
    :func:`sympy_attribute_value` for the string-sympy case).

    :param key: the metric name inside ``extended_metrics``.
    :param transform: optional callable turning the raw metric into a float.
    :return: a :data:`ValueFn`.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        em = rec.get("extended_metrics") or {}
        if key not in em:
            return None
        raw = em[key]
        if raw is None:
            return None
        return float(transform(raw)) if transform is not None else float(raw)
    return _fn


def sympy_attribute_value(
    key: str,
    subs: Dict[str, float],
    *,
    in_extended_metrics: bool = True,
) -> ValueFn:
    """Value function reducing a *string-formatted sympy* attribute to a float.

    Some attributes (e.g. the trajectory **asymptotics**) are stored as a sympy
    expression serialised to a string.  This parses the string with
    ``sympy.sympify`` and substitutes *subs* (e.g. ``{"n": 1e6}``), returning the
    numeric ``evalf`` result.  Trajectories whose attribute is missing or does
    not evaluate to a finite real number are skipped.

    :param key: the attribute name (in ``extended_metrics`` by default, else a
        top-level field when ``in_extended_metrics=False``).
    :param subs: symbol→value substitutions applied before evaluation.
    :param in_extended_metrics: read from ``extended_metrics`` (``True``) vs the
        top-level record (``False``).
    :return: a :data:`ValueFn`.
    """
    import sympy as sp

    sym_subs = {sp.Symbol(k): v for k, v in subs.items()}

    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        container = (rec.get("extended_metrics") or {}) if in_extended_metrics else rec
        raw = container.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            expr = sp.sympify(raw)
            val = complex(expr.subs(sym_subs).evalf())
        except (sp.SympifyError, TypeError, ValueError):
            return None
        if abs(val.imag) > 1e-9:
            return None
        return float(val.real)
    return _fn


def eigenvalue_lognorm_value(index: int = 0) -> ValueFn:
    """Value function: ``log|λ_index| / ||v||₂`` for a trajectory.

    The eigenvalues are stored sorted, as serialised sympy strings, under
    ``extended_metrics["eigenvalues"]`` (so ``index=0`` is λ₁, ``index=1`` is λ₂).
    Each is parsed and the **log of its magnitude** (``log|λ|`` = ``Re log λ``,
    well-defined for negative/complex eigenvalues) is divided by the L2 norm of
    the trajectory direction ``v`` (the raw ``direction`` vector), giving a
    growth rate per unit trajectory length.

    Returns ``None`` (skipped) when the eigenvalue list is missing/too short, the
    eigenvalue does not parse, ``λ`` is 0, or ``v`` is the zero vector.

    :param index: 0-based eigenvalue index (0 → λ₁, 1 → λ₂).
    :return: a :data:`ValueFn`.
    """
    import math
    import sympy as sp

    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        eigs = (rec.get("extended_metrics") or {}).get("eigenvalues")
        if not isinstance(eigs, (list, tuple)) or index >= len(eigs):
            return None
        direction = rec.get("direction")
        if not direction:
            return None
        norm = float(np.linalg.norm([float(x) for x in direction]))
        if norm == 0:
            return None
        try:
            lam = complex(sp.sympify(str(eigs[index])).evalf())
        except (sp.SympifyError, TypeError, ValueError):
            return None
        mag = abs(lam)
        if mag == 0 or not np.isfinite(mag):
            return None
        return math.log(mag) / norm
    return _fn

def convergence_rate(neg: bool) -> ValueFn:
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        result = eigenvalue_lognorm_value(1)(rec) - eigenvalue_lognorm_value(0)(rec)
        return -result if neg else result
    return _fn


# ===========================================================================
# Projection spec (choose the 3-D subspace of a D-dimensional CMF)
# ===========================================================================

@dataclass
class ProjectionSpec:
    r"""Selects the 3-D subspace of a ``D``-dimensional CMF to draw on the sphere.

    The CMF's direction vectors live in ``D`` coordinates (one per CMF symbol,
    in ``shard.symbols`` order).  This spec picks **three** of them to be the
    sphere's ``(x, y, z)`` axes and constrains every remaining ("dependent")
    coordinate to a *linear* combination of those three.  A trajectory is kept
    only when each dependent coordinate matches its linear prediction (within a
    relative tolerance), i.e. the trajectory lies in the chosen subspace.

    :param axes: the three coordinate indices mapped to sphere ``(x, y, z)``.
    :param dependent: ``{coord_index: (a, b, c)}`` - the dependent coordinate at
        ``coord_index`` must equal ``a\cdot x + b\cdot y + c\cdot z`` where ``(x, y, z)`` are
        the free coordinates selected by ``axes``.  Empty for ``D == 3``.
    """

    axes: Tuple[int, int, int] = (0, 1, 2)
    dependent: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    @classmethod
    def identity(cls, dim: int) -> "ProjectionSpec":
        """Trivial spec for a 3-D CMF: coords ``(0, 1, 2)`` → ``(x, y, z)``.

        :param dim: the CMF dimensionality (number of symbols).
        :raises ValueError: if ``dim < 3``.
        """
        if dim < 3:
            raise ValueError(f"Need at least 3 coordinates to project, got {dim}.")
        return cls(axes=(0, 1, 2), dependent={})

    @classmethod
    def from_layout(cls, layout: Sequence) -> "ProjectionSpec":
        r"""Build a spec from the ``(x, y, z, f(x,y,z))`` layout notation.

        ``layout`` has one entry per CMF coordinate (length ``D``):

          * ``"x"`` / ``"y"`` / ``"z"`` - this coordinate is a free sphere axis.
          * a 3-tuple ``(a, b, c)``      - this coordinate is dependent and equals
            ``a\cdot x + b\cdot y + c\cdot z``.

        For example, a 4-D CMF drawn as ``(x, y, z, f)`` with ``f = x - y`` is
        ``ProjectionSpec.from_layout(["x", "y", "z", (1, -1, 0)])``; the
        ``(f, x, y, z)`` variant is ``[(1, -1, 0), "x", "y", "z"]``.

        :param layout: per-coordinate layout as described above.
        :raises ValueError: if the free axes are not exactly ``{x, y, z}``.
        """
        free: Dict[str, int] = {}
        dependent: Dict[int, Tuple[float, float, float]] = {}
        for idx, entry in enumerate(layout):
            if isinstance(entry, str) and entry.lower() in ("x", "y", "z"):
                free[entry.lower()] = idx
            else:
                coeffs = tuple(float(c) for c in entry)
                if len(coeffs) != 3:
                    raise ValueError(
                        f"Dependent coordinate {idx} needs 3 coefficients, got {entry!r}."
                    )
                dependent[idx] = coeffs  # type: ignore[assignment]
        if set(free) != {"x", "y", "z"}:
            raise ValueError(
                f"Layout must mark exactly one each of x, y, z; got {sorted(free)}."
            )
        return cls(axes=(free["x"], free["y"], free["z"]), dependent=dependent)

    def free_to_full(self, xyz: np.ndarray) -> np.ndarray:
        r"""Embed sphere points ``(x, y, z)`` back into full ``D``-dim space.

        Free axes receive ``x/y/z`` directly; dependent axes are filled with
        their linear prediction ``a\cdot x + b\cdot y + c\cdot z``.  Used to test the
        reconstructed grid against the shard's constraint matrix ``A``.

        :param xyz: ``(N, 3)`` array of sphere points.
        :return: ``(N, D)`` array in the CMF coordinate frame.
        """
        n = len(self.axes) + len(self.dependent)
        full = np.zeros((xyz.shape[0], n), dtype=float)
        for slot, ax in enumerate(self.axes):
            full[:, ax] = xyz[:, slot]
        for idx, (a, b, c) in self.dependent.items():
            full[:, idx] = a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2]
        return full

    def project(self, directions: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
        """Select trajectories in the subspace and return their sphere points.

        :param directions: ``(N, D)`` raw direction vectors (CMF-symbol order).
        :param tol: relative tolerance for the dependent-coordinate constraint.
        :return: ``(unit_xyz, mask)`` where ``mask`` is the boolean row-filter of
            kept trajectories and ``unit_xyz`` is their ``(M, 3)`` unit-sphere
            projection (``M = mask.sum()``).
        """
        x = directions[:, self.axes[0]]
        y = directions[:, self.axes[1]]
        z = directions[:, self.axes[2]]

        scale = np.linalg.norm(directions, axis=1)
        scale[scale == 0] = 1.0

        mask = np.ones(directions.shape[0], dtype=bool)
        for idx, (a, b, c) in self.dependent.items():
            predicted = a * x + b * y + c * z
            mask &= np.abs(directions[:, idx] - predicted) <= tol * scale

        xyz = np.column_stack([x, y, z])[mask]
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return xyz / norms, mask


# ===========================================================================
# Data loading (JSONL pipeline)
# ===========================================================================

def load_shards(export_cmfs: str, constant: Constant, cmf_id: Optional[str] = None):
    """Reconstruct live ``Shard`` objects for *constant* from the JSONL export.

    Thin wrapper over :func:`load_shards_from_export` that returns just the
    list of shards for the single constant (or an empty list).  When *cmf_id*
    is given, only shards belonging to that CMF are returned - drawing every
    CMF's shards at once produces an enormous figure (and can exhaust the X
    server's pixmap memory), so restricting to one CMF is the normal mode.

    :param export_cmfs: the ``EXPORT_CMFS`` directory (per-constant formatter
        JSONs + ``<cmf>__shards.jsonl`` files).
    :param constant: the constant whose shards to load.
    :param cmf_id: optional CMF name to keep (e.g. ``"pFq_2_1_-1__0_0_0"`` for
        ``2F1(-1)``); ``None`` keeps every CMF.
    :return: list of reconstructed ``Shard`` objects.
    """
    # Restrict reconstruction to the requested CMF.  Reconstructing every CMF's
    # shards (each needs sympy hyperplane extraction) and discarding all but one
    # is dramatically slower - measured ~58x on the example data - so push the
    # filter down into ``load_shards_from_export`` instead of filtering after.
    relevant = {constant.name: {cmf_id}} if cmf_id is not None else None
    by_const = load_shards_from_export(export_cmfs, [constant], relevant)
    shards = by_const.get(constant, [])

    if cmf_id is not None and not shards:
        # Distinguish "no such CMF" from "CMF has no shards for this constant".
        available = _available_cmf_ids(export_cmfs, constant)
        raise ValueError(
            f"No shards for cmf_id={cmf_id!r} under constant {constant.name!r}. "
            f"Available CMFs: {available}"
        )

    scope = f" of CMF {cmf_id!r}" if cmf_id else ""
    print(f"Loaded {len(shards)} shards{scope} for constant {constant.name!r}.")
    return shards


def _available_cmf_ids(export_cmfs: str, constant: Constant) -> List[str]:
    """List CMF ids available for *constant* by scanning formatter JSON names.

    Cheap (filename scan, no shard reconstruction) - used for error messages
    and the ``--list-cmfs`` CLI command.

    :param export_cmfs: the ``EXPORT_CMFS`` directory.
    :param constant: the constant whose per-constant subdirectory to scan.
    :return: sorted list of CMF ids (formatter JSON stems).
    """
    safe = "".join(c for c in constant.name if c.isalnum() or c in ("-", "_"))
    const_dir = Path(export_cmfs) / safe
    if not const_dir.is_dir():
        return []
    return sorted(f.stem for f in const_dir.glob("*.json"))


def _merge_jsonl(path: Path) -> Dict[str, dict]:
    """Read a per-shard JSONL, folding patch lines into base records.

    Mirrors ``data_utils/search_data.py._merge_jsonl`` (last-write-wins on scalar
    fields, **union** on ``extended_metrics``).  The union is essential: the
    Tier-2/Tier-3 attributes (eigenvalues, gcd_slope, ...) arrive on a *separate*
    patch line written by the background worker, and the base line for the same
    trajectory carries ``extended_metrics: {}``.  A plain ``dict.update`` would
    let whichever line comes last win outright, so an empty ``{}`` could wipe the
    attributes - which is exactly why attribute spheres came out blank.  Unioning
    the two ``extended_metrics`` dicts preserves the populated keys regardless of
    line order.

    :param path: the ``<shard_id>.jsonl`` file.
    :return: ``{trajectory_id: merged_record}``.
    """
    merged: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = record.get("trajectory_id")
            if tid is None:
                continue
            if tid not in merged:
                merged[tid] = record
            else:
                existing_em = dict(merged[tid].get("extended_metrics") or {})
                new_em = dict(record.get("extended_metrics") or {})
                merged[tid].update(record)
                merged[tid]["extended_metrics"] = {**existing_em, **new_em}
    return merged


def load_shard_trajectories(
    shard,
    value_fn: "ValueFn",
    results_root: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a shard's trajectory directions + a scalar value from its JSONL file.

    The file is located by ``derive_cmf_and_shard_ids(shard)`` → ``<shard_id>``
    under *results_root* (the flat ``EXPORT_SEARCH_RESULTS`` layout).  The scalar
    plotted on the sphere is produced by *value_fn* (e.g. :func:`delta_value`,
    :func:`extended_metric_value`, :func:`sympy_attribute_value`); records for
    which it returns ``None`` / a non-finite number are skipped, so attributes
    that are absent or not-yet-computed on some trajectories are dropped cleanly.

    :param shard: a reconstructed ``Shard``.
    :param value_fn: callable ``(record_dict) -> Optional[float]`` extracting the
        scalar to colour the sphere with.
    :param results_root: the ``EXPORT_SEARCH_RESULTS`` directory.
    :return: ``(directions, values)`` - an ``(N, D)`` float array of raw
        directions and an ``(N,)`` array of values (empty arrays if none).
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return np.empty((0, len(shard.symbols))), np.empty((0,))

    merged = _merge_jsonl(path)
    directions: List[List[float]] = []
    values: List[float] = []
    for rec in merged.values():
        direction = rec.get("direction")
        if direction is None:
            continue
        try:
            value = value_fn(rec)
        except Exception:
            value = None
        if value is None or not np.isfinite(value):
            continue
        directions.append([float(v) for v in direction])
        values.append(float(value))

    if not directions:
        return np.empty((0, len(shard.symbols))), np.empty((0,))
    return np.asarray(directions, dtype=float), np.asarray(values, dtype=float)


def load_shard_samples(
    shard,
    value_fn: "ValueFn",
    constant: Constant,
    results_root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load **all** sampled trajectories of a shard (for the scatter view).

    Unlike :func:`load_shard_trajectories` (which keeps only finite-value records
    for the surface interpolation), this returns *every* sampled trajectory with
    a direction, so the scatter plot shows the full sampling - including the
    trajectories that were **not** identified as *constant*.

    The colour value comes from *value_fn* (δ by default, or any attribute - e.g.
    ``extended_metric_value("asymptotic_digits_per_step")``), so the scatter
    colours and the colorbar match the requested attribute, not just δ.

    :param shard: a reconstructed ``Shard``.
    :param value_fn: callable ``(record_dict) -> Optional[float]`` extracting the
        scalar to colour each point with.
    :param constant: the constant whose identification flag to read.
    :param results_root: the ``EXPORT_SEARCH_RESULTS`` directory.
    :return: ``(directions, values, identified)`` - an ``(N, D)`` float array of
        raw directions, an ``(N,)`` value array (``NaN`` where missing/non-finite),
        and an ``(N,)`` boolean array of identification flags.
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return (np.empty((0, len(shard.symbols))), np.empty((0,)), np.empty((0,), bool))

    merged = _merge_jsonl(path)
    directions: List[List[float]] = []
    values: List[float] = []
    identified: List[bool] = []
    for rec in merged.values():
        direction = rec.get("direction")
        if direction is None:
            continue
        try:
            val = value_fn(rec)
        except Exception:
            val = None
        ident = bool((rec.get("identified") or {}).get(constant.name))
        directions.append([float(v) for v in direction])
        values.append(float(val) if val is not None and np.isfinite(val) else np.nan)
        identified.append(ident)

    if not directions:
        return (np.empty((0, len(shard.symbols))), np.empty((0,)), np.empty((0,), bool))
    return (np.asarray(directions, dtype=float),
            np.asarray(values, dtype=float),
            np.asarray(identified, dtype=bool))


def shard_has_identified(shard, constant: Constant, results_root: str) -> bool:
    """Return ``True`` if the shard has ≥1 trajectory identified for *constant*.

    "Identified" = a converging trajectory (LIReC found a p/q for the constant),
    read from each record's per-constant ``identified`` dict.  Used to skip
    shards that contain no converging trajectory at all.

    :param shard: a reconstructed ``Shard``.
    :param constant: the constant to check identification for.
    :param results_root: the ``EXPORT_SEARCH_RESULTS`` directory.
    :return: whether any trajectory in the shard's JSONL is identified.
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return False
    for rec in _merge_jsonl(path).values():
        ident = rec.get("identified") or {}
        if ident.get(constant.name):
            return True
    return False


def load_path_directions(
    shard,
    results_root: str,
) -> np.ndarray:
    """Load an *ordered* list of direction vectors for a shard's overlay path.

    Used for the "second results directory" overlay mode (e.g. a gradient-ascent
    run): the trajectories in that shard's JSONL are returned **in file order**,
    forming the polyline drawn on top of the δ field.

    :param shard: a reconstructed ``Shard``.
    :param results_root: the second ``EXPORT_SEARCH_RESULTS`` directory holding
        the path trajectories.
    :return: ``(K, D)`` array of raw direction vectors in traversal order
        (empty if the file is absent).
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return np.empty((0, len(shard.symbols)))

    # Preserve first-seen order (insertion-ordered dict from the merge).
    merged = _merge_jsonl(path)
    dirs = [
        [float(v) for v in rec["direction"]]
        for rec in merged.values()
        if rec.get("direction") is not None
    ]
    return np.asarray(dirs, dtype=float) if dirs else np.empty((0, len(shard.symbols)))


# ===========================================================================
# Geometry helpers (carried over from the old script)
# ===========================================================================

def get_rotation_matrix(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    r"""Rotation aligning ``vec1`` onto ``vec2`` (Rodrigues' formula).

    Handles the parallel / antiparallel degenerate cases explicitly so the
    180 rotation preserves the right-hand rule.

    :param vec1: source vector.
    :param vec2: target vector.
    :return: a ``3x3`` rotation matrix ``R`` with ``R\cdot vec1 ∥ vec2``.
    """
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    c = np.dot(a, b)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


def _camera_vector(elev: float, azim: float) -> np.ndarray:
    """Unit view direction for a matplotlib 3-D camera at ``(elev, azim)`` deg."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_sphere_horizon(ax, cam_elev: float, cam_azim: float,
                        line_width: float = 1.0, zorder: float = 2) -> None:
    """Draw the silhouette circle of the sphere as seen from the camera.

    :param ax: a 3-D matplotlib axis.
    :param cam_elev: camera elevation in degrees.
    :param cam_azim: camera azimuth in degrees.
    :param line_width: stroke width of the silhouette circle.
    :param zorder: draw order (raised above the surface when ``computed_zorder``
        is disabled, so the border sits on top of the filled patch).
    """
    cam = _camera_vector(cam_elev, cam_azim)
    u = (np.array([-cam[1], cam[0], 0]) if abs(cam[0]) > 0.1 or abs(cam[1]) > 0.1
         else np.array([0, -cam[2], cam[1]]))
    u = u / np.linalg.norm(u)
    v = np.cross(cam, u)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        1.001 * (np.cos(t) * u[0] + np.sin(t) * v[0]),
        1.001 * (np.cos(t) * u[1] + np.sin(t) * v[1]),
        1.001 * (np.cos(t) * u[2] + np.sin(t) * v[2]),
        color="black", linewidth=line_width, alpha=0.8, zorder=zorder,
    )


def plot_hyperplanes_on_sphere(ax, shard, spec: ProjectionSpec,
                               cam_elev: float, cam_azim: float,
                               line_width: float = 1.0, zorder: float = 5) -> None:
    """Draw the great circles where the shard's bounding hyperplanes meet the sphere.

    Each hyperplane normal (a row of ``shard.A``) is projected onto the three
    chosen coordinate axes (``spec.axes``); the camera-facing half of the
    resulting great circle is drawn.

    :param ax: a 3-D matplotlib axis.
    :param shard: the shard whose constraint matrix ``A`` supplies the normals.
    :param spec: the active projection spec (selects the 3 axes).
    :param cam_elev: camera elevation in degrees.
    :param cam_azim: camera azimuth in degrees.
    :param line_width: stroke width of the great circles.
    :param zorder: draw order for the great circles.
    """
    if shard.A is None:
        return
    A = np.array(shard.A, dtype=float)
    cam = _camera_vector(cam_elev, cam_azim)
    theta = np.linspace(0, 2 * np.pi, 300)

    for row in A:
        n = row[list(spec.axes)]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            continue
        N = n / norm_n
        U = (np.array([-N[1], N[0], 0.0]) if abs(N[0]) > 0.1 or abs(N[1]) > 0.1
             else np.array([0.0, -N[2], N[1]]))
        U /= np.linalg.norm(U)
        V = np.cross(N, U)
        cx, cy, cz = (1.001 * (np.cos(theta) * U[i] + np.sin(theta) * V[i]) for i in range(3))
        pts = np.vstack([cx, cy, cz]).T
        hidden = pts.dot(cam) < -0.1
        cx[hidden], cy[hidden], cz[hidden] = np.nan, np.nan, np.nan
        ax.plot(cx, cy, cz, color="black", linewidth=line_width, alpha=0.85, zorder=zorder)


# ===========================================================================
# Core surface rendering
# ===========================================================================

def _best_camera(unit_xyz: np.ndarray, deltas: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Camera (elev, azim) pointing at the highest-δ direction.

    :param unit_xyz: ``(N, 3)`` unit-sphere points.
    :param deltas: ``(N,)`` δ values.
    :return: ``(elev_deg, azim_deg, v_best)``.
    """
    v_best = unit_xyz[int(np.nanargmax(deltas))]
    elev = float(np.degrees(np.arcsin(np.clip(v_best[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(v_best[1], v_best[0])))
    return elev, azim, v_best


def render_shard_surface(
    ax,
    shard,
    unit_xyz: np.ndarray,
    deltas: np.ndarray,
    spec: ProjectionSpec,
    cmap,
    norm,
    *,
    grid_res: int = 300,
    cone_tol: float = 1e-4,
    fill_cone: bool = False,
    smooth_sigma: float = 1.5,
) -> None:
    r"""Interpolate and draw one shard's δ field as a coloured patch on the sphere.

    Pipeline: rotate the best direction to the equator for a well-conditioned
    local ``griddata``, **linearly** interpolate δ (this is naturally bounded by
    the convex hull of the sampled directions - ``griddata`` returns ``NaN``
    outside it, so no extrapolation), smooth, restore to the global frame, clip
    to the shard cone (``A\cdot v ≤ 0``), then ``plot_surface`` with per-face colours.

    The patch boundary is therefore ``hull(samples) ∩ cone``: clean and honest.
    With dense data the hull reaches the great circles (the classic look); with
    sparse data it is a smaller patch covering only where trajectories actually
    landed.

    The old ``nearest``-fill (extrapolating the whole rectangle, then trimming
    to a KNN radius) is *not* used by default: on sparsely-sampled shards it
    paints empty cone corners with a distant sample's δ, producing flat Voronoi
    facets (sharp wedges).  Set ``fill_cone=True`` to opt back into filling the
    entire cone (sensible only when the data densely covers it).

    :param ax: a 3-D matplotlib axis (already created).
    :param shard: the shard (supplies ``A`` for cone trimming).
    :param unit_xyz: ``(N, 3)`` unit-sphere projection of the kept trajectories.
    :param deltas: ``(N,)`` δ values aligned with ``unit_xyz``.
    :param spec: the active projection spec.
    :param cmap: a matplotlib colormap.
    :param norm: a matplotlib ``Normalize`` shared across all spheres.
    :param grid_res: interpolation grid resolution per axis.
    :param cone_tol: tolerance for the ``A\cdot v ≤ 0`` cone-membership trim.
    :param fill_cone: extrapolate (nearest) to fill the whole cone beyond the
        sample hull.  Off by default - only meaningful for densely-sampled shards.
    :param smooth_sigma: NaN-aware Gaussian blur (in grid cells) applied to the
        interpolated field; smooths grid stair-steps without bleeding colour past
        the patch boundary.  ``0`` disables smoothing.
    """
    _, _, v_best = _best_camera(unit_xyz, deltas)
    R = get_rotation_matrix(v_best, np.array([1.0, 0.0, 0.0]))
    rotated = unit_xyz @ R.T

    dt = np.arctan2(rotated[:, 1], rotated[:, 0])
    dp = np.arccos(np.clip(rotated[:, 2], -1, 1))
    gt, gp = np.mgrid[
        np.min(dt) - 0.05:np.max(dt) + 0.05:grid_res * 1j,
        np.min(dp) - 0.05:np.max(dp) + 0.05:grid_res * 1j,
    ]

    # Linear interpolation - NaN outside the convex hull of the samples (no
    # extrapolation).  This hull is the honest, clean patch boundary.
    grid_delta = griddata((dt, dp), deltas, (gt, gp), method="linear")

    if fill_cone:
        # Opt-in: extrapolate into the rest of the cone with a nearest lookup.
        nan_mask = np.isnan(grid_delta)
        if nan_mask.any():
            grid_delta[nan_mask] = griddata(
                (dt, dp), deltas, (gt[nan_mask], gp[nan_mask]), method="nearest"
            )

    # NaN-aware Gaussian smoothing: blur the valid field and the validity mask
    # separately and divide, so colour does not bleed across the NaN boundary
    # (a plain gaussian_filter would smear the patch edge outward).
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        valid = ~np.isnan(grid_delta)
        filled = np.where(valid, grid_delta, 0.0)
        weight = gaussian_filter(valid.astype(float), smooth_sigma, mode="constant")
        blurred = gaussian_filter(filled, smooth_sigma, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            grid_delta = np.where(valid, blurred / np.maximum(weight, 1e-6), np.nan)

    gx = np.cos(gt) * np.sin(gp)
    gy = np.sin(gt) * np.sin(gp)
    gz = np.cos(gp)
    grid_pts = np.c_[gx.ravel(), gy.ravel(), gz.ravel()] @ R  # back to global frame

    # --- clip to the shard cone (embed sphere pts into full CMF dim first) ---
    if shard.A is not None:
        A = np.array(shard.A, dtype=float)
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        full = spec.free_to_full(grid_pts)
        grid_delta.ravel()[np.any(full @ A_norm.T > cone_tol, axis=1)] = np.nan

    colors = cmap(norm(grid_delta))
    colors[np.isnan(grid_delta), 3] = 0.0
    sx, sy, sz = (grid_pts[:, i].reshape(grid_res, grid_res) for i in range(3))
    # antialiased=False: ~3x faster rasterisation of the dense surface, with no
    # visible difference (adjacent quads share colours; the only real edge is the
    # transparent patch boundary).  rcount/ccount = grid_res so every grid cell
    # is drawn (matplotlib would otherwise downsample to 50).
    ax.plot_surface(
        sx, sy, sz, facecolors=colors, shade=False, antialiased=False,
        rcount=grid_res, ccount=grid_res, zorder=5,
    )


def render_shard_scatter(
    ax,
    shard,
    unit_xyz: np.ndarray,
    values: np.ndarray,
    identified: np.ndarray,
    cmap,
    norm,
    *,
    mark_identified: bool = True,
) -> None:
    """Scatter a shard's *sampled trajectory directions* as points on the sphere.

    Shows where trajectories were actually sampled (rather than an interpolated
    field).  Points sit at radius 1.02 so they rest just above the sphere.

    :param ax: a 3-D matplotlib axis (already created).
    :param shard: the shard (unused; kept for signature symmetry).
    :param unit_xyz: ``(N, 3)`` unit-sphere projection of the samples.
    :param values: ``(N,)`` colour value per sample (δ or the chosen attribute;
        ``NaN`` where missing/non-finite).
    :param identified: ``(N,)`` boolean identification flags.
    :param cmap: a matplotlib colormap (for the value when marking).
    :param norm: a matplotlib ``Normalize`` shared across all spheres.
    :param mark_identified: if ``True``, colour identified samples by *values* and
        draw the non-identified ones as faint grey; if ``False``, draw every
        sample in a single neutral colour (just the sampling geometry).
    """
    if unit_xyz.shape[0] == 0:
        return
    pts = unit_xyz * 1.02

    if not mark_identified:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#444444", s=10,
                   alpha=0.85, depthshade=False, zorder=6)
        return

    ident = identified.astype(bool)
    other = ~ident  # non-identified samples
    if other.any():
        ax.scatter(pts[other, 0], pts[other, 1], pts[other, 2], color="#bdbdbd",
                   s=7, alpha=0.6, depthshade=False, zorder=6)
    show = ident & ~np.isnan(values)
    if show.any():
        ax.scatter(pts[show, 0], pts[show, 1], pts[show, 2], c=values[show],
                   cmap=cmap, norm=norm, s=16, alpha=0.95, depthshade=False,
                   edgecolor="none", zorder=7)


def _draw_reference_grid(ax) -> None:
    """Draw a faint static lat/long wireframe (world-z up, 10 spacing)."""
    u = np.linspace(0, 2 * np.pi, 37)
    v = np.linspace(0, np.pi, 19)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="white", alpha=0.0, edgecolor="gray",
        linewidth=0.35, shade=False, zorder=1,
    )


def draw_overlay_path(ax, shard, path_dirs: np.ndarray, spec: ProjectionSpec,
                      tol: float, color: str = "black") -> None:
    """Project an ordered path of directions onto the sphere and draw it.

    Only path points that lie in the chosen subspace (per ``spec``) are drawn;
    they are placed at radius 1.01 so the line and markers sit just above the
    coloured δ surface.

    :param ax: a 3-D matplotlib axis.
    :param shard: the shard the path belongs to (unused for now, kept for API
        symmetry / future per-shard styling).
    :param path_dirs: ``(K, D)`` ordered direction vectors.
    :param spec: the active projection spec.
    :param tol: subspace membership tolerance.
    :param color: line / marker colour.
    """
    if path_dirs.size == 0:
        return
    unit_xyz, _ = spec.project(path_dirs, tol)
    if unit_xyz.shape[0] == 0:
        return
    pts = unit_xyz * 1.01
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.6,
            marker="o", markersize=3.0, zorder=10)
    # Mark the endpoint (the optimum the path climbed to).
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, s=28,
               edgecolor="white", linewidth=0.6, zorder=11)


# ===========================================================================
# Top-level figures
# ===========================================================================

def _nice_step(value_range: float) -> float:
    """Pick a human-friendly colorbar tick step (~5 ticks) for a value range."""
    if value_range <= 0 or not np.isfinite(value_range):
        return 1.0
    raw = value_range / 5.0
    mag = 10 ** np.floor(np.log10(raw))
    for mult in (1, 2, 5, 10):
        if raw <= mult * mag:
            return mult * mag
    return 10 * mag


def _make_norm(values: np.ndarray, step: Optional[float]):
    """Build a shared ``Normalize`` and the tick step for the colorbar.

    :param values: all plotted scalars across every shard.
    :param step: explicit tick step; ``None`` → auto via :func:`_nice_step`.
    :return: ``(norm, vmin, vmax, step)`` with bounds snapped to ``step``.
    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if step is None:
        step = _nice_step(hi - lo)
    vmin = np.floor(lo / step) * step
    vmax = np.ceil(hi / step) * step
    if vmin == vmax:  # constant field - widen so the colorbar is valid
        vmin, vmax = vmin - step, vmax + step
    return plt.Normalize(vmin=vmin, vmax=vmax), vmin, vmax, step


def _add_colorbar(fig, cmap, norm, vmin: float, vmax: float,
                  step: float, label: str) -> None:
    """Add the shared vertical colorbar on the right of the figure.

    :param label: the colorbar axis label (e.g. the δ label or an attribute name).
    :param step: spacing between colorbar ticks.
    """
    cbar_ax = fig.add_axes([0.87, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    ticks = np.arange(vmin, vmax + step * 0.05, step)
    cbar.set_ticks(ticks)
    decimals = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 1
    cbar.ax.set_yticklabels([f"{t:.{decimals}f}" for t in ticks], fontsize=13)
    cbar.set_label(label, fontsize=16, labelpad=15)


def generate_spheres(
    shards,
    constant: Constant,
    results_root: str,
    *,
    one_sphere_per_shard: bool = True,
    mode: str = "surface",
    mark_identified: bool = True,
    spec: Optional[ProjectionSpec] = None,
    subspace_tol: float = 1e-6,
    value_fn: Optional[ValueFn] = None,
    value_label: Optional[str] = None,
    value_step: Optional[float] = None,
    require_identified: bool = True,
    max_atlas_cols: int = 6,
    grid_res: int = 400,
    fill_cone: bool = False,
    smooth_sigma: float = 1.5,
    line_width: float = 1.0,
    zoom: float = 1.4,
    path_root: Optional[str] = None,
    explicit_paths: Optional[Dict[str, np.ndarray]] = None,
    draw_grid: bool = True,
    cmap_name: str = "coolwarm",
    show: bool = True,
):
    """Render the attribute-on-sphere projection(s) for a set of shards.

    By default the irrationality measure δ is drawn (classic behaviour).  Pass a
    *value_fn* to colour the sphere by any other trajectory attribute instead -
    e.g. ``field_value("limit_value")``, ``extended_metric_value("digits_per_step")``,
    or ``sympy_attribute_value("asymptotics", {"n": 1e6})`` for a string-formatted
    sympy attribute reduced to a number by substitution.

    :param shards: reconstructed ``Shard`` objects (e.g. from :func:`load_shards`).
    :param constant: the constant; used for the subspace default and for the
        default δ value function.
    :param results_root: ``EXPORT_SEARCH_RESULTS`` dir for the hedgehog data.
    :param one_sphere_per_shard: ``True`` → one sphere per shard (atlas, like the
        old script); ``False`` → all shards on a single shared sphere.
    :param mode: ``"surface"`` (default) draws the interpolated δ field; ``"scatter"``
        draws the sampled trajectory directions as points.
    :param mark_identified: scatter mode only - if ``True``, colour identified
        trajectories by δ and draw non-identified ones faint grey; if ``False``,
        draw every sample in one neutral colour (sampling geometry only).
    :param spec: projection spec; defaults to the identity ``(0, 1, 2)`` spec for
        a 3-D CMF.  Required (non-identity) for ``D > 3``.
    :param subspace_tol: relative tolerance for the subspace membership test.
    :param value_fn: scalar extractor (see :data:`ValueFn`); ``None`` →
        :func:`delta_value` for *constant*.
    :param value_label: colorbar label; ``None`` → the δ label (or the value
        function's name when a custom one is supplied).
    :param value_step: explicit colorbar tick step; ``None`` → auto-chosen
        (δ keeps its conventional 0.2 step).
    :param require_identified: when ``True`` (default), skip shards that have no
        trajectory identified for *constant* (no converging trajectory).
    :param max_atlas_cols: max spheres per row in the one-sphere-per-shard atlas;
        extra shards wrap onto further rows so the figure never grows so wide it
        exhausts the display's pixmap memory.
    :param grid_res: surface tessellation per axis.  Lower it (e.g. 150) for a
        responsive interactive window; higher (e.g. 400) for a final figure.
    :param fill_cone: extrapolate δ to fill the whole shard cone.  **Off by
        default** - the patch is bounded by the convex hull of the sampled
        directions (honest, no extrapolation).  Turn on only when trajectories
        densely cover the cone, otherwise empty corners get sharp Voronoi wedges.
    :param smooth_sigma: NaN-aware Gaussian smoothing (grid cells) of the δ
        field; ``0`` disables it.
    :param line_width: stroke width of the great-circle / horizon lines.
    :param zoom: ``set_box_aspect`` zoom - larger draws the spheres bigger (which
        also makes the border lines look thinner relative to the sphere).
    :param path_root: optional second ``EXPORT_SEARCH_RESULTS`` dir whose
        trajectories are drawn as an ordered overlay path per shard.
    :param explicit_paths: optional ``{shard_id: (K, D) directions}`` overriding /
        supplementing ``path_root`` for specific shards.
    :param draw_grid: draw the faint reference lat/long wireframe.
    :param cmap_name: matplotlib colormap name.
    :param show: call ``plt.show()`` before returning.
    :return: the matplotlib ``Figure``.
    """
    explicit_paths = explicit_paths or {}
    cmap = plt.get_cmap(cmap_name)

    # Default to the classic δ field; keep its conventional label + 0.2 ticks.
    if value_fn is None:
        value_fn = delta_value(constant)
        if value_label is None:
            value_label = r"Irrationality Measure ($\delta$)"
        if value_step is None:
            value_step = 0.2
    elif value_label is None:
        value_label = "Trajectory attribute"

    if mode not in ("surface", "scatter"):
        raise ValueError(f"mode must be 'surface' or 'scatter', got {mode!r}.")

    # ---- gather per-shard payloads using the projection spec -----------------
    # Each entry: (shard, unit_xyz, cam_scalar, content).  ``cam_scalar`` aims
    # the camera at the highest-δ direction; ``content`` is what the per-shard
    # draw callback consumes.
    per_shard = []
    skipped_unidentified = 0
    for shard in shards:
        if spec is None:
            spec = ProjectionSpec.identity(len(shard.symbols))
        if require_identified and not shard_has_identified(shard, constant, results_root):
            skipped_unidentified += 1
            continue

        if mode == "surface":
            directions, values = load_shard_trajectories(shard, value_fn, results_root)
            if directions.shape[0] == 0:
                continue
            unit_xyz, mask = spec.project(directions, subspace_tol)
            if unit_xyz.shape[0] < 4:  # need a few points to interpolate
                continue
            per_shard.append((shard, unit_xyz, values[mask], ("surface", values[mask])))
        else:  # scatter
            directions, vals, identified = load_shard_samples(
                shard, value_fn, constant, results_root)
            if directions.shape[0] == 0:
                continue
            unit_xyz, mask = spec.project(directions, subspace_tol)
            if unit_xyz.shape[0] == 0:
                continue
            v, ident = vals[mask], identified[mask]
            # Camera centres on identified samples (fall back to all if none).
            cam_scalar = np.where(ident, v, np.nan)
            if not np.isfinite(cam_scalar).any():
                cam_scalar = np.nan_to_num(v, nan=0.0)
            per_shard.append((shard, unit_xyz, cam_scalar, ("scatter", v, ident)))

    if skipped_unidentified:
        print(f"Skipped {skipped_unidentified} shard(s) with no identified "
              f"trajectory for {constant.name!r}.")

    if not per_shard:
        raise RuntimeError("No shard had enough in-subspace trajectories to plot.")
    print(f"Plotting {len(per_shard)} shard(s).")

    # Colour scale from the values actually drawn coloured: the interpolated
    # field (surface), or the coloured scatter points (identified & finite when
    # marking, else every finite sample).
    chunks = []
    for _, _, _, c in per_shard:
        if c[0] == "surface":
            chunks.append(c[1])
        else:
            _, v, ident = c
            sel = (ident & np.isfinite(v)) if mark_identified else np.isfinite(v)
            chunks.append(v[sel])
    all_values = np.concatenate(chunks) if chunks else np.empty((0,))
    if all_values.size == 0:
        print(f"WARNING: no finite values for {value_label!r} on any drawn "
              f"trajectory - the spheres will have no coloured data.  The "
              f"attribute may be absent/None for these records (e.g. not computed "
              f"in this run), or only present on non-identified trajectories.")
        all_values = np.array([0.0, 1.0])  # nothing coloured - neutral scale
    else:
        print(f"{all_values.size} coloured value(s); "
              f"range [{all_values.min():.3g}, {all_values.max():.3g}].")
    norm, vmin, vmax, value_step = _make_norm(all_values, value_step)

    def _resolve_path(shard) -> np.ndarray:
        _, shard_id, _ = derive_cmf_and_shard_ids(shard)
        if shard_id in explicit_paths:
            return np.asarray(explicit_paths[shard_id], dtype=float)
        if path_root is not None:
            return load_path_directions(shard, path_root)
        return np.empty((0, len(shard.symbols)))

    def _draw_content(ax, shard, unit_xyz, content) -> None:
        if content[0] == "surface":
            render_shard_surface(ax, shard, unit_xyz, content[1], spec, cmap, norm,
                                 grid_res=grid_res, fill_cone=fill_cone,
                                 smooth_sigma=smooth_sigma)
        else:
            _, deltas, identified = content
            render_shard_scatter(ax, shard, unit_xyz, deltas, identified, cmap, norm,
                                 mark_identified=mark_identified)

    print(f'generating {"individual spheres" if one_sphere_per_shard else "a single shared sphere"}...')
    # In surface mode the border great circles should sit on top of the filled
    # patch; matplotlib's default depth-sort (``computed_zorder``) can paint the
    # surface over them, so the layout disables it and draws the lines above.
    # Scatter mode keeps depth ordering so the points stay on top of the lines.
    layout_kwargs = dict(line_width=line_width, zoom=zoom, lines_on_top=(mode == "surface"))
    if one_sphere_per_shard:
        fig = _generate_atlas(per_shard, spec, cmap, norm, subspace_tol, draw_grid,
                              _resolve_path, _draw_content, max_atlas_cols, **layout_kwargs)
    else:
        fig = _generate_single(per_shard, spec, cmap, norm, subspace_tol, draw_grid,
                               _resolve_path, _draw_content, **layout_kwargs)

    _add_colorbar(fig, cmap, norm, vmin, vmax, value_step, value_label)
    print(f'Done!')
    if show:
        plt.show()
    return fig


# Draw order used when the border lines must sit on top of the filled surface
# (``computed_zorder=False``): reference grid < surface (zorder 5) < lines.
_LINE_ZORDER_ON_TOP = 12


def _line_zorders(lines_on_top: bool) -> Tuple[float, float]:
    """Return ``(horizon_zorder, hyperplane_zorder)`` for the chosen mode.

    When *lines_on_top* the caller has set ``computed_zorder=False``, so the
    lines need an explicit zorder above the surface (5); otherwise the defaults
    keep matplotlib's depth ordering (scatter points stay on top of the lines).
    """
    if lines_on_top:
        return _LINE_ZORDER_ON_TOP, _LINE_ZORDER_ON_TOP
    return 2, 5


def _setup_sphere_ax(ax, zoom: float, lines_on_top: bool) -> None:
    """Apply the common 3-D axis setup, disabling depth-sort when needed."""
    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    if lines_on_top:
        # Honour manual zorder so the border lines paint over the surface patch
        # instead of being depth-sorted behind it.
        ax.computed_zorder = False


def _generate_atlas(per_shard, spec, cmap, norm, tol, draw_grid, resolve_path,
                    draw_content, max_cols: int = 6, *,
                    line_width: float = 1.0, zoom: float = 1.4,
                    lines_on_top: bool = True):
    """Render one sphere per shard, wrapping into a ``rows x cols`` grid.

    A single row of N spheres becomes impractically wide for large N (and can
    exhaust the X server's pixmap limit), so spheres wrap onto multiple rows
    capped at *max_cols* columns.  ``draw_content(ax, shard, unit_xyz, content)``
    draws the per-shard δ surface or scatter.
    """
    hz, pz = _line_zorders(lines_on_top)
    n = len(per_shard)
    cols = max(1, min(max_cols, n))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols + 1.5, 4.2 * rows), dpi=200)
    for idx, (shard, unit_xyz, cam_scalar, content) in enumerate(per_shard):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _setup_sphere_ax(ax, zoom, lines_on_top)

        draw_content(ax, shard, unit_xyz, content)
        if draw_grid:
            _draw_reference_grid(ax)

        elev, azim, _ = _best_camera(unit_xyz, cam_scalar)
        draw_sphere_horizon(ax, elev, azim, line_width, hz)
        plot_hyperplanes_on_sphere(ax, shard, spec, elev, azim, line_width, pz)
        draw_overlay_path(ax, shard, resolve_path(shard), spec, tol)

        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.03,
                        wspace=-0.15, hspace=0.05)
    return fig


def _generate_single(per_shard, spec, cmap, norm, tol, draw_grid, resolve_path,
                     draw_content, *, line_width: float = 1.0, zoom: float = 1.4,
                     lines_on_top: bool = True):
    """Render every shard on one shared sphere (single global camera)."""
    hz, pz = _line_zorders(lines_on_top)
    fig = plt.figure(figsize=(6.0, 5.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _setup_sphere_ax(ax, zoom, lines_on_top)

    # Global camera = direction of the best δ across all shards.
    global_xyz = np.concatenate([u for _, u, _, _ in per_shard])
    global_scalar = np.concatenate([s for _, _, s, _ in per_shard])
    elev, azim, _ = _best_camera(global_xyz, global_scalar)

    if draw_grid:
        _draw_reference_grid(ax)
    draw_sphere_horizon(ax, elev, azim, line_width, hz)

    for shard, unit_xyz, _cam, content in per_shard:
        draw_content(ax, shard, unit_xyz, content)
        plot_hyperplanes_on_sphere(ax, shard, spec, elev, azim, line_width, pz)
        draw_overlay_path(ax, shard, resolve_path(shard), spec, tol)

    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.03)
    return fig


# ===========================================================================
# CLI
# ===========================================================================

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXPORT_CMFS = str(_HERE / ".." / "examples" / "CMFs")
_DEFAULT_RESULTS = str(_HERE / ".." / "examples" / "search results")


def _resolve_constant(name: str, value_str: Optional[str]) -> Constant:
    """Resolve a constant by name, registering it if not already known.

    Looks first in the registry (importing the ready-made constants so e/π/...
    are available).  Falls back to ``--constant-value`` as a sympy expression,
    with a built-in shortcut for ``log-2`` (the example data's constant).

    :param name: the constant name (must match the JSONL ``delta_estimate`` key).
    :param value_str: optional sympy expression to define an unknown constant.
    :raises SystemExit: if the constant is unknown and no value is given.
    :return: the resolved :class:`Constant`.
    """
    import sympy as sp
    try:  # registers e, pi, euler_gamma, catalan, ... into Constant.registry
        import dreamer.utils.constants.ready_made_consts  # noqa: F401
    except Exception:
        pass

    if name in Constant.registry:
        return Constant.registry[name]
    if value_str:
        return Constant(name, sp.sympify(value_str))
    if name == "log-2":
        return Constant(name, sp.log(2))
    raise SystemExit(
        f"Unknown constant {name!r}.  Pass --constant-value '<sympy expr>' "
        f"(e.g. --constant log-2 --constant-value 'log(2)')."
    )


def _parse_subs(text: Optional[str]) -> Dict[str, float]:
    """Parse ``"n=1e6,m=2"`` into ``{"n": 1e6, "m": 2.0}`` for sympy substitution."""
    subs: Dict[str, float] = {}
    if not text:
        return subs
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, _, val = piece.partition("=")
        subs[key.strip()] = float(val)
    return subs


def _parse_layout(text: Optional[str]) -> Optional[ProjectionSpec]:
    r"""Parse a ``--layout`` string into a :class:`ProjectionSpec` (or ``None``).

    Tokens are comma-separated, one per CMF coordinate: ``x`` / ``y`` / ``z`` for
    the three free sphere axes, and ``a:b:c`` for a dependent coordinate equal to
    ``a\cdot x + b\cdot y + c\cdot z``.  Example (4-D, ``(x, y, z, x − y)``): ``x,y,z,1:-1:0``.

    :param text: the layout string, or ``None`` for the identity 3-D spec.
    :return: the parsed spec, or ``None`` when *text* is empty.
    """
    if not text:
        return None
    layout: List = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok.lower() in ("x", "y", "z"):
            layout.append(tok.lower())
        else:
            layout.append(tuple(float(c) for c in tok.split(":")))
    return ProjectionSpec.from_layout(layout)


def _resolve_value_fn(args) -> Tuple[Optional[ValueFn], Optional[str]]:
    """Build the value function + colorbar label from the CLI value flags.

    :param args: parsed argparse namespace.
    :return: ``(value_fn, value_label)``; ``(None, None)`` selects the δ default.
    """
    if args.field:
        return field_value(args.field), args.value_label or args.field
    if args.metric:
        return extended_metric_value(args.metric), args.value_label or args.metric
    if args.sympy_attr:
        subs = _parse_subs(args.subs)
        label = args.value_label or f"{args.sympy_attr}({args.subs or ''})"
        return sympy_attribute_value(args.sympy_attr, subs), label
    if args.eigen_lognorm:
        idx = args.eigen_lognorm - 1  # 1-based on the CLI (λ1, λ2) → 0-based
        label = args.value_label or rf"$\log|\lambda_{args.eigen_lognorm}| / \|v\|$"
        return eigenvalue_lognorm_value(idx), label
    if args.numeric_convergence_rate:
        label = args.value_label or rf"$(\log|\lambda_2| - \log|\lambda_1|) / \|v\|$"
        return convergence_rate(neg=False), label
    if args.neg_numeric_convergence_rate:
        label = args.value_label or rf"$(\log|\lambda_1| - \log|\lambda_2|) / \|v\|$"
        return convergence_rate(neg=True), label

    return None, args.value_label  # δ default


def _build_cli():
    """Construct the argument parser for the standalone graphing CLI."""
    import argparse

    p = argparse.ArgumentParser(
        prog="shard_delta_sphere_jsonl.py",
        description=(
            "Draw δ (or another trajectory attribute) on sphere projections of a "
            "CMF's shards, read from the JSONL pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--constant", default="log-2",
                   help="Constant name (matches the JSONL delta_estimate key).")
    p.add_argument("--constant-value", default=None,
                   help="Sympy expression defining the constant if it is unknown.")
    p.add_argument("--cmf", default=None,
                   help="CMF id to draw, e.g. pFq_2_1_-1__0_0_0 for 2F1(-1).")
    p.add_argument("--list-cmfs", action="store_true",
                   help="List the CMF ids available for the constant and exit.")
    p.add_argument("--export-cmfs", default=_DEFAULT_EXPORT_CMFS,
                   help="EXPORT_CMFS directory (shards + formatter JSON).")
    p.add_argument("--results", default=_DEFAULT_RESULTS,
                   help="EXPORT_SEARCH_RESULTS directory (per-shard JSONL).")

    p.add_argument("--single", action="store_true",
                   help="Draw all shards on one shared sphere (default: one per shard).")
    p.add_argument("--scatter", action="store_true",
                   help="Plot the sampled trajectory directions as points instead "
                        "of the interpolated δ surface.")
    p.add_argument("--mark-identified", action="store_true",
                   help="Scatter mode: colour identified trajectories by δ and draw "
                        "non-identified ones faint grey (default: all one colour).")
    p.add_argument("--include-unidentified", action="store_true",
                   help="Also draw shards with no identified/converging trajectory.")
    p.add_argument("--line-width", type=float, default=1.0,
                   help="Stroke width of the great-circle / horizon lines.")
    p.add_argument("--zoom", type=float, default=1.4,
                   help="Sphere zoom; larger = bigger spheres / thinner-looking borders.")
    p.add_argument("--max-cols", type=int, default=6,
                   help="Max spheres per row in the atlas (wraps onto more rows).")
    p.add_argument("--grid-res", type=int, default=None,
                   help="Surface tessellation per axis.  Default 400 when saving "
                        "(publication quality), 150 for the interactive window "
                        "(the Tk backend is very slow on dense 3-D surfaces).")
    p.add_argument("--fill-cone", action="store_true",
                   help="Extrapolate δ to fill the whole shard cone.  Default off "
                        "(patch bounded by the data's convex hull - honest, no "
                        "extrapolation).  Use only when the data densely covers "
                        "the cone, else empty corners get sharp Voronoi wedges.")
    p.add_argument("--smooth-sigma", type=float, default=1.5,
                   help="NaN-aware Gaussian smoothing of the field (grid cells); "
                        "0 disables.")
    p.add_argument("--layout", default=None,
                   help="D>3 subspace, e.g. 'x,y,z,1:-1:0' (free axes x/y/z, "
                        "dependent coord a:b:c = a*x+b*y+c*z).")
    p.add_argument("--subspace-tol", type=float, default=1e-6,
                   help="Relative tolerance for the subspace membership test.")

    val = p.add_mutually_exclusive_group()
    val.add_argument("--field", default=None,
                     help="Colour by a top-level record field (e.g. limit_value).")
    val.add_argument("--metric", default=None,
                     help="Colour by an extended_metrics key (e.g. digits_per_step).")
    val.add_argument("--sympy-attr", default=None,
                     help="Colour by a string-sympy attribute (e.g. asymptotics).")
    val.add_argument("--eigen-lognorm", type=int, default=None, metavar="I",
                     help="Colour by log|λ_I| / ||v|| (I is 1-based: 1=λ1, 2=λ2), "
                          "from extended_metrics['eigenvalues'].")
    val.add_argument("--numeric-convergence-rate", action="store_true",
                     help="Colour by (log|λ2| - log|λ1|) / ||v||, the numeric convergence rate estimate derived from the top two "
                          "eigenvalues.")
    val.add_argument("--neg-numeric-convergence-rate", action="store_true",
                     help="Colour by -(log|λ2| - log|λ1|) / ||v||, the numeric convergence rate estimate derived from the top two "
                          "eigenvalues.")
    p.add_argument("--subs", default=None,
                   help="Substitutions for --sympy-attr, e.g. 'n=1e6'.")
    p.add_argument("--value-label", default=None, help="Colorbar label override.")
    p.add_argument("--value-step", type=float, default=None,
                   help="Colorbar tick step (auto for non-δ attributes).")

    p.add_argument("--path-root", default=None,
                   help="Second results dir whose trajectories overlay as a path.")
    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    p.add_argument("--save", default=None,
                   help="Save the figure to this path instead of showing it.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful with --save / headless).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: parse flags, load shards, render the projection(s)."""
    args = _build_cli().parse_args(argv)
    const = _resolve_constant(args.constant, args.constant_value)

    if args.list_cmfs:
        cmfs = _available_cmf_ids(args.export_cmfs, const)
        print(f"CMFs available for {const.name!r} ({len(cmfs)}):")
        for c in cmfs:
            print(f"  {c}")
        return 0

    if args.cmf is None:
        raise SystemExit(
            "Specify --cmf <id> (or --list-cmfs to see the options).  Drawing "
            "every CMF at once is not supported (huge figure / pixmap exhaustion)."
        )

    shards = load_shards(args.export_cmfs, const, cmf_id=args.cmf)
    value_fn, value_label = _resolve_value_fn(args)

    # The Tk interactive backend renders dense 3-D surfaces extremely slowly
    # (the window can stay blank for a long time).  Use a lighter tessellation
    # for on-screen viewing; 300 is plenty for a saved figure (the field is
    # smooth - 400 doubles the save time for no visible gain).  --grid-res
    # overrides, e.g. 400 for a final publication image.
    interactive = not args.save and not args.no_show
    grid_res = args.grid_res if args.grid_res is not None else (150 if interactive else 300)
    if interactive:
        print(f"Rendering interactively at grid_res={grid_res} (use --save for "
              f"a higher-resolution image, or --grid-res to override).")

    fig = generate_spheres(
        shards,
        const,
        args.results,
        one_sphere_per_shard=not args.single,
        mode="scatter" if args.scatter else "surface",
        mark_identified=args.mark_identified,
        spec=_parse_layout(args.layout),
        subspace_tol=args.subspace_tol,
        value_fn=value_fn,
        value_label=value_label,
        value_step=args.value_step,
        require_identified=not args.include_unidentified,
        max_atlas_cols=args.max_cols,
        grid_res=grid_res,
        fill_cone=args.fill_cone,
        smooth_sigma=args.smooth_sigma,
        line_width=args.line_width,
        zoom=args.zoom,
        path_root=args.path_root,
        cmap_name=args.cmap,
        show=False,
    )

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    if not args.no_show and not args.save:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
