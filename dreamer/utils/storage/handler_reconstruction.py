"""
Reconstruct trajectory handlers from stored JSONL records.

Both the Tier-3 post-process stage and the graphing stage need to turn a stored
trajectory record (a plain dict from a per-shard JSONL) back into a live
:class:`TrajectoryAttributesHandler` so attributes can be (re)computed.  That
requires the owning CMF and the ``(start, direction)`` ``Position`` objects.
These helpers are factored out here so both stages share one implementation and
neither has to import the other.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import sympy as sp
from ramanujantools import Position

from dreamer.configs.system import sys_config
from dreamer.utils.storage import Importer


def build_cmf_lookup_from_priorities(priorities) -> Dict[str, object]:
    """Return ``{cmf_name: CMF}`` from in-memory search priorities.

    Falls back to :func:`load_cmfs_from_disk` when priorities carry no usable
    CMF objects (e.g. a stage run standalone).
    """
    lookup: Dict[str, object] = {}
    for searchables in priorities.values():
        for s in searchables:
            cmf_name = getattr(s, "cmf_name", None)
            cmf = getattr(s, "cmf", None)
            if cmf_name and cmf is not None and cmf_name not in lookup:
                lookup[cmf_name] = cmf
    if lookup:
        return lookup
    return load_cmfs_from_disk()


def _iter_cmf_data(data):
    """Yield CMFData-shaped objects from a (possibly nested) imported payload."""
    if data is None:
        return
    if hasattr(data, "cmf") and hasattr(data, "cmf_name"):
        yield data
        return
    if isinstance(data, dict):
        for v in data.values():
            yield from _iter_cmf_data(v)
    elif isinstance(data, (list, tuple, set)):
        for v in data:
            yield from _iter_cmf_data(v)


def load_cmfs_from_disk() -> Dict[str, object]:
    """Best-effort load of CMFs from ``sys_config.EXPORT_CMFS``.

    Returns ``{}`` when the path is not configured or no usable files are found.
    """
    root = sys_config.EXPORT_CMFS
    if not root or not os.path.isdir(root):
        return {}
    lookup: Dict[str, object] = {}
    for const_dir in os.listdir(root):
        const_path = os.path.join(root, const_dir)
        if not os.path.isdir(const_path):
            continue
        for f_name in os.listdir(const_path):
            file_path = os.path.join(const_path, f_name)
            try:
                data = Importer.imprt(file_path)
            except Exception:
                continue
            for item in _iter_cmf_data(data):
                cmf_name = getattr(item, "cmf_name", None)
                cmf = getattr(item, "cmf", None)
                if cmf_name and cmf is not None:
                    lookup.setdefault(cmf_name, cmf)
    return lookup


def reconstruct_positions(cmf, record: dict):
    """Rebuild ``(start, direction)`` ``Position`` objects from JSONL fields.

    Tuples in the record are stored in ``cmf.matrices.keys()`` order, matching
    ``_position_to_tuple`` at write time.  Integers stored as Python ``int`` are
    wrapped back into ``sp.Integer``; any non-integer slot is parsed via
    ``sp.sympify`` so symbolic shifts survive the round-trip.
    """
    symbols = list(cmf.matrices.keys())

    def _to_position(values) -> Position:
        if values is None or len(values) != len(symbols):
            raise ValueError(
                f"Position has {len(values) if values is not None else 0} entries; "
                f"CMF expects {len(symbols)}."
            )
        mapping: Dict[object, object] = {}
        for sym, v in zip(symbols, values):
            mapping[sym] = sp.Integer(v) if isinstance(v, int) else sp.sympify(v)
        return Position(mapping)

    return _to_position(record.get("start_point")), _to_position(record.get("direction"))
