"""Tests for DataManager JSON payloads and JSONable contract behavior."""

import json

import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.jsonable import JSONable
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector


class _BrokenJSONable(JSONable):
    """Intentionally incomplete implementation used to assert abstract contract enforcement."""

    pass


def _build_demo_shard() -> Shard:
    """Create a minimal shard used in DataManager JSON roundtrip tests."""
    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(cmf, e, hps, [1, 1], shift, interior)


def test_jsonable_requires_to_json_implementation():
    """Failure-path: abstract JSONable subclasses without to_json implementation must be non-instantiable."""
    with pytest.raises(TypeError):
        _BrokenJSONable()


def test_data_manager_json_roundtrip_preserves_searchable_space_and_entries():
    """Known-answer/invariant: DataManager JSON roundtrip should preserve searchable context and stored SearchData."""
    space = _build_demo_shard()
    dm = DataManager(use_LIReC=True, searchable_space=space)

    start = space.get_interior_point()
    traj = Position({sym: sp.Integer(0) for sym in space.symbols})
    sd = SearchData(SearchVector(start, traj), delta=1.25)
    dm[sd.sv] = sd

    restored = DataManager.from_json_obj(dm.to_json())

    assert isinstance(restored.searchable_space, Shard)
    assert restored.searchable_space.const.name == space.const.name
    assert len(restored) == 1
    assert next(iter(restored.values())).delta == 1.25


# ---------------------------------------------------------------------------
# Rational start point / interior point serialization
# (regression: rational shifts were silently truncated to whole numbers
#  because int(sympy.Rational(7, 2)) == 3 without raising)
# ---------------------------------------------------------------------------

def test_position_to_tuple_preserves_rational_coordinate():
    """A rational coordinate (from a rational shift) must survive as a
    sympify-able string, not be truncated to a whole number."""
    from dreamer.utils.storage.trajectory_attributes import _position_to_tuple

    pos = Position({sp.Symbol("x"): sp.Rational(7, 2), sp.Symbol("y"): sp.Integer(4)})
    out = _position_to_tuple(pos)

    assert out == ("7/2", 4)
    assert sp.sympify(out[0]) == sp.Rational(7, 2)
    # Integer coordinate stays a plain int (byte-identical to legacy output).
    assert isinstance(out[1], int) and out[1] == 4


def test_position_to_tuple_integer_only_unchanged():
    """Integer-only positions serialize exactly as before (plain ints) so
    integer-shift runs — and their derived trajectory ids — are unchanged."""
    from dreamer.utils.storage.trajectory_attributes import _position_to_tuple

    pos = Position({sp.Symbol("x"): sp.Integer(3), sp.Symbol("y"): sp.Integer(-2)})
    assert _position_to_tuple(pos) == (3, -2)
    assert all(isinstance(v, int) for v in _position_to_tuple(pos))


def test_build_shard_dto_roundtrips_rational_interior_point():
    """build_shard_dto must store a rational interior point as a string that
    from_dict + sympify restores exactly (no truncation to a whole number)."""
    from dreamer.utils.storage.atlas_writer import build_shard_dto
    from dreamer.utils.storage.dtos import ShardDTO

    shard = _build_demo_shard()
    symbols = shard.symbols
    # Simulate a rational shift landing the interior point at 7/2 on one axis.
    shard.start_coord = Position({symbols[0]: sp.Rational(7, 2), symbols[1]: sp.Integer(2)})

    dto = build_shard_dto(shard)
    assert dto.interior_point is not None
    assert "7/2" in dto.interior_point  # not truncated to 3

    # Round through the JSONL line (json-safe) and back, as storage does.
    restored = ShardDTO.from_dict(json.loads(dto.to_json_line()))
    restored_vals = [sp.sympify(v) for v in restored.interior_point]
    assert sp.Rational(7, 2) in restored_vals
    assert sp.Integer(2) in restored_vals


def test_reconstruct_shard_from_dto_rational_interior_point():
    """reconstruct_shard_from_dto must parse a rational interior-point string
    like '7/2' (regression: the no-extractor shard-reload path called int('7/2')
    and crashed).  Mirrors the extractor's own sympify-based reconstruction."""
    from dreamer.extraction.extractor import extract_cmf_hyperplanes
    from dreamer.utils.storage.atlas_writer import (
        build_shard_dto,
        reconstruct_shard_from_dto,
    )
    from dreamer.utils.types import CMFData

    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    cmf_data = CMFData(cmf=cmf, shift=shift)

    # Build a shard whose encoding length matches the CMF's canonical hyperplanes
    # (so reconstruction does not treat it as stale), with a rational interior point.
    hps = extract_cmf_hyperplanes(cmf_data)
    encoding = [1] * len(hps)
    interior = Position({symbols[0]: sp.Rational(7, 2), symbols[1]: sp.Integer(2)})
    shard = Shard.from_cmf_data(cmf_data, [e], hps, encoding, interior)
    dto = build_shard_dto(shard)

    restored = reconstruct_shard_from_dto(dto, cmf_data, [e])
    assert restored is not None  # would be None only if the encoding were stale
    ip = restored.get_interior_point()
    assert ip[symbols[0]] == sp.Rational(7, 2)  # not truncated to 3 / no crash
    assert ip[symbols[1]] == sp.Integer(2)

