import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.extractor import ShardExtractor
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.types import ShiftCMF


def _shift_cmf(selected_points=None, only_selected=False):
    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    return ShiftCMF(cmf=cmf, shift=shift, selected_points=selected_points, only_selected=only_selected)


def test_extract_returns_whole_space_when_no_hyperplanes(monkeypatch):
    extractor = ShardExtractor(e, _shift_cmf())
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: set())

    shards = extractor.extract(call_number=1)

    assert len(shards) == 1
    assert shards[0].is_whole_space


def test_extract_only_selected_requires_points():
    extractor = ShardExtractor(e, _shift_cmf(selected_points=None, only_selected=True))
    with pytest.raises(ValueError, match="No start points were provided for extraction"):
        extractor.extract(call_number=1)


def test_extract_uses_selected_points_to_create_shards(monkeypatch):
    cmf_data = _shift_cmf(selected_points=[(2, 2), (3, 3)], only_selected=True)
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())

    hyperplanes = {
        Hyperplane(symbols[0], symbols=symbols),
        Hyperplane(symbols[1], symbols=symbols),
    }
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hyperplanes)

    shards = extractor.extract(call_number=1)

    assert len(shards) >= 1
    for shard in shards:
        assert shard.get_interior_point() is not None
