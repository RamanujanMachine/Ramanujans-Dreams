from contextlib import contextmanager

import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.extractor import ShardExtractorMod
from dreamer.utils.types import CMFData


def _shift_cmf():
    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    return CMFData(cmf=cmf, shift=shift)


def test_extractor_mod_execute_aggregates_shards(monkeypatch):
    """ShardExtractorMod.execute() aggregates shards per constant.

    Shard pickle exports were removed; only JSONL writes remain (covered by
    TestAtlasWriter).  This test verifies the return-value aggregation only.
    """
    cmf_data = {e: [_shift_cmf(), _shift_cmf()]}

    def _identity_tqdm(iterable, *args, **kwargs):
        return iterable

    def _fake_extract(self, call_number=None):
        return [f"shard-{call_number}"]

    # Suppress the JSONL write side-effects (EXPORT_CMFS not configured).
    monkeypatch.setattr("dreamer.extraction.extractor.SmartTQDM", _identity_tqdm)
    monkeypatch.setattr("dreamer.extraction.extractor.sys_config.EXPORT_CMFS", "")
    monkeypatch.setattr("dreamer.extraction.extractor.ShardExtractor.extract", _fake_extract)

    result = ShardExtractorMod(cmf_data).execute()

    assert result[e] == ["shard-1", "shard-2"]

