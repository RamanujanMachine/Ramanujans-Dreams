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


def test_extractor_mod_execute_aggregates_shards_and_exports(monkeypatch, tmp_path):
    cmf_data = {e: [_shift_cmf(), _shift_cmf()]}
    exported = []

    def _identity_tqdm(iterable, *args, **kwargs):
        return iterable

    @contextmanager
    def _fake_export_stream(root, **_kwargs):
        assert str(tmp_path) in root

        def _writer(chunk, filename):
            exported.append((chunk, filename))

        yield _writer

    def _fake_extract(self, call_number=None):
        return [f"shard-{call_number}"]

    monkeypatch.setattr("dreamer.extraction.extractor.SmartTQDM", _identity_tqdm)
    monkeypatch.setattr("dreamer.extraction.extractor.sys_config.PATH_TO_SEARCHABLES", str(tmp_path))
    monkeypatch.setattr("dreamer.extraction.extractor.Exporter.export_stream", _fake_export_stream)
    monkeypatch.setattr("dreamer.extraction.extractor.ShardExtractor.extract", _fake_extract)

    result = ShardExtractorMod(cmf_data).execute()

    assert result[e] == ["shard-1", "shard-2"]
    assert exported == [(["shard-1"], "UnknownCMF"), (["shard-2"], "UnknownCMF")]

