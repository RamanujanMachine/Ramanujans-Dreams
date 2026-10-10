"""Smoke tests for the copy-paste stage templates in ``examples/``.

``examples/analysis.py`` and ``examples/search.py`` encode the exact signatures
``System`` calls and the current DTO / cache shapes.  They are not imported by
the package, so an API change (e.g. ``build_trajectory_dto`` →
``build_trajectory_dtos``, flat per-constant rows, the nested
``{trajectory_id: {constant: record}}`` cache) silently broke them before.
These tests run each template end to end on a minimal real shard and check
that it writes well-formed flat rows and that a rerun reuses them.
"""

import json

import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


@pytest.fixture
def simple_shard():
    """1F1(z=1) bounded shard with interior point (1, 1)."""
    cmf = rt_pFq(1, 1, sp.Integer(1))
    syms = list(cmf.matrices.keys())
    zero = Position({s: sp.Integer(0) for s in syms})
    hps = [Hyperplane(syms[0], syms), Hyperplane(syms[1], syms)]
    interior = Position({syms[0]: sp.Integer(1), syms[1]: sp.Integer(1)})
    return Shard(cmf, e, hps, [1, 1], zero, interior)


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Route every store into a temp dir and keep the analysis store shared."""
    monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
    monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
    monkeypatch.setattr(config.analysis, "IDENTIFY_THRESHOLD", -1)
    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    return tmp_path


def _rows(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


@pytest.mark.timeout(180)
def test_analysis_template_runs_and_reuses(simple_shard, out_dir):
    """``MyAnalyzerMod`` writes one flat row per (trajectory, constant) and a
    second run appends nothing (cache reuse through the nested cache)."""
    from examples.analysis import MyAnalyzerMod

    result = MyAnalyzerMod({e: [simple_shard]}).execute()
    assert set(result) == {e}
    _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
    path = out_dir / f"{shard_id}.jsonl"
    rows = _rows(path)
    assert rows and all(r["constant"] == e.name and "delta" in r for r in rows)

    before = path.read_text()
    MyAnalyzerMod({e: [simple_shard]}).execute()
    assert path.read_text() == before


@pytest.mark.timeout(180)
def test_search_template_runs_and_reuses(simple_shard, out_dir):
    """``MySearchMod`` writes flat rows for its constant and skips them on rerun."""
    from examples.search import MySearchMod

    MySearchMod({e: [simple_shard]}, use_LIReC=False).execute()
    _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
    path = out_dir / f"{shard_id}.jsonl"
    rows = _rows(path)
    assert rows and all(r["constant"] == e.name and "trajectory_id" in r for r in rows)

    before = path.read_text()
    MySearchMod({e: [simple_shard]}, use_LIReC=False).execute()
    assert path.read_text() == before
