"""Tests for System priority export/import using the shard-ID JSON format.

Since Task 7, analysis priorities are stored as lightweight JSON files
containing only shard IDs (``{"cmf_name": ..., "shard_ids": [...]}``)
instead of full serialised Shard objects.  On import, shards are
reconstructed from the matching ``<cmf>__shards.jsonl`` + formatter JSON in
``EXPORT_CMFS``.

The old pickle / PATH_TO_SEARCHABLES paths have been removed.
"""

from __future__ import annotations

import json
from typing import Dict, List, cast

import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import pi
from dreamer.extraction.extractor import extract_cmf_hyperplanes
from dreamer.extraction.shard import Shard
from dreamer.loading.funcs.pFq_fmt import pFq as pFq_fmt
from dreamer.system import system as system_mod
from dreamer.system.system import System
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage.atlas_writer import (
    build_shard_dto,
    update_found_constants,
    write_cmf_records,
    write_shard_records,
)
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


class _DummySearcher:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self):
        return None


def _setup_export(export_root, formatter, const):
    """Write formatter JSON + full-CMF shards JSONL + cmfs.jsonl to export_root."""
    cmf_data = formatter.to_cmf()
    actual_cmf_name = cmf_data.cmf_name

    safe_key = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
    fmt_dir = export_root / safe_key
    fmt_dir.mkdir(parents=True)
    (fmt_dir / f"{actual_cmf_name}.json").write_text(json.dumps(formatter.to_json_obj()))

    # Build a shard using the actual CMF hyperplanes so encoding length matches.
    hps = extract_cmf_hyperplanes(cmf_data)
    encoding = [1] * len(hps)
    symbols = list(cmf_data.cmf.matrices.keys())
    interior = Position({s: sp.Integer(1) for s in symbols})
    shard = Shard.from_cmf_data(cmf_data, [const], hps, encoding, interior)

    write_shard_records(str(export_root), actual_cmf_name, [shard])
    write_cmf_records(str(export_root), [cmf_data])
    update_found_constants(str(export_root), actual_cmf_name, [const.name])

    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    return shard, shard_id, actual_cmf_name


# ---------------------------------------------------------------------------
# Priority export tests
# ---------------------------------------------------------------------------

def test_system_exports_priorities_as_shard_id_json(monkeypatch, tmp_path):
    """Analysis priorities must be exported as shard-ID JSON (not pickle).

    Each file: ``<priorities_root>/<const>/<safe_cmf>.json``
    Content:   ``{"cmf_name": ..., "const_name": ..., "shard_ids": [...]}``.
    """
    priorities_root = tmp_path / "priorities"

    monkeypatch.setattr(system_mod.sys_config, "EXPORT_CMFS", "")
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    formatter_a = pFq_fmt(const=pi, p=1, q=1, z=sp.Integer(1))
    cmf_data_a = formatter_a.to_cmf()
    hps_a = extract_cmf_hyperplanes(cmf_data_a)
    enc_a = [1] * len(hps_a)
    syms_a = list(cmf_data_a.cmf.matrices.keys())
    pt_a = Position({s: sp.Integer(1) for s in syms_a})
    shard_a1 = Shard.from_cmf_data(cmf_data_a, [pi], hps_a, enc_a, pt_a)
    shard_a2 = Shard.from_cmf_data(cmf_data_a, [pi], hps_a, enc_a, pt_a)

    formatter_b = pFq_fmt(const=pi, p=2, q=1, z=sp.Integer(-1))
    cmf_data_b = formatter_b.to_cmf()
    hps_b = extract_cmf_hyperplanes(cmf_data_b)
    enc_b = [1] * len(hps_b)
    syms_b = list(cmf_data_b.cmf.matrices.keys())
    pt_b = Position({s: sp.Integer(1) for s in syms_b})
    shard_b1 = Shard.from_cmf_data(cmf_data_b, [pi], hps_b, enc_b, pt_b)

    _, sid_a1, _ = derive_cmf_and_shard_ids(shard_a1)
    _, sid_a2, _ = derive_cmf_and_shard_ids(shard_a2)
    _, sid_b1, _ = derive_cmf_and_shard_ids(shard_b1)

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=[],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(
        system, "_System__analysis_stage",
        lambda *_: {pi: [shard_a1, shard_a2, shard_b1]},
    )
    monkeypatch.setattr(system, "_System__search_stage", lambda *_: None)

    system.run(constants=[pi])

    const_dir = priorities_root / pi.name
    cmf_a_safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cmf_data_a.cmf_name).strip("_")
    cmf_b_safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cmf_data_b.cmf_name).strip("_")
    assert (const_dir / f"{cmf_a_safe}.json").is_file()
    assert (const_dir / f"{cmf_b_safe}.json").is_file()

    rec_a = json.loads((const_dir / f"{cmf_a_safe}.json").read_text())
    rec_b = json.loads((const_dir / f"{cmf_b_safe}.json").read_text())

    assert rec_a["cmf_name"] == cmf_data_a.cmf_name
    assert rec_a["const_name"] == pi.name
    assert set(rec_a["shard_ids"]) == {sid_a1, sid_a2}
    assert rec_b["cmf_name"] == cmf_data_b.cmf_name
    assert rec_b["shard_ids"] == [sid_b1]


def test_system_exports_priorities_per_cmf_grouping(monkeypatch, tmp_path):
    """Each CMF produces a separate priority JSON file under the constant dir."""
    priorities_root = tmp_path / "priorities"

    monkeypatch.setattr(system_mod.sys_config, "EXPORT_CMFS", "")
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    formatters = [pFq_fmt(const=pi, p=p, q=1, z=sp.Integer(-1)) for p in [1, 2, 3]]
    shards = []
    for fmt in formatters:
        cd = fmt.to_cmf()
        hps = extract_cmf_hyperplanes(cd)
        enc = [1] * len(hps)
        syms = list(cd.cmf.matrices.keys())
        pt = Position({s: sp.Integer(1) for s in syms})
        shards.append(Shard.from_cmf_data(cd, [pi], hps, enc, pt))

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=[],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__analysis_stage", lambda *_: {pi: shards})
    monkeypatch.setattr(system, "_System__search_stage", lambda *_: None)

    system.run(constants=[pi])

    const_dir = priorities_root / pi.name
    json_files = list(const_dir.glob("*.json"))
    assert len(json_files) == 3  # one per CMF


# ---------------------------------------------------------------------------
# Priority import tests
# ---------------------------------------------------------------------------

def test_system_imports_priorities_from_shard_id_json(monkeypatch, tmp_path):
    """When no analyzers, priorities load from shard-ID JSON → reconstructed Shard."""
    export_root = tmp_path / "export"
    priorities_root = tmp_path / "priorities"

    monkeypatch.setattr(system_mod.sys_config, "EXPORT_CMFS", str(export_root))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    formatter = pFq_fmt(const=pi, p=1, q=1, z=sp.Integer(1))
    shard, shard_id, actual_cmf_name = _setup_export(export_root, formatter, pi)

    prio_const_dir = priorities_root / pi.name
    prio_const_dir.mkdir(parents=True)
    safe_cmf = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in actual_cmf_name).strip("_")
    (prio_const_dir / f"{safe_cmf}.json").write_text(json.dumps({
        "cmf_name": actual_cmf_name,
        "const_name": pi.name,
        "shard_ids": [shard_id],
    }))

    captured: Dict = {}

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=None,
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )

    def _capture_search_stage(priorities):
        captured.update(priorities)

    monkeypatch.setattr(system, "_System__search_stage", _capture_search_stage)
    system.run(constants=[pi])

    assert pi in captured, "Priorities must be loaded for pi"
    assert len(captured[pi]) == 1
    reconstructed = captured[pi][0]
    assert isinstance(reconstructed, Shard)
    assert reconstructed.cmf_name == actual_cmf_name


def test_system_loads_shards_from_jsonl_without_extractor(monkeypatch, tmp_path):
    """When EXPORT_CMFS has formatter JSON + shards JSONL, shards are reconstructed."""
    export_root = tmp_path / "export"

    monkeypatch.setattr(system_mod.sys_config, "EXPORT_CMFS", str(export_root))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", "")

    formatter = pFq_fmt(const=pi, p=1, q=1, z=sp.Integer(1))
    shard, shard_id, actual_cmf_name = _setup_export(export_root, formatter, pi)

    captured: Dict = {}

    def _capture_analysis(cmf_data_arg, *_args, **_kwargs):
        captured["shard_dict"] = cmf_data_arg
        return {}

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=["placeholder"],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__analysis_stage", _capture_analysis)

    system.run(constants=[pi])

    assert "shard_dict" in captured
    assert pi in captured["shard_dict"], "Shard must be loaded for pi"
    assert len(captured["shard_dict"][pi]) == 1
    assert isinstance(captured["shard_dict"][pi][0], Shard)
    assert captured["shard_dict"][pi][0].cmf_name == actual_cmf_name
