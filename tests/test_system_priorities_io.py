"""Regression tests for System priority import/export behavior and CMF relevance filtering."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, cast

from ramanujantools import Position

from dreamer import pi
from dreamer.system import system as system_mod
from dreamer.system.system import System
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage import Exporter, Importer, Formats
from dreamer.utils.types import CMFData


class _DummySearcher:
    """Minimal searcher stub used to isolate System orchestration paths in tests."""

    def __init__(self, *args, **kwargs):
        pass

    def execute(self):
        return None


class _DummyCMF:
    """Minimal CMF-like object satisfying Searchable construction requirements."""

    def dim(self):
        return 1


class _Space(Searchable):
    """Tiny Searchable fixture carrying const/cmf_name metadata for assertions."""

    def __init__(self, const, cmf_name: str, tag: str):
        super().__init__(_DummyCMF(), const, Position({}), use_inv_t=True, cmf_name=cmf_name)
        self.tag = tag

    @classmethod
    def from_cmf_data(cls, cmf_data, constant, *args, **kwargs):
        return cls(constant, cmf_data.cmf_name, "from_cmf")

    def in_space(self, point):
        return True

    def is_valid_trajectory(self, trajectory):
        return True

    def get_interior_point(self):
        return Position({})

    def is_unconstrained(self):
        return True

    def __hash__(self):
        return hash((self.const.name, self.cmf_name, self.tag))


def _write_cmf_input(file_path: Path, cmf_name: str) -> None:
    """
    Create a CMFData pickle file used as a function_sources input.
    :param file_path: Destination path for the pickle file.
    :param cmf_name: cmf_name embedded in the serialized CMFData object.
    :return: None.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = [CMFData(cmf=_DummyCMF(), shift=Position({}), cmf_name=cmf_name)]
    with file_path.open("wb") as f:
        pickle.dump(data, f)


def test_system_imports_priorities_when_analyzers_missing_and_filters_by_cmf(monkeypatch, tmp_path):
    """
    Validate analyzer-omitted flow imports priorities from disk and applies CMF filtering.
    Assumption: priorities are serialized under <root>/<const>/<cmf>.pkl and function_sources include one cmf_name.
    Failure mode trapped: system imports irrelevant CMF priority files when analyzers are absent.
    """
    priorities_root = tmp_path / "priorities"
    (priorities_root / pi.name).mkdir(parents=True)

    wanted = _Space(pi, "wanted_cmf", "wanted")
    ignored = _Space(pi, "ignored_cmf", "ignored")
    Exporter.export(str(priorities_root / pi.name), "wanted_cmf", data=[wanted], fmt=Formats.PICKLE)
    Exporter.export(str(priorities_root / pi.name), "ignored_cmf", data=[ignored], fmt=Formats.PICKLE)

    cmf_input = tmp_path / "cmf_inputs" / pi.name / "cmf_input.pkl"
    _write_cmf_input(cmf_input, "wanted_cmf")

    searchables_root = tmp_path / "searchables"
    searchables_root.mkdir()

    monkeypatch.setattr(system_mod.sys_config, "PATH_TO_SEARCHABLES", str(searchables_root))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    captured: Dict[str, List[Searchable]] = {}

    def _capture_search_stage(priorities):
        captured[pi.name] = priorities[pi]

    system = System(
        function_sources=[str(cmf_input)],
        extractor=None,
        analyzers=None,
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__search_stage", _capture_search_stage)

    system.run(constants=[pi])

    assert pi.name in captured
    tags = {space.tag for space in captured[pi.name]}
    assert tags == {"wanted"}


def test_system_exports_priorities_as_constant_and_cmf_pickles(monkeypatch, tmp_path):
    """
    Verify priorities are exported as one file per constant+CMF.
    Assumption: __analysis_stage returns mixed cmf_name values in one priority list.
    Failure mode trapped: export flattens all priorities into one file or omits per-CMF partitioning.
    """
    priorities_root = tmp_path / "priorities"
    searchables_root = tmp_path / "searchables"
    searchables_root.mkdir()

    monkeypatch.setattr(system_mod.sys_config, "PATH_TO_SEARCHABLES", str(searchables_root))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    cmf_a_1 = _Space(pi, "cmfA", "a1")
    cmf_a_2 = _Space(pi, "cmfA", "a2")
    cmf_b_1 = _Space(pi, "cmfB", "b1")

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=[],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__analysis_stage", lambda *_args, **_kwargs: {pi: [cmf_a_1, cmf_a_2, cmf_b_1]})
    monkeypatch.setattr(system, "_System__search_stage", lambda *_args, **_kwargs: None)

    system.run(constants=[pi])

    const_dir = priorities_root / pi.name
    pickle_ext = Formats.PICKLE.value
    assert (const_dir / f"cmfA.{pickle_ext}").is_file()
    assert (const_dir / f"cmfB.{pickle_ext}").is_file()

    cmf_a_loaded = Importer.imprt(str(const_dir / f"cmfA.{pickle_ext}"))
    cmf_b_loaded = Importer.imprt(str(const_dir / f"cmfB.{pickle_ext}"))
    assert len(cmf_a_loaded) == 2
    assert len(cmf_b_loaded) == 1


def test_system_imports_searchables_only_from_relevant_constant_and_cmf(monkeypatch, tmp_path):
    """
    Validate searchable import path reads only relevant constant/CMF directories.
    Assumption: relevant cmf_name values are inferred from function_sources CMFData input.
    Failure mode trapped: import stage loads unrelated CMF shards for the same constant.
    """
    searchables_root = tmp_path / "searchables"
    priorities_root = tmp_path / "priorities"
    (searchables_root / pi.name / "wanted_cmf").mkdir(parents=True)
    (searchables_root / pi.name / "other_cmf").mkdir(parents=True)

    wanted = _Space(pi, "wanted_cmf", "wanted")
    ignored_same_const = _Space(pi, "other_cmf", "ignored_same_const")

    Exporter.export(
        str(searchables_root / pi.name / "wanted_cmf"),
        "chunk_0000000000",
        data=[wanted],
        fmt=Formats.PICKLE,
    )
    Exporter.export(
        str(searchables_root / pi.name / "other_cmf"),
        "chunk_0000000000",
        data=[ignored_same_const],
        fmt=Formats.PICKLE,
    )

    cmf_input = tmp_path / "cmf_inputs" / pi.name / "cmf_input.pkl"
    _write_cmf_input(cmf_input, "wanted_cmf")

    monkeypatch.setattr(system_mod.sys_config, "PATH_TO_SEARCHABLES", str(searchables_root))
    monkeypatch.setattr(system_mod.sys_config, "EXPORT_ANALYSIS_PRIORITIES", str(priorities_root))

    captured = {}

    def _capture_analysis(cmf_data, *_args, **_kwargs):
        captured.update(cmf_data)
        return {}

    system = System(
        function_sources=[str(cmf_input)],
        extractor=None,
        analyzers=["placeholder"],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__analysis_stage", _capture_analysis)

    system.run(constants=[pi])

    assert pi in captured
    assert len(captured[pi]) == 1
    assert captured[pi][0].tag == "wanted"


