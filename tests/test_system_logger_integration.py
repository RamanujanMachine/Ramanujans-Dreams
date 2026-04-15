from typing import cast

from dreamer.configs.logging import logging_config
from dreamer.system import system as system_mod
from dreamer.system.system import System
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme


class _DummySearcher:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self):
        return None


def test_system_run_calls_logger_start_run_once_per_run(monkeypatch, tmp_path):
    logging_config.GENERATE_LOGS = False
    searchables_dir = tmp_path / "searchables"
    searchables_dir.mkdir()
    monkeypatch.setattr(system_mod.sys_config, "PATH_TO_SEARCHABLES", str(searchables_dir))

    system = System(
        function_sources=[],
        extractor=None,
        analyzers=[],
        searcher=cast(type[SearcherModScheme], _DummySearcher),
    )
    monkeypatch.setattr(system, "_System__validate_constants", lambda _constants: [])
    monkeypatch.setattr(system, "_System__loading_stage", lambda _constants: {})
    monkeypatch.setattr(system, "_System__analysis_stage", lambda _cmf_data: {})

    calls = []

    def _fake_start_run(cls):
        calls.append("start")

    monkeypatch.setattr(Logger, "start_run", classmethod(_fake_start_run))

    system.run(constants=[])
    system.run(constants=[])

    assert calls == ["start", "start"]



