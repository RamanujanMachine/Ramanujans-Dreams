from .system import sys_config
from .database import db_config
from .analysis import analysis_config
from .search import search_config
from .post_process import post_process_config
from .graph import graph_config
from .extraction import extraction_config
from .logging import logging_config
from .monitor import monitor_config
from typing import Dict, List, Any


class ConfigManager:
    """
    Global configuration manager for the search system
    """
    system = sys_config
    database = db_config
    extraction = extraction_config
    analysis = analysis_config
    search = search_config
    post_process = post_process_config
    graph = graph_config
    logging = logging_config
    monitor = monitor_config

    SECTION_ORDER = (
        "system", "database", "extraction", "analysis",
        "search", "post_process", "graph", "logging", "monitor",
    )

    def configure(self, **overrides):
        """
        Override multiple configs at once.

        Example:
            Config.configure(system={"CONSTANT": "X"}, database={"DB_NAME": "foo"})
        """
        from dreamer.utils.logger import Logger
        warned = False
        for section, values in overrides.items():
            cfg = getattr(self, section, None)
            if cfg is not None:
                for key, val in values.items():
                    setattr(cfg, key, val)
            else:
                Logger(
                    f'section {section} is not defined, try: {" / ".join(self.SECTION_ORDER)}',
                    Logger.Levels.exception
                ).log()
                if not warned:
                    warned = True
                    Logger(
                        'Note that these are the builtin sections: ' + ', '.join(self.SECTION_ORDER),
                        Logger.Levels.exception
                    ).log()

    def get_configurables(self) -> Dict[str, List[str]]:
        return {name: getattr(self, name).get_configurations() for name in self.SECTION_ORDER}

    def export_configurations(self) -> Dict[str, Dict[str, Any]]:
        return {name: getattr(self, name).export_configurations() for name in self.SECTION_ORDER}

    def export_configuration_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Export built-in textual descriptions for each configuration field across all sections.
        :return: Mapping from section name to field-description mapping.
        """
        return {name: getattr(self, name).get_configuration_descriptions() for name in self.SECTION_ORDER}

    def export_configurations_with_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Export configuration values together with per-field descriptions across all sections.
        :return: Mapping from section name to per-field metadata dictionaries.
        """
        return {name: getattr(self, name).export_configurations_with_metadata() for name in self.SECTION_ORDER}

    def iter_sections(self):
        """
        Yield configuration sections in a stable order for reproducible presentation.
        :return: Iterator of (section name, configuration object) tuples.
        """
        for section_name in self.SECTION_ORDER:
            yield section_name, getattr(self, section_name)

