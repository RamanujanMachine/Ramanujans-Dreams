from .system import sys_config
from .database import db_config
from .analysis import analysis_config
from .search import search_config
from .post_process import post_process_config
from .graph import graph_config
from .extraction import extraction_config
from .logging import logging_config
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

    SECTION_ORDER = (
        "system", "database", "extraction", "analysis",
        "search", "post_process", "graph", "logging",
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
                    f'section {section} is not defined, try: system / database / analysis / search',
                    Logger.Levels.exception
                ).log()
                if not warned:
                    warned = True
                    Logger(
                        'Note that these are the builtin attributes: '
                        'system\n\ndatabase\n\nextraction\n\nanalysis\n\nsearch\n\nlogging',
                        Logger.Levels.exception
                    ).log()

    def get_configurables(self) -> Dict[str, List[str]]:
        return {
            'system': self.system.get_configurations(),
            'database': self.database.get_configurations(),
            'extraction': self.extraction.get_configurations(),
            'analysis': self.analysis.get_configurations(),
            'search': self.search.get_configurations(),
            'post_process': self.post_process.get_configurations(),
            'graph': self.graph.get_configurations(),
            'logging': self.logging.get_configurations(),
        }

    def export_configurations(self) -> Dict[str, Dict[str, Any]]:
        return {
            'system': self.system.export_configurations(),
            'database': self.database.export_configurations(),
            'extraction': self.extraction.export_configurations(),
            'analysis': self.analysis.export_configurations(),
            'search': self.search.export_configurations(),
            'post_process': self.post_process.export_configurations(),
            'graph': self.graph.export_configurations(),
            'logging': self.logging.export_configurations(),
        }

    def export_configuration_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Export built-in textual descriptions for each configuration field across all sections.
        :return: Mapping from section name to field-description mapping.
        """
        return {
            'system': self.system.get_configuration_descriptions(),
            'database': self.database.get_configuration_descriptions(),
            'extraction': self.extraction.get_configuration_descriptions(),
            'analysis': self.analysis.get_configuration_descriptions(),
            'search': self.search.get_configuration_descriptions(),
            'post_process': self.post_process.get_configuration_descriptions(),
            'graph': self.graph.get_configuration_descriptions(),
            'logging': self.logging.get_configuration_descriptions(),
        }

    def export_configurations_with_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Export configuration values together with per-field descriptions across all sections.
        :return: Mapping from section name to per-field metadata dictionaries.
        """
        return {
            'system': self.system.export_configurations_with_metadata(),
            'database': self.database.export_configurations_with_metadata(),
            'extraction': self.extraction.export_configurations_with_metadata(),
            'analysis': self.analysis.export_configurations_with_metadata(),
            'search': self.search.export_configurations_with_metadata(),
            'post_process': self.post_process.export_configurations_with_metadata(),
            'graph': self.graph.export_configurations_with_metadata(),
            'logging': self.logging.export_configurations_with_metadata(),
        }

    def iter_sections(self):
        """
        Yield configuration sections in a stable order for reproducible presentation.
        :return: Iterator of (section name, configuration object) tuples.
        """
        for section_name in self.SECTION_ORDER:
            yield section_name, getattr(self, section_name)

