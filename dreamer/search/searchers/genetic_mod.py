import os
from typing import List, Optional, Callable

from dreamer.configs.system import sys_config
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.extraction.shard import Shard
from dreamer.configs import search_config
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM


class GeneticSearchMod(SearcherModScheme):
    """
    Searcher module that executes GA trajectory optimization per searchable shard.
    """

    def __init__(
        self,
        searchables: List[Shard],
        use_LIReC: bool = True
    ):
        """
        Initialize GA module-level hyperparameters used for each searchable.
        :param searchables: Shards/searchables to optimize.
        :param use_LIReC: Optional backend flag forwarded to trajectory evaluation.
        """
        super().__init__(
            searchables,
            use_LIReC,
            name="GeneticSearch",
            description="Genetic algorithm optimization module for trajectory discovery",
            version="1.0.0",
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """
        Run GA search for each searchable and export results as serialized DataManager chunks.
        :return: None.
        """
        if not self.searchables:
            return

        export_root = sys_config.EXPORT_SEARCH_RESULTS or ""
        os.makedirs(
            dir_path := os.path.join(export_root, self.searchables[0].const.name),
            exist_ok=True,
        )

        fmt = Formats(sys_config.EXPORT_SEARCH_RESULTS_FORMAT)

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=fmt) as write_chunk:
            for space in SmartTQDM(
                self.searchables,
                desc="Optimizing trajectories via GA: ",
                **sys_config.TQDM_CONFIG,
            ):
                searcher = GeneticSearchMethod(
                    space,
                    space.const,
                    find_limit=search_config.COMPUTE_LIMIT,
                    find_eigen_values=search_config.COMPUTE_EIGEN_VALUES,
                    find_gcd_slope=search_config.COMPUTE_GCD_SLOPE,
                    use_LIReC=self.use_LIReC,
                )
                res = searcher.search()
                space: Searchable
                write_chunk(res, space.cmf_name)
