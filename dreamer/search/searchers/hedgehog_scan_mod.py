from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter, Formats
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from ramanujantools.cmf import CMF
from typing import List
import os

search_config = config.search
sys_config = config.system


class SearcherModV1(SearcherModScheme):
    """
    A searcher module that performs a serial search over a list of searchable spaces.
    """

    def __init__(self, shards: List[Shard], use_LIReC: bool):
        """
        :param shards: A list of searchable spaces to search in.
        :param use_LIReC: If true, LIReC will be used to identify constants within the searchable spaces.
        """
        super().__init__(
            shards,
            use_LIReC,
            description='Searcher module - orchestrating a deep search within a prioritized list of spaces',
            version='0.0.1'
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """
        Executes the search. Computes the results per searchable space and exports them into a file while running.
        :return: A mapping from shards to their search results.
        """
        if not self.searchables:
            return

        os.makedirs(
            dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const.name),
            exist_ok=True
        )

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for shard in SmartTQDM(
                    self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG
            ):
                searcher = SerialSearcher(shard, shard.const, use_LIReC=self.use_LIReC)
                res = searcher.search(
                    None,
                    find_limit=search_config.COMPUTE_LIMIT,
                    find_gcd_slope=search_config.COMPUTE_GCD_SLOPE,
                    find_eigen_values=search_config.COMPUTE_EIGEN_VALUES,
                    trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM
                )

                shard: Searchable
                if shard.cmf.__class__ == CMF:
                    filename = f'generated_cmf_hashed_{hash(shard.cmf)}'
                else:
                    filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in repr(shard.cmf)).strip('_')
                write_chunk(res, filename)
