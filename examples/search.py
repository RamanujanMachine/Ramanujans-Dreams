from dreamer.utils.schemes.searcher_scheme import SearchMethod, SearcherModScheme
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs.system import sys_config
from typing import List, Optional, List
from ramanujantools import Position
from ramanujantools.cmf import CMF
import os


class MySearchMethod(SearchMethod):
    def __init__(self,
                 space: Searchable,
                 constant,  # sympy constant or mp.mpf
                 # TODO: <your arguments here>
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        # TODO: set arguments

    def search(self, starts: Optional[Position | List[Position]] = None) -> DataManager:
        """
        Performs the search in a specific searchable.
        :param starts: A point or a list of points to start the search from.
        :return: A search result object.
        """
        # TODO: compute and search in the space. When finished - return your data manager
        return self.data_manager


class MySearchMod(SearcherModScheme):
    def __init__(
            self,
            searchables: List[Searchable],
            use_LIReC: Optional[bool] = True,
            # TODO: <your arguments here>
            #  Note: that if you add more arguments you would probably need to use functools.partial in System().
            #  This allows you to add as many arguments as you want without any need to change System's impelementation.
    ):
        super().__init__(
            searchables, use_LIReC,
            name='A very witty name',
            description='My super cool and smart module using super smart methods beyond your comprehension',
            version='your version here :)'
        )
        # TODO: set arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """
        Executes the search. Computes the results per searchable space and exports them into a file while running.
        :return: A mapping from searchables to their search results.
        """
        if not self.searchables:
            return

        # Create a folder for the stored results created by the searcher
        os.makedirs(
            dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const.name),
            exist_ok=True
        )

        # This bit of code does the following:
        # When you are searching per searchable (e.g. Shard) the results are stored in automatically in a pickle file
        # in `dir_path`
        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in SmartTQDM(
                    self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG
            ):
                searcher = MySearchMethod(
                    space, space.const   # TODO: add your arguments
                )   # creates an instance of your searcher
                res = searcher.search(
                    # TODO: all you searcher's arguments here
                )
                write_chunk(res, space.cmf_name)    # Writes the search results into a file
