"""
Template: add your own **search** stage to the pipeline.

Copy this file, rename the classes, and fill in the ``TODO``s. A search stage
is two classes (keep them in separate files when contributing — see
CONTRIBUTING.md):

  * ``MySearchMethod`` (a ``SearchMethod``) — the *internal logic*: how your
    algorithm explores trajectory directions inside one searchable (shard) and
    evaluates them.
  * ``MySearchMod`` (a ``SearcherModScheme``) — the *external interface*: how the
    method is driven over all the prioritised shards and where results are
    written.

Wire your module into the pipeline with::

    from dreamer import System
    System(function_sources=[...], searcher=MySearchMod).run(constants=[...])

``System`` constructs the module as ``MySearchMod(priorities, use_LIReC)`` and
calls ``.execute()``; ``execute`` writes its results to disk and returns nothing.

The built-in default, ``dreamer.search.searchers.hedgehog_scan_mod.SearcherModV1``,
is the production-grade reference: it writes one ``<shard_id>.jsonl`` record per
trajectory under ``sys_config.EXPORT_SEARCH_RESULTS`` (which the run summary and
best-δ reporting read back). Follow it if you want your results to flow into the
standard reporting; the simpler ``Exporter`` stream used below is enough for a
self-contained custom searcher.
"""

import os
from typing import List, Optional

from ramanujantools import Position

from dreamer.utils.schemes.searcher_scheme import SearchMethod, SearcherModScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs.system import sys_config


class MySearchMethod(SearchMethod):
    """The search algorithm itself, scoped to a single searchable (shard)."""

    def __init__(self,
                 space: Searchable,
                 constant,                       # sympy constant or mp.mpf
                 # TODO: <your arguments here>
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        """
        :param space: The searchable (shard) to search within.
        :param constant: The constant to look for in ``space``.
        :param data_manager: Optional shared result accumulator.
        :param share_data: If True, work on a copy of ``data_manager`` (the base
            class handles the copy) so methods don't clobber each other's results.
        :param use_LIReC: Identify constants with LIReC while searching.
        """
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        # TODO: store your own arguments

    def search(self, starts: Optional[Position | List[Position]] = None) -> DataManager:
        """
        Perform the search within ``self.space``.

        :param starts: Optional start point(s) to begin the search from. If your
            method samples its own directions you can ignore this.
        :return: A ``DataManager`` holding the trajectories you evaluated and
            their results (delta, identification, …).
        """
        # TODO: explore directions in self.space, evaluate each (compute delta,
        #   identify the constant), and record results into self.data_manager.
        return self.data_manager


class MySearchMod(SearcherModScheme):
    """The stage module — drives ``MySearchMethod`` over the prioritised shards.

    ``System`` instantiates this as ``MySearchMod(priorities, use_LIReC)``.
    ``priorities`` is the ``Dict[Constant, List[Searchable]]`` produced by the
    analysis stage (each constant mapped to the shards that identified it); the
    base class also exposes a deduplicated flat list as ``self.searchables``.
    """

    def __init__(
            self,
            priorities,                          # Dict[Constant, List[Searchable]]
            use_LIReC: Optional[bool] = True,
            # TODO: <your arguments here>
            #   If you add constructor arguments, wire the module in with
            #   functools.partial so System can still construct it with just
            #   (priorities, use_LIReC):
            #       searcher=partial(MySearchMod, my_arg=...)
    ):
        super().__init__(
            priorities, use_LIReC,
            name='A very witty name',
            description='My super cool and smart search module',
            version='your version here :)',
        )
        # TODO: store your own arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """
        Run the search over every prioritised shard and persist the results.

        Results are written to disk as they are produced (so a long run is not
        lost on interruption); this method returns nothing.
        """
        if not self.searchables:
            return

        # One output folder per constant, under the configured results directory.
        for constant, shards in self.priorities.items():
            if not shards:
                continue

            os.makedirs(
                dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, constant.name),
                exist_ok=True,
            )

            # ``export_stream`` streams each result to a file as it is produced.
            # Swap Formats.PICKLE for Formats.JSON if you prefer human-readable
            # output. (The built-in SearcherModV1 instead writes per-shard JSONL
            # records that the run summary reads — see this file's docstring.)
            with Exporter.export_stream(
                    dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE
            ) as write_chunk:
                for space in SmartTQDM(
                        shards, desc=f'Searching shards for {constant.name}: ',
                        **sys_config.TQDM_CONFIG,
                ):
                    searcher = MySearchMethod(
                        space, space.const,      # TODO: pass your own arguments
                    )
                    result: DataManager = searcher.search(
                        # TODO: your search() arguments here
                    )
                    write_chunk(result, space.cmf_name)   # persist this shard's results
