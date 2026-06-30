"""
Template: add your own **analysis** stage to the pipeline.

Copy this file, rename the classes, and fill in the ``TODO``s. An analysis
stage is two classes (keep them in separate files when contributing — see
CONTRIBUTING.md):

  * ``MyAnalyzer`` (an ``AnalyzerScheme``) — the *internal logic*: cheaply
    search each shard and turn the results into a prioritisation.
  * ``MyAnalyzerMod`` (an ``AnalyzerModScheme``) — the *external interface*: run
    the analyzer over every constant's shards and return the ranked shards the
    search stage will consume.

Wire your module into the pipeline with::

    from dreamer import System
    System(function_sources=[...], analyzers=[MyAnalyzerMod]).run(constants=[...])

``System`` constructs the module as ``MyAnalyzerMod(cmf_data)`` and calls
``.execute()``. The goal of analysis is triage: keep the shards that actually
produce the constant and rank them (best first) so the expensive search stage
spends its budget where it matters.

The built-in default is
``dreamer.analysis.analyzers.serial_scan.analyzer_mod.AnalyzerModV1`` — use it
as the production-grade reference.
"""

from typing import List, Dict

from dreamer.utils.schemes.analysis_scheme import AnalyzerScheme, AnalyzerModScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.configs.system import sys_config
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.logger import Logger


class MyAnalyzer(AnalyzerScheme):
    """The analysis algorithm, scoped to one constant and its shards."""

    def __init__(
            self,
            const: Constant,
            shards: List[Searchable],
            # TODO: <your arguments here>
    ):
        """
        :param const: The constant being analysed.
        :param shards: The shards (searchables) extracted for ``const``.
        """
        self.const = const
        self.shards = shards

    def search(self) -> Dict[Searchable, DataManager]:
        """
        Cheaply probe each shard (e.g. sample a few trajectories, compute their
        delta and whether they identify the constant).

        :return: A mapping from each searched shard to its results.
        """
        # TODO: probe each shard and collect results; return {shard: DataManager}.
        pass

    def prioritize(
            self,
            managers: Dict[Searchable, DataManager],
            *args,   # TODO: <your arguments here>
    ) -> Dict[Searchable, Dict[str, int]]:
        """
        Turn search results into a per-shard prioritisation score.

        The convention used by the built-in analyzer is to score each shard by
        ``{'delta_rank': int, 'dim': int}`` — rank by the best delta found, and
        keep the space dimension as a secondary key so that lower-dimensional
        (cheaper) shards are searched before high-dimensional ones. You are free
        to use any scoring you like, as long as ``execute`` below knows how to
        sort by it.

        :param managers: Mapping from each shard to its search results.
        :return: Mapping from each shard to its prioritisation score.
        """
        # TODO: rank the shards in `managers` and return the score per shard.
        pass


class MyAnalyzerMod(AnalyzerModScheme):
    """The stage module — runs ``MyAnalyzer`` per constant and ranks the shards.

    ``System`` instantiates this as ``MyAnalyzerMod(cmf_data)``, where
    ``cmf_data`` is ``Dict[Constant, List[Searchable]]`` from extraction.
    """

    def __init__(
            self,
            cmf_data: Dict[Constant, List[Searchable]],
            # TODO: <your arguments here>
            #   If you add constructor arguments, wire the module in with
            #   functools.partial so System can still construct it with just
            #   cmf_data:  analyzers=[partial(MyAnalyzerMod, my_arg=...)]
    ):
        super().__init__(
            cmf_data,
            name='A very witty name',
            desc='My super cool and smart analysis module',
            version='your version here :)',
        )
        # TODO: store your own arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Analyse every constant's shards and return them filtered + ranked.

        :return: Mapping from each constant to its prioritised list of shards
            (best first). The template below is a starting point, not a contract.
        """
        queues: Dict[Constant, List[Searchable]] = {c: [] for c in self.cmf_data.keys()}

        for constant, shards in SmartTQDM(
                self.cmf_data.items(), desc='Analyzing constants and their CMFs',
                **sys_config.TQDM_CONFIG,
        ):
            Logger(
                Logger.buffer_print(
                    sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant.name}', '='
                ), Logger.Levels.message,
            ).log()

            analyzer = MyAnalyzer(constant, shards)
            managers = analyzer.search()
            prioritization: Dict[Searchable, Dict[str, int]] = analyzer.prioritize(
                managers,
                # TODO: any other arguments your prioritize() needs
            )

            # TODO: drop shards that don't pass your threshold, then sort the
            #   survivors by `prioritization` and assign them to queues[constant].
            #   e.g. with the {'delta_rank', 'dim'} convention:
            #       queues[constant] = sorted(
            #           prioritization,
            #           key=lambda s: (prioritization[s]['delta_rank'], prioritization[s]['dim']),
            #       )

        return queues
