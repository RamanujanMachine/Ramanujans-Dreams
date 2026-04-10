from dreamer.extraction import shard
from dreamer.utils.schemes.analysis_scheme import AnalyzerScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.configs import config
from dreamer.extraction.shard import Shard
from typing import Dict, List
from ramanujantools.cmf import CMF


analysis_config = config.analysis


class Analyzer(AnalyzerScheme):
    """
    The analyzer is an implementation of a method to perform a small search of sorts and
    prioritize the Shards according to relevance.
    """

    def __init__(self, const: Constant, shards: List[Shard]):
        """
        :param const: Constant to analyze for
        :param shards: Shards to analyze
        """
        self.const = const
        self.shards = shards

    def search(self) -> Dict[Shard, DataManager]:
        """
        Preforms a small search inside the shards
        :return: The results per shard
        """
        managers = {}

        # Notice we used the naming of the subspaces as "shards" but this might change in the future
        for i, space in enumerate(SmartTQDM(self.shards, desc='Analyzing shards', **config.system.TQDM_CONFIG)):
            start = space.get_interior_point()
            Logger(f'{"=" * 10} SHARD NO. {i + 1} {"=" * 10}', Logger.Levels.message).log(msg_prefix='\n')
            if analysis_config.SHOW_START_POINT:
                Logger(f'Chosen shard start point: {start}', Logger.Levels.info).log()
            if analysis_config.SHOW_SEARCHABLE:
                Logger(f'Shard: \n{space}', Logger.Levels.info).log()

            Logger(f'Shard description (Ax < b):\n{str(space)}', Logger.Levels.debug).log()

            searcher = SerialSearcher(space, self.const, use_LIReC=analysis_config.USE_LIReC)
            dm = searcher.search(
                start,
                find_limit=analysis_config.ANALYZE_LIMIT,
                find_gcd_slope=analysis_config.ANALYZE_GCD_SLOPE,
                find_eigen_values=analysis_config.ANALYZE_EIGEN_VALUES,
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM
            )

            identified = dm.identified_percentage
            best_delta, best_trajectory = dm.best_delta

            if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                if best_delta is None:
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"',
                        Logger.Levels.info
                    ).log()
                else:
                    cmf_msg = f'{repr(space.cmf)}'\
                        if type(space.cmf) is not CMF else '<raw cmf>'
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"'
                        f'\n    > Best delta:\t {best_delta:.4f}'
                        f'\n    > Trajectory:\t {best_trajectory}'
                        f'\n    > CMF:\t\t {cmf_msg}'
                        f'\n    > p,q vectors:\t{dm[best_trajectory].initial_values.tolist()}',
                        Logger.Levels.info
                    ).log()
            if identified > analysis_config.IDENTIFY_THRESHOLD and best_delta is not None:
                managers[space] = dm
            else:
                if best_delta is not None:
                    Logger(
                        f'Ignoring shard - identified <= {analysis_config.IDENTIFY_THRESHOLD * 100}% '
                        'of tested trajectories',
                        Logger.Levels.info
                    ).log()
                else:
                    Logger('No best delta was found', Logger.Levels.warning).log()
        return managers

    # TODO: add tests for this function
    def prioritize(self, managers: Dict[Searchable, DataManager], ranks: int = 3) -> Dict[Searchable, Dict[str, int]]:
        """
        Prioritizes the searchables according to how big is their delta and the CMF dimension
        :param managers: A mapping from a searchable to its search result.
        :param ranks: Number of delta ranks (we divide the interval [-1, 0^+] into #ranks).
        :return: A mapping from searchables to their prioritization object:
            ``{'delta_rank':<rank>,'dim':<searchable-dim>}``
        """
        if ranks < 3:
            Logger('prioritization ranks must be >= 3 (resulting to default = 3), '
                   'continuing to prevent data loss', Logger.Levels.exception).log()

        def match_rank(n: int, num):
            step = 1 / n
            ranks = [-1 + k * step for k in range(n + 1)]
            left = 0
            right = len(ranks) - 1

            while left < right:
                mid = (left + right) // 2
                if num > ranks[mid]:
                    left = mid + 1
                else:
                    right = mid
            return left + 1

        ranked = {}
        for shard, dm in managers.items():
            best_delta = dm.best_delta[0]
            ranked[shard] = {
                'delta_rank': match_rank(ranks, best_delta if best_delta else -1),
                'dim': shard.dim
            }
        return ranked
