from dreamer.utils.multi_processing import create_pool
from dreamer.extraction.samplers import ShardSamplingOrchestator
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.storage_objects import DataManager, SearchVector, SearchData
from dreamer.utils.schemes.searchable import Searchable
from dreamer.configs import config

import mpmath as mp
from functools import partial
from typing import Optional, Callable, List, cast
from ramanujantools import Position


search_config = config.search


class SerialSearcher(SearchMethod):
    """
    Serial trajectory searcher. \n
    A naive searcher.
    """

    def __init__(self,
                 space: Searchable,
                 constant,  # sympy constant or mp.mpf
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        """
        :param space: The searchable to search in.
        :param constant: The constant to look for in the subspace.
        :param data_manager: The data manager to store search results in.
            If no data manager is provided, a new one will be created, and it will not be shared.
        :param share_data: If true, the data manager will be shared between searchables, otherwise it will be cloned.
        :param use_LIReC: Use LIReC to identify constants within the searchable.
        """
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        self.data_manager = data_manager if data_manager else DataManager(use_LIReC)
        self.parallel = search_config.PARALLEL_SEARCH
        self.pool = create_pool() if self.parallel else None

    def search(self,
               starts: Optional[object] = None,
               find_limit: bool = True,
               find_eigen_values: bool = True,
               find_gcd_slope: bool = True,
               trajectory_generator: Callable[[int], int] = search_config.NUM_TRAJECTORIES_FROM_DIM
               ) -> DataManager:
        """
        Performs the search.
        :param starts: A start point within the searchable.
        :param find_limit: If true, compute the limit of the trajectory matrix.
        :param find_eigen_values: If ture, compute the eigenvalues of the trajectory matrix.
        :param find_gcd_slope: If true, compute the GCD slope.
        :param trajectory_generator: A function that given the dimension of the searchable,
            returns the number of trajectories to sample.
        :return: The data manager containing the search results.
        """
        if not starts:
            starts = self.space.get_interior_point()
        if starts is None:
            raise ValueError('Search requires at least one valid start point')
        starts_list: List[Position]
        if isinstance(starts, list):
            starts_list = []
            for s in starts:
                starts_list.append(cast(Position, s))
        else:
            starts_list = [cast(Position, starts)]

        trajectories = ShardSamplingOrchestator(self.space).sample_trajectories(trajectory_generator)

        pairs = [(t, start) for start in starts_list for t in trajectories if
                 SearchVector(start, t) not in self.data_manager]
        traj_lst = [p[0] for p in pairs]
        start_lst = [p[1] for p in pairs]

        if self.parallel:
            results = self.pool.map(
                partial(
                    self.space.compute_trajectory_data,
                    use_LIReC=self.use_LIReC,
                    find_limit=find_limit,
                    find_eigen_values=find_eigen_values,
                    find_gcd_slope=find_gcd_slope
                ),
                traj_lst, start_lst, chunksize=search_config.SEARCH_VECTOR_CHUNK)
            for res in results:
                if res is not None:
                    res_data = cast(SearchData, res)
                    res_data.gcd_slope = mp.mpf(res_data.gcd_slope) if res_data.gcd_slope else None
                    if isinstance(res_data.delta, str):
                        res_data.delta = mp.mpf(res_data.delta)
                    self.data_manager[res_data.sv] = res_data
        else:
            for t, start in pairs:
                sd = self.space.compute_trajectory_data(
                    t, start,
                    use_LIReC=self.use_LIReC,
                    find_limit=find_limit,
                    find_eigen_values=find_eigen_values,
                    find_gcd_slope=find_gcd_slope
                )
                self.data_manager[sd.sv] = sd
        return self.data_manager
