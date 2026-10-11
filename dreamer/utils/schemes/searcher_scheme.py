from abc import ABC, abstractmethod
from copy import copy
import mpmath as mp

from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.module import Module
from ramanujantools import Position
from typing import Dict, List, Optional


class SearchMethod(ABC):
    """
    The SearchMethod represents a general search method. \n
    A search method is a minimal object that its sole purpose is to perform the search on a given searchable. \n
    This is the base class for all search methods that will be used by the search modules.
    """

    def __init__(self,
                 space: Searchable,
                 const: mp.mpf,
                 use_LIReC: bool,
                 data_manager: DataManager = None,
                 share_data: bool = True):
        """
        :param space: A subspace the search method is designated to work search in
        :param const: The constant to look for in the subspace
        :param use_LIReC: Preform search by computing data with LIReC
        :param data_manager: A data manager to be used by the search method
        :param share_data: Decide whether to share data between methods or not
        """
        self.space = space
        self.const = const
        self.use_LIReC = use_LIReC
        self.best_delta = -1
        self.trajectories = set()
        self.start_points = set()
        self.data_manager = data_manager if not share_data else copy(data_manager)

    @abstractmethod
    def search(self, starts: Optional[Position | List[Position]] = None) -> DataManager:
        """
        Performs the search in a specific searchable.
        :param starts: A point or a list of points to start the search from.
        :return: A search result object.
        """
        raise NotImplementedError


class SearcherModScheme(Module):
    """
    A Scheme for all search modules.
    """
    def __init__(self, priorities,
                 use_LIReC: bool = True,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 version: Optional[str] = None):
        """
        :param priorities: Either a ``Dict[Constant, List[Searchable]]`` mapping
            each constant to the shards that identified it (the modern interface),
            or a bare ``List[Searchable]`` for backward compatibility.
        :param use_LIReC: While searching, identify constants using LIReC
        :param name: Optional - the name of the module.
        :param description: Optional - module description.
        :param version: Optional - the module version.
        """
        super().__init__(name, description, version)
        if isinstance(priorities, dict):
            self.priorities: Dict = priorities
            # Flat list for backward-compatible subclasses that read self.searchables.
            seen: set = set()
            flat: List[Searchable] = []
            for shards in priorities.values():
                for s in shards:
                    if id(s) not in seen:
                        seen.add(id(s))
                        flat.append(s)
            self.searchables: List[Searchable] = flat
        else:
            # Legacy path: bare list — wrap in a synthetic dict keyed by primary const.
            # Guard against unhashable or mock const values (e.g. tests using SimpleNamespace).
            self.searchables = list(priorities)
            from collections import defaultdict
            d: Dict = defaultdict(list)
            for s in self.searchables:
                try:
                    d[s.const].append(s)
                except TypeError:
                    # const is not hashable (e.g. a mock SimpleNamespace) — use object id.
                    d[id(s.const)].append(s)
            self.priorities = dict(d)
        self.use_LIReC = use_LIReC

    @abstractmethod
    def execute(self) -> Optional[Dict[Searchable, DataManager]]:
        """
        Executes the search.
        :return: A mapping from searchables to their search results.
        """
        raise NotImplementedError
