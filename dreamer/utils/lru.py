"""
Small bounded containers for long-lived caches.
"""

from collections import OrderedDict
from typing import Any, Hashable


class LRUDict(OrderedDict):
    """A ``dict`` that keeps at most *maxsize* entries, evicting the least
    recently used one (lookups via ``[]`` / ``get`` count as use).

    Used for caches whose values are large (e.g. trajectory handlers holding
    walked matrices) and whose keys keep arriving for the whole run — an
    unbounded ``dict`` there grows the process' memory without limit.  A miss
    after eviction only costs a recomputation, never correctness.

    :param maxsize: Maximum number of entries (``None`` or ``<= 0`` = unbounded).
    """

    def __init__(self, maxsize: int = 64, *args, **kwargs):
        self.maxsize = maxsize if maxsize and maxsize > 0 else None
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: Hashable) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key: Hashable, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key: Hashable, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        if self.maxsize is not None:
            while len(self) > self.maxsize:
                self.popitem(last=False)


def make_handler_cache() -> LRUDict:
    """The per-shard trajectory-handler cache of the search methods, bounded by
    ``search.HANDLER_CACHE_SIZE``."""
    from dreamer.configs import config
    return LRUDict(config.search.HANDLER_CACHE_SIZE)
