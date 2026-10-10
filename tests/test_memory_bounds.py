"""Tests for the memory bounds of long-lived caches.

  - ``LRUDict``: never exceeds ``maxsize``; lookups refresh recency; ``get`` /
    ``in`` work; ``maxsize <= 0`` is unbounded.
  - ``make_handler_cache`` follows ``search.HANDLER_CACHE_SIZE``.
  - ``FrequencyList`` with a shared log fetches only the **new tail** of the
    log (not the whole list) and still folds every published value in once.
"""

import multiprocessing as mp

import pytest

from dreamer.configs import config
from dreamer.utils.lru import LRUDict, make_handler_cache
from dreamer.utils.storage.frequency_list import FrequencyList


class TestLRUDict:
    def test_bounded_and_evicts_least_recent(self):
        d = LRUDict(3)
        for k in "abc":
            d[k] = k.upper()
        _ = d["a"]                      # refresh "a"
        d["d"] = "D"                    # evicts "b" (least recently used)
        assert list(d) == ["c", "a", "d"] and len(d) == 3
        assert "b" not in d and d.get("b") is None and d.get("a") == "A"

    def test_unbounded_when_nonpositive(self):
        d = LRUDict(0)
        for i in range(1000):
            d[i] = i
        assert len(d) == 1000

    def test_make_handler_cache_uses_config(self, monkeypatch):
        monkeypatch.setattr(config.search, "HANDLER_CACHE_SIZE", 5)
        cache = make_handler_cache()
        for i in range(20):
            cache[i] = object()
        assert len(cache) == 5 and list(cache) == list(range(15, 20))


class _CountingList(list):
    """A list that records the slices / full copies requested from it."""

    def __init__(self, *a):
        super().__init__(*a)
        self.full_copies = 0
        self.slices = []

    def __iter__(self):
        self.full_copies += 1
        return super().__iter__()

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.slices.append((item.start, item.stop))
        return super().__getitem__(item)


class TestSharedLogSync:
    def test_fetches_only_the_new_tail(self):
        log = _CountingList()
        fl = FrequencyList(max_size=100, shared_log=log)
        for v in range(5):
            log.append((v,))
        fl._sync_from_shared()
        log.append((5,))
        log.append((6,))
        fl._sync_from_shared()
        fl._sync_from_shared()          # nothing new → no fetch
        assert log.slices == [(0, 5), (5, 7)]
        assert log.full_copies == 0
        assert fl.find(lambda v: v == (6,)) == (6,)

    @pytest.mark.timeout(120)
    def test_real_manager_list(self):
        with mp.Manager() as manager:
            shared = manager.list()
            a = FrequencyList(max_size=100, shared_log=shared)
            b = FrequencyList(max_size=100, shared_log=shared)
            a.append(((1, 2), (3, 4)))
            assert b.find(lambda v: v == ((1, 2), (3, 4))) == ((1, 2), (3, 4))
            a.append(((5,), (6,)))
            assert b.find(lambda v: v == ((5,), (6,))) == ((5,), (6,))
