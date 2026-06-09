"""
Tests for the two large-machine efficiency features:

1. **Shared ``(p, q)`` cache** — :class:`FrequencyList` with an optional
   cross-process ``shared_log`` (local cache + shared append log).  A value
   appended in one list becomes visible to another sharing the same log after a
   :meth:`~FrequencyList.find` sync, without local duplication, and the
   ``shared_log=None`` path is unchanged.

2. **Single global core budget** —
   :func:`dreamer.utils.multi_processing.search_worker_budget` reserves Tier-2
   cores only when Tier-2 is active, and
   :func:`dreamer.search.methods.flatland.parallel_eval.resolve_eval_workers`
   treats the per-method knobs as optional caps against that budget.
"""

import multiprocessing as mp

from dreamer.configs import config
from dreamer.utils.storage.frequency_list import FrequencyList
from dreamer.utils import multi_processing as mpc
from dreamer.search.methods.flatland.parallel_eval import resolve_eval_workers


# ---------------------------------------------------------------------------
# Shared FrequencyList
# ---------------------------------------------------------------------------

class TestSharedFrequencyList:
    def test_value_propagates_across_lists(self):
        """A value appended in list A is found by list B sharing the log."""
        log = []  # plain list satisfies the proxy contract (append + iteration)
        a = FrequencyList(max_size=10, shared_log=log)
        b = FrequencyList(max_size=10, shared_log=log)

        a.append(((1, 2), (3,)))
        # B has not seen it locally yet ...
        assert all(item[0] != ((1, 2), (3,)) for item in b.items)
        # ... but find() syncs from the shared log first, then matches.
        hit = b.find(lambda v: v == ((1, 2), (3,)))
        assert hit == ((1, 2), (3,))
        assert b._synced_len == 1
        assert any(item[0] == ((1, 2), (3,)) for item in b.items)

    def test_no_local_duplication_or_double_publish(self):
        """Re-appending a synced value neither duplicates locally nor in the log."""
        log = []
        a = FrequencyList(max_size=10, shared_log=log)
        b = FrequencyList(max_size=10, shared_log=log)

        a.append(("p", "q"))
        b.find(lambda v: v == ("p", "q"))  # syncs ("p","q") into b
        # b already has it locally → append is a no-op and must not re-publish.
        b.append(("p", "q"))
        assert sum(item[0] == ("p", "q") for item in b.items) == 1
        assert log.count(("p", "q")) == 1

    def test_find_miss_still_advances_cursor(self):
        log = []
        a = FrequencyList(max_size=10, shared_log=log)
        b = FrequencyList(max_size=10, shared_log=log)
        a.append((1,))
        a.append((2,))
        assert b.find(lambda v: v == (99,)) is None  # no match
        assert b._synced_len == 2  # but both shared entries were folded in
        assert {item[0] for item in b.items} == {(1,), (2,)}

    def test_shared_log_none_is_unchanged_behaviour(self):
        """Without a shared log the cache behaves exactly as before."""
        fl = FrequencyList(max_size=2)
        fl.append((1,))
        fl.append((1,))  # dedup
        assert len(fl.items) == 1
        fl.append((2,))
        fl.append((3,))  # evicts least-frequent (the last/coldest item)
        assert len(fl.items) == 2
        assert fl._synced_len == 0  # never touched

    def test_real_manager_round_trip(self):
        """End-to-end with a genuine Manager list proxy (cross-process safe)."""
        manager = mp.Manager()
        try:
            log = manager.list()
            a = FrequencyList(max_size=10, shared_log=log)
            b = FrequencyList(max_size=10, shared_log=log)
            a.append(((7,), (1,)))
            assert b.find(lambda v: v == ((7,), (1,))) == ((7,), (1,))
            assert list(log) == [((7,), (1,))]
        finally:
            manager.shutdown()


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------

class TestSearchWorkerBudget:
    def test_tier2_inactive_uses_all_cores(self, monkeypatch):
        monkeypatch.setattr(mpc.sys_config, "TOTAL_CORES", 10)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        assert mpc.search_worker_budget() == 10

    def test_tier2_active_reserves_workers_plus_writer(self, monkeypatch):
        monkeypatch.setattr(mpc.sys_config, "TOTAL_CORES", 10)
        monkeypatch.setattr(mpc.sys_config, "NUM_BACKGROUND_WORKERS", 4)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ("eigenvalues",))
        assert mpc.search_worker_budget() == 10 - (4 + 1)

    def test_budget_floored_at_one(self, monkeypatch):
        monkeypatch.setattr(mpc.sys_config, "TOTAL_CORES", 2)
        monkeypatch.setattr(mpc.sys_config, "NUM_BACKGROUND_WORKERS", 8)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ("eigenvalues",))
        assert mpc.search_worker_budget() == 1

    def test_none_total_falls_back_to_cpu_count(self, monkeypatch):
        monkeypatch.setattr(mpc.sys_config, "TOTAL_CORES", None)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        monkeypatch.setattr(mpc.os, "cpu_count", lambda: 7)
        assert mpc.search_worker_budget() == 7


class TestResolveEvalWorkers:
    def test_zero_and_none_use_full_budget(self, monkeypatch):
        monkeypatch.setattr(
            "dreamer.search.methods.flatland.parallel_eval.search_worker_budget",
            lambda: 6,
        )
        assert resolve_eval_workers(0) == 6
        assert resolve_eval_workers(None) == 6

    def test_positive_value_caps_at_budget(self, monkeypatch):
        monkeypatch.setattr(
            "dreamer.search.methods.flatland.parallel_eval.search_worker_budget",
            lambda: 6,
        )
        assert resolve_eval_workers(2) == 2     # below budget → honoured
        assert resolve_eval_workers(100) == 6   # above budget → capped
