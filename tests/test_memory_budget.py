"""Tests for memory-aware worker counts.

  - Parsers: Linux ``/proc/meminfo``, cgroup v2 / v1 limits ("max" = none),
    macOS ``vm_stat``.
  - ``available_memory_mb`` with ``MEMORY_LIMIT_MB``: limit minus the RSS of
    this process and its live children (a real budget).
  - ``memory_worker_cap``: ``min(cores, floor((avail - reserve) / per_worker))``,
    floored at 1, disabled by ``MEMORY_AWARE_WORKERS = False``.
  - ``ResilientPool`` applies the cap at construction and **shrinks** a running
    pool when available memory falls below the reserve — while every task still
    completes with the right result (the heuristic must never corrupt a run).
  - The measured worker baseline is plausible on this machine.
"""

import os

import pytest

from dreamer.configs import config
from dreamer.utils import memory_budget as mb
from dreamer.utils.resilient_pool import ResilientPool

MEMINFO = """MemTotal:       65536000 kB
MemFree:         1000000 kB
MemAvailable:   40960000 kB
"""
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100000.
Pages active:                                 370342.
Pages inactive:                               200000.
Pages speculative:                              4000.
"""


class TestParsers:
    def test_meminfo(self):
        assert mb.parse_meminfo_available_mb(MEMINFO) == pytest.approx(40000.0)
        assert mb.parse_meminfo_available_mb("nothing") is None

    def test_vm_stat(self):
        assert mb.parse_vm_stat_available_mb(VM_STAT) == pytest.approx(304000 * 16384 / 2**20)
        assert mb.parse_vm_stat_available_mb("garbage") is None

    def test_cgroup_v2_and_v1(self, tmp_path):
        (tmp_path / "memory.max").write_text(str(8 * 2**30) + "\n")
        (tmp_path / "memory.current").write_text(str(2 * 2**30) + "\n")
        assert mb.cgroup_available_mb(str(tmp_path)) == pytest.approx(6 * 1024)
        (tmp_path / "memory.max").write_text("max\n")
        assert mb.cgroup_available_mb(str(tmp_path)) is None   # no limit
        v1 = tmp_path / "v1" / "memory"
        v1.mkdir(parents=True)
        (v1 / "memory.limit_in_bytes").write_text(str(4 * 2**30))
        (v1 / "memory.usage_in_bytes").write_text(str(1 * 2**30))
        assert mb.cgroup_available_mb(str(tmp_path / "v1")) == pytest.approx(3 * 1024)


class TestAvailable:
    def test_limit_is_a_budget_minus_current_usage(self, monkeypatch):
        monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", 100_000)
        own = mb.rss_mb([os.getpid()])
        assert own > 10
        assert mb.available_memory_mb() == pytest.approx(100_000 - own, rel=0.05)

    def test_detection_is_plausible(self, monkeypatch):
        monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", None)
        avail = mb.available_memory_mb()
        assert 0 < avail <= mb.total_memory_mb()


class TestCap:
    @pytest.mark.parametrize("requested, avail, reserve, each, expected", [
        (80, 64_000, 1_000, 600, 80),     # memory is not the limit
        (80, 20_000, 1_000, 600, 31),     # memory caps it: floor(19000 / 600)
        (80, 1_200, 1_000, 600, 1),       # floored at 1
        (1, 100, 1_000, 600, 1),
    ])
    def test_formula(self, monkeypatch, requested, avail, reserve, each, expected):
        monkeypatch.setattr(mb, "available_memory_mb", lambda: avail)
        monkeypatch.setattr(mb, "per_worker_mb", lambda: each)
        monkeypatch.setattr(config.system, "MEMORY_RESERVE_MB", reserve)
        monkeypatch.setattr(config.system, "MEMORY_AWARE_WORKERS", True)
        assert mb.memory_worker_cap(requested, log=False) == expected

    def test_disabled(self, monkeypatch):
        monkeypatch.setattr(mb, "available_memory_mb", lambda: 10)
        monkeypatch.setattr(config.system, "MEMORY_AWARE_WORKERS", False)
        assert mb.memory_worker_cap(64, log=False) == 64

    def test_per_worker_configured_or_measured(self, monkeypatch):
        monkeypatch.setattr(config.system, "WORKER_MEMORY_MB", 700)
        assert mb.per_worker_mb() == 700
        monkeypatch.setattr(config.system, "WORKER_MEMORY_MB", None)
        monkeypatch.setattr(config.system, "WORKER_TASK_HEADROOM_MB", 100)
        assert mb.per_worker_mb() == pytest.approx(mb.measured_worker_baseline_mb() + 100)

    @pytest.mark.timeout(180)
    def test_measured_baseline_is_plausible(self):
        base = mb.measured_worker_baseline_mb()
        assert 40 < base < 400


# ---------------------------------------------------------------------------
# Pool integration
# ---------------------------------------------------------------------------

def _hog(args):
    """Hold ~``mb_`` MB for a moment, return a checksum (memory-hungry task)."""
    i, mb_ = args
    import time
    block = bytearray(mb_ * 2**20)
    block[::4096] = b"x" * len(block[::4096])
    time.sleep(0.3)
    return i * i


@pytest.mark.timeout(300)
class TestPoolIntegration:
    def test_pool_is_capped_at_construction(self, monkeypatch):
        monkeypatch.setattr(config.system, "MEMORY_AWARE_WORKERS", True)
        monkeypatch.setattr(mb, "available_memory_mb", lambda: 1_000 + 2 * 300)
        monkeypatch.setattr(mb, "per_worker_mb", lambda: 300)
        monkeypatch.setattr(config.system, "MEMORY_RESERVE_MB", 1_000)
        with ResilientPool(16, name="t") as pool:
            assert pool._processes == 2
            assert list(pool.imap(_hog, [(i, 1) for i in range(5)])) == [i * i for i in range(5)]

    def test_running_pool_shrinks_and_stays_correct(self, monkeypatch):
        """A tight budget: while hungry tasks run, available memory (limit minus
        our process tree) falls below the reserve, so the pool shrinks — and
        every result is still correct and in order."""
        monkeypatch.setattr(config.system, "MEMORY_AWARE_WORKERS", True)
        own = mb.rss_mb([os.getpid()])
        # Budget: main + reserve + room for ~2 idle workers; four hungry workers exceed it.
        monkeypatch.setattr(config.system, "MEMORY_RESERVE_MB", 200)
        monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", int(own + 200 + 2 * 150 + 200))
        monkeypatch.setattr(mb, "per_worker_mb", lambda: 1)       # start wide on purpose
        with ResilientPool(4, name="t") as pool:
            assert pool._processes == 4
            out = list(pool.imap(_hog, [(i, 120) for i in range(24)]))
        assert out == [i * i for i in range(24)]
        assert pool.shrinks >= 1 and pool._processes < 4


@pytest.mark.timeout(300)
def test_pool_grows_in_place_after_warm_up(monkeypatch):
    """Started below the core count by a conservative estimate, a pool grows in
    place (no executor replacement) once warm workers measure small — and all
    results stay correct.  Needs dynamically spawned workers (spawn/forkserver)."""
    import multiprocessing as mp
    from dreamer.utils import resilient_pool as rp

    if mp.get_start_method() == "fork":
        pytest.skip("in-place growth needs spawn/forkserver workers")
    monkeypatch.setattr(config.system, "MEMORY_AWARE_WORKERS", True)
    monkeypatch.setattr(config.system, "MEMORY_RESERVE_MB", 100)
    monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", None)
    monkeypatch.setattr(mb, "available_memory_mb", lambda: 100 + 2 * 10_000)   # 2 by estimate...
    monkeypatch.setattr(mb, "per_worker_mb", lambda: 10_000)                   # ...pessimistic
    monkeypatch.setattr(rp.ResilientPool, "_GROW_INTERVAL", 0.0)
    with ResilientPool(4, name="t") as pool:
        assert pool._processes == 2
        executor = pool._executor
        # warm workers measure ~100-200 MB, far below the estimate → room to grow
        monkeypatch.setattr(mb, "available_memory_mb", lambda: 100 + 4 * 1_000)
        out = list(pool.imap(_hog, [(i, 1) for i in range(40)]))
    assert out == [i * i for i in range(40)]
    assert pool.grows >= 1 and pool._processes == 4
    assert pool._executor is executor          # grown in place
