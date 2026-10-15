"""Tests for :mod:`dreamer.utils.allocation` — sizing to the batch job, not the node.

Motivation (measured on the Zeus PBS cluster, 2026-10-15): a job with
``ncpus=4`` on an 80-core node sees ``os.cpu_count() == 80`` and an 80-CPU
affinity mask, and no memory cgroup.  The run must size its pools to the
*request*.

  - Scheduler detection (PBS / Slurm / none) and the core request per scheduler.
  - Cores: ``TOTAL_CORES`` wins; otherwise the smallest of scheduler request,
    affinity mask and ``os.cpu_count()``; garbage values are ignored.
  - Every pool-sizing entry point (search / stage / extraction) follows it.
  - Memory: PBS size strings and ``qstat -f`` parsing; Slurm per-node / per-CPU;
    ``MEMORY_LIMIT_MB`` wins; the PBS lookup runs ``qstat`` once per process
    tree (cached + inherited through the environment).
  - ``available_memory_mb`` becomes a budget inside a job (not the node's RAM).
  - Worker threading: ``OMP_NUM_THREADS = $NCPUS`` set by PBS is reduced to 1
    for children; a deliberate user value is kept; nothing happens outside a job.
"""

import os

import pytest

from dreamer.configs import config
from dreamer.utils import allocation as al
from dreamer.utils import memory_budget as mb

QSTAT_F = """Job Id: 4797556.zeus-master
    Job_Name = zeus_smoke
    resources_used.walltime = 00:00:00
    Resource_List.mem = 8gb
    Resource_List.ncpus = 4
    Resource_List.walltime = 00:10:00
"""


@pytest.fixture
def clean_env(monkeypatch):
    """No scheduler variables, no cached allocation, no core/memory overrides."""
    for var in ("PBS_JOBID", "NCPUS", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_CPUS_PER_NODE", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
                al._MEM_ENV, *al.THREAD_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(al, "_mem_cache", False)
    monkeypatch.setattr(config.system, "TOTAL_CORES", None)
    monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", None)
    return monkeypatch


def _pbs(monkeypatch, ncpus="4"):
    monkeypatch.setenv("PBS_JOBID", "4797556.zeus-master")
    monkeypatch.setenv("NCPUS", ncpus)


class TestScheduler:
    def test_detection(self):
        assert al.scheduler({}) is None
        assert al.scheduler({"PBS_JOBID": "1.zeus-master"}) == "pbs"
        assert al.scheduler({"SLURM_JOB_ID": "77"}) == "slurm"

    def test_pbs_cores_only_inside_a_job(self):
        assert al.scheduler_cores({"PBS_JOBID": "1", "NCPUS": "4"}) == 4
        # NCPUS alone (no job id) is not trusted as an allocation.
        assert al.scheduler_cores({"NCPUS": "4"}) is None

    def test_slurm_cores(self):
        assert al.scheduler_cores({"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "8"}) == 8
        assert al.scheduler_cores({"SLURM_JOB_ID": "1", "SLURM_JOB_CPUS_PER_NODE": "16(x2),8"}) == 16

    @pytest.mark.parametrize("bad", ["", "0", "-3", "four", "  "])
    def test_garbage_values_ignored(self, bad):
        assert al.scheduler_cores({"PBS_JOBID": "1", "NCPUS": bad}) is None


class TestCores:
    def test_job_request_caps_an_unconfined_node(self, clean_env, monkeypatch):
        """The Zeus case: 80 visible cores, 4 granted -> 4."""
        _pbs(monkeypatch, "4")
        monkeypatch.setattr(al.os, "cpu_count", lambda: 80)
        monkeypatch.setattr(al, "affinity_cores", lambda: 80)
        assert al.available_cores() == 4

    def test_affinity_mask_caps_a_larger_request(self, clean_env, monkeypatch):
        _pbs(monkeypatch, "32")
        monkeypatch.setattr(al.os, "cpu_count", lambda: 80)
        monkeypatch.setattr(al, "affinity_cores", lambda: 8)
        assert al.available_cores() == 8

    def test_outside_a_job_uses_the_machine(self, clean_env, monkeypatch):
        monkeypatch.setattr(al.os, "cpu_count", lambda: 12)
        monkeypatch.setattr(al, "affinity_cores", lambda: None)
        assert al.available_cores() == 12

    def test_total_cores_wins(self, clean_env, monkeypatch):
        _pbs(monkeypatch, "4")
        monkeypatch.setattr(config.system, "TOTAL_CORES", 2)
        assert al.available_cores() == 2

    def test_never_below_one(self, clean_env, monkeypatch):
        monkeypatch.setattr(al.os, "cpu_count", lambda: None)
        monkeypatch.setattr(al, "affinity_cores", lambda: None)
        assert al.available_cores() == 1

    def test_every_pool_budget_follows_the_allocation(self, clean_env, monkeypatch):
        from dreamer.extraction.v2.ray_extractor import _resolve_workers
        from dreamer.utils.multi_processing import search_worker_budget
        from dreamer.utils.stage_pool import total_core_budget
        _pbs(monkeypatch, "4")
        monkeypatch.setattr(al.os, "cpu_count", lambda: 80)
        monkeypatch.setattr(al, "affinity_cores", lambda: 80)
        assert search_worker_budget() == 4
        assert total_core_budget() == 4
        assert _resolve_workers(None) == 4
        assert _resolve_workers(3) == 3   # an explicit worker count is kept

    def test_executor_is_sized_not_defaulted(self, clean_env, monkeypatch):
        from dreamer.utils.multi_processing import create_process_pool_executor
        monkeypatch.setattr(config.system, "TOTAL_CORES", 2)
        ex = create_process_pool_executor()
        try:
            assert ex._max_workers == 2
        finally:
            ex.shutdown()


class TestMemory:
    @pytest.mark.parametrize("text, expected", [
        ("8gb", 8192.0), ("8GB", 8192.0), ("8192mb", 8192.0), ("1048576kb", 1024.0),
        ("1tb", 1024.0 ** 2), (str(2 * 2**20), 2.0), ("lots", None), ("8 pb", None),
    ])
    def test_pbs_sizes(self, text, expected):
        got = al.parse_pbs_size_mb(text)
        assert got == (pytest.approx(expected) if expected is not None else None)

    def test_qstat_parsing(self):
        assert al.parse_qstat_mem_mb(QSTAT_F) == pytest.approx(8192.0)
        assert al.parse_qstat_mem_mb("Job Id: 1\n    Resource_List.ncpus = 4\n") is None

    def test_pbs_lookup_runs_qstat_once_and_children_inherit(self, clean_env, monkeypatch):
        _pbs(monkeypatch)
        calls = []

        class Done:
            stdout = QSTAT_F

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return Done()

        monkeypatch.setattr(al.subprocess, "run", fake_run)
        assert al.scheduler_memory_mb() == pytest.approx(8192.0)
        assert al.scheduler_memory_mb() == pytest.approx(8192.0)
        assert calls == [["qstat", "-f", "4797556.zeus-master"]]
        # Exported for worker processes, which then never call qstat.
        assert float(os.environ[al._MEM_ENV]) == pytest.approx(8192.0)
        monkeypatch.setattr(al, "_mem_cache", False)
        assert al.scheduler_memory_mb() == pytest.approx(8192.0)
        assert len(calls) == 1

    def test_qstat_unavailable_means_unknown(self, clean_env, monkeypatch):
        _pbs(monkeypatch)

        def boom(*a, **k):
            raise FileNotFoundError("qstat")

        monkeypatch.setattr(al.subprocess, "run", boom)
        assert al.scheduler_memory_mb() is None

    def test_slurm_memory(self):
        env = {"SLURM_JOB_ID": "1", "SLURM_MEM_PER_NODE": "16000"}
        assert al.scheduler_memory_mb(env) == pytest.approx(16000.0)
        env = {"SLURM_JOB_ID": "1", "SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_PER_TASK": "4"}
        assert al.scheduler_memory_mb(env) == pytest.approx(8000.0)

    def test_no_job_no_allocation(self, clean_env):
        assert al.allocated_memory_mb() is None

    def test_memory_limit_wins(self, clean_env, monkeypatch):
        _pbs(monkeypatch)
        monkeypatch.setenv(al._MEM_ENV, "8192")
        monkeypatch.setattr(config.system, "MEMORY_LIMIT_MB", 3000)
        assert al.allocated_memory_mb() == 3000.0

    def test_available_memory_is_the_job_budget(self, clean_env, monkeypatch):
        """Inside a job the budget is the request minus our RSS — not the node's free RAM."""
        _pbs(monkeypatch)
        monkeypatch.setenv(al._MEM_ENV, "100000")
        monkeypatch.setattr(mb, "_machine_available_mb", lambda: 344_000.0)
        own = mb.rss_mb([os.getpid()])
        avail = mb.available_memory_mb()
        assert avail <= 100_000 - own + 1
        assert avail > 100_000 - own - 500    # only this process (+ maybe a child)


class TestWorkerThreads:
    def test_pbs_thread_count_reduced_to_one(self, clean_env, monkeypatch):
        _pbs(monkeypatch, "4")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")      # what PBS sets
        changed = al.limit_worker_threads()
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"     # unset -> 1 as well
        assert changed["OMP_NUM_THREADS"] == "1"

    def test_deliberate_user_value_kept(self, clean_env, monkeypatch):
        _pbs(monkeypatch, "4")
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        al.limit_worker_threads()
        assert os.environ["OMP_NUM_THREADS"] == "2"

    def test_nothing_outside_a_job(self, clean_env):
        assert al.limit_worker_threads() == {}
        assert "OMP_NUM_THREADS" not in os.environ

    def test_pool_context_applies_it(self, clean_env, monkeypatch):
        from dreamer.utils.mp_context import pool_context
        _pbs(monkeypatch, "4")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        pool_context()
        assert os.environ["OMP_NUM_THREADS"] == "1"


def test_describe_mentions_every_source(clean_env, monkeypatch):
    _pbs(monkeypatch, "4")
    monkeypatch.setenv(al._MEM_ENV, "8192")
    text = al.describe()
    assert "scheduler=pbs" in text and "cores=" in text and "8192 MB" in text
