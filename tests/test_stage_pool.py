"""Tests for the stage-wide process pool and buffered single writer.

Coverage:
  - Core budget / ``STAGE_POOL_WORKERS`` cap resolution (incl. edge values).
  - ``StagePool``: serial and multi-process paths, ordered results, the stage
    context (shards / extra) inside tasks, error isolation, state cleanup.
  - ``BufferedJsonlWriter``: batching (one open per batch), timeout flush,
    multiple files, ``write_jsonl_line`` compatibility, empty edge case.
  - ``run_tier1_task`` known answer: identical rows to a direct serial walk.
  - Stage equivalence: the analyzer, ``SearcherModV1`` and the Tier-3
    post-process write the **same rows** (and the analyzer the same
    priorities) whether their stage pool runs serially or on 2 processes.

The autouse ``serial_stage_pools`` fixture (conftest) pins stage pools to serial;
tests of the multi-process path override ``STAGE_POOL_WORKERS`` explicitly.
"""

import builtins
import json
import os

import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.configs.system import sys_config
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.multi_processing import write_jsonl_line
from dreamer.utils.stage_pool import (
    BufferedJsonlWriter,
    StagePool,
    TaskError,
    Tier1Result,
    Tier1Task,
    resolve_stage_workers,
    run_tier1_task,
    stage_extra,
    stage_shard,
    total_core_budget,
)
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    build_trajectory_dtos,
    derive_cmf_and_shard_ids,
)


# ---------------------------------------------------------------------------
# Module-level task functions (must be picklable for the spawned workers)
# ---------------------------------------------------------------------------

class _FakeShard:
    """Minimal shard stand-in (picklable): a CMF reference, a cache and a label."""

    def __init__(self, label, cmf):
        from dreamer.utils.storage.frequency_list import FrequencyList
        self.label = label
        self.cmf = cmf
        self.cache = FrequencyList(10)


from collections import namedtuple  # noqa: E402

_T = namedtuple("_T", ["shard_idx", "x"])


def _echo_task(t):
    """Return the task value with the stage context it was executed under."""
    if isinstance(t, _T):
        sh = stage_shard(t.shard_idx)
        return t.x, (sh.label, sh.cmf, type(sh.cache).__name__), stage_extra("tag"), os.getpid()
    return t, None, stage_extra("tag"), os.getpid()


def _worker_shard_count(_):
    """How many shards does this worker hold?"""
    from dreamer.utils.stage_pool import _STAGE_STATE
    return len(_STAGE_STATE["shards"])


def _maybe_fail_task(x):
    """Fail (by returning a TaskError) on odd inputs."""
    if x % 2:
        return TaskError(x, "odd")
    return x * 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cmf():
    """1F1(z=1) — a minimal 2-symbol CMF."""
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def simple_shard(simple_cmf, symbols):
    """A bounded shard with two hyperplanes and an interior point at (1,1)."""
    zero = Position({s: sp.Integer(0) for s in symbols})
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero, interior)


@pytest.fixture
def two_workers(monkeypatch):
    """Allow a real 2-process stage pool for this test."""
    monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", 2)
    monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)


def _rows(path):
    """All JSON rows of a JSONL file as a sorted list of canonical strings."""
    with open(path) as f:
        return sorted(json.dumps(json.loads(ln), sort_keys=True) for ln in f if ln.strip())


# ---------------------------------------------------------------------------
# Core budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_total_core_budget_uses_total_cores(self, monkeypatch):
        monkeypatch.setattr(sys_config, "TOTAL_CORES", 7)
        assert total_core_budget() == 7

    def test_total_core_budget_defaults_to_cpu_count(self, monkeypatch):
        monkeypatch.setattr(sys_config, "TOTAL_CORES", None)
        assert total_core_budget() == max(1, os.cpu_count() or 1)

    @pytest.mark.parametrize("cap, budget, expected", [
        (None, 8, 8),   # no cap → full budget
        (0, 8, 8),      # 0 → full budget
        (-3, 8, 8),     # negative → full budget
        (3, 8, 3),      # cap below budget
        (20, 8, 8),     # cap above budget
        (1, 8, 1),      # serial
    ])
    def test_resolve_stage_workers(self, monkeypatch, cap, budget, expected):
        monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", cap)
        assert resolve_stage_workers(budget) == expected

    def test_resolve_stage_workers_floors_at_one(self, monkeypatch):
        monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", None)
        assert resolve_stage_workers(0) == 1


# ---------------------------------------------------------------------------
# StagePool
# ---------------------------------------------------------------------------

class TestStagePoolSerial:
    def test_serial_runs_in_process_with_context(self):
        shard = _FakeShard("shard-A", "CMF")
        with StagePool([shard], workers=1, extra={"tag": "T"}) as pool:
            assert not pool.parallel
            out = list(pool.imap(_echo_task, [_T(0, x) for x in (1, 2, 3)]))
        assert [o[0] for o in out] == [1, 2, 3]
        assert all(o[1] == ("shard-A", "CMF", "FrequencyList") and o[2] == "T" for o in out)
        assert {o[3] for o in out} == {os.getpid()}

    def test_state_cleared_after_exit(self):
        with StagePool(["shard-A"], workers=1):
            assert stage_shard(0) == "shard-A"
        with pytest.raises(KeyError):
            stage_shard(0)

    def test_empty_task_list(self):
        with StagePool([], workers=1) as pool:
            assert list(pool.imap(_echo_task, [])) == []

    def test_task_errors_do_not_abort_stream(self):
        with StagePool([], workers=1) as pool:
            out = list(pool.imap(_maybe_fail_task, range(5)))
        assert out[0] == 0 and out[2] == 20 and out[4] == 40
        assert isinstance(out[1], TaskError) and out[1].key == 1


@pytest.mark.timeout(180)
class TestStagePoolParallel:
    def test_results_ordered_and_context_in_workers(self, two_workers):
        shards = [_FakeShard("shard-A", "CMF-1"), _FakeShard("shard-B", "CMF-1")]
        tasks = [_T(i % 2, i) for i in range(20)]
        with StagePool(shards, extra={"tag": "T"}) as pool:
            assert pool.parallel and pool.workers == 2
            out = list(pool.imap(_echo_task, tasks))
        assert [o[0] for o in out] == list(range(20))   # delivery is in task order
        for t, o in zip(tasks, out):
            # The worker got the right shard, re-attached to the CMF sent once,
            # with a fresh (p, q) cache bound to its shared log.
            assert o[1] == (shards[t.shard_idx].label, "CMF-1", "FrequencyList") and o[2] == "T"
        assert os.getpid() not in {o[3] for o in out}  # ran in worker processes

    def test_workers_hold_only_the_shards_they_used(self, two_workers):
        """Workers never receive the whole shard list (a 9D CMF has >300k shards):
        only the shards of tasks they ran, at most a small LRU."""
        shards = [_FakeShard(f"s{i}", "CMF") for i in range(500)]
        with StagePool(shards) as pool:
            assert list(pool.imap(_echo_task, [_T(0, 1), _T(1, 2)]))[0][0] == 1
            counts = list(pool.imap(_worker_shard_count, range(4)))
        assert max(counts) <= 2

    def test_task_errors_do_not_abort_parallel_stream(self, two_workers):
        with StagePool([]) as pool:
            out = list(pool.imap(_maybe_fail_task, range(6)))
        assert [o for o in out if not isinstance(o, TaskError)] == [0, 20, 40]
        assert [o.key for o in out if isinstance(o, TaskError)] == [1, 3, 5]


# ---------------------------------------------------------------------------
# BufferedJsonlWriter
# ---------------------------------------------------------------------------

class TestBufferedJsonlWriter:
    def test_batches_opens(self, tmp_path, monkeypatch):
        path = str(tmp_path / "a.jsonl")
        opens = [0]
        real_open = builtins.open

        def counting_open(p, mode="r", *a, **k):
            if str(p) == path and "a" in mode:
                opens[0] += 1
            return real_open(p, mode, *a, **k)

        monkeypatch.setattr(builtins, "open", counting_open)
        with BufferedJsonlWriter(batch_size=4, flush_timeout=1e9) as w:
            for i in range(10):
                w.write_line(path, json.dumps({"i": i}))
            assert opens[0] == 2           # two full batches flushed
        assert opens[0] == 3               # remainder flushed on close
        with real_open(path) as f:
            assert [json.loads(x)["i"] for x in f] == list(range(10))

    def test_timeout_flushes_everything(self, tmp_path):
        a, b = str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")
        w = BufferedJsonlWriter(batch_size=1000, flush_timeout=0.0)
        w.write_line(a, "{}")
        w.write_line(b, "{}")  # timeout elapsed → flushes both paths
        assert os.path.exists(a) and os.path.exists(b)
        w.close()

    def test_write_jsonl_line_compat_drops_structural_patch(self, tmp_path):
        path = str(tmp_path / "p.jsonl")
        with BufferedJsonlWriter() as w:
            write_jsonl_line({"trajectory_id": "t", "constant": "e"}, w.sink(path))  # no-op patch
            write_jsonl_line({"trajectory_id": "t", "constant": "e", "x": 1}, w.sink(path))
        with open(path) as f:
            assert [json.loads(x) for x in f] == [{"trajectory_id": "t", "constant": "e", "x": 1}]

    def test_no_writes_creates_no_file(self, tmp_path):
        with BufferedJsonlWriter():
            pass
        assert list(tmp_path.iterdir()) == []

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "dir" / "x.jsonl")
        with BufferedJsonlWriter() as w:
            w.write_line(path, "{}")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# run_tier1_task — known answer against the direct serial walk
# ---------------------------------------------------------------------------

class TestRunTier1Task:
    def _task(self, shard, symbols, constants):
        cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
        traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        return Tier1Task("k", 0, traj, start, constants, cmf_id, shard_id, enc), traj, start

    def test_matches_direct_walk(self, simple_shard, symbols):
        task, traj, start = self._task(simple_shard, symbols, (e,))
        with StagePool([simple_shard], workers=1):
            res = run_tier1_task(task, want_matrix=True)
        assert isinstance(res, Tier1Result) and res.key == "k"

        handler = TrajectoryAttributesHandler.from_cmf(
            simple_shard.cmf, traj, start, constant=None, searchable=simple_shard,
        )
        expected = build_trajectory_dtos(
            handler, cmf_id=task.cmf_id, shard_id=task.shard_id,
            cmf_name=simple_shard.cmf_name, shard_encoding_str=task.shard_encoding_str,
            start=start, direction=traj, constants=[e],
        )
        assert [d.to_json_line() for d in res.dtos] == [d.to_json_line() for d in expected]
        assert res.matrix == handler.trajectory_matrix

    def test_no_constants_returns_matrix_only(self, simple_shard, symbols):
        task, _, _ = self._task(simple_shard, symbols, ())
        with StagePool([simple_shard], workers=1):
            res = run_tier1_task(task, want_matrix=True)
        assert res.dtos == [] and res.matrix is not None

    def test_failure_returns_task_error(self, simple_shard, symbols):
        task, _, _ = self._task(simple_shard, symbols, (e,))

        class _Boom:
            @staticmethod
            def from_cmf(*a, **k):
                raise RuntimeError("boom")

        with StagePool([simple_shard], workers=1):
            res = run_tier1_task(task, handler_cls=_Boom)
        assert isinstance(res, TaskError) and res.key == "k" and "boom" in res.message


# ---------------------------------------------------------------------------
# Stage equivalence: serial vs 2-process pool
# ---------------------------------------------------------------------------

def _run_analyzer(shard, out_dir, monkeypatch):
    from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
    from dreamer.configs.analysis import analysis_config

    monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out_dir))
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", None)
    monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)
    monkeypatch.setattr(analysis_config, "STORE_TRAJECTORIES_SEPARATELY", False)
    return AnalyzerModV1({e: [shard]}).execute()


@pytest.mark.timeout(300)
class TestStageEquivalence:
    def test_analyzer_parallel_matches_serial(self, simple_shard, tmp_path, monkeypatch):
        _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        serial_dir, par_dir = tmp_path / "serial", tmp_path / "parallel"

        serial = _run_analyzer(simple_shard, serial_dir, monkeypatch)
        monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", 2)
        monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
        parallel = _run_analyzer(simple_shard, par_dir, monkeypatch)

        rows = _rows(serial_dir / f"{shard_id}.jsonl")
        assert rows, "analysis wrote no rows"
        assert rows == _rows(par_dir / f"{shard_id}.jsonl")
        # Ordered delivery ⇒ identical file order too.
        assert (serial_dir / f"{shard_id}.jsonl").read_text() == (par_dir / f"{shard_id}.jsonl").read_text()
        assert {c.name: len(v) for c, v in serial.items()} == {c.name: len(v) for c, v in parallel.items()}

    def test_analyzer_rerun_reuses_cache_without_pool(self, simple_shard, tmp_path, monkeypatch):
        """A second run over a fully-cached shard computes nothing (no pool, no new rows)."""
        import dreamer.analysis.analyzers.serial_scan.analyzer_mod as am

        _run_analyzer(simple_shard, tmp_path, monkeypatch)
        _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        before = (tmp_path / f"{shard_id}.jsonl").read_text()

        def _no_pool(*a, **k):
            raise AssertionError("pool must not be created when everything is cached")

        monkeypatch.setattr(am, "StagePool", _no_pool)
        _run_analyzer(simple_shard, tmp_path, monkeypatch)
        assert (tmp_path / f"{shard_id}.jsonl").read_text() == before

    def test_search_parallel_matches_serial(self, simple_shard, tmp_path, monkeypatch):
        from dreamer.configs import config
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1

        _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)

        outs = {}
        for name, workers in (("serial", 1), ("parallel", 2)):
            out = tmp_path / name
            monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out))
            monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", workers)
            monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
            SearcherModV1({e: [simple_shard]}, use_LIReC=False).execute()
            outs[name] = out / f"{shard_id}.jsonl"
        assert _rows(outs["serial"]), "search wrote no rows"
        assert outs["serial"].read_text() == outs["parallel"].read_text()

    def test_tier3_parallel_matches_serial(self, simple_shard, tmp_path, monkeypatch):
        from dreamer.configs import config
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1

        cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        monkeypatch.setattr(config.post_process, "TIER3_ATTRIBUTES", ("traj_size",))
        base = [
            {"trajectory_id": f"t{i}", "cmf_id": cmf_id, "shard_id": shard_id,
             "constant": e.name, "start_point": [1, 1], "direction": [1, i]}
            for i in (1, 2, 3)
        ]
        outs = {}
        for name, workers in (("serial", 1), ("parallel", 2)):
            out = tmp_path / name
            out.mkdir()
            path = out / f"{shard_id}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in base))
            monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out))
            monkeypatch.setattr(sys_config, "STAGE_POOL_WORKERS", workers)
            monkeypatch.setattr(sys_config, "TOTAL_CORES", 2)
            Tier3PostProcessModV1({e: [simple_shard]}).execute()
            outs[name] = path.read_text()
        patches = outs["serial"].splitlines()[len(base):]
        assert len(patches) == len(base)
        assert all("traj_size" in json.loads(p) for p in patches), patches
        assert outs["serial"] == outs["parallel"]


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
class TestFailurePaths:
    def test_pool_terminated_and_state_cleared_on_exception(self, two_workers):
        """An exception inside the ``with`` block terminates the workers, shuts the
        manager down and clears the stage state (no leaked processes)."""
        with pytest.raises(RuntimeError):
            with StagePool([_FakeShard("shard-A", "CMF")]) as pool:
                assert list(pool.imap(_echo_task, [_T(0, 1), _T(0, 2)]))[0][0] == 1   # workers started
                procs = list(pool._pool._executor._processes.values())
                assert procs
                raise RuntimeError("producer failed")
        assert pool._pool is None and pool._manager is None
        assert all(not p.is_alive() for p in procs)
        with pytest.raises(KeyError):
            stage_shard(0)

    def test_analyzer_skips_failed_walks(self, simple_shard, tmp_path, monkeypatch):
        """A walk that raises inside a task is logged and skipped: no rows are
        written for it and the stage still completes (mirrors the serial path)."""
        def boom(*a, **k):
            raise RuntimeError("walk failed")

        monkeypatch.setattr(TrajectoryAttributesHandler, "from_cmf", boom)
        result = _run_analyzer(simple_shard, tmp_path, monkeypatch)
        _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        assert not (tmp_path / f"{shard_id}.jsonl").exists() or _rows(tmp_path / f"{shard_id}.jsonl") == []
        assert result == {e: []}

    def test_search_skips_failed_walks(self, simple_shard, tmp_path, monkeypatch):
        """``SearcherModV1`` logs and skips a failing trajectory instead of aborting."""
        from dreamer.configs import config
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1

        def boom(*a, **k):
            raise RuntimeError("walk failed")

        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
        monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(TrajectoryAttributesHandler, "from_cmf", boom)
        SearcherModV1({e: [simple_shard]}, use_LIReC=False).execute()
        _, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        path = tmp_path / f"{shard_id}.jsonl"
        assert not path.exists() or _rows(path) == []
