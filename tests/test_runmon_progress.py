"""Tests for :mod:`runmon.progress` — the progress tree.

  - Nesting follows the open/close order per thread; paths and the running stage.
  - Fractions: counts, the open child's share (3.5 / 12), explicit estimates
    override counts, unknown totals give ``None``, done nodes read 1.
  - ETA from elapsed time and fraction.
  - Closing: context manager marks failures, dangling inner nodes are stopped,
    finished children are bounded (KEEP_FINISHED).
  - Snapshots are JSON-serialisable.
  - Hot-path cost: ``advance`` stays in the microsecond range.
"""

import json
import threading
import time

import pytest

from runmon import progress as pg


@pytest.fixture
def tree():
    return pg.ProgressTree()


def test_nesting_and_paths(tree):
    with tree.task("search", total=2, kind="stage") as stage:
        with tree.task("annealing", total=4, kind="method") as m:
            with tree.task("shard 1", kind="shard") as s:
                assert tree.current() is s
                assert s.path() == "search/annealing/shard 1"
                assert tree.active_path() == "search/annealing/shard 1"
                assert tree.stage() == "search"
            assert tree.current() is m
        assert stage.children[0] is m
    assert tree.current() is None
    assert tree.stage() is None


def test_fraction_includes_open_child(tree):
    with tree.task("stage", total=12) as stage:
        stage.set(done=3)
        with tree.task("shard 4", total=10) as shard:
            shard.set(done=5)
            assert stage.fraction() == pytest.approx(3.5 / 12)
        stage.advance()
        assert stage.fraction() == pytest.approx(4 / 12)


def test_estimate_overrides_count_and_clamps(tree):
    node = tree.open("sa", total=100)
    node.set(done=10)
    node.estimate(0.42, T=0.3)
    assert node.fraction() == pytest.approx(0.42) and node.estimated
    assert node.metrics["T"] == 0.3
    node.estimate(7.0)
    assert node.fraction() == 1.0
    node.estimate(None)
    assert node.fraction() == pytest.approx(0.10) and not node.estimated


def test_unknown_total_and_done(tree):
    node = tree.open("open-ended")
    assert node.fraction() is None and node.eta_seconds() is None
    node.close()
    assert node.fraction() == 1.0 and node.status == "done"


def test_eta(tree, monkeypatch):
    node = tree.open("x", total=4)
    node.started = time.time() - 30
    node.set(done=1)                                  # 25 % after 30 s -> 90 s left
    assert node.eta_seconds() == pytest.approx(90, rel=0.05)


def test_failure_and_dangling_children(tree):
    with pytest.raises(RuntimeError):
        with tree.task("outer") as outer:
            tree.open("inner-never-closed")
            raise RuntimeError("x")
    assert outer.status == "failed"
    assert outer.children[0].status == "stopped"
    assert tree.current() is None


def test_close_is_idempotent(tree):
    n = tree.open("x")
    n.close("failed")
    n.close("done")
    assert n.status == "failed"


def test_finished_children_are_bounded(tree):
    with tree.task("stage", total=100) as stage:
        for i in range(50):
            with tree.task(f"shard {i}"):
                pass
            stage.advance()
        assert len(stage.children) == pg.KEEP_FINISHED
        assert stage.children[-1].name == "shard 49"


def test_threads_have_separate_stacks(tree):
    seen = {}
    main = tree.open("main-task")

    def worker():
        n = tree.open("thread-task")
        seen["parent"] = n.parent
        n.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["parent"] is None                     # not nested under main's task
    assert tree.current() is main


def test_snapshot_is_json(tree):
    with tree.task("search", total=3, kind="stage") as s:
        s.metric(best=0.0167, obj=object())
        snap = tree.snapshot()
    text = json.dumps(snap, default=str)
    assert '"search"' in text and snap[0]["children"] == []


def test_advance_is_cheap(tree):
    node = tree.open("hot", total=10**6)
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        node.advance()
    per_call = (time.perf_counter() - t0) / n
    assert node.done == n
    assert per_call < 5e-6, f"advance() costs {per_call * 1e6:.2f} µs"
