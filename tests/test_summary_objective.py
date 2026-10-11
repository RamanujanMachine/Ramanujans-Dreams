"""The markdown summary must rank/report by the active optimisation objective."""
import json

import pytest

from dreamer.configs import config
from dreamer.utils.storage.summary import build_summary_markdown


def _rec(tid, direction, delta, conv_rate, *, const="e"):
    # One flat row per (trajectory, constant): every metric is a top-level column.
    return {
        "trajectory_id": tid,
        "constant": const,
        "start_point": [1, 1],
        "direction": list(direction),
        "delta": delta,
        "identified": True,
        "convergence_rate": conv_rate,
        "config_fingerprint": "fp",
    }


def _write_shard(tmp_path, shard_id, records):
    path = tmp_path / f"{shard_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def shard_dir(tmp_path):
    # Two trajectories: rec A has the higher δ, rec B the higher convergence_rate.
    _write_shard(
        tmp_path, "cmfA__deadbeef",
        [
            _rec("cmfA__deadbeef__aaa", (3, 1), delta=0.9, conv_rate=0.10),
            _rec("cmfA__deadbeef__bbb", (1, 3), delta=0.2, conv_rate=0.50),
        ],
    )
    return tmp_path


def test_delta_objective_reports_delta(shard_dir, monkeypatch):
    monkeypatch.setattr(config.system, "OPTIMIZATION_OBJECTIVE", "delta")
    md = build_summary_markdown(search_results_root=str(shard_dir))
    assert "Best δ" in md
    assert "Optimisation objective: `delta`" in md
    # δ-best is 0.9 (rec A).
    assert "0.9000" in md


def test_convergence_rate_objective_reports_and_ranks_by_it(shard_dir, monkeypatch):
    monkeypatch.setattr(config.system, "OPTIMIZATION_OBJECTIVE", "convergence_rate")
    md = build_summary_markdown(search_results_root=str(shard_dir))
    assert "Best convergence_rate" in md
    assert "Optimisation objective: `convergence_rate`" in md
    # The best trajectory is now rec B (conv_rate 0.50), not the higher-δ rec A.
    assert "0.5000" in md
    assert "`[1, 3]`" in md  # rec B's direction is surfaced as the best trajectory
