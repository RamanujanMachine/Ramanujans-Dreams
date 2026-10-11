"""Tests for the post-process graphing stage (bumpiness math + Grapher I/O)."""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from dreamer.configs import config
from dreamer.graphing.bumpiness import (
    angular_distance,
    empirical_semivariogram,
    median_total_variation,
    total_variation,
)
from dreamer.graphing import Grapher
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


# ---------------------------------------------------------------------------
# (A) Total variation
# ---------------------------------------------------------------------------

class TestTotalVariation:
    def test_hand_value(self):
        assert total_variation([1, 2, 1, 3]) == 4.0  # 1 + 1 + 2

    def test_drops_non_finite(self):
        assert total_variation([0, float("-inf"), 0, 5]) == 5.0

    def test_too_short_is_nan(self):
        assert math.isnan(total_variation([1]))
        assert math.isnan(total_variation([]))

    def test_median(self):
        med, n = median_total_variation([[1, 2, 1], [0, 0, 0, 5]])  # TVs 2, 5
        assert med == 3.5 and n == 2

    def test_median_empty(self):
        med, n = median_total_variation([[1], []])
        assert math.isnan(med) and n == 0


# ---------------------------------------------------------------------------
# (B) Semivariogram
# ---------------------------------------------------------------------------

class TestSemivariogram:
    def test_angular_distance(self):
        assert abs(angular_distance([1, 0], [0, 1]) - math.pi / 2) < 1e-9
        assert angular_distance([0, 0], [1, 1]) is None

    def test_smooth_vs_needle_relative_nugget(self):
        n = 80
        angles = np.linspace(0, 1.0, n)
        dirs = [[math.cos(a), math.sin(a)] for a in angles]
        smooth = list(angles)                               # δ smooth in angle
        needle = list(np.random.default_rng(0).normal(size=n))  # δ pure noise

        vs = empirical_semivariogram(dirs, smooth, rng=np.random.default_rng(1))
        vn = empirical_semivariogram(dirs, needle, rng=np.random.default_rng(1))

        # Smooth field: little variance at short lag → small relative nugget.
        assert vs["relative_nugget"] < 0.2
        # Needle field: short-lag variance ≈ total variance → large nugget.
        assert vn["relative_nugget"] > 0.6
        assert vs["relative_nugget"] < vn["relative_nugget"]

    def test_degenerate_inputs(self):
        v = empirical_semivariogram([], [])
        assert v["n_points"] == 0 and math.isnan(v["relative_nugget"])
        v1 = empirical_semivariogram([[1, 0]], [0.5])
        assert v1["n_points"] == 1 and math.isnan(v1["nugget"])

    def test_subsampling_is_reproducible(self):
        n = 60
        dirs = [[math.cos(i), math.sin(i)] for i in np.linspace(0, 2, n)]
        deltas = list(np.random.default_rng(2).normal(size=n))
        a = empirical_semivariogram(dirs, deltas, max_pairs=50, rng=np.random.default_rng(7))
        b = empirical_semivariogram(dirs, deltas, max_pairs=50, rng=np.random.default_rng(7))
        assert a["nugget"] == b["nugget"] and a["n_pairs"] == b["n_pairs"]


# ---------------------------------------------------------------------------
# Grapher I/O (histograms + bumpiness table; no CMF object required)
# ---------------------------------------------------------------------------

def _shard(cmf_name, encoding):
    return SimpleNamespace(cmf_name=cmf_name, encoding=encoding, cmf=None)


class _Const:
    """Hashable stand-in for a Constant (SimpleNamespace defines __eq__ → unhashable)."""

    def __init__(self, name):
        self.name = name
        self.value_sympy = None


def _write_shard(dir_path, shard_id, records):
    with open(os.path.join(dir_path, shard_id + ".jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(tid, cmf, sid, delta, direction, seq=None):
    # One flat per-(trajectory, constant) row; metrics are top-level columns.
    rec = {
        "trajectory_id": tid, "cmf_id": cmf, "shard_id": sid,
        "constant": "pi",
        "start_point": [0, 0], "direction": list(direction),
        "delta": delta, "identified": True,
    }
    if seq is not None:
        rec["delta_sequence"] = seq
    return rec


class TestGrapherIO:
    def _setup(self, tmp_path, monkeypatch, **flags):
        monkeypatch.setattr(config.system, "EXPORT_SEARCH_RESULTS", str(tmp_path / "search"))
        monkeypatch.setattr(config.system, "EXPORT_GRAPHS", str(tmp_path / "out"))
        os.makedirs(str(tmp_path / "search"), exist_ok=True)
        for k, v in flags.items():
            monkeypatch.setattr(config.graph, k, v)

    def test_histograms_and_bumpiness_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch,
                    PLOT_DELTA_HISTOGRAMS=True, WRITE_BUMPINESS_TABLE=True,
                    PLOT_BEST_DELTA_SEQUENCE=False)
        search = str(tmp_path / "search")

        cmf = "G"
        shard = _shard(cmf, (1, -1))
        _, sid, enc = derive_cmf_and_shard_ids(shard)
        recs = [
            _rec("a", cmf, sid, 0.2, (1, 0), seq=[0.1, 0.2, 0.15]),
            _rec("b", cmf, sid, 0.6, (0, 1), seq=[0.0, 0.5, 0.5]),
            _rec("c", cmf, sid, 0.4, (1, 1), seq=[0.2, 0.2, 0.2]),
        ]
        _write_shard(search, sid, recs)

        const = _Const("pi")
        Grapher({const: [shard]}).generate()

        out = str(tmp_path / "out")
        files = os.listdir(out)
        # one per-shard hist, one whole-CMF hist, and the bumpiness table.
        assert any(f.startswith("hist_delta_") and "shard_" in f for f in files)
        assert any(f.endswith("__CMF.png") for f in files)
        assert "bumpiness.csv" in files and "bumpiness.md" in files

        with open(os.path.join(out, "bumpiness.csv")) as f:
            content = f.read()
        assert "relative_nugget" in content and "median_delta_seq_TV" in content
        assert cmf in content

    def test_disabled_is_noop(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch,
                    PLOT_DELTA_HISTOGRAMS=False, WRITE_BUMPINESS_TABLE=False,
                    PLOT_BEST_DELTA_SEQUENCE=False)
        const = _Const("pi")
        Grapher({const: [_shard("G", (1,))]}).generate()
        # No output directory is created when nothing is enabled.
        assert not os.path.isdir(str(tmp_path / "out"))

    def test_post_process_execute_runs_graphing_with_empty_tier3(self, tmp_path, monkeypatch):
        """Tier3PostProcessModV1.execute() runs graphing even when no Tier-3
        attributes are configured (graphing is an independent sub-stage)."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1

        self._setup(tmp_path, monkeypatch,
                    PLOT_DELTA_HISTOGRAMS=True, WRITE_BUMPINESS_TABLE=True,
                    PLOT_BEST_DELTA_SEQUENCE=False)
        monkeypatch.setattr(config.post_process, "TIER3_ATTRIBUTES", ())  # nothing to compute
        search = str(tmp_path / "search")

        cmf = "H"
        shard = _shard(cmf, (1, -1))
        _, sid, _ = derive_cmf_and_shard_ids(shard)
        _write_shard(search, sid, [
            _rec("a", cmf, sid, 0.2, (1, 0)), _rec("b", cmf, sid, 0.6, (0, 1)),
        ])

        Tier3PostProcessModV1({_Const("pi"): [shard]}).execute()

        files = os.listdir(str(tmp_path / "out"))
        assert "bumpiness.csv" in files
        assert any(f.startswith("hist_delta_") for f in files)
