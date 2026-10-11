"""Tests for the Tier-3 post-process two-phase top-N ranking pipeline."""
import json
import os
from types import SimpleNamespace

import pytest

from dreamer.configs import config
from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
from dreamer.utils.storage.predicate_specs import parse_predicate_spec
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


def _shard(cmf_name, encoding):
    return SimpleNamespace(cmf_name=cmf_name, encoding=encoding, cmf=None)


class _Const:
    """Hashable stand-in for a Constant (SimpleNamespace defines __eq__ → unhashable)."""

    def __init__(self, name):
        self.name = name
        self.value_sympy = None


def _const(name):
    return _Const(name)


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(tid, cmf_id, shard_id, delta, direction=(1, 0, 0), const="pi", **extended):
    # One flat per-(trajectory, constant) row; metrics are top-level columns.
    return {
        "trajectory_id": tid,
        "cmf_id": cmf_id,
        "shard_id": shard_id,
        "constant": const,
        "start_point": [0, 0, 0],
        "direction": list(direction),
        "delta": delta,
        **extended,
    }


def _nest(*recs):
    """Nest flat records into ``{trajectory_id: {constant: record}}``."""
    out = {}
    for r in recs:
        out.setdefault(r["trajectory_id"], {})[r.get("constant")] = r
    return out


# ---------------------------------------------------------------------------
# Pure ranking helper
# ---------------------------------------------------------------------------

class TestRankInto:
    def test_top_n_highest(self):
        target = set()
        records = _nest(
            _rec("a", "C", "S", 0.1), _rec("b", "C", "S", 0.9),
            _rec("c", "C", "S", 0.5), _rec("d", "C", "S", 0.3),
        )
        sel = parse_predicate_spec("top 2 highest delta in shard")
        Tier3PostProcessModV1._rank_into(target, records, sel, "pi")
        assert target == {"b", "c"}

    def test_top_n_lowest(self):
        target = set()
        records = _nest(*[_rec(k, "C", "S", d) for k, d in [("a", 0.1), ("b", 0.9), ("c", 0.5)]])
        sel = parse_predicate_spec("top 1 lowest delta in shard")
        Tier3PostProcessModV1._rank_into(target, records, sel, "pi")
        assert target == {"a"}

    def test_skips_missing_metric(self):
        target = set()
        records = _nest(
            _rec("a", "C", "S", 0.5),
            {"trajectory_id": "b", "constant": "pi"},  # no delta → excluded
        )
        sel = parse_predicate_spec("top 2 highest delta in shard")
        Tier3PostProcessModV1._rank_into(target, records, sel, "pi")
        assert target == {"a"}


# ---------------------------------------------------------------------------
# Pre-skip + context construction
# ---------------------------------------------------------------------------

class TestSurvivesAndContext:
    def test_survives_only_when_in_top_n(self):
        resolved = [("asymptotics", parse_predicate_spec("top 1 highest delta in shard"))]
        key = "top_1_highest_delta_in_shard"
        sets = {key: {"t1"}}
        assert Tier3PostProcessModV1._survives_top_n({"asymptotics"}, "t1", sets, resolved) is True
        assert Tier3PostProcessModV1._survives_top_n({"asymptotics"}, "t9", sets, resolved) is False

    def test_handler_only_predicate_always_survives(self):
        resolved = [("relation", parse_predicate_spec("max_degree below 5"))]
        # Can't decide without the handler → must survive (worker decides).
        assert Tier3PostProcessModV1._survives_top_n({"relation"}, "t9", {}, resolved) is True

    def test_build_context_singleton_membership(self):
        resolved = [("asymptotics", parse_predicate_spec("top 1 highest delta in shard"))]
        key = "top_1_highest_delta_in_shard"
        sets = {key: {"t1", "t2"}}
        ctx_in = Tier3PostProcessModV1._build_context("t1", sets, resolved)
        ctx_out = Tier3PostProcessModV1._build_context("t9", sets, resolved)
        assert ctx_in["top_n_sets"][key] == {"t1"}   # singleton, not the full set
        assert ctx_out["top_n_sets"][key] == set()


# ---------------------------------------------------------------------------
# End-to-end Phase-1 ranking over real JSONL files
# ---------------------------------------------------------------------------

class TestComputeTopNSets:
    def test_shard_and_cmf_scope(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.system, "EXPORT_SEARCH_RESULTS", str(tmp_path))

        cmf = "myCMF"
        shard1 = _shard(cmf, (1, -1, 1))
        shard2 = _shard(cmf, (-1, 1, 1))
        _, sid1, _ = derive_cmf_and_shard_ids(shard1)
        _, sid2, _ = derive_cmf_and_shard_ids(shard2)

        # shard1: deltas 0.1, 0.9 ; shard2: deltas 0.5, 0.95
        _write_jsonl(os.path.join(str(tmp_path), sid1 + ".jsonl"), [
            _rec("a", cmf, sid1, 0.1), _rec("b", cmf, sid1, 0.9),
        ])
        _write_jsonl(os.path.join(str(tmp_path), sid2 + ".jsonl"), [
            _rec("c", cmf, sid2, 0.5), _rec("d", cmf, sid2, 0.95),
        ])

        const = _const("pi")
        mod = Tier3PostProcessModV1({const: [shard1, shard2]})

        shard_sel = parse_predicate_spec("top 1 highest delta in shard")
        cmf_sel = parse_predicate_spec("top 2 highest delta in cmf")
        sets = mod._compute_top_n_sets([shard_sel, cmf_sel])

        # Per shard: best of each shard → {b, d}.
        assert sets[shard_sel.key] == {"b", "d"}
        # Whole CMF top 2 → 0.95 (d), 0.9 (b).
        assert sets[cmf_sel.key] == {"b", "d"}

    def test_cmf_scope_pools_across_shards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.system, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        cmf = "C2"
        shard1 = _shard(cmf, (1, 1))
        shard2 = _shard(cmf, (-1, -1))
        _, sid1, _ = derive_cmf_and_shard_ids(shard1)
        _, sid2, _ = derive_cmf_and_shard_ids(shard2)
        _write_jsonl(os.path.join(str(tmp_path), sid1 + ".jsonl"),
                     [_rec("a", cmf, sid1, 0.2), _rec("b", cmf, sid1, 0.3)])
        _write_jsonl(os.path.join(str(tmp_path), sid2 + ".jsonl"),
                     [_rec("c", cmf, sid2, 0.8), _rec("d", cmf, sid2, 0.1)])

        mod = Tier3PostProcessModV1({_const("pi"): [shard1, shard2]})
        cmf_sel = parse_predicate_spec("top 1 highest delta in cmf")
        sets = mod._compute_top_n_sets([cmf_sel])
        assert sets[cmf_sel.key] == {"c"}  # global max 0.8 lives in shard2
