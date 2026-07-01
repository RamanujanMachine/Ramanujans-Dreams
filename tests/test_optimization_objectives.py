"""Tests for the optimization-objective registry (optimization_objectives)."""
import pytest

from dreamer.utils.storage.optimization_objectives import (
    OBJECTIVES,
    Objective,
    get_objective,
    is_valid_objective,
    objective_raw_value,
    objective_score,
    score_record,
    signed_score,
)


class _FakeHandler:
    """Minimal stand-in exposing the handler methods objectives read."""

    def __init__(self, delta=1.5, convergence_rate=0.25):
        self._delta = delta
        self._cr = convergence_rate

    def delta(self):
        return self._delta

    def convergence_rate(self):
        return self._cr


class TestRegistry:
    def test_builtin_objectives_present_and_maximise(self):
        # Both shipped objectives are "larger is better".
        for name in ("delta", "convergence_rate"):
            assert is_valid_objective(name)
            assert get_objective(name).direction == "max"

    def test_unknown_objective_raises(self):
        assert not is_valid_objective("not_an_objective")
        with pytest.raises(KeyError, match="Unknown optimization objective"):
            get_objective("not_an_objective")

    def test_default_system_objective_is_valid(self):
        from dreamer.configs.system import sys_config
        assert is_valid_objective(sys_config.OPTIMIZATION_OBJECTIVE)


class TestValueExtraction:
    def test_delta_raw_value(self):
        h = _FakeHandler(delta=2.0)
        assert objective_raw_value("delta", h) == 2.0

    def test_convergence_rate_raw_value(self):
        h = _FakeHandler(convergence_rate=0.4)
        assert objective_raw_value("convergence_rate", h) == 0.4

    def test_delta_neg_inf_sentinel_preserved(self):
        # δ's non-convergence sentinel is a valid worst-value for a max objective.
        h = _FakeHandler(delta=float("-inf"))
        assert objective_raw_value("delta", h) == float("-inf")
        assert signed_score("delta", float("-inf")) == float("-inf")


class TestSignedScore:
    def test_max_passes_through(self):
        assert signed_score("delta", 1.5) == 1.5

    def test_none_propagates(self):
        assert signed_score("delta", None) is None
        assert objective_score("convergence_rate", _FakeHandler(convergence_rate=None)) is None

    def test_min_direction_is_negated(self):
        # Register a temporary "min" objective to prove the sign flip; the search
        # loop still maximises, so a min-objective's raw value is driven down.
        OBJECTIVES["_tmp_min"] = Objective(
            "_tmp_min", "min", lambda h: 3.0
        )
        try:
            assert signed_score("_tmp_min", 3.0) == -3.0
            assert objective_score("_tmp_min", _FakeHandler()) == -3.0
        finally:
            del OBJECTIVES["_tmp_min"]

    def test_objective_score_matches_manual_composition(self):
        h = _FakeHandler(convergence_rate=0.7)
        assert objective_score("convergence_rate", h) == signed_score(
            "convergence_rate", objective_raw_value("convergence_rate", h)
        )


class TestScoreRecord:
    """The shared stored-record scorer used by every stage."""

    def test_reads_objective_value_under_matching_objective(self):
        rec = {"objective_name": "convergence_rate",
               "objective_value": {"e": 0.4}, "identified": {"e": True}}
        assert score_record(rec, "e", "convergence_rate") == (0.4, True)

    def test_delta_fallback_for_legacy_record(self):
        rec = {"delta_estimate": {"e": 1.7}, "identified": {"e": True}}
        assert score_record(rec, "e", "delta") == (1.7, True)

    def test_no_fallback_for_non_delta_objective(self):
        rec = {"delta_estimate": {"e": 1.7}, "identified": {"e": True}}
        assert score_record(rec, "e", "convergence_rate") is None

    def test_none_raw_maps_to_worst_score(self):
        rec = {"objective_name": "convergence_rate",
               "objective_value": {"e": None}, "identified": {"e": False}}
        assert score_record(rec, "e", "convergence_rate") == (float("-inf"), False)

    def test_min_objective_record_is_negated(self):
        OBJECTIVES["_tmp_min"] = Objective("_tmp_min", "min", lambda h: 0.0)
        try:
            rec = {"objective_name": "_tmp_min",
                   "objective_value": {"e": 2.0}, "identified": {"e": True}}
            assert score_record(rec, "e", "_tmp_min") == (-2.0, True)
        finally:
            del OBJECTIVES["_tmp_min"]
