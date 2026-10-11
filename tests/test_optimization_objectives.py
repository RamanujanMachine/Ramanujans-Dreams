"""Tests for the optimization-objective registry (optimization_objectives)."""
import pytest

from dreamer.utils.storage.optimization_objectives import (
    OBJECTIVES,
    Objective,
    get_objective,
    is_valid_objective,
    objective_display_label,
    objective_metric_attribute,
    record_raw_value,
    score_record,
    signed_score,
)


class TestRegistry:
    def test_builtin_objectives_present_and_maximise(self):
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

    def test_objective_metric_attribute(self):
        # δ is a core column → no extra metric to compute; others are their own key.
        assert objective_metric_attribute("delta") is None
        assert objective_metric_attribute("convergence_rate") == "convergence_rate"

    def test_display_label(self):
        assert objective_display_label("delta") == "δ"
        assert objective_display_label("convergence_rate") == "convergence_rate"


class TestSignedScore:
    def test_max_passes_through(self):
        assert signed_score("delta", 1.5) == 1.5

    def test_none_propagates(self):
        assert signed_score("delta", None) is None

    def test_min_direction_is_negated(self):
        OBJECTIVES["_tmp_min"] = Objective("_tmp_min", "min")
        try:
            assert signed_score("_tmp_min", 3.0) == -3.0
        finally:
            del OBJECTIVES["_tmp_min"]


class TestScoreRecord:
    """The shared per-(trajectory, constant) flat-record scorer."""

    def test_reads_delta_core_column(self):
        rec = {"delta": 0.5, "identified": True}
        assert score_record(rec, "delta") == (0.5, True)

    def test_reads_flat_metric_column(self):
        rec = {"convergence_rate": 0.4, "identified": True}
        assert score_record(rec, "convergence_rate") == (0.4, True)

    def test_missing_column_returns_none(self):
        # No convergence_rate column at all → recompute signal.
        assert score_record({"delta": 0.1, "identified": True}, "convergence_rate") is None

    def test_present_but_nonfinite_maps_to_worst(self):
        rec = {"convergence_rate": None, "identified": False}
        assert score_record(rec, "convergence_rate") == (float("-inf"), False)
        rec2 = {"delta": float("-inf"), "identified": False}
        assert score_record(rec2, "delta") == (float("-inf"), False)

    def test_min_objective_record_is_negated(self):
        OBJECTIVES["_tmp_min"] = Objective("_tmp_min", "min")
        try:
            rec = {"_tmp_min": 2.0, "identified": True}
            assert score_record(rec, "_tmp_min") == (-2.0, True)
        finally:
            del OBJECTIVES["_tmp_min"]

    def test_record_raw_value(self):
        assert record_raw_value({"convergence_rate": 0.4}, "convergence_rate") == 0.4
        assert record_raw_value({}, "convergence_rate") is None
        assert record_raw_value({"convergence_rate": None}, "convergence_rate") is None
