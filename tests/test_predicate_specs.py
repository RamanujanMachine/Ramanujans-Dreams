"""Tests for the Tier-3 predicate grammar (predicate_specs) and ranking metrics."""
import pytest

from dreamer.utils.storage.predicate_specs import (
    TopNSelector,
    iter_top_n_selectors,
    parse_predicate_spec,
)
from dreamer.utils.storage.record_metrics import (
    METRIC_EXTRACTORS,
    delta_metric,
)
from dreamer.utils.storage.attribute_registry import (
    compute_attributes,
    register_attribute,
)


class _FakeHandler:
    """Minimal handler exposing just what the grammar predicates touch."""

    def __init__(self, degrees):
        self._degrees = degrees

    def coeff_degrees(self):
        return self._degrees


# ---------------------------------------------------------------------------
# Grammar parsing
# ---------------------------------------------------------------------------

class TestTopNGrammar:
    def test_parses_all_template_forms(self):
        p = parse_predicate_spec("top 3 highest convergence_rate in shard")
        assert isinstance(p, TopNSelector)
        assert (p.metric, p.n, p.highest, p.scope) == ("convergence_rate", 3, True, "shard")

        q = parse_predicate_spec("top 5 lowest delta in cmf")
        assert (q.metric, q.n, q.highest, q.scope) == ("delta", 5, False, "cmf")

    def test_case_insensitive(self):
        p = parse_predicate_spec("TOP 2 HIGHEST DELTA IN CMF")
        assert isinstance(p, TopNSelector)
        assert p.metric == "delta" and p.scope == "cmf"

    def test_key_is_stable_and_descriptive(self):
        p = parse_predicate_spec("top 3 highest convergence_rate in shard")
        assert p.key == "top_3_highest_convergence_rate_in_shard"
        # Equal selectors → equal keys (used to look up the same id-set).
        assert parse_predicate_spec("top 3 highest convergence_rate in shard").key == p.key

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            parse_predicate_spec("top 3 highest nonsense in shard")

    def test_zero_count_raises(self):
        with pytest.raises(ValueError):
            parse_predicate_spec("top 0 highest delta in shard")

    def test_membership_via_context(self):
        sel = parse_predicate_spec("top 2 highest delta in shard")
        ctx_in = {"trajectory_id": "t1", "top_n_sets": {sel.key: {"t1", "t2"}}}
        ctx_out = {"trajectory_id": "t9", "top_n_sets": {sel.key: {"t1", "t2"}}}
        assert sel(None, ctx_in) is True
        assert sel(None, ctx_out) is False
        # No context / missing key → never qualifies (fail closed).
        assert sel(None, None) is False
        assert sel(None, {"trajectory_id": "t1", "top_n_sets": {}}) is False


class TestMaxDegreeGrammar:
    def test_below(self):
        p = parse_predicate_spec("max_degree below 5")
        assert p(_FakeHandler([1, 2, 3])) is True
        assert p(_FakeHandler([1, 5, 2])) is False  # max 5 not < 5

    def test_above(self):
        p = parse_predicate_spec("max_degree above 2")
        assert p(_FakeHandler([1, 2, 3])) is True
        assert p(_FakeHandler([0, 1, 2])) is False

    def test_empty_degrees_is_false(self):
        assert parse_predicate_spec("max_degree below 5")(_FakeHandler([])) is False


class TestBackwardCompatibility:
    def test_named_predicate_fallthrough(self):
        # A registered name still resolves (to the registry callable).
        pred = parse_predicate_spec("if_identified")
        assert callable(pred)

    def test_callable_passthrough(self):
        f = lambda h: True
        assert parse_predicate_spec(f) is f

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            parse_predicate_spec("totally_unknown_predicate")


class TestIterTopNSelectors:
    def test_collects_only_top_n(self):
        specs = [
            ("asymptotics", "top 3 highest delta in shard"),
            ("relation", "if_identified"),
            ("order", "max_degree below 4"),
            "bare_attribute",
        ]
        keys = [s.key for s in iter_top_n_selectors(specs)]
        assert keys == ["top_3_highest_delta_in_shard"]


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------

class TestMetricExtractors:
    def test_delta_metric_reads_per_constant(self):
        # Flat per-constant row: δ is the core ``delta`` column.
        rec = {"constant": "pi", "delta": 0.5}
        assert delta_metric(rec, "pi") == 0.5
        assert delta_metric({}, "pi") is None

    def test_delta_metric_drops_non_finite(self):
        rec = {"constant": "pi", "delta": float("-inf")}
        assert delta_metric(rec, "pi") is None

    def test_convergence_rate_reads_stored_value(self):
        # Single definition: read the handler-computed convergence_rate straight
        # out of its flat top-level column (no recomputation).
        extractor = METRIC_EXTRACTORS["convergence_rate"]
        rec = {"convergence_rate": 1.25}
        assert extractor(rec, None) == 1.25

    def test_convergence_rate_missing_returns_none(self):
        extractor = METRIC_EXTRACTORS["convergence_rate"]
        assert extractor({"direction": [1, 0]}, None) is None
        # Non-finite stored value is dropped like every other numeric column.
        assert extractor({"convergence_rate": float("inf")}, None) is None

    def test_registry_complete(self):
        for name in ("delta", "convergence_rate", "approximated_digits_per_step",
                     "spectral_gap", "gcd_slope", "precision_at"):
            assert name in METRIC_EXTRACTORS


# ---------------------------------------------------------------------------
# compute_attributes honours the top-N context gate
# ---------------------------------------------------------------------------

class TestComputeAttributesTopNGate:
    def test_gate_passes_and_blocks(self):
        register_attribute("_dummy_one", lambda h: 1)
        spec = ("_dummy_one", "top 1 highest delta in shard")
        key = "top_1_highest_delta_in_shard"

        passing = compute_attributes(
            _FakeHandler([1]), [spec],
            context={"trajectory_id": "t1", "top_n_sets": {key: {"t1"}}},
        )
        assert passing.get("_dummy_one") == 1

        blocked = compute_attributes(
            _FakeHandler([1]), [spec],
            context={"trajectory_id": "t9", "top_n_sets": {key: {"t1"}}},
        )
        assert "_dummy_one" not in blocked
