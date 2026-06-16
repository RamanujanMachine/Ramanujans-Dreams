"""
Tests for the attributes-management pipeline (Task 3 and follow-ups).

Coverage:
  - ShardDTO field-order fix and all-DTO serialization round-trips
  - Stable id helpers (_stable_id, _position_to_tuple, _serialize_encoding,
    derive_cmf_and_shard_ids)
  - TrajectoryAttributesHandler stub methods (p_vector, q_vector, identified)
  - build_trajectory_dto factory
  - SerialSearcher.sample_pairs()
  - load_seen_trajectory_ids / cross-run deduplication
  - JSONL Exporter / Importer round-trip
  - Central attribute registry: known names, errors, custom registration
  - System.__best_trajectory_record scanning JSONL outputs
"""

import json
import multiprocessing as mp
import os

import numpy as np
import pytest
import sympy as sp

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.storage.dtos import CmfDTO, CmfFamilyDTO, ShardDTO, TrajectoryDTO
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    _serialize_encoding,
    _stable_id,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.utils.storage.attribute_registry import (
    ATTRIBUTE_REGISTRY,
    PREDICATES,
    attribute_name,
    compute_attribute,
    compute_attributes,
    register_attribute,
    register_predicate,
)
from dreamer.utils.storage import Exporter, Importer, Formats
from dreamer.utils.multi_processing import (
    load_seen_shards,
    load_seen_trajectories,
    load_seen_trajectory_ids,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cmf():
    """1F1(z=1) — a minimal 2-symbol CMF."""
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def zero_shift(symbols):
    return Position({s: sp.Integer(0) for s in symbols})


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    """A bounded shard with two hyperplanes and an interior point at (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    """An unconstrained (whole-space) shard."""
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def minimal_handler(simple_cmf, simple_shard, symbols):
    """TrajectoryAttributesHandler for a concrete (traj, start) pair.

    Built with ``constant=e.value_sympy`` and ``searchable=simple_shard`` so
    Tier-1 attributes (delta, limit, p/q vectors, identified) are computable
    and the shard's p/q cache is exercised — matching real producer usage.
    """
    traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return TrajectoryAttributesHandler.from_cmf(
        simple_cmf, traj, start,
        constant=e.value_sympy,
        searchable=simple_shard,
    )


# ---------------------------------------------------------------------------
# 1. DTO field-order fix and serialization round-trips
# ---------------------------------------------------------------------------

class TestDTOFieldOrdering:

    def test_shard_dto_required_fields_before_optional(self):
        """ShardDTO can be instantiated with only required fields — no TypeError."""
        dto = ShardDTO(
            shard_id="s1",
            cmf_id="c1",
            shard_encoding=(1, -1),
            dimensionality=2,
            dimension=2,
            found_constants=["pi"],
        )
        assert dto.shard_id == "s1"
        assert dto.orthogonality_defect is None
        assert dto.interior_point is None

    def test_shard_dto_optional_fields_can_be_set(self):
        dto = ShardDTO(
            shard_id="s2",
            cmf_id="c2",
            shard_encoding=(1,),
            dimensionality=1,
            dimension=1,
            found_constants=[],
            interior_point=(3, 4),
            orthogonality_defect=0.1,
        )
        assert dto.interior_point == (3, 4)
        assert dto.orthogonality_defect == 0.1


class TestDTOSerializationRoundTrips:

    def test_trajectory_dto_round_trip(self):
        """JSON serialise → deserialise → fields are equal."""
        dto = TrajectoryDTO(
            trajectory_id="abc123",
            cmf_id="4F3",
            shard_id="sh1",
            start_point=(1, 2),
            direction=(0, 1),
            recurrence_relation="a(n)*f(n) + b(n)*f(n-1) = 0",
            recurrence_order=1,
            limit_value=2.718,
            delta_estimate={"e": 1.5, "log2": 0.7},
            p_vector={"e": (1, 0), "log2": (2, 1)},
            q_vector={"e": (0, 1), "log2": (1, 0)},
            identified={"e": True, "log2": False},
            walk_type=2,
        )
        restored = TrajectoryDTO.from_dict(json.loads(dto.to_json_line()))

        assert restored == dto
        assert isinstance(restored.start_point, tuple)
        assert isinstance(restored.direction, tuple)
        assert isinstance(restored.delta_estimate, dict)
        assert isinstance(restored.p_vector, dict)
        # New reconstruction-critical field survives the round trip:
        assert restored.walk_type == 2

    def test_trajectory_dto_walk_type_default(self):
        """Older JSONL records missing ``walk_type`` deserialise with default 1."""
        d = {
            "trajectory_id": "old",
            "cmf_id": "c",
            "shard_id": "s",
            "start_point": [1],
            "direction": [0],
            "recurrence_relation": "",
            "recurrence_order": 1,
            "limit_value": 1.0,
            "delta_estimate": {"e": 1.0},
            "identified": {"e": True},
        }
        restored = TrajectoryDTO.from_dict(d)
        assert restored.walk_type == 1

    def test_trajectory_dto_round_trip_with_extended_metrics(self):
        """extended_metrics survive the round-trip intact."""
        dto = TrajectoryDTO(
            trajectory_id="xyz",
            cmf_id="c",
            shard_id="s",
            start_point=(0,),
            direction=(1,),
            recurrence_relation="",
            recurrence_order=2,
            limit_value=3.14,
            delta_estimate={"e": 1.1},
            p_vector={"e": ()},
            q_vector={"e": ()},
            extended_metrics={"eigenvalues": ["1+0j", "0.5"], "spectral_gap": 0.5},
        )
        restored = TrajectoryDTO.from_dict(json.loads(dto.to_json_line()))
        assert restored.extended_metrics["spectral_gap"] == 0.5
        assert restored.extended_metrics["eigenvalues"] == ["1+0j", "0.5"]

    def test_shard_dto_round_trip(self):
        dto = ShardDTO(
            shard_id="sh",
            cmf_id="cf",
            shard_encoding=(1, -1, 1),
            dimensionality=3,
            dimension=2,
            found_constants=["e"],
            interior_point=(2, 3, 4),
        )
        restored = ShardDTO.from_dict(json.loads(dto.to_json_line()))
        assert restored == dto
        assert isinstance(restored.shard_encoding, tuple)
        assert isinstance(restored.interior_point, tuple)

    def test_cmf_dto_round_trip(self):
        dto = CmfDTO(
            cmf_id="4F3_shift1",
            family_id="4F3",
            cmf_hyperplanes=["x=0", "y+z=1"],
            coordinate_shift=(1, 0, -1),
            found_constants=["zeta3"],
        )
        restored = CmfDTO.from_dict(json.loads(dto.to_json_line()))
        assert restored == dto
        assert isinstance(restored.coordinate_shift, tuple)

    def test_cmf_family_dto_round_trip(self):
        dto = CmfFamilyDTO(
            family_id="4F3",
            global_family_id="pFq",
            matrix_definitions={"x": "[[1, n], [0, 1]]", "y": "[[1, 0], [n, 1]]"},
            dimensions=2,
        )
        restored = CmfFamilyDTO.from_dict(json.loads(dto.to_json_line()))
        assert restored == dto

    def test_trajectory_dto_missing_p_q_vectors(self):
        """Omitting p_vector / q_vector in the dict produces None."""
        d = {
            "trajectory_id": "t",
            "cmf_id": "c",
            "shard_id": "s",
            "start_point": [1],
            "direction": [0],
            "recurrence_relation": "",
            "recurrence_order": 1,
            "limit_value": 1.0,
            "delta_estimate": {"e": 1.0},
            "identified": {"e": True},
        }
        dto = TrajectoryDTO.from_dict(d)
        assert dto.p_vector is None
        assert dto.q_vector is None
        assert dto.extended_metrics == {}


# ---------------------------------------------------------------------------
# 2. Stable-id helpers
# ---------------------------------------------------------------------------

class TestStableId:

    def test_same_inputs_same_output(self):
        assert _stable_id("a", "b", "c") == _stable_id("a", "b", "c")

    def test_different_inputs_different_output(self):
        assert _stable_id("a", "b") != _stable_id("b", "a")

    def test_length_is_respected(self):
        assert len(_stable_id("x", length=8)) == 8
        assert len(_stable_id("x", length=32)) == 32

    def test_hex_characters_only(self):
        result = _stable_id("hello", "world")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_parts(self):
        # Empty string is a valid input; result should still be hex
        assert len(_stable_id("")) == 16


class TestPositionToTuple:

    def test_sympy_integers_become_python_ints(self, symbols):
        pos = Position({symbols[0]: sp.Integer(3), symbols[1]: sp.Integer(-1)})
        result = _position_to_tuple(pos)
        assert result == (3, -1)
        assert all(isinstance(v, int) for v in result)

    def test_python_ints_pass_through(self, symbols):
        pos = Position({symbols[0]: 5, symbols[1]: 7})
        assert _position_to_tuple(pos) == (5, 7)

    def test_symbolic_value_falls_back_to_str(self, symbols):
        n = sp.Symbol("n")
        pos = Position({symbols[0]: n, symbols[1]: sp.Integer(2)})
        result = _position_to_tuple(pos)
        assert isinstance(result[0], str)   # symbolic → str
        assert result[1] == 2


class TestSerializeEncoding:
    """Encoding-based serialisation (replaces the old inequality-blob path)."""

    def test_bounded_shard_produces_stable_string(self, simple_shard):
        s1 = _serialize_encoding(simple_shard)
        s2 = _serialize_encoding(simple_shard)
        assert s1 == s2
        assert s1 != "whole_space"

    def test_whole_space_shard_produces_placeholder(self, whole_space_shard):
        assert _serialize_encoding(whole_space_shard) == "whole_space"

    def test_encoding_matches_sign_vector(self, simple_shard):
        """The serialised string is the comma-joined ±1 encoding."""
        s = _serialize_encoding(simple_shard)
        assert s == ",".join(str(int(v)) for v in simple_shard.encoding)

    def test_different_shards_produce_different_strings(self, simple_cmf, symbols, zero_shift):
        hps_a = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
        hps_b = [
            Hyperplane(symbols[0], symbols),
            Hyperplane(symbols[1], symbols),
            Hyperplane(symbols[0] + symbols[1] - 10, symbols),
        ]
        interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        shard_a = Shard(simple_cmf, e, hps_a, [1, 1], zero_shift, interior)

        interior_b = Position({symbols[0]: sp.Integer(3), symbols[1]: sp.Integer(3)})
        shard_b = Shard(simple_cmf, e, hps_b, [1, 1, -1], zero_shift, interior_b)

        assert _serialize_encoding(shard_a) != _serialize_encoding(shard_b)


class TestDeriveCmfAndShardIds:

    def test_returns_three_strings(self, simple_shard):
        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)
        assert isinstance(cmf_id, str)
        assert isinstance(shard_id, str)
        assert isinstance(enc_str, str)

    def test_cmf_id_equals_cmf_name(self, simple_shard):
        cmf_id, _, _ = derive_cmf_and_shard_ids(simple_shard)
        assert cmf_id == simple_shard.cmf_name

    def test_shard_id_is_deterministic(self, simple_shard):
        _, id1, _ = derive_cmf_and_shard_ids(simple_shard)
        _, id2, _ = derive_cmf_and_shard_ids(simple_shard)
        assert id1 == id2

    def test_whole_space_shard_works(self, whole_space_shard):
        cmf_id, shard_id, enc = derive_cmf_and_shard_ids(whole_space_shard)
        assert enc == "whole_space"
        # Structural format: ``"{cmf_id}__{16-char hash}"``.
        assert shard_id.startswith(f"{cmf_id}__")
        assert len(shard_id.rsplit("__", 1)[1]) == 16

    def test_shard_id_embeds_cmf_id(self, simple_shard):
        """shard_id must literally start with cmf_id so any record's shard id
        discloses its parent CMF without a separate lookup."""
        cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        assert shard_id.startswith(f"{cmf_id}__"), (
            f"shard_id={shard_id!r} should start with '{cmf_id}__'"
        )


# ---------------------------------------------------------------------------
# 3. Handler stub methods
# ---------------------------------------------------------------------------

class TestHandlerStubs:

    def test_p_vector_shape_or_none(self, minimal_handler):
        """``p_vector()`` is either ``None`` (LIReC couldn't identify) or a
        list of length ``traj_size`` — the handler no longer fabricates a
        canonical-axis fallback."""
        result = minimal_handler.p_vector()
        if result is None:
            assert minimal_handler.identified() is False
        else:
            assert isinstance(result, list)
            assert len(result) == minimal_handler.traj_size()

    def test_q_vector_shape_or_none(self, minimal_handler):
        """``q_vector()`` mirrors ``p_vector()``."""
        result = minimal_handler.q_vector()
        if result is None:
            assert minimal_handler.identified() is False
        else:
            assert isinstance(result, list)
            assert len(result) == minimal_handler.traj_size()

    def test_identified_returns_bool(self, minimal_handler):
        """``identified()`` is a Python bool.

        The exact value depends on whether LIReC can identify ``e`` from
        the small ``1F1`` trajectory and whether the path converges, which
        is environment-dependent — so we only assert the type.  Calling
        ``identified()`` before any other attribute must work because it
        drives ``delta()`` (and hence ``_pq_vector`` and the convergence
        check) itself.
        """
        assert isinstance(minimal_handler.identified(), bool)

    def test_identified_iff_finite_delta(self, minimal_handler):
        """``identified()`` ↔ ``math.isfinite(delta())`` — the design invariant.

        A trajectory is identified iff it (a) found p/q, (b) converges to
        the constant, and (c) yields a well-defined delta.  All three
        collapse to ``delta != -inf``.
        """
        import math
        assert minimal_handler.identified() == math.isfinite(minimal_handler.delta())

    def test_identified_implies_pq_exists(self, minimal_handler):
        """When ``identified`` is True, ``p_vector`` and ``q_vector`` are populated."""
        if minimal_handler.identified():
            assert minimal_handler.p_vector() is not None
            assert minimal_handler.q_vector() is not None

    def test_identified_false_without_constant(self, simple_cmf, symbols):
        """A handler built without a constant cannot identify — returns False."""
        traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        handler = TrajectoryAttributesHandler.from_cmf(
            simple_cmf, traj, start, constant=None,
        )
        assert handler.identified() is False

    def test_from_cmf_produces_handler(self, simple_cmf, symbols):
        traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        handler = TrajectoryAttributesHandler.from_cmf(
            simple_cmf, traj, start, constant=e.value_sympy,
        )
        assert isinstance(handler, TrajectoryAttributesHandler)

    def test_delta_is_finite(self, minimal_handler):
        """Handler built from a real CMF returns either a finite delta or
        ``-inf`` (the documented non-convergence sentinel when the LIReC
        fallback p/q vectors don't reconstruct the target constant)."""
        delta = minimal_handler.delta()
        assert delta is not None
        assert isinstance(delta, float)
        assert not (delta != delta)  # NaN check
        # Either finite, or the documented -inf sentinel.
        assert abs(delta) < 1e9 or delta == float("-inf")

    def test_order_is_positive_int(self, minimal_handler):
        order = minimal_handler.order()
        assert isinstance(order, int)
        assert order >= 1

    def test_formula_str_is_string(self, minimal_handler):
        assert isinstance(minimal_handler.formula_str(), str)

    def test_limit_is_finite(self, minimal_handler):
        limit = minimal_handler.limit()
        assert limit is not None
        assert abs(float(limit)) < 1e15


# ---------------------------------------------------------------------------
# 4. build_trajectory_dto factory
# ---------------------------------------------------------------------------

class TestBuildTrajectoryDto:

    def test_produces_trajectory_dto(self, minimal_handler, symbols):
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        dto = build_trajectory_dto(
            minimal_handler,
            cmf_id="1F1",
            shard_id="sh1",
            cmf_name="1F1",
            shard_encoding_str="test_encoding",
            start=start,
            direction=direction,
        )
        assert isinstance(dto, TrajectoryDTO)

    def test_trajectory_id_is_deterministic(self, minimal_handler, symbols):
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        kwargs = dict(
            cmf_id="1F1", shard_id="sh", cmf_name="1F1",
            shard_encoding_str="enc", start=start, direction=direction,
        )
        dto_a = build_trajectory_dto(minimal_handler, **kwargs)
        dto_b = build_trajectory_dto(minimal_handler, **kwargs)
        assert dto_a.trajectory_id == dto_b.trajectory_id

    def test_different_starts_give_different_ids(self, minimal_handler, symbols, simple_cmf):
        # Use direction=(1,1) — the (1,0) direction is degenerate for 1F1
        # (singular trajectory matrix → ZeroDivisionError on walk).
        start_a = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        start_b = Position({symbols[0]: sp.Integer(2), symbols[1]: sp.Integer(3)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        handler_b = TrajectoryAttributesHandler.from_cmf(
            simple_cmf, direction, start_b, constant=e.value_sympy,
        )
        kwargs_base = dict(cmf_id="c", shard_id="s", cmf_name="c", shard_encoding_str="e")
        dto_a = build_trajectory_dto(minimal_handler, **kwargs_base, start=start_a, direction=direction)
        dto_b = build_trajectory_dto(handler_b, **kwargs_base, start=start_b, direction=direction)
        assert dto_a.trajectory_id != dto_b.trajectory_id

    def test_base_tier1_fields_populated(self, minimal_handler, symbols):
        """build_trajectory_dto fills the cheap Tier-1 fields.

        ``limit_value`` and ``delta_estimate`` (a dict) are Tier-1.  The
        recurrence (``recurrence_relation`` / ``recurrence_order``) is **Tier-2**
        and stays ``None`` unless ``compute_recurrence=True`` (it builds the
        expensive symbolic ``LinearRecurrence``).  ``extended_metrics`` stays
        empty until Tier-2 workers (if any) write to it.
        """
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        dto = build_trajectory_dto(
            minimal_handler,
            cmf_id="c", shard_id="s", cmf_name="c",
            shard_encoding_str="enc", start=start, direction=direction,
        )
        # Recurrence is NOT computed on the default (hot) path.
        assert dto.recurrence_relation is None
        assert dto.recurrence_order is None
        # ``delta_estimate`` is a dict; each value is finite or the -inf sentinel.
        assert isinstance(dto.delta_estimate, dict)
        for delta_val in dto.delta_estimate.values():
            assert abs(delta_val) < 1e9 or delta_val == float("-inf")
        assert abs(float(dto.limit_value)) < 1e15
        assert dto.extended_metrics == {}   # workers haven't run yet

    def test_compute_recurrence_opt_in_populates_recurrence(self, minimal_handler, symbols):
        """``compute_recurrence=True`` populates the Tier-2 recurrence fields."""
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        dto = build_trajectory_dto(
            minimal_handler,
            cmf_id="c", shard_id="s", cmf_name="c",
            shard_encoding_str="enc", start=start, direction=direction,
            compute_recurrence=True,
        )
        assert isinstance(dto.recurrence_relation, str)
        assert dto.recurrence_relation != ""
        assert dto.recurrence_order >= 1

    def test_p_and_q_vectors_are_dicts_or_none(self, minimal_handler, symbols):
        """``build_trajectory_dto`` stores p/q as dicts (const→tuple) or ``None``."""
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        dto = build_trajectory_dto(
            minimal_handler,
            cmf_id="c", shard_id="s", cmf_name="c",
            shard_encoding_str="enc", start=start, direction=direction,
        )
        if dto.p_vector is None:
            assert dto.q_vector is None
        else:
            assert isinstance(dto.p_vector, dict)
            assert isinstance(dto.q_vector, dict)
            for pv in dto.p_vector.values():
                if pv is not None:
                    assert isinstance(pv, tuple)
                    assert len(pv) == minimal_handler.traj_size()


# ---------------------------------------------------------------------------
# 5. SerialSearcher.sample_pairs
# ---------------------------------------------------------------------------

class TestSamplePairs:

    def test_returns_list_of_pairs(self, simple_shard):
        from dreamer.search.methods.hedgehog_scan import SerialSearcher
        searcher = SerialSearcher(simple_shard, e, use_LIReC=False)
        pairs = searcher.sample_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        traj, start = pairs[0]
        assert isinstance(traj, Position)
        assert isinstance(start, Position)

    def test_falls_back_to_origin_without_interior_point(self, simple_cmf, symbols, zero_shift):
        """When interior_point=None, get_interior_point() returns the origin — sample_pairs uses it."""
        from dreamer.search.methods.hedgehog_scan import SerialSearcher
        hps = [Hyperplane(symbols[0], symbols)]
        shard_no_point = Shard(simple_cmf, e, hps, [1], zero_shift, interior_point=None)
        # Shard.get_interior_point() falls back to origin (all-zero Position), not None.
        # sample_pairs must succeed and use that origin as the start.
        searcher = SerialSearcher(shard_no_point, e, use_LIReC=False)
        pairs = searcher.sample_pairs()
        assert len(pairs) > 0
        origin_values = {s: sp.Integer(0) for s in symbols}
        for _, start in pairs:
            assert start == Position(origin_values)

    def test_trajectory_pairs_are_within_shard(self, simple_shard):
        """Every start in the returned pairs should be inside the shard."""
        from dreamer.search.methods.hedgehog_scan import SerialSearcher
        searcher = SerialSearcher(simple_shard, e, use_LIReC=False)
        pairs = searcher.sample_pairs()
        for _, start in pairs:
            assert simple_shard.in_space(start)

    def test_custom_start_is_used(self, simple_shard, symbols):
        """Providing an explicit start returns that start in every pair."""
        from dreamer.search.methods.hedgehog_scan import SerialSearcher
        custom_start = Position({symbols[0]: sp.Integer(2), symbols[1]: sp.Integer(2)})
        searcher = SerialSearcher(simple_shard, e, use_LIReC=False)
        pairs = searcher.sample_pairs(starts=custom_start)
        starts_in_pairs = [s for _, s in pairs]
        assert all(s == custom_start for s in starts_in_pairs)


# ---------------------------------------------------------------------------
# 6. load_seen_trajectory_ids
# ---------------------------------------------------------------------------

class TestLoadSeenTrajectoryIds:

    def test_returns_empty_set_for_nonexistent_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        result = load_seen_trajectory_ids(path)
        assert result == set()

    def test_reads_ids_from_existing_file(self, tmp_path):
        path = tmp_path / "trajectories.jsonl"
        lines = [
            json.dumps({"trajectory_id": "aaa", "delta_estimate": 1.0}),
            json.dumps({"trajectory_id": "bbb", "delta_estimate": 2.0}),
            json.dumps({"trajectory_id": "ccc", "delta_estimate": 3.0}),
        ]
        path.write_text("\n".join(lines) + "\n")
        ids = load_seen_trajectory_ids(str(path))
        assert ids == {"aaa", "bbb", "ccc"}

    def test_skips_malformed_lines_gracefully(self, tmp_path):
        path = tmp_path / "partial.jsonl"
        path.write_text(
            json.dumps({"trajectory_id": "good"}) + "\n"
            "NOT JSON AT ALL\n"
            + json.dumps({"no_id_field": "x"}) + "\n"
        )
        ids = load_seen_trajectory_ids(str(path))
        assert ids == {"good"}

    def test_returns_set_not_list(self, tmp_path):
        path = tmp_path / "ids.jsonl"
        path.write_text(json.dumps({"trajectory_id": "x"}) + "\n")
        result = load_seen_trajectory_ids(str(path))
        assert isinstance(result, set)

    def test_dedup_within_file(self, tmp_path):
        """Duplicate ids in the file collapse to a single set entry."""
        path = tmp_path / "dup.jsonl"
        path.write_text(
            json.dumps({"trajectory_id": "dup"}) + "\n"
            + json.dumps({"trajectory_id": "dup"}) + "\n"
        )
        assert load_seen_trajectory_ids(str(path)) == {"dup"}


# ---------------------------------------------------------------------------
# 7. extended_metrics mutation on frozen TrajectoryDTO
# ---------------------------------------------------------------------------

class TestExtendedMetricsMutation:

    def test_frozen_dto_extended_metrics_is_mutable(self):
        """frozen=True blocks field reassignment but not in-place dict mutation."""
        dto = TrajectoryDTO(
            trajectory_id="t",
            cmf_id="c",
            shard_id="s",
            start_point=(0,),
            direction=(1,),
            recurrence_relation="",
            recurrence_order=1,
            limit_value=1.0,
            delta_estimate={"e": 1.0},
            p_vector={"e": ()},
            q_vector={"e": ()},
        )
        dto.extended_metrics["eigenvalues"] = ["1+0j"]
        assert dto.extended_metrics["eigenvalues"] == ["1+0j"]

    def test_frozen_dto_field_reassignment_raises(self):
        """Reassigning a field on a frozen DTO must raise FrozenInstanceError."""
        dto = TrajectoryDTO(
            trajectory_id="t",
            cmf_id="c",
            shard_id="s",
            start_point=(0,),
            direction=(1,),
            recurrence_relation="",
            recurrence_order=1,
            limit_value=1.0,
            delta_estimate={"e": 1.0},
            p_vector={"e": ()},
            q_vector={"e": ()},
        )
        with pytest.raises(Exception):  # FrozenInstanceError is a dataclasses internal
            dto.trajectory_id = "new_id"


# ---------------------------------------------------------------------------
# 8. JSONL Exporter / Importer round-trip
# ---------------------------------------------------------------------------

def _make_dto(trajectory_id: str = "t1", delta: float = 1.0) -> TrajectoryDTO:
    """Build a minimal TrajectoryDTO with the given id and delta."""
    return TrajectoryDTO(
        trajectory_id=trajectory_id,
        cmf_id="cmf",
        shard_id="shard",
        start_point=(1, 2),
        direction=(0, 1),
        recurrence_relation="a*f(n) + b*f(n-1) = 0",
        recurrence_order=1,
        limit_value=2.7,
        delta_estimate={"e": delta},
        p_vector={"e": ()},
        q_vector={"e": ()},
        identified={"e": True},
    )


class TestJsonlRoundTrip:
    """Exporter.export(JSONL) writes a file Importer.imprt(JSONL) reads back."""

    def test_jsonl_export_then_import_dtos(self, tmp_path):
        dtos = [_make_dto("a", 1.0), _make_dto("b", 2.0), _make_dto("c", 3.0)]
        Exporter.export(
            root=str(tmp_path), f_name="traj", fmt=Formats.JSONL, data=dtos,
        )
        path = tmp_path / "traj.jsonl"
        assert path.exists()
        records = Importer.imprt(str(path))
        assert len(records) == 3
        ids = [r["trajectory_id"] for r in records]
        assert ids == ["a", "b", "c"]
        deltas = [r["delta_estimate"] for r in records]
        assert deltas == [{"e": 1.0}, {"e": 2.0}, {"e": 3.0}]

    def test_jsonl_export_then_dto_from_dict(self, tmp_path):
        """Records returned by Importer can be rebuilt into typed DTOs."""
        original = _make_dto("rebuild_me", 4.2)
        Exporter.export(
            root=str(tmp_path), f_name="t", fmt=Formats.JSONL, data=[original],
        )
        records = Importer.imprt(str(tmp_path / "t.jsonl"))
        restored = TrajectoryDTO.from_dict(records[0])
        assert restored == original

    def test_jsonl_export_accepts_plain_dicts(self, tmp_path):
        """Items without to_json_line() fall back to json.dumps."""
        data = [{"a": 1}, {"a": 2}]
        Exporter.export(root=str(tmp_path), f_name="d", fmt=Formats.JSONL, data=data)
        records = Importer.imprt(str(tmp_path / "d.jsonl"))
        assert records == data

    def test_jsonl_export_rejects_non_iterable(self, tmp_path):
        with pytest.raises(TypeError):
            Exporter.export(
                root=str(tmp_path), f_name="bad", fmt=Formats.JSONL, data=42,
            )

    def test_jsonl_import_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            json.dumps({"x": 1}) + "\n"
            "completely not json\n"
            + json.dumps({"x": 2}) + "\n"
            "\n"  # blank line ignored
        )
        records = Importer.imprt(str(path))
        assert records == [{"x": 1}, {"x": 2}]

    def test_jsonl_import_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert Importer.imprt(str(path)) == []


# ---------------------------------------------------------------------------
# 9. Central attribute registry
# ---------------------------------------------------------------------------

class TestAttributeRegistry:

    def test_known_attributes_are_registered(self):
        """Every public handler method that yields JSON-safe output must
        be reachable through the registry — keeps the registry in sync
        with :class:`TrajectoryAttributesHandler`'s public surface."""
        expected = {
            # Tier-1 — core scalars / vectors.
            "delta", "limit", "order", "formula", "identified",
            "p_vector", "q_vector", "traj_size", "limit_rational",
            # Tier-2 — heavier numerical / spectral attributes.
            "eigenvalues", "eigenvalue_errors", "spectral_gap", "gcd_slope",
            "convergence_class", "coeff_degrees",
            "asymptotic_digits_per_step", "precision_at",
            "companion_coboundary_rank",
            # Tier-3 — symbolic / expensive attributes.
            "asymptotics", "kamidelta", "delta_sequence", "digits_per_step",
            "relation", "recurrence_coeffs",
        }
        assert expected <= set(ATTRIBUTE_REGISTRY)

    def test_unknown_attribute_raises_keyerror(self, minimal_handler):
        with pytest.raises(KeyError, match="not_a_real_attr"):
            compute_attribute(minimal_handler, "not_a_real_attr")

    def test_compute_attribute_delta_is_finite_float(self, minimal_handler):
        """``delta`` is a float — either finite or the -inf sentinel."""
        value = compute_attribute(minimal_handler, "delta")
        assert isinstance(value, float)
        assert np.isfinite(value) or value == float("-inf")

    def test_compute_attribute_identified_is_bool(self, minimal_handler):
        """``identified`` resolves through the registry to a Python bool.

        Because ``identified`` is derived from ``_pq_vector()`` (rather
        than tracked as a separate flag), the registry path produces the
        same result regardless of when it's called relative to other
        attributes.
        """
        value = compute_attribute(minimal_handler, "identified")
        assert isinstance(value, bool)

    def test_identified_independent_of_call_order(self, simple_cmf, simple_shard, symbols):
        """Asking for ``identified`` before any other attribute must match
        asking for it after.

        Builds two fresh handlers from the same fixture: computes
        ``identified`` first on one (registry-driven, before delta/p_vector),
        and last on the other (after ``p_vector`` ran the identification
        path).  Both must agree — that's the order-independence the
        derived-from-``_pq_vector`` design guarantees.
        """
        traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        kwargs = dict(constant=e.value_sympy, searchable=simple_shard)
        h_first = TrajectoryAttributesHandler.from_cmf(simple_cmf, traj, start, **kwargs)
        h_last = TrajectoryAttributesHandler.from_cmf(simple_cmf, traj, start, **kwargs)

        ident_first = compute_attribute(h_first, "identified")  # before anything else
        h_last.p_vector()
        ident_last = compute_attribute(h_last, "identified")

        assert ident_first == ident_last

    def test_compute_attributes_collects_dict(self, minimal_handler):
        out = compute_attributes(minimal_handler, ("delta", "order", "formula"))
        assert set(out) == {"delta", "order", "formula"}
        assert isinstance(out["delta"], float)
        assert isinstance(out["order"], int)
        assert isinstance(out["formula"], str)

    def test_compute_attributes_empty_list(self, minimal_handler):
        assert compute_attributes(minimal_handler, ()) == {}

    def test_compute_attributes_stores_error_by_default(self, minimal_handler):
        """A failing computation is captured as <name>_error, others still run."""
        register_attribute("always_fails", lambda _h: (_ for _ in ()).throw(ValueError("boom")))
        try:
            out = compute_attributes(minimal_handler, ("delta", "always_fails"))
            assert "delta" in out
            assert "always_fails_error" in out
            assert "boom" in out["always_fails_error"]
        finally:
            del ATTRIBUTE_REGISTRY["always_fails"]

    def test_compute_attributes_raises_when_requested(self, minimal_handler):
        register_attribute("always_fails", lambda _h: (_ for _ in ()).throw(ValueError("boom")))
        try:
            with pytest.raises(ValueError):
                compute_attributes(minimal_handler, ("always_fails",), on_error="raise")
        finally:
            del ATTRIBUTE_REGISTRY["always_fails"]

    def test_register_attribute_then_use(self, minimal_handler):
        register_attribute("custom_const", lambda _h: 42)
        try:
            assert compute_attribute(minimal_handler, "custom_const") == 42
        finally:
            del ATTRIBUTE_REGISTRY["custom_const"]

    def test_registry_outputs_are_json_serializable(self, minimal_handler):
        """Every default-registered attribute must yield JSON-safe output.

        Round-trips every entry through ``json.dumps`` — catches any new
        registry entry whose serializer leaks SymPy / mpmath / numpy.
        Per-attribute failures (e.g. expensive ones unable to compute on
        a degenerate fixture) are tolerated; what matters is that the
        returned value is JSON-safe when computation succeeds.
        """
        for name in ATTRIBUTE_REGISTRY:
            try:
                value = compute_attribute(minimal_handler, name)
            except Exception:
                continue  # computation failure is orthogonal to serialization
            json.dumps(value)


# ---------------------------------------------------------------------------
# 9b. Conditional attribute computation — predicates and (name, predicate) specs
# ---------------------------------------------------------------------------

class TestConditionalAttributes:
    """``compute_attributes`` accepts mixed string / ``(name, predicate)``
    specs.  Predicates may be callables or string keys into PREDICATES.
    """

    def test_attribute_name_extracts_from_string_and_tuple(self):
        """``attribute_name`` returns the bare name for both spec forms."""
        assert attribute_name("delta") == "delta"
        assert attribute_name(("eigenvalues", "if_identified")) == "eigenvalues"
        assert attribute_name(("eigenvalues", lambda _h: True)) == "eigenvalues"

    def test_if_identified_predicate_registered(self):
        """The default ``if_identified`` predicate exists and tracks the handler."""
        assert "if_identified" in PREDICATES

    def test_unknown_predicate_name_raises(self, minimal_handler):
        """A misspelled predicate name must fail loudly (matches the
        attribute-name policy)."""
        with pytest.raises(KeyError, match="no_such_predicate"):
            compute_attributes(
                minimal_handler,
                (("eigenvalues", "no_such_predicate"),),
            )

    def test_predicate_true_runs_attribute(self, minimal_handler):
        """A truthy predicate produces the attribute in the output dict."""
        out = compute_attributes(
            minimal_handler,
            (("order", lambda _h: True),),
        )
        assert "order" in out
        assert isinstance(out["order"], int)

    def test_predicate_false_skips_attribute(self, minimal_handler):
        """A falsy predicate omits the attribute entirely — no value, no
        ``<name>_error``: the signal is "we decided not to compute"."""
        out = compute_attributes(
            minimal_handler,
            (("order", lambda _h: False),),
        )
        assert out == {}

    def test_mixed_spec_list(self, minimal_handler):
        """Plain strings and tuples can coexist in one specs list."""
        out = compute_attributes(
            minimal_handler,
            (
                "order",
                ("formula", lambda _h: True),
                ("delta", lambda _h: False),
            ),
        )
        assert "order" in out
        assert "formula" in out
        assert "delta" not in out

    def test_if_identified_gates_on_handler(self, minimal_handler):
        """The named ``if_identified`` predicate gates on the handler's
        own identification status.  Whichever way it evaluates, the
        attribute presence in the output must agree."""
        out = compute_attributes(
            minimal_handler,
            (("order", "if_identified"),),
        )
        assert ("order" in out) == minimal_handler.identified()

    def test_register_predicate_then_use(self, minimal_handler):
        """User-registered named predicates are honoured by ``compute_attributes``."""
        register_predicate("always_yes", lambda _h: True)
        try:
            out = compute_attributes(
                minimal_handler,
                (("order", "always_yes"),),
            )
            assert "order" in out
        finally:
            del PREDICATES["always_yes"]

    # ------------------------------------------------------------------
    # Handler-only complex predicate (no shard context needed)
    # ------------------------------------------------------------------

    def test_if_has_degree_2_built_in(self, minimal_handler):
        """``if_has_degree_2`` is shipped and tracks ``coeff_degrees()``."""
        assert "if_has_degree_2" in PREDICATES
        try:
            expected = 2 in minimal_handler.coeff_degrees()
        except Exception:
            pytest.skip("coeff_degrees() unavailable on the fixture")
        out = compute_attributes(
            minimal_handler,
            (("order", "if_has_degree_2"),),
        )
        assert ("order" in out) == expected

    def test_inline_handler_only_complex_predicate(self, minimal_handler):
        """A user-supplied lambda answering a structural question works the
        same as a named predicate.  Demonstrates the "polynomial coefficient
        of degree k" idiom for arbitrary ``k``."""
        try:
            degrees = minimal_handler.coeff_degrees()
        except Exception:
            pytest.skip("coeff_degrees() unavailable on the fixture")
        expected = 1 in degrees
        out = compute_attributes(
            minimal_handler,
            (("order", lambda h: 1 in h.coeff_degrees()),),
        )
        assert ("order" in out) == expected

    # ------------------------------------------------------------------
    # Context-aware (shard-level) predicate
    # ------------------------------------------------------------------

    def test_context_aware_predicate_uses_two_args(self, minimal_handler):
        """Predicates with arity 2 receive ``(handler, context)``.  Single-arg
        predicates registered in the same list still take only the handler —
        proving the dispatch is per-predicate, not per-call."""
        seen = {}
        def gate(h, ctx):
            seen["h"] = h
            seen["ctx"] = ctx
            return ctx.get("go", False)

        out = compute_attributes(
            minimal_handler,
            (("order", gate),),
            context={"go": True, "marker": "ok"},
        )
        assert "order" in out
        assert seen["h"] is minimal_handler
        assert seen["ctx"] == {"go": True, "marker": "ok"}

    def test_context_aware_predicate_false_skips(self, minimal_handler):
        """A falsy two-arg predicate omits the attribute."""
        out = compute_attributes(
            minimal_handler,
            (("order", lambda _h, ctx: ctx.get("go", False)),),
            context={"go": False},
        )
        assert out == {}

    def test_if_top_n_delta_named_predicate(self, minimal_handler):
        """``if_top_n_delta`` is the shipped shard-level predicate; verifies
        the canonical context shape ``{trajectory_id, top_n_ids}``."""
        assert "if_top_n_delta" in PREDICATES
        ctx_in = {"trajectory_id": "winner", "top_n_ids": {"winner", "other"}}
        out_in = compute_attributes(
            minimal_handler,
            (("order", "if_top_n_delta"),),
            context=ctx_in,
        )
        assert "order" in out_in

        ctx_out = {"trajectory_id": "loser", "top_n_ids": {"winner"}}
        out_out = compute_attributes(
            minimal_handler,
            (("order", "if_top_n_delta"),),
            context=ctx_out,
        )
        assert "order" not in out_out

    def test_top_n_pattern_end_to_end(self, minimal_handler):
        """End-to-end exercise of the "compute extras for top-N" idiom.

        Models a tiny shard with three pre-computed records; selects the
        top-2 by delta; runs ``compute_attributes`` once per record with a
        context built from that selection.  Only the top-2 should get the
        gated attribute computed.
        """
        records = [
            {"trajectory_id": "t1", "delta_estimate": 0.3},
            {"trajectory_id": "t2", "delta_estimate": 1.7},
            {"trajectory_id": "t3", "delta_estimate": 2.5},
        ]
        top_n_ids = {
            r["trajectory_id"]
            for r in sorted(records, key=lambda r: -r["delta_estimate"])[:2]
        }
        # Reusing the same handler is fine — the predicate only cares about
        # the trajectory_id we thread through the context.
        results = {}
        for r in records:
            ctx = {"trajectory_id": r["trajectory_id"], "top_n_ids": top_n_ids}
            results[r["trajectory_id"]] = compute_attributes(
                minimal_handler,
                ("delta_estimate" not in r and "delta" or "order", ("formula", "if_top_n_delta")),
                context=ctx,
            )
        assert "formula" in results["t2"]
        assert "formula" in results["t3"]
        assert "formula" not in results["t1"]


# ---------------------------------------------------------------------------
# 10. System.__best_trajectory_record — JSONL scan logic
# ---------------------------------------------------------------------------

class TestBestTrajectoryRecord:
    """Verifies the system stage's JSONL scan picks the maximum delta.

    JSONL files now live in the flat EXPORT_SEARCH_RESULTS dir (no constant
    subdir).  ``delta_estimate`` is a ``{const_name: float}`` dict.
    ``__best_trajectory_record`` returns ``(record, delta_float)`` or
    ``(None, None)``.
    """

    def test_finds_max_delta_across_files(self, tmp_path, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.system.system import System

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))

        # Two shards, three trajectories — best delta lives in shard B.
        (tmp_path / "cmfA__sh1.jsonl").write_text(
            json.dumps({"trajectory_id": "a1", "delta_estimate": {"e": 1.2},
                        "start_point": [0, 0], "direction": [1, 0]}) + "\n"
            + json.dumps({"trajectory_id": "a2", "delta_estimate": {"e": 2.5},
                          "start_point": [0, 1], "direction": [1, 0]}) + "\n"
        )
        (tmp_path / "cmfB__sh2.jsonl").write_text(
            json.dumps({"trajectory_id": "b1", "delta_estimate": {"e": 4.7},
                        "start_point": [2, 2], "direction": [0, 1]}) + "\n"
        )

        class _Const:
            name = "e"
        record, delta_val = System._System__best_trajectory_record(
            _Const(), {"cmfA__sh1", "cmfB__sh2"},
        )
        assert record is not None
        assert record["trajectory_id"] == "b1"
        assert delta_val == pytest.approx(4.7)

    def test_scoped_to_run_shards_ignores_other_runs(self, tmp_path, monkeypatch):
        """Only shards in *this run's* shard_ids are scanned — a leftover JSONL
        from a previous run on a different CMF (same constant) must not bleed
        its (larger) delta into this run's best-delta report.
        """
        from dreamer.configs.system import sys_config
        from dreamer.system.system import System

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))

        # Run 1 (CMF A) left a high-delta file behind in the flat dir.
        (tmp_path / "cmfA__sh1.jsonl").write_text(
            json.dumps({"trajectory_id": "a1", "delta_estimate": {"e": 9.9},
                        "start_point": [0, 0], "direction": [1, 0]}) + "\n"
        )
        # Run 2 (CMF B) — the only shard this run actually searched.
        (tmp_path / "cmfB__sh2.jsonl").write_text(
            json.dumps({"trajectory_id": "b1", "delta_estimate": {"e": 1.0},
                        "start_point": [2, 2], "direction": [0, 1]}) + "\n"
        )

        class _Const:
            name = "e"
        # Scope to run 2's shard only.
        record, delta_val = System._System__best_trajectory_record(
            _Const(), {"cmfB__sh2"},
        )
        assert record is not None
        assert record["trajectory_id"] == "b1", "must not pick CMF-A's leftover record"
        assert delta_val == pytest.approx(1.0)

    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.system.system import System

        # Use a path that doesn't exist as EXPORT_SEARCH_RESULTS.
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path / "nonexistent"))

        class _Const:
            name = "missing_const"
        record, delta_val = System._System__best_trajectory_record(_Const(), {"any__sh"})
        assert record is None
        assert delta_val is None

    def test_returns_none_when_no_jsonl_files(self, tmp_path, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.system.system import System

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        (tmp_path / "stray.txt").write_text("not a jsonl file")

        class _Const:
            name = "e"
        record, delta_val = System._System__best_trajectory_record(_Const(), {"stray"})
        assert record is None
        assert delta_val is None

    def test_skips_records_with_no_delta(self, tmp_path, monkeypatch):
        from dreamer.configs.system import sys_config
        from dreamer.system.system import System

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        (tmp_path / "f.jsonl").write_text(
            json.dumps({"trajectory_id": "no_delta"}) + "\n"
            + json.dumps({"trajectory_id": "has_delta", "delta_estimate": {"e": 1.0},
                          "start_point": [0], "direction": [1]}) + "\n"
        )

        class _Const:
            name = "e"
        record, delta_val = System._System__best_trajectory_record(_Const(), {"f"})
        assert record["trajectory_id"] == "has_delta"
        assert delta_val == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. Config-driven attribute selection integration
# ---------------------------------------------------------------------------

class TestConfigAttributeSelection:
    """End-to-end: registry honours config-listed attribute names."""

    def test_search_config_default_tier2_attrs_in_registry(self):
        """Every default TIER2_ATTRIBUTES name must be registered.

        Specs may be bare strings or ``(name, predicate)`` tuples; only the
        resolved attribute name needs to live in the registry.
        """
        from dreamer.configs.search import search_config
        for spec in search_config.TIER2_ATTRIBUTES:
            name = attribute_name(spec)
            assert name in ATTRIBUTE_REGISTRY, (
                f"TIER2_ATTRIBUTES default {name!r} missing from registry"
            )

    def test_compute_attributes_with_known_tier2_names_works(self, minimal_handler):
        """A representative Tier-2 attribute list resolves through the registry."""
        names = ('eigenvalues', 'spectral_gap', 'gcd_slope', 'convergence_class')
        out = compute_attributes(minimal_handler, names)
        for name in names:
            assert name in out or f"{name}_error" in out


# ---------------------------------------------------------------------------
# 12. Merge-on-read + patch semantics
# ---------------------------------------------------------------------------

class TestMergeOnRead:
    """Append-only JSONL with per-trajectory patch records merged on read.

    The search stage emits a partial "patch" dict when an already-computed
    trajectory is missing some configured attributes.  Readers merge all
    records sharing the same ``trajectory_id`` to reconstruct the full logical
    record.
    """

    # ------------------------------------------------------------------
    # load_seen_trajectories()
    # ------------------------------------------------------------------

    def test_single_record_fully_intact(self, tmp_path):
        path = tmp_path / "t.jsonl"
        base = {
            "trajectory_id": "abc",
            "delta_estimate": 1.5,
            "extended_metrics": {"eigenvalues": ["1+0j"]},
        }
        path.write_text(json.dumps(base) + "\n")
        result = load_seen_trajectories(str(path))
        assert set(result) == {"abc"}
        assert result["abc"]["delta_estimate"] == 1.5
        assert result["abc"]["extended_metrics"] == {"eigenvalues": ["1+0j"]}

    def test_patch_merges_new_extended_metrics_key(self, tmp_path):
        """Patch line adds a new key to extended_metrics without removing existing ones."""
        path = tmp_path / "t.jsonl"
        base = {"trajectory_id": "t1", "extended_metrics": {"eigenvalues": ["1+0j"]}}
        patch = {"trajectory_id": "t1", "extended_metrics": {"spectral_gap": 0.5}}
        path.write_text(json.dumps(base) + "\n" + json.dumps(patch) + "\n")
        merged = load_seen_trajectories(str(path))["t1"]
        assert merged["extended_metrics"] == {"eigenvalues": ["1+0j"], "spectral_gap": 0.5}

    def test_patch_overwrites_conflicting_extended_metrics_key(self, tmp_path):
        """Later patch wins when a key exists in both base and patch."""
        path = tmp_path / "t.jsonl"
        base = {"trajectory_id": "t1", "extended_metrics": {"eigenvalues": ["old"]}}
        patch = {"trajectory_id": "t1", "extended_metrics": {"eigenvalues": ["new"]}}
        path.write_text(json.dumps(base) + "\n" + json.dumps(patch) + "\n")
        merged = load_seen_trajectories(str(path))["t1"]
        assert merged["extended_metrics"]["eigenvalues"] == ["new"]

    def test_missing_file_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        assert load_seen_trajectories(path) == {}

    def test_load_seen_trajectory_ids_backward_compat(self, tmp_path):
        """load_seen_trajectory_ids is a thin wrapper — still returns a set of ids."""
        path = tmp_path / "t.jsonl"
        path.write_text(
            json.dumps({"trajectory_id": "x"}) + "\n"
            + json.dumps({"trajectory_id": "y"}) + "\n"
        )
        ids = load_seen_trajectory_ids(str(path))
        assert isinstance(ids, set)
        assert ids == {"x", "y"}

    # ------------------------------------------------------------------
    # Importer._read_jsonl(merge=True)
    # ------------------------------------------------------------------

    def test_importer_read_jsonl_merge_combines_same_id(self, tmp_path):
        path = tmp_path / "t.jsonl"
        r1 = {"trajectory_id": "t1", "delta_estimate": 1.0,
              "extended_metrics": {"eigenvalues": ["1"]}}
        r2 = {"trajectory_id": "t1", "extended_metrics": {"spectral_gap": 0.3}}
        r3 = {"trajectory_id": "t2", "delta_estimate": 2.0, "extended_metrics": {}}
        path.write_text("\n".join(json.dumps(r) for r in [r1, r2, r3]) + "\n")
        merged = Importer._read_jsonl(str(path), merge=True)
        assert len(merged) == 2
        t1 = next(r for r in merged if r.get("trajectory_id") == "t1")
        assert t1["delta_estimate"] == 1.0
        assert t1["extended_metrics"]["eigenvalues"] == ["1"]
        assert t1["extended_metrics"]["spectral_gap"] == 0.3

    def test_importer_read_jsonl_merge_false_returns_raw_lines(self, tmp_path):
        """merge=False (default) returns one entry per JSON line, including duplicates."""
        path = tmp_path / "t.jsonl"
        r1 = {"trajectory_id": "t1", "extended_metrics": {"eigenvalues": ["1"]}}
        r2 = {"trajectory_id": "t1", "extended_metrics": {"spectral_gap": 0.3}}
        path.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")
        raw = Importer._read_jsonl(str(path), merge=False)
        assert len(raw) == 2

    # ------------------------------------------------------------------
    # Worker: only compute missing attributes
    # ------------------------------------------------------------------

    def test_worker_skips_already_present_t2_attr(self, monkeypatch):
        """``compute_tier2_for_item`` must not overwrite already-present attrs."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import compute_tier2_for_item

        monkeypatch.setattr(
            config.search,
            "TIER2_ATTRIBUTES",
            ("eigenvalues", "spectral_gap", "gcd_slope", "convergence_class"),
        )

        patch = {
            "trajectory_id": "existing_t1",
            "extended_metrics": {
                "eigenvalues": ["pre-computed"],
                "spectral_gap": 0.5,
                "gcd_slope": 0.1,
                "convergence_class": "linear",
            },
        }
        out = compute_tier2_for_item((None, None, patch))

        assert out["trajectory_id"] == "existing_t1"
        # Pre-computed values must not be replaced by error entries.
        assert out["extended_metrics"]["eigenvalues"] == ["pre-computed"]
        assert "eigenvalues_error" not in out["extended_metrics"]

    def test_worker_full_dto_input_is_passthrough_when_nothing_missing(
        self, minimal_handler, symbols, monkeypatch,
    ):
        """Full DTO input with no missing TIER2 attrs is returned unchanged."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import compute_tier2_for_item

        # No attrs requested → nothing to compute → DTO untouched.
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())

        start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        direction = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
        dto = build_trajectory_dto(
            minimal_handler,
            cmf_id="c", shard_id="s", cmf_name="c",
            shard_encoding_str="enc", start=start, direction=direction,
        )

        out = compute_tier2_for_item((None, None, dto))
        assert out is dto, "Worker must return the same DTO object unchanged"
        assert out.extended_metrics == {}

    def test_worker_with_none_traj_matrix_does_not_crash(self, monkeypatch):
        """When traj_matrix=None and attrs are missing, the worker is a no-op.

        The producer passes ``None`` to short-circuit work; the worker must
        forward the patch unchanged without raising.
        """
        from dreamer.configs import config
        from dreamer.utils.multi_processing import compute_tier2_for_item

        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ("eigenvalues",))

        patch = {"trajectory_id": "t1", "extended_metrics": {}}
        out = compute_tier2_for_item((None, None, patch))
        # No computation happened; extended_metrics stays empty.
        assert out["extended_metrics"] == {}

    def test_tier3_worker_direct_call(self, simple_shard, monkeypatch):
        """``compute_tier3_for_item`` computes registered attrs into a patch dict."""
        from dreamer.configs import config
        from dreamer.post_process.tier3_post_process_mod import compute_tier3_for_item

        monkeypatch.setattr(config.post_process, "TIER3_ATTRIBUTES", ("kamidelta",))

        # Build a real handler from simple_shard so the worker has something
        # symbolic to walk through.
        from dreamer.search.methods.hedgehog_scan import SerialSearcher
        pairs = SerialSearcher(simple_shard, e, use_LIReC=False).sample_pairs()
        if not pairs:
            pytest.skip("No trajectory pairs available")
        traj_p, start_p = pairs[0]
        handler = TrajectoryAttributesHandler.from_cmf(
            simple_shard.cmf, traj_p, start_p,
            constant=e.value_sympy,
            searchable=simple_shard,
        )

        patch = {"trajectory_id": "t1", "extended_metrics": {}}
        out = compute_tier3_for_item((handler.trajectory_matrix(), e.value_sympy, patch))

        assert out is patch  # same dict, mutated in place
        # kamidelta either computed successfully or recorded as an error;
        # either path is acceptable — what matters is that one of them is present.
        assert (
            "kamidelta" in out["extended_metrics"]
            or "kamidelta_error" in out["extended_metrics"]
        )

    # ------------------------------------------------------------------
    # Writer: handles plain patch dicts
    # ------------------------------------------------------------------

    def test_write_jsonl_line_writes_patch_dict(self, tmp_path):
        """``write_jsonl_line`` must serialise a dict patch as a JSON line."""
        from dreamer.utils.multi_processing import write_jsonl_line

        output_path = tmp_path / "out.jsonl"
        patch = {"trajectory_id": "p1", "extended_metrics": {"spectral_gap": 0.7}}
        with open(output_path, "a") as fout:
            write_jsonl_line(patch, fout)

        lines = [ln for ln in output_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["trajectory_id"] == "p1"
        assert record["extended_metrics"]["spectral_gap"] == 0.7

    # ------------------------------------------------------------------
    # Producer: smart deduplication
    # ------------------------------------------------------------------

    def _collecting_sink(self):
        """Return ``(sink, items)`` where *items* is a list each sink call appends to.

        The producer now invokes ``sink(item)`` with a single argument (the
        same item shape ``worker_pool``'s ``push`` accepts).
        """
        items: list = []

        def sink(item):
            items.append(item)

        return sink, items

    @staticmethod
    def _freeze_sample_pairs(monkeypatch, shard, **kw):
        """Sample once and pin ``sample_pairs`` to return that exact list.

        The Sampling Orchestrator is non-deterministic, so calling
        ``sample_pairs`` twice (once to build the test's seen-trajectories
        map, once inside ``_produce`` / analyzer) can yield different pairs
        and break the test invariants.  Pinning the method removes the
        non-determinism for the duration of the test.
        """
        from dreamer.search.methods.hedgehog_scan import SerialSearcher

        pairs = SerialSearcher(shard, e, use_LIReC=False).sample_pairs(**kw)
        monkeypatch.setattr(
            SerialSearcher, "sample_pairs",
            lambda self, **_kw: list(pairs),
        )
        return pairs

    def test_producer_skips_fully_covered_trajectory(self, simple_shard, monkeypatch):
        """If all desired attrs are present, the producer must invoke sink zero times."""
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.configs import config

        # Force a non-empty TIER2_ATTRIBUTES so "fully covered" is a non-trivial
        # property — otherwise everything is trivially covered.
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ("eigenvalues",))

        pairs = self._freeze_sample_pairs(monkeypatch, simple_shard)
        if not pairs:
            pytest.skip("No trajectory pairs available for this shard")

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        # Pre-populate every trajectory as fully covered for "eigenvalues".
        seen_trajectories = {}
        for traj_p, start_p in pairs:
            start_t = tuple(int(v) for v in start_p.values())
            dir_t = tuple(int(v) for v in traj_p.values())
            tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
            fp = tier1_config_fingerprint(walk_depth_for(simple_shard.cmf, traj_p))
            seen_trajectories[tid] = {
                "trajectory_id": tid,
                "extended_metrics": {"eigenvalues": "dummy"},
                "config_fingerprint": fp,
            }

        sink, items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
        )

        assert items == [], (
            f"Expected no sink calls when all attrs are present, got {len(items)}"
        )

    def test_producer_does_not_build_handler_for_fully_covered_trajectory(
        self, simple_shard, monkeypatch,
    ):
        """The early-skip path must avoid constructing the handler entirely.

        Building the handler triggers the trajectory walk through
        ``build_trajectory_dto``, which is the costly step we want to avoid
        on re-runs.
        """
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.configs import config

        # Empty TIER2_ATTRIBUTES → every seen trajectory is "fully covered".
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())

        pairs = self._freeze_sample_pairs(monkeypatch, simple_shard)
        if not pairs:
            pytest.skip("No trajectory pairs available for this shard")

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        seen_trajectories = {}
        for traj_p, start_p in pairs:
            start_t = tuple(int(v) for v in start_p.values())
            dir_t = tuple(int(v) for v in traj_p.values())
            tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
            fp = tier1_config_fingerprint(walk_depth_for(simple_shard.cmf, traj_p))
            seen_trajectories[tid] = {
                "trajectory_id": tid, "extended_metrics": {}, "config_fingerprint": fp,
            }

        # Count handler constructions.
        calls = [0]
        original_from_cmf = TrajectoryAttributesHandler.from_cmf

        def counting_from_cmf(*args, **kwargs):
            calls[0] += 1
            return original_from_cmf(*args, **kwargs)

        monkeypatch.setattr(
            TrajectoryAttributesHandler, "from_cmf", counting_from_cmf,
        )

        sink, items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
        )

        assert calls[0] == 0, (
            f"Expected zero handler builds for fully-covered trajectories, got {calls[0]}"
        )
        assert items == []

    def test_producer_emits_full_dto_for_new_trajectory(self, simple_shard):
        """A trajectory not seen before must be passed to sink as a full TrajectoryDTO."""
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.utils.storage.dtos import TrajectoryDTO

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        sink, items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories={},  # nothing seen — every pair is new
        )

        assert len(items) > 0, "Expected new trajectories to reach the sink"
        for traj_matrix, constant, payload in items:
            assert isinstance(payload, TrajectoryDTO), (
                f"Expected TrajectoryDTO for a new trajectory, got {type(payload).__name__}"
            )
            assert traj_matrix is not None, (
                "New trajectories must ship the trajectory matrix to workers."
            )
            assert constant is not None, (
                "Producer must propagate the sympy constant alongside each item."
            )

    def test_producer_updates_seen_trajectories_after_emit(self, simple_shard):
        """After a new trajectory is emitted, it must appear in seen_trajectories."""
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)
        seen: dict = {}
        sink, _items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories=seen,
        )

        assert len(seen) > 0, "Producer must record emitted trajectories in seen_trajectories"
        for record in seen.values():
            assert "extended_metrics" in record

    def test_load_seen_trajectories_patch_only_no_base(self, tmp_path):
        """A file containing only a patch (no prior base record) is still readable.

        This represents a corrupted/partial run where only patches survived;
        the merge logic must not crash and the patch becomes the merged record.
        """
        path = tmp_path / "patch_only.jsonl"
        patch = {"trajectory_id": "orphan", "extended_metrics": {"spectral_gap": 0.5}}
        path.write_text(json.dumps(patch) + "\n")
        merged = load_seen_trajectories(str(path))
        assert merged["orphan"]["extended_metrics"] == {"spectral_gap": 0.5}

    def test_load_seen_trajectories_three_way_merge(self, tmp_path):
        """Three records sharing the same id are merged left-to-right."""
        path = tmp_path / "three.jsonl"
        records = [
            {"trajectory_id": "t", "delta_estimate": 1.0,
             "extended_metrics": {"a": 1}},
            {"trajectory_id": "t", "extended_metrics": {"b": 2}},
            {"trajectory_id": "t", "delta_estimate": 9.9,
             "extended_metrics": {"c": 3}},
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        merged = load_seen_trajectories(str(path))["t"]
        assert merged["extended_metrics"] == {"a": 1, "b": 2, "c": 3}
        # Top-level keys: later wins.
        assert merged["delta_estimate"] == 9.9

    def test_producer_patch_path_does_not_compute_recurrence_relation(
        self, simple_shard, monkeypatch,
    ):
        """Patch path (case 2) builds the handler but must skip ``formula_str``/``order``.

        Those are Tier-1 DTO fields written once, in the new-trajectory case
        only.  When a trajectory is already in the JSONL and only Tier-2
        attrs are missing, the producer must ship the trajectory matrix to
        the worker *without* triggering the linear-recurrence symbolic work.
        """
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.configs import config

        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ("eigenvalues",))

        pairs = self._freeze_sample_pairs(monkeypatch, simple_shard)
        if not pairs:
            pytest.skip("No trajectory pairs available for this shard")

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        # Mark every trajectory as known with the configured Tier-2 attr missing.
        seen_trajectories: dict = {}
        for traj_p, start_p in pairs:
            start_t = tuple(int(v) for v in start_p.values())
            dir_t = tuple(int(v) for v in traj_p.values())
            tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
            seen_trajectories[tid] = {
                "trajectory_id": tid,
                "extended_metrics": {},  # eigenvalues missing → patch path
            }

        calls = {"formula_str": 0, "order": 0}
        original_formula = TrajectoryAttributesHandler.formula_str
        original_order = TrajectoryAttributesHandler.order

        def counting_formula(self_):
            calls["formula_str"] += 1
            return original_formula(self_)

        def counting_order(self_):
            calls["order"] += 1
            return original_order(self_)

        monkeypatch.setattr(TrajectoryAttributesHandler, "formula_str", counting_formula)
        monkeypatch.setattr(TrajectoryAttributesHandler, "order", counting_order)

        sink, items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
        )

        # Patches must have been emitted, but neither Tier-1 symbolic field was touched.
        assert len(items) > 0
        assert calls["formula_str"] == 0
        assert calls["order"] == 0

    def test_producer_emits_patch_for_missing_tier2_attr(self, simple_shard, monkeypatch):
        """Trajectories with a missing Tier-2 attr must produce patch dicts, not full DTOs."""
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.configs import config

        # Configure two Tier-2 attributes; mark only the first as already present
        # so each trajectory has exactly one missing attr → triggers patch path.
        configured = ("eigenvalues", "spectral_gap")
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", configured)

        pairs = self._freeze_sample_pairs(monkeypatch, simple_shard)
        if not pairs:
            pytest.skip("No trajectory pairs available for this shard")

        cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)
        present_attr = configured[0]

        seen_trajectories = {}
        for traj_p, start_p in pairs:
            start_t = tuple(int(v) for v in start_p.values())
            dir_t = tuple(int(v) for v in traj_p.values())
            tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
            fp = tier1_config_fingerprint(walk_depth_for(simple_shard.cmf, traj_p))
            seen_trajectories[tid] = {
                "trajectory_id": tid,
                "extended_metrics": {present_attr: "pre-computed"},
                "config_fingerprint": fp,
            }

        sink, items = self._collecting_sink()
        SearcherModV1([simple_shard], use_LIReC=False)._produce(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=enc_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
        )

        assert len(items) > 0, "Expected at least one patch to be emitted"
        for _traj_matrix, _constant, payload in items:
            assert isinstance(payload, dict), (
                f"Expected patch dict, got {type(payload).__name__}"
            )
            assert "trajectory_id" in payload
            # The patch is empty (workers fill it); the already-present attr
            # must not be in it.
            assert present_attr not in payload["extended_metrics"]


# ---------------------------------------------------------------------------
# 13. Direct-write path (no Tier-2 attributes configured)
# ---------------------------------------------------------------------------

class TestDirectWritePath:
    """When TIER2_ATTRIBUTES is empty, ``_run_shard`` must skip the MPMC
    subprocess setup and write straight to the JSONL from the main thread."""

    def test_direct_write_creates_jsonl_without_subprocesses(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        from dreamer.search.searchers.hedgehog_scan_mod import SearcherModV1
        from dreamer.configs import config

        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())

        # Spy on mp.Process so we can assert it is never invoked.
        process_calls = [0]
        original_process = mp.Process

        def spy_process(*args, **kwargs):
            process_calls[0] += 1
            return original_process(*args, **kwargs)

        monkeypatch.setattr("multiprocessing.Process", spy_process)
        # ``worker_pool`` creates Process via ``mp.Process`` in multi_processing.py.
        monkeypatch.setattr(
            "dreamer.utils.multi_processing.mp.Process",
            spy_process,
        )

        from dreamer.configs.system import sys_config
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))

        searcher = SearcherModV1([simple_shard], use_LIReC=False)
        searcher._run_shard(
            shard=simple_shard,
            identified_consts=list(simple_shard.consts),
            num_workers=4,  # ignored on the direct-write path
            config_overrides=config.export_configurations(),
        )

        assert process_calls[0] == 0, (
            "Direct-write path must not spawn any subprocess"
        )

        # A JSONL file must have been created and contain at least one record.
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        records = jsonl_files[0].read_text().strip().splitlines()
        assert len(records) > 0
        for line in records:
            record = json.loads(line)
            assert "trajectory_id" in record


# ---------------------------------------------------------------------------
# 14. Analyzer cross-run dedup
# ---------------------------------------------------------------------------

class TestAnalyzerDedup:
    """Analysis stage must skip shards already represented in the per-constant JSONL."""

    def test_load_seen_shards_reads_records(self, tmp_path):
        path = tmp_path / "e.jsonl"
        records = [
            {"shard_id": "s1", "best_delta": 1.5, "identified_pct": 0.8},
            {"shard_id": "s2", "best_delta": 2.5, "identified_pct": 1.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = load_seen_shards(str(path))
        assert set(result) == {"s1", "s2"}
        assert result["s1"]["best_delta"] == 1.5

    def test_load_seen_shards_missing_file(self, tmp_path):
        assert load_seen_shards(str(tmp_path / "missing.jsonl")) == {}

    def test_load_seen_shards_skips_records_without_shard_id(self, tmp_path):
        path = tmp_path / "e.jsonl"
        path.write_text(
            json.dumps({"shard_id": "s1", "best_delta": 1.0}) + "\n"
            + json.dumps({"no_id_here": True}) + "\n"
        )
        assert set(load_seen_shards(str(path))) == {"s1"}

    def test_analyzer_always_samples_pairs(self, simple_shard, tmp_path, monkeypatch):
        """The analyzer must call ``sample_pairs`` even when every trajectory
        is already on file — different runs may sample differently and
        the per-trajectory dedup happens after sampling.
        """
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config
        from dreamer.search.methods.hedgehog_scan import SerialSearcher

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)

        # Seed every sampled trajectory as fully cached so no walks happen
        # — but sampling itself must still occur.
        _cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)
        pairs = SerialSearcher(simple_shard, e, use_LIReC=False).sample_pairs(
            trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
        )
        # JSONL now lives flat under EXPORT_SEARCH_RESULTS.
        jsonl_path = tmp_path / f"{shard_id}.jsonl"
        with open(jsonl_path, "w") as fout:
            for traj_p, start_p in pairs:
                start_t = tuple(int(v) for v in start_p.values())
                dir_t = tuple(int(v) for v in traj_p.values())
                tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
                fout.write(json.dumps({
                    "trajectory_id": tid,
                    "delta_estimate": {e.name: 1.0},
                    "identified": {e.name: True},
                }) + "\n")

        sample_calls = [0]
        original_sample = SerialSearcher.sample_pairs

        def counting_sample(self_, *args, **kwargs):
            sample_calls[0] += 1
            return original_sample(self_, *args, **kwargs)

        monkeypatch.setattr(SerialSearcher, "sample_pairs", counting_sample)

        AnalyzerModV1({e: [simple_shard]}).execute()

        assert sample_calls[0] >= 1, (
            "Analyzer must always call sample_pairs, even with a populated cache"
        )

    def test_analyzer_skips_walks_for_cached_trajectories(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """When every sampled trajectory is on file with dict-format delta + identified,
        no handler is constructed (no trajectory walk happens).
        """
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config
        from dreamer.search.methods.hedgehog_scan import SerialSearcher

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)

        _cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        # Capture one fixed set of pairs and pin ``sample_pairs`` to it, so the
        # trajectories the analyzer processes are *exactly* the ones we seed.
        # The samplers are not yet seeded for determinism (see roadmap backlog),
        # and the search-stage default (``pt``) differs from
        # ``analysis.SAMPLING_METHOD`` (``raycast``); without pinning, two
        # separate sample calls would draw different trajectories and the cache
        # would miss, defeating the point of this test.
        fixed_pairs = SerialSearcher(simple_shard, e, use_LIReC=False).sample_pairs(
            trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
            sampling_method=analysis_config.SAMPLING_METHOD,
        )
        monkeypatch.setattr(
            SerialSearcher, "sample_pairs",
            lambda self_, *args, **kwargs: fixed_pairs,
        )

        # JSONL now lives flat under EXPORT_SEARCH_RESULTS.
        jsonl_path = tmp_path / f"{shard_id}.jsonl"
        with open(jsonl_path, "w") as fout:
            for traj_p, start_p in fixed_pairs:
                start_t = tuple(int(v) for v in start_p.values())
                dir_t = tuple(int(v) for v in traj_p.values())
                tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)
                fp = tier1_config_fingerprint(walk_depth_for(simple_shard.cmf, traj_p))
                fout.write(json.dumps({
                    "trajectory_id": tid,
                    "delta_estimate": {e.name: 2.5},
                    "identified": {e.name: True},
                    "config_fingerprint": fp,
                }) + "\n")

        calls = [0]
        original = TrajectoryAttributesHandler.from_cmf

        def counting(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(TrajectoryAttributesHandler, "from_cmf", counting)

        result = AnalyzerModV1({e: [simple_shard]}).execute()

        assert calls[0] == 0, (
            f"All trajectories cached → zero handler builds expected, got {calls[0]}"
        )
        # And the cached best_delta is used in ranking.
        assert simple_shard in result[e]

    def test_analyzer_walks_uncached_trajectories(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """A trajectory missing from the JSONL must trigger a fresh handler build."""
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)

        calls = [0]
        original = TrajectoryAttributesHandler.from_cmf

        def counting(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(TrajectoryAttributesHandler, "from_cmf", counting)

        AnalyzerModV1({e: [simple_shard]}).execute()
        # No cache → every sampled pair must produce one handler build.
        assert calls[0] > 0, "Uncached run must build handlers for new trajectories"

    def test_analyzer_writes_per_trajectory_records(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """Output is per-trajectory at ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl`` (flat)."""
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)

        AnalyzerModV1({e: [simple_shard]}).execute()

        _cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        # JSONL now lives flat (no constant subdir).
        jsonl_path = tmp_path / f"{shard_id}.jsonl"
        assert jsonl_path.exists(), (
            "Analyzer must write to the flat per-shard JSONL location"
        )
        lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        assert len(lines) > 0

        # Every line must be a valid TrajectoryDTO-shaped record.
        for line in lines:
            record = json.loads(line)
            assert "trajectory_id" in record
            assert "delta_estimate" in record
            assert "identified" in record
            assert "shard_id" in record
            assert record["shard_id"] == shard_id

    def test_analyzer_writes_to_separate_store_when_flag_on(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """With ``STORE_TRAJECTORIES_SEPARATELY`` the analyzer writes to
        ``EXPORT_ANALYSIS_RESULTS`` and leaves ``EXPORT_SEARCH_RESULTS`` empty."""
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config

        search_dir = tmp_path / "search"
        analysis_dir = tmp_path / "analysis"
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(search_dir))
        monkeypatch.setattr(sys_config, "EXPORT_ANALYSIS_RESULTS", str(analysis_dir))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)
        monkeypatch.setattr(analysis_config, "STORE_TRAJECTORIES_SEPARATELY", True)

        AnalyzerModV1({e: [simple_shard]}).execute()

        _cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        analysis_jsonl = analysis_dir / f"{shard_id}.jsonl"
        search_jsonl = search_dir / f"{shard_id}.jsonl"

        assert analysis_jsonl.exists(), "Trajectories must go to the analysis store"
        assert [ln for ln in analysis_jsonl.read_text().splitlines() if ln.strip()]
        assert not search_jsonl.exists(), (
            "Search-results dir must not receive analysis trajectories when the "
            "separate-store flag is on"
        )

    def test_analyzer_partial_cache_only_walks_missing(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """When only some trajectories are cached, only the uncached ones get walked."""
        from dreamer.analysis.analyzers.serial_scan.analyzer_mod import AnalyzerModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs.analysis import analysis_config
        from dreamer.search.methods.hedgehog_scan import SerialSearcher

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(analysis_config, "IDENTIFY_THRESHOLD", -1)

        _cmf_id, shard_id, enc_str = derive_cmf_and_shard_ids(simple_shard)

        # Pin sample_pairs so the analyzer's internal call returns the same
        # list the test caches against — sampling is non-deterministic
        # otherwise (see TestMergeOnRead._freeze_sample_pairs).
        pairs = SerialSearcher(simple_shard, e, use_LIReC=False).sample_pairs(
            trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
        )
        monkeypatch.setattr(
            SerialSearcher, "sample_pairs",
            lambda self, **_kw: list(pairs),
        )
        if len(pairs) < 2:
            pytest.skip("Need >=2 sampled pairs to exercise partial-cache path")

        # Cache exactly the first pair.
        cached_pair, *_ = pairs
        traj_p, start_p = cached_pair
        start_t = tuple(int(v) for v in start_p.values())
        dir_t = tuple(int(v) for v in traj_p.values())
        tid = derive_trajectory_id(shard_id, simple_shard.cmf_name, enc_str, start_t, dir_t)

        # JSONL now lives flat under EXPORT_SEARCH_RESULTS.
        jsonl_path = tmp_path / f"{shard_id}.jsonl"
        fp = tier1_config_fingerprint(walk_depth_for(simple_shard.cmf, traj_p))
        with open(jsonl_path, "w") as fout:
            fout.write(json.dumps({
                "trajectory_id": tid,
                "delta_estimate": {e.name: 1.0},
                "identified": {e.name: True},
                "config_fingerprint": fp,
            }) + "\n")

        calls = [0]
        original = TrajectoryAttributesHandler.from_cmf

        def counting(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(TrajectoryAttributesHandler, "from_cmf", counting)

        AnalyzerModV1({e: [simple_shard]}).execute()

        expected_walks = len(pairs) - 1
        assert calls[0] == expected_walks, (
            f"Expected exactly {expected_walks} handler builds (one per uncached pair), "
            f"got {calls[0]}"
        )


# ---------------------------------------------------------------------------
# 15. worker_pool generic abstraction
# ---------------------------------------------------------------------------

# Module-level worker/writer fns so multiprocessing can pickle them.

def _double_worker(item):
    """Worker for tests: returns ``(label, value*2)``."""
    label, value = item
    return (label, value * 2)


def _id_worker(item):
    return item


def _write_repr(item, fout):
    fout.write(repr(item) + "\n")


class TestWorkerPool:
    """Generic queue/process abstraction must run both direct and MPMC modes."""

    def test_direct_mode_no_subprocess(self, tmp_path, monkeypatch):
        """``worker_fn=None`` runs producer → writer on the main thread."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import worker_pool

        process_calls = [0]

        def spy_process(*args, **kwargs):
            process_calls[0] += 1
            raise AssertionError("Direct mode must not spawn subprocesses")

        monkeypatch.setattr("dreamer.utils.multi_processing.mp.Process", spy_process)

        output = tmp_path / "out.txt"
        with worker_pool(
            num_workers=4,
            worker_fn=None,
            writer_fn=_write_repr,
            output_path=str(output),
            config_overrides=config.export_configurations(),
        ) as push:
            push(("a", 1))
            push(("b", 2))

        assert process_calls[0] == 0
        lines = output.read_text().strip().splitlines()
        assert lines == ["('a', 1)", "('b', 2)"]

    def test_parallel_false_runs_worker_fn_inline(self, tmp_path, monkeypatch):
        """``parallel=False`` applies the worker_fn on the main thread.

        Crucial for the Search direct-write path: the producer pushes
        ``(traj_matrix, payload)`` tuples that the worker must unwrap
        before they reach the writer — and this still has to happen
        when no subprocess is created.
        """
        from dreamer.configs import config
        from dreamer.utils.multi_processing import worker_pool

        def spy_no_process(*args, **kwargs):
            raise AssertionError("parallel=False must not spawn subprocesses")

        monkeypatch.setattr(
            "dreamer.utils.multi_processing.mp.Process", spy_no_process,
        )

        output = tmp_path / "out.txt"
        with worker_pool(
            num_workers=4,
            worker_fn=_double_worker,
            writer_fn=_write_repr,
            output_path=str(output),
            config_overrides=config.export_configurations(),
            parallel=False,
        ) as push:
            push(("x", 1))
            push(("x", 2))

        lines = output.read_text().strip().splitlines()
        assert lines == ["('x', 2)", "('x', 4)"]

    def test_mpmc_mode_applies_worker_fn(self, tmp_path):
        """MPMC mode pipes every item through ``worker_fn`` before writing."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import worker_pool

        output = tmp_path / "out.txt"
        with worker_pool(
            num_workers=2,
            worker_fn=_double_worker,
            writer_fn=_write_repr,
            output_path=str(output),
            config_overrides=config.export_configurations(),
        ) as push:
            for i in range(5):
                push(("x", i))

        lines = output.read_text().strip().splitlines()
        # Order across workers is not guaranteed — compare as a set.
        assert set(lines) == {
            "('x', 0)", "('x', 2)", "('x', 4)", "('x', 6)", "('x', 8)",
        }

    def test_mpmc_subprocesses_cleaned_up_even_if_producer_raises(self, tmp_path):
        """The finally-block must drain queues and join workers + writer."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import worker_pool

        output = tmp_path / "out.txt"
        with pytest.raises(RuntimeError, match="producer-side"):
            with worker_pool(
                num_workers=2,
                worker_fn=_id_worker,
                writer_fn=_write_repr,
                output_path=str(output),
                config_overrides=config.export_configurations(),
            ) as push:
                push(("x", 1))
                raise RuntimeError("producer-side failure")
        # Reaching this line proves all subprocesses joined (the with-block
        # would hang on __exit__ otherwise).

    def test_empty_producer_writes_no_lines(self, tmp_path):
        """No ``push`` calls → output file is created but empty."""
        from dreamer.configs import config
        from dreamer.utils.multi_processing import worker_pool

        output = tmp_path / "empty.txt"
        with worker_pool(
            num_workers=2,
            worker_fn=_id_worker,
            writer_fn=_write_repr,
            output_path=str(output),
            config_overrides=config.export_configurations(),
        ) as _push:
            pass  # nothing pushed

        # The writer opens the file in append mode, so it exists even with no input.
        assert output.exists()
        assert output.read_text() == ""

    def test_direct_mode_batches_flushes(self, tmp_path, monkeypatch):
        """Direct mode flushes every WRITER_BATCH_SIZE items, not per push.

        We monkeypatch ``builtins.open`` to wrap the returned file so we can
        count ``flush`` calls.  With ``WRITER_BATCH_SIZE=3``, three pushes
        should produce exactly one flush — not three.
        """
        from dreamer.configs import config
        from dreamer.configs.system import sys_config
        from dreamer.utils.multi_processing import worker_pool

        monkeypatch.setattr(sys_config, "WRITER_BATCH_SIZE", 3)

        flush_calls = [0]
        real_open = open

        class FlushCountingFile:
            def __init__(self, f):
                self._f = f
            def write(self, *a, **kw):
                return self._f.write(*a, **kw)
            def flush(self):
                flush_calls[0] += 1
                return self._f.flush()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return self._f.__exit__(*a)
            def close(self):
                return self._f.close()

        def counting_open(*args, **kwargs):
            return FlushCountingFile(real_open(*args, **kwargs))

        monkeypatch.setattr(
            "dreamer.utils.multi_processing.open", counting_open, raising=False,
        )

        output = tmp_path / "batched.txt"
        with worker_pool(
            num_workers=1,
            worker_fn=None,
            writer_fn=_write_repr,
            output_path=str(output),
            config_overrides=config.export_configurations(),
            parallel=False,
        ) as push:
            push(("x", 1))
            push(("x", 2))
            assert flush_calls[0] == 0, "Pre-batch pushes must not flush"
            push(("x", 3))  # third push fills the batch → exactly one flush
            assert flush_calls[0] == 1
            push(("x", 4))  # fourth starts a new batch — no extra flush yet
            assert flush_calls[0] == 1


# ---------------------------------------------------------------------------
# 16. Tier-3 post-process stage
# ---------------------------------------------------------------------------

class TestTier3PostProcess:
    """Tier-3 stage runs after Search, patches existing JSONL files."""

    def test_short_circuit_when_no_tier3_attrs_configured(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """Empty ``TIER3_ATTRIBUTES`` → execute returns immediately, no file touched."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs import config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(config.post_process, "TIER3_ATTRIBUTES", ())

        # Seed a JSONL the stage would otherwise process — assert it's untouched.
        jsonl = tmp_path / "anything.jsonl"
        seeded = json.dumps({
            "trajectory_id": "t1",
            "cmf_id": simple_shard.cmf_name,
            "extended_metrics": {},
        }) + "\n"
        jsonl.write_text(seeded)

        Tier3PostProcessModV1({e: [simple_shard]}).execute()
        assert jsonl.read_text() == seeded

    def test_skips_fully_covered_trajectory_without_worker(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """Trajectories that already have every TIER3 attr present must not spawn workers."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs import config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(
            config.post_process,
            "TIER3_ATTRIBUTES",
            ("asymptotics",),
        )

        cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        # Flat layout: JSONL directly in EXPORT_SEARCH_RESULTS (no const subdir).
        jsonl = tmp_path / f"{shard_id}.jsonl"
        # All trajectories already carry the only configured Tier-3 attr.
        jsonl.write_text(
            json.dumps({
                "trajectory_id": "t1",
                "cmf_id": cmf_id,
                "start_point": [1, 1],
                "direction": [1, 1],
                "extended_metrics": {"asymptotics": ["pre-computed"]},
            }) + "\n"
        )

        # Spy on subprocess spawn — we must not even enter the worker_pool MPMC path.
        process_calls = [0]
        original = mp.Process
        def counting_process(*args, **kwargs):
            process_calls[0] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr("dreamer.utils.multi_processing.mp.Process", counting_process)

        Tier3PostProcessModV1({e: [simple_shard]}).execute()

        assert process_calls[0] == 0, (
            "Fully-covered shard must short-circuit before spawning workers"
        )
        # Original file unchanged — no patches appended.
        assert json.loads(jsonl.read_text().strip())["extended_metrics"] == {
            "asymptotics": ["pre-computed"],
        }

    def test_skips_shards_not_in_priorities(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """JSONLs whose shard isn't in ``priorities`` must be left untouched.

        Extraction may write data for shards that didn't pass the analysis
        identification threshold (so they aren't in ``priorities``).  Tier-3
        compute on those is wasted work — the post-process stage should
        process only the prioritised shards.
        """
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs import config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        monkeypatch.setattr(
            config.post_process,
            "TIER3_ATTRIBUTES",
            ("kamidelta",),
        )

        _cmf_id, prioritised_shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        # An orphan shard JSONL — same cmf prefix, different encoding hash.
        # The structural shard_id format is ``<cmf_id>__<hash>``.
        orphan_shard_id = prioritised_shard_id.rsplit("__", 1)[0] + "__deadbeefdeadbeef"
        orphan_jsonl = tmp_path / f"{orphan_shard_id}.jsonl"
        orphan_record = {
            "trajectory_id": f"{orphan_shard_id}__cafef00dcafef00d",
            "cmf_id": prioritised_shard_id.rsplit("__", 1)[0],
            "shard_id": orphan_shard_id,
            "start_point": [1, 1],
            "direction": [1, 1],
            "extended_metrics": {},
        }
        original_orphan_text = json.dumps(orphan_record) + "\n"
        orphan_jsonl.write_text(original_orphan_text)

        # ``priorities`` contains *only* simple_shard — the orphan isn't in it.
        Tier3PostProcessModV1({e: [simple_shard]}).execute()

        # The orphan file is byte-for-byte unchanged: post-process didn't even
        # open it, let alone append a patch.
        assert orphan_jsonl.read_text() == original_orphan_text

    def test_appends_patch_for_missing_tier3_attr(
        self, simple_shard, tmp_path, monkeypatch,
    ):
        """A trajectory missing a Tier-3 attr must receive an appended patch line."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
        from dreamer.configs.system import sys_config
        from dreamer.configs import config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
        # ``kamidelta`` is registered and cheap-ish on the trivial 1F1 shard;
        # falling back to a registry error is acceptable — the patch line itself
        # is what's under test.
        monkeypatch.setattr(
            config.post_process,
            "TIER3_ATTRIBUTES",
            ("kamidelta",),
        )

        cmf_id, shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        # Flat layout: JSONL directly in EXPORT_SEARCH_RESULTS (no const subdir).
        jsonl = tmp_path / f"{shard_id}.jsonl"
        base_record = {
            "trajectory_id": "t-needs-tier3",
            "cmf_id": cmf_id,
            "start_point": [1, 1],
            "direction": [1, 1],
            "extended_metrics": {},
        }
        jsonl.write_text(json.dumps(base_record) + "\n")

        Tier3PostProcessModV1({e: [simple_shard]}).execute()

        lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2, "Expected at least one patch appended"
        # The last line should be the patch.
        patch = json.loads(lines[-1])
        assert patch["trajectory_id"] == "t-needs-tier3"
        assert "extended_metrics" in patch
        # Either kamidelta computed, or it errored — either is fine; what matters
        # is that the patch line exists and carries the trajectory id.
        em = patch["extended_metrics"]
        assert "kamidelta" in em or "kamidelta_error" in em

    def test_cmf_lookup_built_from_priorities(self, simple_shard):
        """Searchables in priorities feed the in-memory CMF lookup."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1

        mod = Tier3PostProcessModV1({e: [simple_shard]})
        assert simple_shard.cmf_name in mod._cmf_lookup
        # Same object — the searchable's CMF is reused, not re-loaded.
        assert mod._cmf_lookup[simple_shard.cmf_name] is simple_shard.cmf

    def test_cmf_lookup_falls_back_to_disk(self, tmp_path, monkeypatch):
        """Empty priorities → look in sys_config.EXPORT_CMFS instead."""
        from dreamer.post_process.tier3_post_process_mod import Tier3PostProcessModV1
        from dreamer.configs.system import sys_config

        # Empty path → empty lookup, no crash.
        monkeypatch.setattr(sys_config, "EXPORT_CMFS", str(tmp_path))
        mod = Tier3PostProcessModV1({})
        assert mod._cmf_lookup == {}


# ---------------------------------------------------------------------------
# Atlas writer (CmfDTO / CmfFamilyDTO / ShardDTO JSONL storage)
# ---------------------------------------------------------------------------

class TestAtlasWriter:
    """Tests for ``atlas_writer.py`` — the DB-ready DTO storage layer.

    Covers the loading-stage CMF/family writer and the extraction-stage
    shard writer, including idempotent rerun behaviour (skip-if-present).
    """

    def test_build_cmf_family_dto(self, simple_cmf):
        from dreamer.utils.storage.atlas_writer import build_cmf_family_dto

        dto = build_cmf_family_dto(simple_cmf)
        assert isinstance(dto, CmfFamilyDTO)
        assert dto.family_id == "1F1"
        assert dto.global_family_id == "pFq"
        assert dto.dimensions == len(simple_cmf.matrices)
        assert dto.matrix_definitions  # non-empty
        # Round-trip
        assert CmfFamilyDTO.from_dict(json.loads(dto.to_json_line())) == dto

    def test_build_cmf_dto(self, simple_cmf, zero_shift):
        from dreamer.utils.storage.atlas_writer import build_cmf_dto
        from dreamer.utils.types import CMFData

        data = CMFData(cmf=simple_cmf, shift=zero_shift, cmf_name="test_cmf")
        dto = build_cmf_dto(data, [e])
        assert isinstance(dto, CmfDTO)
        assert dto.cmf_id == "test_cmf"
        assert dto.family_id == "1F1"
        assert dto.found_constants == [e.name]
        assert dto.cmf_hyperplanes == []
        # Shift is zero for all symbols
        assert all(v == 0 for v in dto.coordinate_shift)

    def test_build_shard_dto_matches_derive_ids(self, simple_shard):
        from dreamer.utils.storage.atlas_writer import build_shard_dto

        dto = build_shard_dto(simple_shard)
        expected_cmf_id, expected_shard_id, _ = derive_cmf_and_shard_ids(simple_shard)
        assert dto.shard_id == expected_shard_id
        assert dto.cmf_id == expected_cmf_id
        assert e.name in dto.found_constants
        # Interior point present (simple_shard fixture passes one in)
        assert dto.interior_point == (1, 1)
        # Encoding is the ±1 sign vector the shard was constructed with —
        # simple_shard uses encoding=[1, 1] (above both hyperplanes).
        assert dto.shard_encoding == (1, 1)

    def test_build_shard_dto_whole_space(self, whole_space_shard):
        from dreamer.utils.storage.atlas_writer import build_shard_dto

        dto = build_shard_dto(whole_space_shard)
        assert dto.shard_encoding == ()
        assert dto.dimensionality == len(whole_space_shard.symbols)

    def test_append_dtos_jsonl_writes_new(self, tmp_path):
        from dreamer.utils.storage.atlas_writer import append_dtos_jsonl

        path = str(tmp_path / "cmfs.jsonl")
        dtos = [
            CmfDTO(cmf_id="a", family_id="1F1", cmf_hyperplanes=[], coordinate_shift=(0,), found_constants=["e"]),
            CmfDTO(cmf_id="b", family_id="1F1", cmf_hyperplanes=[], coordinate_shift=(0,), found_constants=["e"]),
        ]
        written = append_dtos_jsonl(path, dtos, "cmf_id")
        assert written == 2
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_append_dtos_jsonl_skips_existing(self, tmp_path):
        """Idempotent rerun — same ids are not re-appended."""
        from dreamer.utils.storage.atlas_writer import append_dtos_jsonl

        path = str(tmp_path / "cmfs.jsonl")
        dto = CmfDTO(
            cmf_id="a", family_id="1F1", cmf_hyperplanes=[],
            coordinate_shift=(0,), found_constants=["e"],
        )
        # First write
        assert append_dtos_jsonl(path, [dto], "cmf_id") == 1
        # Second write with same id → 0 new records
        assert append_dtos_jsonl(path, [dto], "cmf_id") == 0
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_append_dtos_jsonl_appends_only_newcomers(self, tmp_path):
        """Mixed batch — only previously-unseen ids are appended."""
        from dreamer.utils.storage.atlas_writer import append_dtos_jsonl

        path = str(tmp_path / "cmfs.jsonl")
        dto_a = CmfDTO(cmf_id="a", family_id="1F1", cmf_hyperplanes=[], coordinate_shift=(0,), found_constants=["e"])
        dto_b = CmfDTO(cmf_id="b", family_id="1F1", cmf_hyperplanes=[], coordinate_shift=(0,), found_constants=["e"])

        append_dtos_jsonl(path, [dto_a], "cmf_id")
        # Second call has dto_a (existing) and dto_b (new); only dto_b appended.
        assert append_dtos_jsonl(path, [dto_a, dto_b], "cmf_id") == 1
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
        assert len(lines) == 2
        ids = {json.loads(ln)["cmf_id"] for ln in lines}
        assert ids == {"a", "b"}

    def test_write_cmf_records_creates_both_files(self, tmp_path, simple_cmf, zero_shift):
        """Loading-stage helper emits flat cmfs.jsonl + cmf_families.jsonl."""
        from dreamer.utils.storage.atlas_writer import write_cmf_records
        from dreamer.utils.types import CMFData

        data = CMFData(cmf=simple_cmf, shift=zero_shift, cmf_name="test_cmf")
        write_cmf_records(str(tmp_path), [data])

        assert (tmp_path / "cmfs.jsonl").exists()
        assert (tmp_path / "cmf_families.jsonl").exists()

        cmf_records = [
            json.loads(ln) for ln in (tmp_path / "cmfs.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        family_records = [
            json.loads(ln) for ln in (tmp_path / "cmf_families.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(cmf_records) == 1
        assert cmf_records[0]["cmf_id"] == "test_cmf"
        # found_constants is written empty initially (filled in post-analysis).
        assert cmf_records[0]["found_constants"] == []
        assert len(family_records) == 1
        assert family_records[0]["family_id"] == "1F1"

    def test_write_cmf_records_idempotent(self, tmp_path, simple_cmf, zero_shift):
        """Re-running the loading stage doesn't grow the JSONL files."""
        from dreamer.utils.storage.atlas_writer import write_cmf_records
        from dreamer.utils.types import CMFData

        data = CMFData(cmf=simple_cmf, shift=zero_shift, cmf_name="test_cmf")
        write_cmf_records(str(tmp_path), [data])
        write_cmf_records(str(tmp_path), [data])  # rerun

        cmf_lines = [ln for ln in (tmp_path / "cmfs.jsonl").read_text().splitlines() if ln.strip()]
        family_lines = [ln for ln in (tmp_path / "cmf_families.jsonl").read_text().splitlines() if ln.strip()]
        assert len(cmf_lines) == 1
        assert len(family_lines) == 1

    def test_write_shard_records_creates_file(self, tmp_path, simple_shard):
        """Extraction-stage helper emits flat ``<cmf>__shards.jsonl``."""
        from dreamer.utils.storage.atlas_writer import write_shard_records

        written = write_shard_records(
            str(tmp_path), simple_shard.cmf_name, [simple_shard]
        )
        assert written == 1

        files = list(tmp_path.glob("*__shards.jsonl"))
        assert len(files) == 1
        records = [json.loads(ln) for ln in files[0].read_text().splitlines() if ln.strip()]
        assert len(records) == 1
        assert records[0]["cmf_id"] == simple_shard.cmf_name

    def test_write_shard_records_idempotent(self, tmp_path, simple_shard):
        """Same shard written twice → file still has one record."""
        from dreamer.utils.storage.atlas_writer import write_shard_records

        write_shard_records(str(tmp_path), simple_shard.cmf_name, [simple_shard])
        # Second write with same shard → no growth
        new_written = write_shard_records(
            str(tmp_path), simple_shard.cmf_name, [simple_shard]
        )
        assert new_written == 0

        files = list(tmp_path.glob("*__shards.jsonl"))
        lines = [ln for ln in files[0].read_text().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_shard_dto_round_trip_through_jsonl(self, tmp_path, simple_shard):
        """ShardDTO survives JSONL serialise → parse → from_dict."""
        from dreamer.utils.storage.atlas_writer import write_shard_records

        write_shard_records(str(tmp_path), simple_shard.cmf_name, [simple_shard])
        path = next(tmp_path.glob("*__shards.jsonl"))
        record = json.loads(path.read_text().splitlines()[0])
        restored = ShardDTO.from_dict(record)
        assert restored.shard_id == derive_cmf_and_shard_ids(simple_shard)[1]
        assert restored.cmf_id == simple_shard.cmf_name

    def test_update_cmf_hyperplanes_populates_existing_record(
        self, tmp_path, simple_cmf, zero_shift, symbols,
    ):
        """After loading writes an empty-hyperplanes CmfDTO, the extraction
        backfill must populate ``cmf_hyperplanes`` on the same line.
        """
        from dreamer.utils.storage.atlas_writer import (
            write_cmf_records, update_cmf_hyperplanes,
        )
        from dreamer.utils.types import CMFData

        data = CMFData(cmf=simple_cmf, shift=zero_shift, cmf_name="cmf_a")
        write_cmf_records(str(tmp_path), [data])
        path = tmp_path / "cmfs.jsonl"
        record_before = json.loads(path.read_text().splitlines()[0])
        assert record_before["cmf_hyperplanes"] == []

        hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
        updated = update_cmf_hyperplanes(str(tmp_path), "cmf_a", hps)
        assert updated is True

        record_after = json.loads(path.read_text().splitlines()[0])
        assert len(record_after["cmf_hyperplanes"]) == 2
        # Other fields preserved.
        assert record_after["cmf_id"] == "cmf_a"
        assert record_after["family_id"] == record_before["family_id"]

    def test_update_cmf_hyperplanes_no_matching_record(
        self, tmp_path, simple_cmf, zero_shift, symbols,
    ):
        """Unknown cmf_name → no-op, returns False."""
        from dreamer.utils.storage.atlas_writer import (
            write_cmf_records, update_cmf_hyperplanes,
        )
        from dreamer.utils.types import CMFData

        data = CMFData(cmf=simple_cmf, shift=zero_shift, cmf_name="cmf_a")
        write_cmf_records(str(tmp_path), [data])

        hps = [Hyperplane(symbols[0], symbols)]
        updated = update_cmf_hyperplanes(str(tmp_path), "nope", hps)
        assert updated is False

    def test_update_cmf_hyperplanes_missing_file(self, tmp_path, symbols):
        """No cmfs.jsonl yet → returns False, no crash."""
        from dreamer.utils.storage.atlas_writer import update_cmf_hyperplanes

        hps = [Hyperplane(symbols[0], symbols)]
        updated = update_cmf_hyperplanes(str(tmp_path), "anything", hps)
        assert updated is False

    def test_cmf_name_does_not_contain_constant(self, simple_cmf):
        """``cmf_name`` must be constant-agnostic so a CMF can host more
        than one constant without spawning a fresh id.
        """
        from dreamer.loading.funcs.base_cmf import BaseCMF

        bc = BaseCMF(const=e, cmf_name="my_cmf", cmf=simple_cmf, shifts=[0, 0])
        assert "e" not in bc.cmf_name.split("__")
        assert bc.cmf_name.startswith("my_cmf")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

class TestSummaryWriter:
    """Tests for ``dreamer.utils.storage.summary``.

    Each test builds a tiny EXPORT_SEARCH_RESULTS tree by hand (so the test
    is independent of the rest of the pipeline) and then asserts on the
    rendered markdown.
    """

    @staticmethod
    def _write_jsonl(path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_returns_none_when_root_missing(self, tmp_path):
        from dreamer.utils.storage.summary import write_summary

        missing = tmp_path / "does_not_exist"
        out = write_summary(search_results_root=str(missing))
        assert out is None

    def test_empty_root_produces_minimal_report(self, tmp_path):
        """An empty search-results directory still produces a valid markdown file."""
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        root.mkdir()
        out = write_summary(search_results_root=str(root))
        assert out is not None
        text = (root / "summary.md").read_text()
        assert "Pipeline Summary" in text
        assert "nothing to summarise" in text.lower()

    def test_single_shard_renders_full_report(self, tmp_path):
        """A small synthetic JSONL renders all sections with correct stats.

        JSONL files now live flat under EXPORT_SEARCH_RESULTS (no constant
        subdir).  Per-constant attributes (delta_estimate, identified) are
        dicts keyed by constant name.
        """
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        cmf_id = "pFq_2_1_-1__0_0_0"
        shard_id = f"{cmf_id}__deadbeefdeadbeef"
        records = [
            {
                "trajectory_id": f"{shard_id}__aaaaaaaaaaaaaaaa",
                "cmf_id": cmf_id,
                "shard_id": shard_id,
                "start_point": [-3, 1, -1],
                "direction": [-1, 1, 0],
                "delta_estimate": {"log-2": 0.28},
                "identified": {"log-2": True},
            },
            {
                "trajectory_id": f"{shard_id}__bbbbbbbbbbbbbbbb",
                "cmf_id": cmf_id,
                "shard_id": shard_id,
                "start_point": [-3, 1, -1],
                "direction": [0, 1, 0],
                "delta_estimate": {"log-2": -0.5},
                "identified": {"log-2": True},
            },
            {
                "trajectory_id": f"{shard_id}__cccccccccccccccc",
                "cmf_id": cmf_id,
                "shard_id": shard_id,
                "start_point": [-3, 1, -1],
                "direction": [-1, 0, -1],
                # ``-inf`` is the documented non-convergence sentinel.
                "delta_estimate": {"log-2": float("-inf")},
                "identified": {"log-2": False},
            },
        ]
        # Flat layout: <root>/<shard_id>.jsonl
        self._write_jsonl(str(root / f"{shard_id}.jsonl"), records)

        out = write_summary(search_results_root=str(root))
        assert out is not None
        text = (root / "summary.md").read_text()

        # Header + overview
        assert "Pipeline Summary" in text
        assert "log-2" in text
        # Best-δ value rounded to 4 decimals
        assert "0.2800" in text
        # Per-CMF section appeared
        assert f"CMF `{cmf_id}`" in text
        # Per-shard table row: trajectories=3, identified=2, positive δ=1
        assert "| 3 | 2 | 1 |" in text
        # Best trajectory line carries start → direction from the winning record
        assert "`[-3, 1, -1]` → `[-1, 1, 0]`" in text
        # Overall-best block also names the start and direction
        assert "start point: `[-3, 1, -1]`" in text
        assert "direction: `[-1, 1, 0]`" in text

    def test_this_run_shards_filter_drops_orphan_files(self, tmp_path):
        """Stale JSONLs (from a previous run with different sampling) must
        be dropped when ``this_run_shards`` is supplied.

        JSONL files now live flat under EXPORT_SEARCH_RESULTS.
        """
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        cmf_id = "pFq_X"

        current_shards = {
            f"{cmf_id}__1111111111111111",
            f"{cmf_id}__2222222222222222",
        }
        orphan_shard = f"{cmf_id}__3333333333333333"

        for sid in list(current_shards) + [orphan_shard]:
            self._write_jsonl(
                str(root / f"{sid}.jsonl"),
                [{
                    "trajectory_id": f"{sid}__deadbeefdeadbeef",
                    "cmf_id": cmf_id, "shard_id": sid,
                    "start_point": [0, 0], "direction": [1, 0],
                    "delta_estimate": {"log-2": 0.5}, "identified": {"log-2": True},
                }],
            )

        # Without the filter — all three appear (legacy behaviour)
        write_summary(search_results_root=str(root))
        all_text = (root / "summary.md").read_text()
        assert orphan_shard.rsplit("__", 1)[-1][:18] in all_text

        # With the filter — only the two this-run shards appear
        write_summary(
            search_results_root=str(root),
            this_run_shards={"log-2": current_shards},
        )
        filtered_text = (root / "summary.md").read_text()
        assert orphan_shard.rsplit("__", 1)[-1][:18] not in filtered_text
        for sid in current_shards:
            assert sid.rsplit("__", 1)[-1][:18] in filtered_text
        # Overview shard count must be 2, not 3
        assert "| 2 |" in filtered_text
        assert "this run only" in filtered_text

    def test_this_run_shards_drops_unknown_constants(self, tmp_path):
        """A constant absent from ``this_run_shards`` must not leak into the summary.

        With the flat layout, both shards live in root/. The filter restricts
        which constant names and shard IDs are shown.
        """
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        # Old shard (pi) still on disk from a previous run.
        old_shard = "pFq_old__deadbeefdeadbeef"
        self._write_jsonl(
            str(root / f"{old_shard}.jsonl"),
            [{
                "trajectory_id": f"{old_shard}__aaaaaaaaaaaaaaaa",
                "cmf_id": "pFq_old", "shard_id": old_shard,
                "delta_estimate": {"pi": 0.9}, "identified": {"pi": True},
            }],
        )
        # Current run touched only "log-2".
        new_shard = "pFq_new__cafef00dcafef00d"
        self._write_jsonl(
            str(root / f"{new_shard}.jsonl"),
            [{
                "trajectory_id": f"{new_shard}__bbbbbbbbbbbbbbbb",
                "cmf_id": "pFq_new", "shard_id": new_shard,
                "delta_estimate": {"log-2": 0.4}, "identified": {"log-2": True},
            }],
        )

        write_summary(
            search_results_root=str(root),
            this_run_shards={"log-2": {new_shard}},
        )
        text = (root / "summary.md").read_text()
        assert "log-2" in text
        # The "pi" constant section must not appear — it's stale.
        assert "## pi" not in text

    def test_overall_best_picks_max_across_constants(self, tmp_path):
        """Best δ in the overview must aggregate across every constant.

        With the flat layout, each shard's JSONL has dict-valued delta.
        Different constants can appear in different shards.
        """
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"

        def make_shard(const_name: str, cmf_id: str, hash_suffix: str, delta: float):
            shard_id = f"{cmf_id}__{hash_suffix}"
            rec = {
                "trajectory_id": f"{shard_id}__deadbeefdeadbeef",
                "cmf_id": cmf_id,
                "shard_id": shard_id,
                "delta_estimate": {const_name: delta},
                "identified": {const_name: True},
            }
            self._write_jsonl(str(root / f"{shard_id}.jsonl"), [rec])

        make_shard("log-2", "pFq_A", "1111111111111111", 0.10)
        make_shard("pi",    "pFq_B", "2222222222222222", 0.50)
        make_shard("e",     "pFq_C", "3333333333333333", 0.30)

        out = write_summary(search_results_root=str(root))
        text = (root / "summary.md").read_text()

        # Each constant got a row
        for const in ("log-2", "pi", "e"):
            assert f"`{const}`" in text
        # Overall best comes from the "pi" shard
        assert "Overall best δ" in text
        assert "0.5000" in text
        assert "constant: `pi`" in text

    def test_empty_patch_lines_do_not_inflate_trajectory_counts(self, tmp_path):
        """Patches merge into base records — they must not be counted as new trajectories."""
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        cmf_id = "pFq_X"
        shard_id = f"{cmf_id}__deadbeefdeadbeef"
        tid = f"{shard_id}__cafef00dcafef00d"
        records = [
            {
                "trajectory_id": tid,
                "cmf_id": cmf_id, "shard_id": shard_id,
                "delta_estimate": {"log-2": 0.7},
                "identified": {"log-2": True},
            },
            {"trajectory_id": tid, "extended_metrics": {"gcd_slope": 0.5}},
            {"trajectory_id": tid, "extended_metrics": {"asymptotics": ["1"]}},
        ]
        self._write_jsonl(str(root / f"{shard_id}.jsonl"), records)

        write_summary(search_results_root=str(root))
        text = (root / "summary.md").read_text()
        # One unique trajectory_id → 1 trajectory, 1 identified, 1 positive δ
        assert "| 1 | 1 | 1 |" in text

    def test_summary_surfaces_shard_interior_point(self, tmp_path):
        """When the EXPORT_CMFS sidecar carries an interior_point, the
        per-shard table must render it under a 'Start point' column.

        JSONL lives flat under search_root; sidecar under cmfs_root.
        """
        from dreamer.utils.storage.summary import write_summary

        search_root = tmp_path / "search results"
        cmfs_root = tmp_path / "CMFs"
        cmf_id = "pFq_X"
        shard_id = f"{cmf_id}__deadbeefdeadbeef"
        tid = f"{shard_id}__cafef00dcafef00d"

        self._write_jsonl(
            str(search_root / f"{shard_id}.jsonl"),
            [{
                "trajectory_id": tid,
                "cmf_id": cmf_id, "shard_id": shard_id,
                "start_point": [7, -3], "direction": [1, 0],
                "delta_estimate": {"log-2": 0.42}, "identified": {"log-2": True},
            }],
        )
        # ShardDTO sidecar — only interior_point is what the summary reads.
        self._write_jsonl(
            str(cmfs_root / "log-2" / f"{cmf_id}__shards.jsonl"),
            [{
                "shard_id": shard_id,
                "cmf_id": cmf_id,
                "shard_encoding": [1, -1],
                "dimensionality": 2,
                "found_constants": ["log-2"],
                "interior_point": [7, -3],
            }],
        )

        write_summary(
            search_results_root=str(search_root),
            export_cmfs_root=str(cmfs_root),
        )
        text = (search_root / "summary.md").read_text()
        assert "Start point" in text  # header column rendered
        assert "`[7, -3]`" in text    # per-shard row carries the witness

    def test_summary_start_point_shows_dash_when_sidecar_missing(self, tmp_path):
        """Missing ShardDTO sidecar => column rendered as em-dash (no crash)."""
        from dreamer.utils.storage.summary import write_summary

        root = tmp_path / "search results"
        cmf_id = "pFq_Y"
        shard_id = f"{cmf_id}__1234123412341234"
        tid = f"{shard_id}__abcdabcdabcdabcd"
        self._write_jsonl(
            str(root / f"{shard_id}.jsonl"),
            [{
                "trajectory_id": tid,
                "cmf_id": cmf_id, "shard_id": shard_id,
                "delta_estimate": {"log-2": 0.1}, "identified": {"log-2": True},
            }],
        )
        # No export_cmfs_root passed => no sidecar to load.
        write_summary(search_results_root=str(root))
        text = (root / "summary.md").read_text()
        assert "Start point" in text
        # Row should contain an em-dash for the missing point.
        assert "| — |" in text
