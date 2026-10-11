"""Joint (common-denominator) δ: unit tests and the acceptance cases A–E.

Reference numbers come from the validated proof of concept
(``delta-conjecture-search/poc``, RT 0.0.8) and the RT_search_master kit
(``examples/expected_targets.json``).  Conventions: ``MeijerG(p, 2, p, p, 1)``,
W_N = M(0)…M(N−1) walked from n = 0, S_N = W_N^{-T} (walk type 1), readout
x_N = F · S_N e_j.  Joint δ is numerical evidence, not proof.

Cases needing the deepest walks (B at N = 500/1000) are opt-in:
``RD_RUN_SLOW=1 pytest tests/test_joint_delta.py``.
"""
import json
import os
from fractions import Fraction

import pytest
import sympy as sp
from flint import fmpq

from dreamer.utils.constants.constant import Constant, JointConstant, is_joint_value, joint_name
from dreamer.utils.constants.ready_made_consts import joint, pi, zeta
from dreamer.utils.joint.approximation import (
    EVIDENCE_LABEL, JointRow, JointSeries, apply_forms, casoratian, decimal_digits,
    primitive_vector, score_vector,
)
from dreamer.utils.joint.recognition import (
    JointForms, common_denominator_forms, direction, forms_hold, rational_dependencies,
)
from dreamer.utils.storage.optimization_objectives import get_objective, score_record

R = sp.Rational
SLOW = pytest.mark.skipif(not os.environ.get("RD_RUN_SLOW"), reason="slow; set RD_RUN_SLOW=1")


# ---------------------------------------------------------------------------
# Helpers: build a real handler through RD's own CMF path
# ---------------------------------------------------------------------------

def _handler(p, start, direction_, walk_depth=100):
    from ramanujantools import Position
    from ramanujantools.cmf import MeijerG
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

    axes = sp.symbols(f"a0:{p}") + sp.symbols(f"b0:{p}")
    return TrajectoryAttributesHandler.from_cmf(
        MeijerG(p, 2, p, p, 1), Position(dict(zip(axes, direction_))),
        Position(dict(zip(axes, start))), walk_depth=walk_depth, walk_type=1,
    )


def _joint_handler(target, *args, forms=None, **kw):
    h = _handler(*args, **kw)
    h._constant = target.value_sympy
    if forms is not None:
        h._cache["joint_forms"] = forms
    return h


CASE_B = (5, [0, 0, 1, 1, 2, 0, 0, 0, 0, 0], [-16, -17, 11, 12, 10, -2, -1, 0, 1, 2])
CASE_B_FORMS = JointForms(column=0, q_form=[0, 0, 6, 0],
                          p_forms={"zeta-3": [0, -3, 3, 0], "zeta-4": [2, -2, 0, 0]})
CASE_C = (5, [1, 1, 0, 0, R(5, 2), 0, 0, 0, 0, R(1, 2)], [13, 12, -16, -17, 8, 0, 1, 2, -1, -2])
CASE_D = (6, [R(3, 2), 1, 0, 0, 2, 1, 0, 0, 0, 0, R(1, 2), 0], [7, 12, -16, -17, 11, 4, 0, 1, 2, -1, -2, 1])
KIT_RAY = [3, 4, -5, -6, 4, 0, 1, 2, -1, -2]


@pytest.fixture(scope="module")
def case_b():
    return _joint_handler(joint(zeta(3), zeta(4)), *CASE_B, forms=CASE_B_FORMS)


# ---------------------------------------------------------------------------
# E. Unit level
# ---------------------------------------------------------------------------

class TestPrimitiveVector:
    def test_one_gcd_after_common_denominator(self):
        assert primitive_vector([fmpq(1, 6), fmpq(1, 4)]) == [2, 3]

    def test_accepts_fraction_sympy_and_int(self):
        assert primitive_vector([Fraction(1, 6), R(1, 4)]) == [2, 3]
        assert primitive_vector([4, 6, 10]) == [2, 3, 5]

    def test_joint_gcd_differs_from_per_coordinate_reduction(self):
        # Rule 2: (q, p1, p2) = (4, 2, 3) is already primitive jointly, although
        # reducing p1/q = 2/4 on its own would give the smaller denominator 2.
        assert primitive_vector([fmpq(4), fmpq(2), fmpq(3)]) == [4, 2, 3]

    def test_sign_normalised_on_first_nonzero(self):
        assert primitive_vector([0, -2, 4]) == [0, 1, -2]

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError):
            primitive_vector([0, fmpq(0)])

    def test_rejects_floats(self):
        with pytest.raises(TypeError):
            primitive_vector([0.5, 1])


class TestCasoratianAndForms:
    def test_dependent_set_is_zero(self):
        assert casoratian([[1, 2], [2, 4]]) == 0
        assert casoratian([[1, 2, 3], [4, 5, 6], [5, 7, 9]]) == 0

    def test_independent_set_nonzero(self):
        assert casoratian([[1, 0], [0, 1]]) == 1

    def test_casoratian_shape_guard(self):
        with pytest.raises(ValueError):
            casoratian([[1, 2, 3], [4, 5, 6]])

    def test_apply_forms_exact(self):
        col = [fmpq(1, 2), fmpq(1, 3)]
        assert apply_forms(col, [[2, 0], [0, 3]]) == [1, 1]
        with pytest.raises(ValueError):
            apply_forms(col, [[1, 2, 3]])

    def test_decimal_digits_without_str(self):
        assert [decimal_digits(x) for x in (0, 9, 10, -99, 10 ** 5000)] == [0, 1, 2, 2, 5001]

    def test_common_denominator_rescaling(self):
        q, ps = common_denominator_forms([[0, 2, 0], [0, 3, 0]], [[1, 0, 0], [0, 0, 1]])
        assert q == [0, 6, 0] and ps == [[3, 0, 0], [0, 0, 2]]

    def test_non_proportional_denominators_rejected(self):
        assert common_denominator_forms([[1, 0], [0, 1]], [[1, 1], [1, 1]]) is None

    def test_rational_dependency_detected(self):
        u = direction([fmpq(1), fmpq(2), fmpq(3)], 60)    # u3 = u1 + u2
        kept, deps = rational_dependencies(u, 60, 10 ** 4)
        assert len(kept) == 1 and len(deps) == 2

    def test_forms_roundtrip(self):
        assert JointForms.from_dict(json.loads(json.dumps(CASE_B_FORMS.as_dict()))) == CASE_B_FORMS


class TestScoring:
    def test_score_vector_matches_definition(self):
        # 22/7 against pi: delta = -log|pi - 22/7| / log 7 - 1
        import math
        row = score_vector(1, [7, 22], [sp.pi])
        expected = -math.log(abs(math.pi - 22 / 7)) / math.log(7) - 1
        assert row.delta == pytest.approx(expected, rel=1e-9)
        assert row.q_digits == 1

    def test_error_ball_must_exclude_zero(self):
        with pytest.raises(ArithmeticError):
            score_vector(1, [1, 1], [sp.Integer(1)])   # exact hit -> zero error

    def test_length_guard(self):
        with pytest.raises(ValueError):
            score_vector(1, [1, 2, 3], [sp.pi])

    def test_series_intervals_and_slope(self):
        s = JointSeries(["x"], [JointRow(1, 1.0, 2.0, [2.0], 1.0, 1), JointRow(2, 2.0, 4.0, [4.0], 1.0, 1),
                                JointRow(3, 3.0, 6.0, [6.0], 1.0, 1)])
        assert s.intervals == [1.0, 1.0] and s.ls_slope == pytest.approx(1.0)
        d = s.as_dict()
        assert d["note"] == EVIDENCE_LABEL and json.dumps(d)


# ---------------------------------------------------------------------------
# JointConstant target
# ---------------------------------------------------------------------------

class TestJointConstant:
    def test_name_value_and_registry_reuse(self):
        j = joint(zeta(3), zeta(4))
        assert j.name == "joint[zeta-3,zeta-4]" == joint_name(["zeta-3", "zeta-4"])
        assert j.member_names == ["zeta-3", "zeta-4"]
        assert is_joint_value(j.value_sympy) and not is_joint_value(zeta(3).value_sympy)
        assert joint(zeta(3), zeta(4)) is j
        assert Constant.get_constant(j.name) is j

    def test_order_is_part_of_the_key(self):
        assert joint(zeta(4), zeta(3)).name == "joint[zeta-4,zeta-3]"

    @pytest.mark.parametrize("members", [[zeta(3)], [zeta(3), zeta(3)]])
    def test_invalid_members(self, members):
        with pytest.raises(ValueError):
            JointConstant(members)

    def test_no_nesting_and_no_arithmetic(self):
        j = joint(zeta(2), pi)
        with pytest.raises(ValueError):
            JointConstant([j, zeta(3)])
        with pytest.raises(TypeError):
            j + 1

    def test_objective_registered(self):
        assert get_objective("joint_delta").direction == "max"


# ---------------------------------------------------------------------------
# A. Kit scalar controls (k = 1 through the joint machinery), to 1e-8
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
@pytest.mark.parametrize("start,target,expected", [
    ([1, 1, 0, 0, 2, 0, 0, 0, 0, 0], 1 - 2 * sp.zeta(3),
     {8: 0.16843812782846504, 32: 0.08409385361772036}),
    ([1, 1, 0, 0, R(5, 2), 0, 0, 0, 0, R(1, 2)], sp.pi ** 2 - R(34, 3),
     {32: 0.11512953539162162}),
])
def test_case_a_kit_controls(start, target, expected):
    h = _handler(5, start, KIT_RAY)
    for N, ref in expected.items():
        col = h._oriented_column(N, 1)
        v = primitive_vector(apply_forms(col, [[0, 0, 1, 0], [0, 1, 0, 0]]))
        assert score_vector(N, v, [target]).delta == pytest.approx(ref, abs=1e-8)


# ---------------------------------------------------------------------------
# B. Sheevy's {ζ(3), ζ(4)} construction
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
class TestCaseB:
    @pytest.mark.parametrize("N,ref", [(8, 0.031448), (32, 0.026528), (100, 0.019188)])
    def test_joint_delta(self, case_b, N, ref):
        assert case_b.joint_delta(N) == pytest.approx(ref, abs=1e-6)

    def test_scalar_readout_15z3_plus_z4_control(self, case_b):
        # Row (2B, -3A-2B, 3A, 0)/(0,0,6,0), A=15, B=1 -> 15ζ(3)+ζ(4) (Sheevy's scalar number at 1000).
        col = case_b._oriented_column(100, 0)
        v = primitive_vector(apply_forms(col, [[0, 0, 6, 0], [2, -47, 45, 0]]))
        assert score_vector(100, v, [15 * sp.zeta(3) + sp.zeta(4)]).delta > 0

    def test_recognition_recovers_forms(self):
        h = _joint_handler(joint(zeta(3), zeta(4)), *CASE_B)
        f = h.joint_forms()
        assert (f.column, f.q_form, f.p_forms) == (0, CASE_B_FORMS.q_form, CASE_B_FORMS.p_forms)
        assert all(r < -1000 for r in f.log10_residuals.values())
        assert h.identified() and h.delta() == float("-inf")   # never scalar δ / LIReC

    def test_casoratian_nonzero(self, case_b):
        res = case_b.joint_casoratian((20, 50, 100))
        assert all(r["nonzero"] for r in res.values())

    def test_series(self, case_b):
        s = case_b.joint_series()
        assert s["depths"][-1] == 100 and s["delta"][-1] == pytest.approx(0.019188, abs=1e-6)
        assert len(s["intervals"]) == len(s["depths"]) - 1 and s["note"] == EVIDENCE_LABEL

    def test_forms_hold_check(self, case_b):
        from dreamer.configs import config
        dps = 300
        col = case_b._oriented_column(config.search.JOINT_RECOGNITION_DEPTH, 0)
        targets = case_b._joint_targets()
        assert forms_hold(col, CASE_B_FORMS, targets, dps)
        wrong = JointForms(0, [0, 0, 6, 0], {"zeta-3": [0, -3, 4, 0], "zeta-4": [2, -2, 0, 0]})
        assert not forms_hold(col, wrong, targets, dps)

    @SLOW
    @pytest.mark.timeout(1800)
    @pytest.mark.parametrize("N,ref", [(500, 0.016162), (1000, 0.016794)])
    def test_joint_delta_deep(self, case_b, N, ref):
        assert case_b.joint_delta(N) == pytest.approx(ref, abs=1e-6)


# ---------------------------------------------------------------------------
# C. Rational dependency (paired-half field), D. three constants (MG66)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_case_c_dependency_and_joint_delta():
    h = _joint_handler(joint(zeta(2), zeta(3)), *CASE_C)
    f = h.joint_forms()
    assert f.dependencies == [{1: 9, 2: 18, 3: 8}]
    assert f.q_form == [0, 0, -18, 0]
    assert f.p_forms == {"zeta-2": [0, -3, -34, 0], "zeta-3": [-3, 8, -4, 0]}
    assert h.joint_delta(500) == pytest.approx(0.032686129242429727502, abs=1e-12)


@pytest.mark.timeout(600)
def test_case_d_three_constants_negative():
    h = _joint_handler(joint(zeta(2), zeta(3), zeta(4)), *CASE_D)
    f = h.joint_forms()
    assert f is not None and len(f.dependencies) == 1
    assert h.joint_delta(256) == pytest.approx(-0.1923, abs=1e-3)


# ---------------------------------------------------------------------------
# Pipeline integration: flat rows, reuse and seeding
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
class TestPipelineIntegration:
    def _row(self, constants):
        from dreamer.utils.storage.trajectory_attributes import build_trajectory_dtos
        h = _handler(*CASE_B)
        return h, build_trajectory_dtos(
            h, cmf_id="cmf", shard_id="shard", cmf_name="MG55", shard_encoding_str="enc",
            start=h._start_point, direction=h._trajectory, constants=constants,
        )

    def test_joint_row_is_flat_and_carries_forms(self):
        _, (dto,) = self._row([joint(zeta(3), zeta(4))])
        rec = json.loads(dto.to_json_line())
        assert rec["constant"] == "joint[zeta-3,zeta-4]"
        assert rec["identified"] is True and rec["p_vector"] is None and rec["q_vector"] is None
        assert rec["joint_forms"]["q_form"] == [0, 0, 6, 0]
        assert rec["joint_delta"] == pytest.approx(0.019188, abs=1e-6)
        assert "joint_forms" in rec["fingerprints"] and "joint_delta" in rec["fingerprints"]
        score, ided = score_record(rec, "joint_delta")
        assert ided and score == pytest.approx(rec["joint_delta"])

    def test_seed_restores_forms_without_recognition(self, monkeypatch):
        from dreamer.utils.joint import recognition
        _, (dto,) = self._row([joint(zeta(3), zeta(4))])
        rec = json.loads(dto.to_json_line())
        monkeypatch.setattr(recognition, "recognise_forms",
                            lambda *a, **k: pytest.fail("recognition must not rerun on a seeded row"))
        h = _handler(*CASE_B)
        _, _, _, ided, extra = h.compute_for_constant(
            joint(zeta(3), zeta(4)).value_sympy, sync_attributes=("joint_delta",), seed=rec,
        )
        assert ided and extra["joint_delta"] == pytest.approx(rec["joint_delta"])

    def test_shard_cache_reuses_forms(self, monkeypatch):
        from types import SimpleNamespace
        from dreamer.utils.joint import recognition
        target = joint(zeta(3), zeta(4))
        shard = SimpleNamespace(joint_forms_cache={})
        h1 = _joint_handler(target, *CASE_B)
        h1._searchable = shard
        assert h1.joint_forms() is not None and len(shard.joint_forms_cache) == 1
        monkeypatch.setattr(recognition, "recognise_forms",
                            lambda *a, **k: pytest.fail("cached forms should have been reused"))
        h2 = _joint_handler(target, *CASE_B)
        h2._searchable = shard
        assert h2.joint_forms() is shard.joint_forms_cache[("zeta-3", "zeta-4")][0]

    def test_scalar_rows_unaffected(self):
        h = _handler(*CASE_B)
        h._constant = zeta(3).value_sympy
        assert h.joint_forms() is None and h.joint_delta() == float("-inf")
        assert h.joint_series() is None and h.joint_casoratian() is None
