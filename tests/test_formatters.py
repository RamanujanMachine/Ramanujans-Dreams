"""Tests for the CMF formatter classes (pFq, MeijerG, BaseCMF).

Covers:
- JSON round-trip serialization for pFq and MeijerG
- CMF creation: correct dimension, symbol count
- Formatter registry completeness
- Input validation (bad p/q values, wrong shift length)
- MeijerG super().__init__ fix verification
"""
import pytest
import sympy as sp

from dreamer.loading.funcs.formatter import Formatter
from dreamer.loading.funcs.pFq_fmt import pFq
from dreamer.loading.funcs.meijerG_fmt import MeijerG
from dreamer.loading.funcs.base_cmf import BaseCMF
from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def log2():
    """Return or create the log-2 constant."""
    from dreamer import log
    return log(2)


@pytest.fixture
def const_e():
    from dreamer import e
    return e


# ---------------------------------------------------------------------------
# 1. Formatter registry
# ---------------------------------------------------------------------------
class TestFormatterRegistry:

    def test_all_formatters_registered(self):
        """pFq, MeijerG, and BaseCMF must be registered on import."""
        for name in ("pFq", "MeijerG", "BaseCMF"):
            assert name in Formatter.registry, f"{name} missing from registry"

    def test_fetch_from_registry(self):
        cls = Formatter.fetch_from_registry("pFq")
        assert cls is pFq

    def test_fetch_unknown_raises(self):
        with pytest.raises(KeyError):
            Formatter.fetch_from_registry("UnknownFormatter")


# ---------------------------------------------------------------------------
# 2. pFq
# ---------------------------------------------------------------------------
class TestpFq:

    def test_creation_2f1(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        assert fmt.p == 2
        assert fmt.q == 1
        assert fmt.const == "log-2"
        assert len(fmt.shifts) == 3  # p + q

    def test_default_shifts(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        assert fmt.shifts == [0, 0, 0]

    def test_custom_shifts(self, log2):
        fmt = pFq(log2, 2, 1, -1, shifts=[0, sp.Rational(1, 2), 0])
        assert fmt.shifts[1] == sp.Rational(1, 2)

    def test_json_roundtrip(self, log2):
        original = pFq(log2, 2, 1, -1)
        json_obj = original.to_json_obj()
        restored = Formatter.from_json_obj(json_obj)
        assert isinstance(restored, pFq)
        assert restored.p == original.p
        assert restored.q == original.q
        assert restored.const == original.const

    def test_to_cmf_dimension(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        shift_cmf = fmt.to_cmf()
        assert shift_cmf.cmf.dim() == 3  # 2F1 → 3 symbols

    def test_to_cmf_has_matrices(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        shift_cmf = fmt.to_cmf()
        assert len(shift_cmf.cmf.matrices) == 3

    def test_invalid_p_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 0, 1, -1)

    def test_invalid_q_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 2, 0, -1)

    def test_wrong_shift_length_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 2, 1, -1, shifts=[0, 0])  # needs 3, given 2


# ---------------------------------------------------------------------------
# 3. MeijerG
# ---------------------------------------------------------------------------
class TestMeijerG:

    def test_creation(self, const_e):
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        assert fmt.m == 1
        assert fmt.n == 1
        assert fmt.p == 1
        assert fmt.q == 2

    def test_default_shifts_length(self, const_e):
        """Shifts should default to [0]*(p+q)."""
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        assert len(fmt.shifts) == 3  # p + q = 1 + 2

    def test_super_init_passes_correct_args(self, const_e):
        """After the fix, Formatter.__init__ should receive shifts as a list, not a bool."""
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        # The parent class (Formatter) stores shifts — check it's a list
        assert isinstance(fmt.shifts, list), (
            f"shifts should be list, got {type(fmt.shifts)}"
        )

    def test_json_roundtrip(self, const_e):
        original = MeijerG(const_e, 1, 1, 1, 2, 1)
        json_obj = original.to_json_obj()
        restored = Formatter.from_json_obj(json_obj)
        assert isinstance(restored, MeijerG)
        assert restored.m == original.m
        assert restored.p == original.p
        assert restored.q == original.q

    def test_to_cmf(self, const_e):
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        shift_cmf = fmt.to_cmf()
        # MeijerG(1,1,1,2) has p+q = 3 symbols
        assert shift_cmf.cmf.dim() == 3

    def test_invalid_pq_raises(self, const_e):
        with pytest.raises(ValueError):
            MeijerG(const_e, 1, 1, 0, 2, 1)


# ---------------------------------------------------------------------------
# 4. BaseCMF
# ---------------------------------------------------------------------------
class TestBaseCMF:

    def test_creation_from_raw_cmf(self, log2):
        from ramanujantools.cmf import pFq as rt_pFq
        raw_cmf = rt_pFq(2, 1, -1)
        fmt = BaseCMF(log2, raw_cmf)
        assert fmt.cmf is raw_cmf

    def test_json_roundtrip_not_supported(self, log2):
        """BaseCMF wraps arbitrary CMFs; round-trip may have limitations but shouldn't crash."""
        from ramanujantools.cmf import pFq as rt_pFq
        raw_cmf = rt_pFq(2, 1, -1)
        fmt = BaseCMF(log2, raw_cmf)
        json_obj = fmt.to_json_obj()
        assert json_obj is not None
