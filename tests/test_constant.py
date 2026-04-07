"""Tests for the Constant registry and value computation.

Covers:
- Registration and retrieval of static and dynamic constants
- mpmath value computation (the fixed cached_property path)
- Arithmetic operations between constants
- Edge cases: re-registration, unknown constant lookup
"""
import mpmath
import sympy as sp
import pytest

from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the Constant registry between tests."""
    snapshot = dict(Constant.registry)
    yield
    Constant.registry.clear()
    Constant.registry.update(snapshot)


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------
class TestConstantRegistry:

    def test_static_constants_registered_on_import(self):
        """Importing dreamer registers the six static constants."""
        import dreamer  # noqa: F401 — triggers registration
        for name in ("e", "pi", "euler_gamma", "pi_squared", "catalan", "gompertz"):
            assert Constant.is_registered(name), f"{name} not registered"

    def test_dynamic_zeta_registration(self):
        from dreamer import zeta
        z3 = zeta(3)
        assert Constant.is_registered("zeta-3")
        assert z3 is Constant.get_constant("zeta-3")

    def test_dynamic_log_registration(self):
        from dreamer import log
        ln2 = log(2)
        assert Constant.is_registered("log-2")
        assert ln2 is Constant.get_constant("log-2")

    def test_dynamic_sqrt_registration(self):
        from dreamer import sqrt
        s2 = sqrt(2)
        assert Constant.is_registered("sqrt(2)")

    def test_idempotent_creation(self):
        """Creating the same constant twice returns the same registry entry."""
        from dreamer import zeta
        z3a = zeta(3)
        z3b = zeta(3)
        assert z3a.name == z3b.name

    def test_available_constants_returns_list(self):
        names = Constant.available_constants()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_unknown_constant_raises(self):
        with pytest.raises(KeyError):
            Constant.get_constant("nonexistent_constant_xyz")


# ---------------------------------------------------------------------------
# 2. Value computation
# ---------------------------------------------------------------------------
class TestConstantValues:

    @pytest.mark.parametrize("name,sympy_expr,prefix", [
        ("e", sp.E, "2.71828182845904"),
        ("pi", sp.pi, "3.14159265358979"),
        ("euler_gamma", sp.EulerGamma, "0.57721566490153"),
        ("catalan", sp.Catalan, "0.91596559417721"),
    ])
    def test_value_mpmath_matches_known_digits(self, name, sympy_expr, prefix):
        """value_mpmath must match the first 14 digits of known constants."""
        mpmath.mp.dps = 50
        c = Constant(f"_test_{name}", sympy_expr)
        val = c.value_mpmath
        val_str = mpmath.nstr(val, 20)
        assert val_str.startswith(prefix), f"{name}: {val_str} doesn't start with {prefix}"

    def test_explicit_mpmath_value_used_when_provided(self):
        """If an explicit mpmath value is given at construction, it should be used."""
        mpmath.mp.dps = 50
        explicit = mpmath.mpf("3.14")
        c = Constant("_test_explicit", sp.pi, value_mpmath=explicit)
        assert c.value_mpmath == explicit

    def test_value_mpmath_no_infinite_recursion(self):
        """The old code had infinite recursion in the cached_property; verify the fix."""
        mpmath.mp.dps = 50
        c = Constant("_test_no_recurse", sp.E)
        # This would have raised RecursionError before the fix
        val = c.value_mpmath
        assert mpmath.almosteq(val, mpmath.e, 1e-30)


# ---------------------------------------------------------------------------
# 3. Arithmetic
# ---------------------------------------------------------------------------
class TestConstantArithmetic:

    def test_mul_two_constants(self):
        a = Constant("_ta", sp.Integer(2))
        b = Constant("_tb", sp.Integer(3))
        c = a * b
        assert c.value_sympy == 6

    def test_mul_constant_and_int(self):
        a = Constant("_ta2", sp.Integer(5))
        c = a * 3
        assert c.value_sympy == 15

    def test_add_two_constants(self):
        a = Constant("_tadd1", sp.Integer(10))
        b = Constant("_tadd2", sp.Integer(7))
        c = a + b
        assert c.value_sympy == 17

    def test_sub_two_constants(self):
        a = Constant("_tsub1", sp.Integer(10))
        b = Constant("_tsub2", sp.Integer(3))
        c = a - b
        assert c.value_sympy == 7

    def test_unsupported_type_raises(self):
        a = Constant("_tbad", sp.Integer(1))
        with pytest.raises(TypeError):
            a * 3.14  # float not supported
