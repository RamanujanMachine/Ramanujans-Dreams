import sympy as sp
import mpmath as mp
from typing import Union, Dict, Optional
from dreamer.utils.caching import cached_property


class Constant:
    """
    Registry DP for Constant management and converting between mpmath and sympy numbers.
    """
    registry: Dict[str, "Constant"] = dict()

    def __init__(
            self,
            name: str,
            value_sympy: sp.Expr,
            value_mpmath: Optional[Union[mp.mpf, mp.mpc]] = None
    ):
        """
        :param name: name of the constant
        :param value_sympy: sympy expression representing the constant
        :param value_mpmath: optional - mpmath expression representing the constant
            (it must be equivalent to the sympy one)
        """
        self.name = name
        self.value_sympy = value_sympy

        # Store explicit mpmath value; if not given, the cached_property below
        # will compute it lazily from the sympy expression.
        if value_mpmath is not None:
            self._explicit_mpmath = value_mpmath
        
        Constant.registry[self.name] = self

    @cached_property
    def value_mpmath(self):
        """
        :return: The mpmath value of the constant (computed once and cached).
        """
        if hasattr(self, '_explicit_mpmath'):
            return self._explicit_mpmath
        try:
            return sp.lambdify([], self.value_sympy, modules=['mpmath'])()
        except Exception as e:
            print(f"Warning: Could not auto-convert {self.name} to mpmath. Error: {e}")
            return mp.mpf(0)

    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}*{other.name}', self.value_sympy * other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}*{other}', self.value_sympy * other)
        raise TypeError(f"Unsupported operand for __mul__: '{type(other)}'")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}+{other.name}', self.value_sympy + other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}+{other}', self.value_sympy + other)
        raise TypeError(f"Unsupported operand for __add__: '{type(other)}'")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{self.name}-{other.name}', self.value_sympy - other.value_sympy)
        if isinstance(other, int):
            return Constant(f'{self.name}-{other}', self.value_sympy - other)
        raise TypeError(f"Unsupported operand for __sub__: '{type(other)}'")

    def __rsub__(self, other):
        if isinstance(other, Constant):
            return Constant(f'{other.name}-{self.name}', other.value_sympy - self.value_sympy)
        if isinstance(other, int):
            return Constant(f'{other}-{self.name}', other - self.value_sympy)
        raise TypeError(f"Unsupported operand for __rsub__: '{type(other)}'")

    @staticmethod
    def is_registered(name: str) -> bool:
        """
        Checks if a constant is already defined.
        :param name: Name of constant to check.
        :return: True if constant was already defined, false otherwise.
        """
        return name in Constant.registry

    @staticmethod
    def available_constants():
        """
        :return: A list of all defined constants in the registry
        """
        return list(Constant.registry.keys())

    @staticmethod
    def get_constant(name: str) -> 'Constant':
        """
        Retrieve a constant from the registry.
        :param name: Name of constant to retrieve
        :return: The constant
        """
        return Constant.registry[name]

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        # Stable, content-based identity.  The default ``object.__repr__`` embeds
        # the memory address (``<Constant object at 0x...>``), which differs every
        # process — anything deriving state from ``str(constant)`` (e.g. a
        # reproducible RNG seed) would then be nondeterministic across runs.
        return self.name

    def __str__(self):
        return self.name


class JointConstant(Constant):
    """A *set* of constants targeted jointly (one common denominator).

    Used for simultaneous rational approximation: a trajectory row for a joint
    target is scored by the joint (common-denominator) δ of its members rather
    than by scalar δ.  It is keyed like any constant, by a stable composite
    name ``joint[<m1>,<m2>,...]`` (members in the given order), so result rows
    stay flat with the composite key ``(trajectory_id, constant)``.

    ``value_sympy`` is the ``sympy.Tuple`` of the member values — the handler
    recognises a joint target by that type (it must never reach LIReC).
    """

    def __init__(self, members):
        """
        :param members: Two or more distinct :class:`Constant` objects.
        :raises ValueError: for fewer than two members, duplicates, or nested joints.
        """
        members = list(members)
        if len(members) < 2:
            raise ValueError("a joint target needs at least two constants")
        if any(isinstance(m, JointConstant) or not isinstance(m, Constant) for m in members):
            raise ValueError("joint members must be plain Constant objects")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joint members: {names}")
        self.members = members
        super().__init__(joint_name(names), sp.Tuple(*[m.value_sympy for m in members]))

    @property
    def member_names(self):
        """Member constant names, in order."""
        return [m.name for m in self.members]

    def member_values(self) -> dict:
        """``{member name: sympy value}`` in member order."""
        return {m.name: m.value_sympy for m in self.members}

    def __add__(self, other):
        raise TypeError("arithmetic is not defined on a joint target")

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__


def joint_name(names) -> str:
    """Stable composite name of a joint target, e.g. ``joint[zeta-3,zeta-4]``."""
    return "joint[" + ",".join(names) + "]"


def is_joint_value(value) -> bool:
    """Whether a handler-level constant value denotes a joint target."""
    return isinstance(value, sp.Tuple)
