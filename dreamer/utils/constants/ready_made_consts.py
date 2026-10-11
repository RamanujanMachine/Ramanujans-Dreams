import sympy as sp
from .constant import Constant
from sympy.functions.special.hyper import hyper


e = Constant('e', sp.E)
pi = Constant('pi', sp.pi)
euler_gamma = Constant('euler_gamma', sp.EulerGamma)
pi_squared = Constant('pi_squared', sp.pi ** 2)
catalan = Constant('catalan', sp.Catalan)
gompertz = Constant('gompertz', -sp.exp(1) * sp.Ei(-1))

_ginzburg_num = hyper(
    (sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)),
    (sp.Rational(3, 2), sp.Rational(3, 2)),
    sp.Rational(1,4)
)
_calegary = (sp.Rational(4, 3) / sp.sqrt(3)) * _ginzburg_num
calegary = Constant('calagari', _calegary)


def zeta(n: int):
    if f'zeta-{n}' not in Constant.registry:
        return Constant(f'zeta-{n}', sp.zeta(n))
    return Constant.registry[f'zeta-{n}']


def sqrt(v: Constant | int | float):
    if isinstance(v, Constant):
        return Constant(f'sqrt({v.name})', sp.sqrt(v.value_sympy))
    if isinstance(v, float | int):
        return Constant(f'sqrt({v})', sp.sqrt(v))
    raise TypeError(f"Unsupported operand for sqrt: '{type(v)}'")


def power(v: Constant, n: int):
    return Constant(f'{v.name}^{n}', v.value_sympy ** n)


def log(n: int):
    if f'log-{n}' not in Constant.registry:
        return Constant(f'log-{n}', sp.log(n))
    return Constant.registry[f'log-{n}']
