"""Joint (common-denominator) δ: exact primitive vectors and their scores.

Ported from the validated proof of concept (``delta-conjecture-search/poc/joint.py``).

A *joint readout* is an integer form matrix F with k+1 rows over the state
vector: row 0 is the denominator form (q-form), rows 1..k the numerator forms
(p-forms).  On a fixed column j of the oriented walk S_N it gives the rational
vector x_N = F · S_N e_j.  All denominators are cleared and the vector is
divided by **one** gcd of the whole vector, giving the primitive integer vector

    v_N = (q_N, p_{1,N}, ..., p_{k,N}).

It is scored against independently evaluated targets ξ_i (Arb balls)::

    H_N = log |q_N|,   E_N = -log max_i |ξ_i - p_{i,N}/q_N|,   δ_N = E_N / H_N - 1.

k = 1 is the ordinary scalar δ.  Approximants are exact (python-flint
``fmpq``/``fmpz``); floats are used only for the logarithms.

Sustained δ_N > 0 together with a non-zero Casoratian is *numerical evidence*
for the Q-linear independence of 1, ξ_1..ξ_k — not a proof.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

import sympy as sp

#: Label attached to every stored joint-δ output (rule: evidence, not proof).
EVIDENCE_LABEL = "numerical evidence, not proof"


def primitive_vector(values: Sequence) -> List[int]:
    """Clear all denominators, then divide by the gcd of the **whole** vector.

    :param values: Rationals (``fmpq``, ``Fraction``, sympy ``Rational`` or int).
    :return: Primitive integer vector, sign-normalised so the first non-zero
        entry is positive.
    :raises ValueError: for the zero vector.
    """
    pairs = [_as_pq(x) for x in values]
    lcm = 1
    for _, q in pairs:
        lcm = lcm * q // math.gcd(lcm, q)
    ints = [p * (lcm // q) for p, q in pairs]
    g = 0
    for v in ints:
        g = math.gcd(g, v)
    if g == 0:
        raise ValueError("zero vector")
    lead = next(v for v in ints if v)
    if lead < 0:
        g = -g
    return [v // g for v in ints]


def _as_pq(x) -> tuple:
    """``(numerator, denominator)`` of an exact rational, denominator > 0."""
    if isinstance(x, int):
        return x, 1
    if hasattr(x, "p") and hasattr(x, "q"):          # fmpq, sympy Rational
        return int(x.p), int(x.q)
    if hasattr(x, "numerator"):                      # fractions.Fraction
        return int(x.numerator), int(x.denominator)
    raise TypeError(f"not an exact rational: {x!r} ({type(x).__name__})")


def apply_forms(column: Sequence, forms: Sequence[Sequence[int]]) -> List:
    """x = F · u for an exact column u (one entry per form row).

    :raises ValueError: if a form's length differs from the state dimension.
    """
    d = len(column)
    out = []
    for row in forms:
        if len(row) != d:
            raise ValueError(f"form length {len(row)} does not match state dimension {d}")
        acc = 0
        for c, x in zip(row, column):
            if c:
                acc = acc + int(c) * x
        out.append(acc)
    return out


@dataclass
class JointRow:
    """One scored depth."""
    N: int
    H: float                 # log q_N of the primitive joint vector
    E: float                 # -log of the max coordinate error
    E_coords: List[float]    # -log error of each coordinate
    delta: float             # E/H - 1
    q_digits: int


@dataclass
class JointSeries:
    """Joint δ at several depths, with interval scores and the LS slope."""
    labels: List[str]
    rows: List[JointRow] = field(default_factory=list)

    @property
    def intervals(self) -> List[float]:
        """Successive interval scores (E_b - E_a) / (H_b - H_a) - 1."""
        return [
            (b.E - a.E) / (b.H - a.H) - 1 if b.H > a.H else float("nan")
            for a, b in zip(self.rows, self.rows[1:])
        ]

    @property
    def ls_slope(self) -> float:
        """Least-squares slope of E against H, minus 1 (``nan`` for < 2 rows)."""
        if len(self.rows) < 2:
            return float("nan")
        xs = [r.H for r in self.rows]
        ys = [r.E for r in self.rows]
        xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - xm) ** 2 for x in xs)
        if den == 0:
            return float("nan")
        return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den - 1

    def as_dict(self) -> dict:
        """JSON-safe summary for storage."""
        return {
            "labels": list(self.labels),
            "depths": [r.N for r in self.rows],
            "delta": [r.delta for r in self.rows],
            "E_coords_per_step": [[e / r.N for e in r.E_coords] for r in self.rows],
            "q_digits": [r.q_digits for r in self.rows],
            "intervals": self.intervals,
            "ls_slope": self.ls_slope,
            "note": EVIDENCE_LABEL,
        }


def decimal_digits(x: int) -> int:
    """Number of decimal digits of ``|x|`` (0 for 0) without ``str()``.

    ``str()`` of an int above 4300 digits raises under Python's default
    ``int_max_str_digits`` limit, and joint denominators reach ~10^5 digits.
    """
    x = abs(int(x))
    if x == 0:
        return 0
    d = int(x.bit_length() * 0.30102999566398120)   # floor(log10) or one less
    return d + 1 if x >= 10 ** d else d


def _log(x) -> float:
    """Natural log of a positive Arb ball, refusing balls that contain zero."""
    if x.contains(0):
        raise ArithmeticError("error ball contains zero: increase precision")
    return float(x.log().mid())


def score_vector(N: int, v: Sequence[int], targets: Sequence[sp.Expr]) -> JointRow:
    """Score one primitive vector (q, p_1..p_k) against sympy targets.

    Targets are evaluated with Arb at ≥ 3·digits(q) + 60 digits, which resolves
    any error down to q^-3 (rule 4); the error ball must exclude zero.
    """
    from flint import arb, ctx, fmpq
    from dreamer.utils.constants.high_precision import _to_arb

    if len(v) != len(targets) + 1:
        raise ValueError("vector length must be number of targets + 1")
    q = abs(int(v[0]))
    if q == 0:
        raise ZeroDivisionError("joint denominator q_N is zero")
    digits = decimal_digits(q)
    old = ctx.dps
    try:
        ctx.dps = max(old, 3 * digits + 60)
        errs = [abs(_to_arb(t) - arb(fmpq(int(p), int(v[0])))) for t, p in zip(targets, v[1:])]
        E_coords = [-_log(e) for e in errs]
    finally:
        ctx.dps = old
    H = math.log(q)
    E = min(E_coords)
    return JointRow(N=N, H=H, E=E, E_coords=E_coords, delta=E / H - 1 if H > 0 else float("nan"),
                    q_digits=digits)


def score_vectors(vectors: Mapping[int, Sequence[int]], targets: Sequence[sp.Expr],
                  labels: Sequence[str]) -> JointSeries:
    """Score primitive vectors ``{N: (q, p_1..p_k)}`` in depth order."""
    series = JointSeries(list(labels))
    for N in sorted(vectors):
        series.rows.append(score_vector(N, vectors[N], targets))
    return series


def casoratian(vectors: Sequence[Sequence[int]]) -> int:
    """Exact determinant of k+1 consecutive primitive vectors (each of length k+1)."""
    from flint import fmpz_mat

    m = len(vectors)
    if m == 0 or any(len(v) != m for v in vectors):
        raise ValueError("need k+1 vectors of length k+1")
    return int(fmpz_mat([[int(x) for x in v] for v in vectors]).det())
