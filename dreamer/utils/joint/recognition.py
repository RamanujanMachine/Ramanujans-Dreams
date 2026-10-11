"""Recognise common-denominator joint forms on a CMF column (homogeneous PSLQ).

Ported from the validated proof of concept (``delta-conjecture-search/poc/recognize.py``).

The column S_N e_j converges projectively to a direction u whose entries are
typically *ratios* of linear forms in constants, so LIReC's scalar ``identify``
is not enough.  We look for homogeneous integer relations

    a · u + ξ (b · u) = 0      i.e.   ξ = -(a·u) / (b·u),   b ≠ 0,

so the readout for ξ is denominator form ``b`` and numerator form ``-a``.

Safeguards:

* **Rational dependencies** among the entries of u are found first and the
  dependent coordinates dropped; otherwise PSLQ returns dependency-only
  relations (case C of the acceptance tests).
* Every relation is **re-verified on an independent deeper walk** at twice the
  precision, with freshly evaluated (Arb) constant digits.
* A joint readout exists only when all denominator forms are **proportional**
  (:func:`common_denominator_forms`).  Non-proportional denominators would need
  a lattice intersection of the relation sets — not implemented; such a column
  is reported as not recognised.

A failed search is not evidence of independence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence

import mpmath as mp
import sympy as sp

from dreamer.utils.constants.high_precision import high_precision


@dataclass
class JointForms:
    """A frozen common-denominator readout: ``p_i·u / q·u → ξ_i`` on ``column``.

    Stored JSON-safe on the trajectory row (``joint_forms``) so a row reproduces
    the exact approximant sequence; the readout is never re-selected per depth.
    """
    column: int
    q_form: List[int]
    p_forms: Dict[str, List[int]]           # member constant name -> p-form
    dependencies: List[Dict[int, int]] = field(default_factory=list)
    log10_residuals: Dict[str, float] = field(default_factory=dict)
    found_depth: int = 0
    verify_depth: int = 0
    dps: int = 0

    def forms(self, names: Sequence[str]) -> List[List[int]]:
        """Form matrix F: q-form first, then the p-forms in *names* order."""
        return [list(self.q_form)] + [list(self.p_forms[n]) for n in names]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dependencies"] = [{str(k): v for k, v in dep.items()} for dep in self.dependencies]
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "JointForms":
        return cls(
            column=int(d["column"]),
            q_form=[int(x) for x in d["q_form"]],
            p_forms={k: [int(x) for x in v] for k, v in d["p_forms"].items()},
            dependencies=[{int(k): int(v) for k, v in dep.items()} for dep in d.get("dependencies", [])],
            log10_residuals={k: float(v) for k, v in d.get("log10_residuals", {}).items()},
            found_depth=int(d.get("found_depth", 0)),
            verify_depth=int(d.get("verify_depth", 0)),
            dps=int(d.get("dps", 0)),
        )


def direction(column: Sequence, dps: int) -> List[mp.mpf]:
    """Exact column -> mpf direction scaled to max-norm 1 (call inside nothing; sets dps)."""
    with mp.workdps(dps):
        u = [mp.mpf(int(x.p)) / int(x.q) if hasattr(x, "q") else mp.mpf(int(x)) for x in column]
        scale = max(abs(x) for x in u)
        if scale == 0:
            raise ValueError("zero column")
        return [x / scale for x in u]


def _value(expr: sp.Expr, dps: int) -> mp.mpf:
    """Independent Arb evaluation of *expr* as an mpf at the current mp precision."""
    return mp.mpf(high_precision(sp.sympify(expr), int(dps) + 20)._mpf_)


def _pslq(vec, dps: int, maxcoeff: int):
    return mp.pslq(vec, maxcoeff=maxcoeff, maxsteps=10 ** 6, tol=mp.mpf(10) ** (-(dps // 2)))


def rational_dependencies(u: Sequence[mp.mpf], dps: int, maxcoeff: int):
    """Repeatedly find integer relations among u alone and drop a coordinate per relation.

    :return: ``(kept_indices, dependencies)``; each dependency is ``{index: coeff}``.
    """
    kept = list(range(len(u)))
    deps: List[Dict[int, int]] = []
    with mp.workdps(dps):
        while len(kept) > 1:
            sub = [u[i] for i in kept]
            zeros = [k for k, x in enumerate(sub) if x == 0]
            if zeros:
                i0 = kept[zeros[0]]
                deps.append({i0: 1})
                kept.remove(i0)
                continue
            rel = _pslq(sub, dps, maxcoeff)
            if rel is None:
                break
            deps.append({kept[k]: int(c) for k, c in enumerate(rel) if c})
            drop = max((k for k, c in enumerate(rel) if c), key=lambda k: kept[k])
            kept.pop(drop)
    return kept, deps


def common_denominator_forms(q_forms: Sequence[Sequence[int]], p_forms: Sequence[Sequence[int]]):
    """Rescale proportional denominator forms to one integer q-form.

    :param q_forms: Denominator form b_i of each relation.
    :param p_forms: Numerator form of each relation (same order).
    :return: ``(q_form, [p_forms])`` with p_i·u / q·u = ξ_i, or ``None`` if the
        denominator forms are not all proportional.
    """
    base = list(q_forms[0])
    piv = next(i for i, x in enumerate(base) if x)
    scales = []
    for b in q_forms:
        if b[piv] == 0:
            return None
        s = Fraction(b[piv], base[piv])
        if any(Fraction(x) != s * y for x, y in zip(b, base)):
            return None
        scales.append(s)
    # p_i/q = (p_i'/s_i)/base: bring every p_i over a common integer multiple L of base.
    L = 1
    for s in scales:
        L = L * s.numerator // math.gcd(L, s.numerator)
    q = [x * L for x in base]
    ps = []
    for p, s in zip(p_forms, scales):
        row = [Fraction(x) * Fraction(L) / s for x in p]
        if any(x.denominator != 1 for x in row):
            return None
        ps.append([int(x) for x in row])
    return q, ps


def relation_log10_residual(u: Sequence[mp.mpf], q_form: Sequence[int], p_form: Sequence[int],
                            value: mp.mpf, dps: int) -> float:
    """log10 |p·u - ξ q·u| for a max-norm-1 direction u (−dps when exactly 0)."""
    with mp.workdps(dps):
        res = abs(sum(int(c) * x for c, x in zip(p_form, u)) - value * sum(int(c) * x for c, x in zip(q_form, u)))
        return float(mp.log10(res)) if res else -float(dps)


def recognise_forms(find_column: Sequence, verify_column: Sequence, column: int,
                    targets: Mapping[str, sp.Expr], dps: int, maxcoeff: int,
                    found_depth: int = 0, verify_depth: int = 0) -> Optional[JointForms]:
    """Dependency-aware recognition of a common-denominator readout for **all** targets.

    :param find_column: Exact column at the recognition depth.
    :param verify_column: Exact column of the same index at a deeper, independent depth.
    :param column: The column index (stored with the forms).
    :param targets: ``{name: sympy value}`` of the member constants.
    :param dps: PSLQ precision; verification runs at ``2·dps``.
    :param maxcoeff: PSLQ coefficient bound.
    :return: Frozen :class:`JointForms`, or ``None`` when any target is not
        recognised and verified, or the denominators are not proportional.
    """
    u = direction(find_column, dps)
    w = direction(verify_column, 2 * dps)
    d = len(u)
    kept, deps = rational_dependencies(u, dps, maxcoeff)
    q_forms, p_forms, residuals = [], [], {}
    for name, expr in targets.items():
        with mp.workdps(dps):
            xi = _value(expr, dps)
            us = [u[i] for i in kept]
            rel = _pslq(us + [xi * x for x in us], dps, maxcoeff)
        if rel is None or not any(rel[len(kept):]):
            return None
        a, b = [0] * d, [0] * d
        for k, i in enumerate(kept):
            a[i], b[i] = int(rel[k]), int(rel[len(kept) + k])
        p = [-x for x in a]
        with mp.workdps(2 * dps):
            lres = relation_log10_residual(w, b, p, _value(expr, 2 * dps), 2 * dps)
        if not lres < -0.9 * dps:
            return None
        q_forms.append(b)
        p_forms.append(p)
        residuals[name] = lres
    cd = common_denominator_forms(q_forms, p_forms)
    if cd is None:
        return None
    q, ps = cd
    return JointForms(column=column, q_form=q, p_forms=dict(zip(targets, ps)), dependencies=deps,
                      log10_residuals=residuals, found_depth=found_depth, verify_depth=verify_depth,
                      dps=dps)


def forms_hold(column_values: Sequence, forms: JointForms, targets: Mapping[str, sp.Expr],
               dps: int) -> bool:
    """Cheap per-trajectory check that frozen *forms* still read the targets.

    Used to reuse forms recognised on another trajectory of the same shard: every
    relation's residual on this column must be below ``10^(-0.9·dps)``.
    """
    u = direction(column_values, dps)
    with mp.workdps(dps):
        if sum(int(c) * x for c, x in zip(forms.q_form, u)) == 0:
            return False
        for name, expr in targets.items():
            if name not in forms.p_forms:
                return False
            lres = relation_log10_residual(u, forms.q_form, forms.p_forms[name], _value(expr, dps), dps)
            if not lres < -0.9 * dps:
                return False
    return True
