"""User-specified **trajectory direction constraints** for the extraction stage.

A constraint is a map ``{variable_name: value}`` (e.g. ``{"x0": 12, "y1": 28}``) that
pins the *direction* of every sampled trajectory: "always take 12 steps in ``x0`` and 28
in ``y1``".  Because downstream length-scaling is allowed, only the **scale-invariant**
content is enforced — the **ratio** between the fixed coordinates and their **sign** — so
the constrained trajectory set stays a *homogeneous sub-cone* of the shard's recession
cone ``A·v ≤ 0`` (see ``context/algorithms/06_trajectory_constraints.md``).

For fixed coordinates ``S`` with integer values ``c_S`` the conditions are:

* **Ratio** (``|S| − 1`` equalities): for the anchor ``a = S[0]`` and each other
  ``b ∈ S``, ``c_a·v_b − c_b·v_a = 0`` (gcd-reduced).
* **Sign** (strict): ``sign(c_i)·v_i > 0`` for every ``i ∈ S`` — a direction with
  ``v_i = 0`` does not "take ``c_i`` steps" in coordinate ``i``.
* A fixed value of **0** is the degenerate equality ``v_i = 0`` (no sign row).

All conditions are homogeneous, so they are appended to the cone matrix ``A`` as extra
rows and conditioned by the *unchanged* :class:`HyperSpaceConditioner`; the **strict**
``v_i ≠ 0`` part (the closed cone admits the ``v_i = 0`` facet) is enforced by a cheap
post-harvest filter via :func:`fixed_sign_mask`.
"""

from __future__ import annotations

from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt


def get_trajectory_constraints() -> Optional[Dict[str, int]]:
    """Return the configured extraction-stage trajectory constraints (or ``None``).

    Read lazily from the global config so a run that sets
    ``config.extraction.TRAJECTORY_CONSTRAINTS`` is honoured by every stage that
    conditions a cone.  An empty dict is treated as "no constraints" (``None``).
    """
    from dreamer.configs import config

    constraints = getattr(config.extraction, "TRAJECTORY_CONSTRAINTS", None)
    return constraints or None


def resolve_constraint_indices(
    symbols: Sequence, constraints: Dict[str, int]
) -> Dict[int, int]:
    """Map ``{variable_name: value}`` to ``{column_index: value}`` using ``symbols``.

    :param symbols: Ordered shard symbols (``shard.symbols`` == the column order of ``A``).
    :param constraints: ``{name: value}`` over a subset of the symbol names.
    :return: ``{index: int(value)}``.
    :raises ValueError: if a constraint names a variable not in ``symbols``.
    """
    name_to_idx = {str(s): i for i, s in enumerate(symbols)}
    resolved: Dict[int, int] = {}
    for name, value in constraints.items():
        key = str(name)
        if key not in name_to_idx:
            raise ValueError(
                f"Unknown trajectory-constraint variable {name!r}. "
                f"Valid variables: {sorted(name_to_idx)}."
            )
        resolved[name_to_idx[key]] = int(value)
    return resolved


def constraint_rows(
    symbols: Sequence, constraints: Dict[str, int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build the homogeneous cone rows enforcing the direction constraints.

    :param symbols: Ordered shard symbols (column order of ``A``).
    :param constraints: ``{name: value}`` direction constraints.
    :return: ``(rows, fixed)`` where ``rows`` is a ``(k, d)`` float64 array (each row
        ``r`` means the wall ``r·v ≤ 0``; ratio equalities appear as paired ``±r`` rows,
        sign constraints as single rows) and ``fixed`` maps each constrained index to a
        sign in ``{+1, -1}`` (or ``0`` for a fixed value of ``0`` ⇒ ``v_i = 0`` equality).
    """
    d = len(symbols)
    resolved = resolve_constraint_indices(symbols, constraints)

    rows: List[np.ndarray] = []
    fixed: Dict[int, int] = {}

    zeros = [i for i, v in resolved.items() if v == 0]
    nonzero = sorted((i, v) for i, v in resolved.items() if v != 0)

    # Fixed value 0 ⇒ exact equality v_i = 0 (paired ±e_i, no sign).
    for i in zeros:
        e = np.zeros(d)
        e[i] = 1.0
        rows.append(e.copy())
        rows.append(-e)
        fixed[i] = 0

    if nonzero:
        anchor_i, anchor_c = nonzero[0]
        # Ratio equalities to the anchor: c_anchor·v_b − c_b·v_anchor = 0 (gcd-reduced).
        for b_i, b_c in nonzero[1:]:
            r = np.zeros(d)
            r[b_i] = float(anchor_c)
            r[anchor_i] = float(-b_c)
            g = gcd(int(abs(anchor_c)), int(abs(b_c))) or 1
            r /= g
            rows.append(r.copy())
            rows.append(-r)
        # Sign walls: sign(c_i)·v_i ≥ 0  ⇔  −sign(c_i)·e_i ≤ 0.
        for i, c in nonzero:
            s = 1 if c > 0 else -1
            e = np.zeros(d)
            e[i] = -float(s)
            rows.append(e)
            fixed[i] = s

    if rows:
        return np.vstack(rows).astype(np.float64), fixed
    return np.zeros((0, d), dtype=np.float64), fixed


def augment_cone(
    A: Optional[np.ndarray], symbols: Sequence, constraints: Optional[Dict[str, int]]
) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    """Append the constraint rows to a shard's cone matrix ``A``.

    :param A: ``(rows, d)`` recession-cone matrix, or ``None`` for a whole-space shard.
    :param symbols: Ordered shard symbols.
    :param constraints: ``{name: value}`` constraints, or ``None``/empty for no-op.
    :return: ``(A_aug, fixed)``.  With no constraints, ``A_aug is A`` (unchanged) and
        ``fixed`` is empty.  For a whole-space shard *with* constraints, ``A_aug`` is the
        constraint rows alone (so a cone sampler can replace the sphere sampler).
    """
    if not constraints:
        return A, {}
    rows, fixed = constraint_rows(symbols, constraints)
    if len(rows) == 0:
        return A, fixed
    if A is None or len(A) == 0:
        return rows, fixed
    return np.vstack([np.asarray(A, dtype=np.float64), rows]), fixed


def fixed_sign_mask(samples: np.ndarray, fixed: Dict[int, int]) -> np.ndarray:
    """Boolean mask of rows obeying the **strict** fixed-coordinate sign/non-zero rule.

    The closed augmented cone admits the ``v_i = 0`` facet, so this post-filter drops any
    sampled direction whose constrained coordinate is zero or has the wrong sign.

    :param samples: ``(n, d)`` integer directions.
    :param fixed: ``{index: sign}`` from :func:`constraint_rows`/:func:`augment_cone`.
    :return: ``(n,)`` boolean mask (all-``True`` when ``fixed`` is empty).
    """
    samples = np.asarray(samples)
    mask = np.ones(samples.shape[0], dtype=bool)
    for i, s in fixed.items():
        col = samples[:, i]
        if s == 0:
            mask &= col == 0
        else:
            mask &= np.sign(col) == s
    return mask


def constrained_cone_feasible(
    A: Optional[np.ndarray],
    symbols: Sequence,
    constraints: Dict[str, int],
    *,
    tol: float = 1e-7,
) -> bool:
    """Test whether the constrained sub-cone admits a non-trivial direction (LP).

    Solves ``max t`` over ``‖v‖_∞ ≤ 1`` subject to the cone ``A·v ≤ 0``, the constraint
    walls (ratio equalities + sign walls), and a **strict-margin** target so the optimum
    is positive only when a genuine interior direction exists:

    * with ≥1 non-zero fixed coordinate, ``sign(c_i)·v_i ≥ t`` for each (forces every
      fixed coord strictly into its half-space, away from ``0``);
    * with only ``v_i = 0`` constraints, ``A·v ≤ −t`` (a strict interior of the cone).

    :return: ``True`` iff the LP is feasible with optimum ``> tol``.
    """
    rows, fixed = constraint_rows(symbols, constraints)
    d = len(symbols)
    has_ineq = A is not None and len(A) > 0
    pos = [(i, s) for i, s in fixed.items() if s != 0]

    # Whole-space shard with no positivity target: any free direction works.
    if not has_ineq and not pos:
        return d - len(rows) // 2 >= 1

    A_ub_parts: List[np.ndarray] = []
    # Objective: minimise −t (maximise the strict margin t).
    c = np.zeros(d + 1)
    c[-1] = -1.0

    if has_ineq:
        Am = np.asarray(A, dtype=np.float64)
        A_ub_parts.append(np.hstack([Am, np.zeros((len(Am), 1))]))
    if len(rows):
        A_ub_parts.append(np.hstack([rows, np.zeros((len(rows), 1))]))

    if pos:
        for i, s in pos:
            row = np.zeros(d + 1)
            row[i] = -float(s)
            row[-1] = 1.0  # −sign·v_i + t ≤ 0  ⇔  sign·v_i ≥ t
            A_ub_parts.append(row[None, :])
    elif has_ineq:
        Am = np.asarray(A, dtype=np.float64)
        A_ub_parts.append(np.hstack([Am, np.ones((len(Am), 1))]))  # A·v + t ≤ 0

    A_ub = np.vstack(A_ub_parts)
    b_ub = np.zeros(A_ub.shape[0])
    bounds = [(-1.0, 1.0)] * d + [(0.0, None)]

    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return bool(res.success and res.fun is not None and (-res.fun) > tol)
