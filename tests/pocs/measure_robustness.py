"""
Robustness / anti-overfitting check for the heuristic's new plateau
defaults (plateau_ratio=1e-4, plateau_patience=3, num_rays=2M).

The defaults were calibrated on a few pFq cases; this confirms they
generalise to STRUCTURALLY DIFFERENT arrangements (other p,q and other z,
none of them 4F3(1) or 2F1(-1)).  For each CMF we report:
  - OLD defaults (ratio=1e-3, patience=1, 1M rays)  shards + time
  - NEW defaults                                    shards + time
  - exact ground-truth total (when enumeration finishes in the budget)
so we can see (a) new >= old everywhere, (b) time stays bounded, (c) new
is a high fraction of the true total where we can check it.

Run::  python examples/measure_robustness.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.base import BaseExtractor  # noqa: E402
from dreamer.extraction.v2 import cells  # noqa: E402
from dreamer.extraction.v2.milp import find_integer_point  # noqa: E402
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402

EXACT_DEADLINE = 150.0

# Diverse (p, q, z): different dimensions (D=p+q) and different z, none of
# them the cases the defaults were tuned on.
CASES = [
    (2, 1, 2),
    (2, 2, 1),
    (3, 1, 1),
    (4, 1, 1),
    (3, 2, 1),
]


def build(p, q, z):
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    cmf = pFq(constant, p, q, z).to_cmf()
    ex = ShardExtractor(constant, cmf)
    sh = [h.apply_shift(cmf.shift) for h in ex._extract_cmf_hps()]
    return BaseExtractor.hyperplanes_to_matrix(sh)


def run(ext, hps):
    t0 = time.perf_counter()
    n = len(ext.extract(hps))
    return n, time.perf_counter() - t0


def exact_total(A, c):
    try:
        all_cells = cells.enumerate_cells(A, c, seed=0,
                                          deadline=time.time() + EXACT_DEADLINE)
    except cells.ExtractionTimeout:
        return None
    is_unb = cells.make_unbounded_checker(A)
    tot = 0
    for s in all_cells:
        sv = np.asarray(s, dtype=np.int64)
        if is_unb(sv) and find_integer_point(A, c, sv) is not None:
            tot += 1
    return tot


def hps_for(p, q, z):
    A, c = build(p, q, z)
    # rebuild Hyperplane list for extract() (it takes Hyperplane objects)
    config.configure(extraction={'INIT_POINT_MAX_COORD': 3,
                                 'IGNORE_DUPLICATE_SEARCHABLES': False},
                     logging={'GENERATE_LOGS': False})
    cmf = pFq(zeta(2), p, q, z).to_cmf()
    ex = ShardExtractor(zeta(2), cmf)
    hps = [h.apply_shift(cmf.shift) for h in ex._extract_cmf_hps()]
    return hps, A, c


def main() -> int:
    print(f"{'CMF':>14} {'D':>2} {'N':>3} | {'old':>6} {'t_old':>6} | "
          f"{'new':>6} {'t_new':>6} | {'exact':>6} {'new/exact':>9}")
    print("-" * 86)
    for p, q, z in CASES:
        hps, A, c = hps_for(p, q, z)
        old_ext = RayShootingExtractor(num_rays=1_000_000, plateau_ratio=1e-3,
                                       plateau_patience=1, seed=0)
        new_ext = RayShootingExtractor(seed=0)  # NEW defaults
        old_n, old_t = run(old_ext, hps)
        new_n, new_t = run(new_ext, hps)
        tot = exact_total(A, c)
        tot_s = str(tot) if tot is not None else "timeout"
        frac = f"{100.0*new_n/tot:.1f}%" if tot else "-"
        name = f"pFq({p},{q},{z})"
        print(f"{name:>14} {p+q:>2} {A.shape[0]:>3} | {old_n:>6} {old_t:>5.1f}s | "
              f"{new_n:>6} {new_t:>5.1f}s | {tot_s:>6} {frac:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
