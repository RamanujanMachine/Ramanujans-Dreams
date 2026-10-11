"""
Disentangle the heuristic's D=5 coverage gap: is it UNDER-SAMPLING (the
baseline plateaus early but more rays would find the missing full-dim
cones) or STRUCTURAL (cones so thin that rays never realistically hit
them)?  And how does the baseline's ray cost compare to face-aligned's?

We sweep the baseline ray budget (plateau OFF) and watch the shard count
climb, comparing against the face-aligned-augmented total from the
prototype (490 on pFq(3,2)).

Run::  python examples/measure_ray_budget.py [p q]
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
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


def build(p, q):
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    cmf = pFq(constant, p, q, 1).to_cmf()
    ex = ShardExtractor(constant, cmf)
    sh = [h.apply_shift(cmf.shift) for h in ex._extract_cmf_hps()]
    return BaseExtractor.hyperplanes_to_matrix(sh)


def baseline_at_budget(A, c, total_rays, batch=20000, seed=0):
    sh = RayShootingExtractor(seed=seed)
    rng = np.random.default_rng(seed)
    out, D, shot = {}, A.shape[1], 0
    while shot < total_rays:
        b = min(batch, total_rays - shot)
        V = sh._sample_rays(rng, D, b)
        shot += b
        if V.shape[0] == 0:
            continue
        sh._collect_unique_cells_into(sh._shoot(V, A, c), A, c, out)
    return len(out)


def main() -> int:
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    A, c = build(p, q)
    print(f"pFq({p},{q})  N={A.shape[0]}, D={A.shape[1]}")
    print("baseline shard count vs ray budget (plateau OFF):\n")
    print(f"  {'rays':>12}  {'shards':>7}  {'time':>7}")
    for budget in (20_000, 100_000, 500_000, 2_000_000, 10_000_000):
        t0 = time.perf_counter()
        n = baseline_at_budget(A, c, budget)
        print(f"  {budget:>12,}  {n:>7}  {time.perf_counter()-t0:>6.1f}s")
    print("\n  (prototype: default-plateau baseline=431, +face-aligned=490)")
    print("  If the count climbs toward/past 490 -> under-sampling (just shoot")
    print("  more / relax the plateau). If it stalls well below -> the missing")
    print("  cones are effectively unreachable by rays and face-aligned earns")
    print("  its keep. Compare ray cost to face-aligned's ~32k shoots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
