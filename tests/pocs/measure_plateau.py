"""
Pick a good plateau-stopping default for the heuristic.  The current
default (plateau_ratio=1e-3) quits on the first low-yield batch and loses
the long, low-but-nonzero tail of cells.  Sweep plateau_ratio and watch
shards vs time, so we can relax the stop without paying for runaway rays.

Reference (pFq(3,2), D=5): exact truth = 489; ray-saturation = 476.

Run::  python examples/measure_plateau.py [p q]
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


def build_hps(p, q):
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    cmf = pFq(constant, p, q, 1).to_cmf()
    ex = ShardExtractor(constant, cmf)
    return [h.apply_shift(cmf.shift) for h in ex._extract_cmf_hps()]


def main() -> int:
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    hps = build_hps(p, q)
    print(f"pFq({p},{q})  ({len(hps)} hyperplanes)")
    print("shards vs plateau_ratio (num_rays=2M cap):\n")
    print(f"  {'plateau_ratio':>14}  {'shards':>7}  {'time':>7}")
    for ratio in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        ext = RayShootingExtractor(
            num_rays=2_000_000, batch_size=20_000, plateau_ratio=ratio, seed=0
        )
        t0 = time.perf_counter()
        res = ext.extract(hps)
        print(f"  {ratio:>14.0e}  {len(res):>7}  {time.perf_counter()-t0:>6.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
