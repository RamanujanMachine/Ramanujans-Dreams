"""Joint δ for {ζ(3), ζ(4)} on the integer MG55 field ``MeijerG(m=5, n=2, p=5, q=5, z=1)``.

* ``case B``  — Sheevy's construction: start (0,0,1,1,2; 0^5),
  direction (−16,−17,11,12,10; −2,−1,0,1,2).  Expected: forms q = (0,0,6,0),
  p_ζ3 = (0,−3,3,0), p_ζ4 = (2,−2,0,0) on column 0; joint δ positive
  (≈ +0.0192 at N = 100, ≈ +0.0168 at N = 1000).
* ``kit ray`` — start (1,1,0,0,2; 0^5), direction (3,4,−5,−6,4; 0,1,2,−1,−2).
  The POC measured −0.14 at N = 500 on column 1; RD freezes the first column
  on which both constants are recognised, so this re-measures it.

Usage (from the repository root, conda env ``rama``)::

    python examples/joint_delta/run_mg55_integer_zeta3_zeta4.py --pipeline   # full System on the touching shard(s)

    python examples/joint_delta/run_mg55_integer_zeta3_zeta4.py            # pipeline walk depth
    python examples/joint_delta/run_mg55_integer_zeta3_zeta4.py --depth 500
"""
import argparse
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ramanujantools.cmf import MeijerG  # noqa: E402

from dreamer.utils.constants.ready_made_consts import joint, zeta  # noqa: E402
from _joint_run import run, run_pipeline  # noqa: E402
from dreamer.loading import MeijerG as MeijerGSource  # noqa: E402

CASES = [
    ('case B (Sheevy)', (0, 0, 1, 1, 2, 0, 0, 0, 0, 0), (-16, -17, 11, 12, 10, -2, -1, 0, 1, 2)),
    ('kit ray', (1, 1, 0, 0, 2, 0, 0, 0, 0, 0), (3, 4, -5, -6, 4, 0, 1, 2, -1, -2)),
]
SHIFTS = [0] * 10


def relative(start):
    """Selected start points are given relative to the formatter's shifts (absolute = point + shift)."""
    return tuple(start)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline', action='store_true',
                    help='full System (extraction, analysis, search) on the shard(s) the trajectory touches')
    ap.add_argument('--depth', type=int, default=None, help='walk depth (default: pipeline default)')
    ap.add_argument('--out', default='./joint_runs/mg55_integer_zeta3_zeta4')
    args = ap.parse_args()
    if args.pipeline:
        run_pipeline(MeijerGSource(
            joint(zeta(3), zeta(4)), 5, 2, 5, 5, 1, shifts=SHIFTS,
            selected_start_points=[relative(start) for _, start, _ in CASES],
            selected_trajectories=[direction for _, _, direction in CASES],
            only_selected=True, use_inv_t=True,   # the formatter default is read at import time
        ), joint(zeta(3), zeta(4)), args.out)
    else:
        run(MeijerG(5, 2, 5, 5, 1), 'MeijerG_5_2_5_5_1', joint(zeta(3), zeta(4)), CASES, args.out, args.depth)
