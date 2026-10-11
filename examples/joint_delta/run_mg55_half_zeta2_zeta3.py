"""Joint δ for {ζ(2), ζ(3)} on the paired-half MG55 field (acceptance case C).

``MeijerG(m=5, n=2, p=5, q=5, z=1)``, start (1,1,0,0,5/2; 0,0,0,0,1/2),
direction (13,12,−16,−17,8; 0,1,2,−1,−2).  The column limit has a rational
dependency 9u₁ + 18u₂ + 8u₃ = 0 that recognition must detect and drop.
Expected: forms q = (0,0,−18,0), p_ζ2 = (0,−3,−34,0), p_ζ3 = (−3,8,−4,0);
joint δ_500 = 0.0326861292424297.  {1, ζ(2), ζ(3)} is a known theorem, so this
is a positive control, not a new result.

Usage (from the repository root, conda env ``rama``)::

    python examples/joint_delta/run_mg55_half_zeta2_zeta3.py --pipeline   # full System on the touching shard(s)

    python examples/joint_delta/run_mg55_half_zeta2_zeta3.py --depth 500
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
    ('case C (paired half)', (1, 1, 0, 0, '5/2', 0, 0, 0, 0, '1/2'), (13, 12, -16, -17, 8, 0, 1, 2, -1, -2)),
]
SHIFTS = [0, 0, 0, 0, sp.Rational(1, 2), 0, 0, 0, 0, sp.Rational(1, 2)]


def relative(start):
    """Selected start points are given relative to the formatter's shifts (absolute = point + shift)."""
    return tuple(sp.sympify(x) - s for x, s in zip(start, SHIFTS))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline', action='store_true',
                    help='full System (extraction, analysis, search) on the shard(s) the trajectory touches')
    ap.add_argument('--depth', type=int, default=None, help='walk depth (default: pipeline default)')
    ap.add_argument('--out', default='./joint_runs/mg55_half_zeta2_zeta3')
    args = ap.parse_args()
    if args.pipeline:
        run_pipeline(MeijerGSource(
            joint(zeta(2), zeta(3)), 5, 2, 5, 5, 1, shifts=SHIFTS,
            selected_start_points=[relative(start) for _, start, _ in CASES],
            selected_trajectories=[direction for _, _, direction in CASES],
            only_selected=True, use_inv_t=True,   # the formatter default is read at import time
        ), joint(zeta(2), zeta(3)), args.out)
    else:
        run(MeijerG(5, 2, 5, 5, 1), 'MeijerG_5_2_5_5_1_half', joint(zeta(2), zeta(3)), CASES, args.out, args.depth)
