"""
Step 0 of the heuristic coverage plan: does the baseline ray-shooting
heuristic actually MISS integer-containing unbounded cells, and if so, are
the missed cells the predicted lower-dimensional-recession ("tube/slab")
cells?

Method (needs a CMF small enough for exact to finish, for ground truth):
  1. exact: enumerate ALL cells, keep the unbounded ones that contain an
     integer point (the same find_integer_point the pipeline uses) -> the
     ground-truth shard set.
  2. baseline heuristic: generic random rays from the origin.
  3. miss set = exact_shards \\ heuristic_shards.
  4. classify each missed cell by its recession-cone interior margin:
         max t  s.t.  s_i (A_i . d) >= t  for all i,  d in [-1,1]^D
     t > 0  => FULL-dimensional recession cone (heuristic *should* find it;
              a miss here is just finite-ray-budget bad luck).
     t == 0 => LOWER-dimensional recession cone (tube/slab) -> the heuristic
              *structurally* cannot find it with generic rays.

Run::  python examples/measure_heuristic_coverage.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

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


# (p, q) from argv (default 3 1).  D = p + q.  Want the largest D where exact
# still finishes within EXACT_DEADLINE.  D=3 (pFq(2,1)) is too easy.
P = int(sys.argv[1]) if len(sys.argv) > 1 else 3
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1
EXACT_DEADLINE = 600.0  # seconds for the full exact enumeration


def build_arrangement(p, q):
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    formatter = pFq(constant, p, q, 1)
    cmf_data = formatter.to_cmf()
    extractor = ShardExtractor(constant, cmf_data)
    hps = extractor._extract_cmf_hps()
    shifted = [hp.apply_shift(cmf_data.shift) for hp in hps]
    A, c = BaseExtractor.hyperplanes_to_matrix(shifted)
    return A, c, cmf_data.cmf_name


def recession_interior_margin(A, sign):
    """max t s.t. s_i (A_i . d) >= t for all i, d in [-1,1]^D.  Returns t*."""
    A = np.asarray(A, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    n, d = A.shape
    sA = s[:, None] * A                      # rows s_i A_i
    # vars = [d_1..d_D, t]; maximise t == minimise -t
    obj = np.zeros(d + 1); obj[-1] = -1.0
    # s_i A_i . d - t >= 0  ->  -(s_i A_i).d + t <= 0
    A_ub = np.hstack([-sA, np.ones((n, 1))])
    b_ub = np.zeros(n)
    bounds = [(-1.0, 1.0)] * d + [(0.0, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return float(res.x[-1]) if res.success else 0.0


def main() -> int:
    A, c, name = build_arrangement(P, Q)
    n, d = A.shape
    print(f"CMF: {name}   N={n}, D={d}\n")

    # --- 1. Exact ground truth (must finish for this to be valid). ---
    print(f"running exact (deadline {EXACT_DEADLINE:.0f}s)...", flush=True)
    t0 = time.perf_counter()
    try:
        all_cells = cells.enumerate_cells(
            A, c, seed=0, deadline=time.time() + EXACT_DEADLINE
        )
    except cells.ExtractionTimeout as exc:
        print(f"  exact DID NOT FINISH ({exc}); pick a smaller (p,q) for "
              "ground truth.")
        return 1
    is_unbounded = cells.make_unbounded_checker(A)
    exact_shards = set()
    for s in all_cells:
        sv = np.asarray(s, dtype=np.int64)
        if is_unbounded(sv) and find_integer_point(A, c, sv) is not None:
            exact_shards.add(s)
    print(f"  exact finished in {time.perf_counter()-t0:.1f}s: "
          f"{len(all_cells)} cells, {len(exact_shards)} integer-containing "
          f"unbounded shards (ground truth)\n")

    # --- 2. Baseline heuristic (generic rays from origin). ---
    heur_shards = _baseline_shards(A, c)
    print(f"baseline heuristic found {len(heur_shards)} shards")

    # --- 3. Miss set. ---
    missed = exact_shards - heur_shards
    spurious = heur_shards - exact_shards   # should be 0 (heuristic is sound)
    print(f"  missed by heuristic : {len(missed)} / {len(exact_shards)}")
    print(f"  heuristic spurious  : {len(spurious)} (should be 0)\n")

    # --- 4. Classify missed cells by recession-cone dimension. ---
    structural = sampling = 0
    for s in missed:
        t = recession_interior_margin(A, np.asarray(s, dtype=np.int64))
        if t > 1e-7:
            sampling += 1       # full-dim cone: missed by finite-ray luck
        else:
            structural += 1     # low-dim cone: structurally unreachable
    print("missed cells by recession-cone type:")
    print(f"  LOW-dim recession (structural miss, tube/slab) : {structural}")
    print(f"  FULL-dim recession (just finite-ray budget)    : {sampling}")
    print("\nverdict:")
    if not missed:
        print("  baseline already complete here -> coverage not a problem at "
              "this size; try a larger D.")
    elif structural > 0:
        print(f"  CONFIRMED: {structural} integer-containing shards are "
              "structurally missed (low-dim recession). Face-aligned shooting "
              "targets exactly these.")
    else:
        print("  all misses are full-dim cones -> just need more/smarter rays, "
              "not face-aligned shooting.")
    return 0


def _baseline_shards(A, c):
    """Run the generic ray-shooter on a raw (A, c) and return its sign set."""
    shooter = RayShootingExtractor(seed=0)
    rng = np.random.default_rng(0)
    out = {}
    d = A.shape[1]
    for _ in range(50):  # plenty of batches to let it plateau
        V = shooter._sample_rays(rng, d, shooter.batch_size)
        if V.shape[0] == 0:
            continue
        pts = shooter._shoot(V, A, c)
        before = len(out)
        shooter._collect_unique_cells_into(pts, A, c, out)
        if len(out) - before < shooter.plateau_ratio * shooter.batch_size:
            break
    return set(out.keys())


if __name__ == "__main__":
    sys.exit(main())
