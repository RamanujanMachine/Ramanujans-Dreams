"""
Verification: does HyperSpaceConditioner correctly reduce a lower-dimensional
(corridor) recession cone into its relative-interior subspace, so that the
strict PT sampler (a) does not regress on fat cones and (b) handles corridors
where every valid trajectory must be parallel to the forced walls?

Convention used by the sampler/conditioner:
    cone          = { v : A_prime v >= 0 }
    strict interior = { v : A_prime v >  0 }
A facet i is an *implicit equality* iff  max{ A_i v : A_prime v >= 0, |v|<=1 } = 0,
i.e. it is tight on the entire cone.  Those rows are folded into E; Z spans the
integer null space of E (the corridor's own subspace); the remaining facets B are
the only ones the strict test  B z > 0  acts on.
"""
import numpy as np
from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler as PT


def analyse(name, A_prime, run_harvest=True, quota=200):
    A_prime = np.asarray(A_prime, dtype=np.int64)
    print("=" * 78)
    print(f"[{name}]  A_prime shape = {A_prime.shape}  (rows x d_orig)")

    cond = HyperSpaceConditioner(A_prime, max_beta=25, defect_tolerance=5.0)
    E, B = cond._extract_constraints()
    Z = cond._compute_integer_basis(E)
    d_orig = A_prime.shape[1]
    d_flat = Z.shape[1]
    rankE = np.linalg.matrix_rank(E) if len(E) else 0

    print(f"  implicit-equality rows detected : {len(E):>3}  (rank {rankE})")
    print(f"  genuinely-strict-able facets B  : {len(B):>3}")
    print(f"  d_orig = {d_orig}   ->   d_flat = {d_flat}   (reduction {d_orig - d_flat})")

    # E rows must be exactly tight on the whole subspace: E @ Z == 0
    if len(E):
        EZ = E.astype(np.int64) @ Z
        print(f"  E @ Z all zero (forced walls hold exactly in subspace): {np.all(EZ == 0)}")

    if not run_harvest:
        return

    try:
        sampler = PT(A_prime, rng_seed=12345)
        rays = sampler.harvest(quota)
    except Exception as exc:  # NarrowConeError etc.
        print(f"  harvest raised: {type(exc).__name__}: {exc}")
        return

    if len(rays) == 0:
        print("  harvest returned 0 rays")
        return

    rays = np.asarray(rays, dtype=np.int64)
    vals = rays @ A_prime.T                      # (n, rows)  exact integer A_prime v

    # 1. every ray legal under the EXACT cone  A_prime v <= 0  (strict interior B v < 0)
    legal = np.all(vals <= 0)
    from math import gcd
    from functools import reduce
    primitive = all(reduce(gcd, (abs(int(x)) for x in r)) == 1 for r in rays)
    # 2. implicit-equality rows must be EXACTLY 0 on every ray (parallel to walls)
    if len(E):
        eq_idx = [i for i in range(len(A_prime))
                  if any(np.array_equal(A_prime[i], e) for e in E.astype(np.int64))]
        eq_tight = np.all(vals[:, eq_idx] == 0)
    else:
        eq_tight = True
    # 3. how many rays sit ON a NON-forced facet (B_i v == 0)?  -> what strict excludes
    if len(B):
        b_idx = [i for i in range(len(A_prime))
                 if any(np.array_equal(A_prime[i], bb) for bb in B.astype(np.int64))]
        on_face = np.any(vals[:, b_idx] == 0, axis=1)
        n_on_face = int(on_face.sum())
    else:
        n_on_face = 0

    print(f"  harvested rays                  : {len(rays)}")
    print(f"  all rays satisfy A_prime v <= 0 : {legal}   (in-cone -> no regression)")
    print(f"  all rays primitive (gcd == 1)   : {primitive}")
    print(f"  forced-wall rows exactly 0      : {eq_tight}   (all trajectories parallel to walls)")
    print(f"  rays lying on a NON-forced facet: {n_on_face}   (what strict B z>0 currently excludes)")
    print(f"  example ray: {rays[0]}")


# --- Case 1: FAT baseline (regression control) -------------------------------
# explicit equality  x0 + x1 + x2 = 0  (encoded as +/- rows) + 3 real inequalities
E1 = np.array([[1, 1, 1, 0, 0]])
B1 = np.array([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]])
A1 = np.vstack([E1, -E1, B1])
analyse("Fat baseline (explicit equality + real inequalities)", A1)

# --- Case 2: HIDDEN corridor -------------------------------------------------
# 3 inequality facets, none is the literal negation of another, yet they force
# v0 = v1 = 0, leaving only the v2 axis -> recession cone is 1-dimensional.
#   v0+v1>=0 , v0-v1>=0 , -v0>=0   ==>  v0=0 then v1=0
A2 = np.array([[1, 1, 0],
               [1, -1, 0],
               [-1, 0, 0]])
analyse("Hidden corridor (1-D recession cone, no explicit equalities)", A2)

# --- Case 3: corridor with a surviving genuine inequality --------------------
# force v0=v1=0 (hidden, as above) in R^4, keep v2>=0 genuine, v3 free.
A3 = np.array([[1, 1, 0, 0],
               [1, -1, 0, 0],
               [-1, 0, 0, 0],
               [0, 0, 1, 0]])
analyse("Corridor + surviving inequality (d_flat should be 2)", A3)
