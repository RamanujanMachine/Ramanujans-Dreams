"""Degeneracy + impact census on REAL shards (pFq(3, 2, 1/2), 5D, 60 shards).

Two questions, measured on real recession cones rather than synthetic ones:

  1. DEGENERACY — are real shard recession cones full-dimensional, or corridor-like?
     Per shard we report d_orig -> d_flat and the implicit-equality (forced) vs
     genuinely-strict-able (non-forced) facet split, using the production conditioner.

  2. IMPACT — with the now-closed cone, what fraction of harvested directions are
     *face-parallel* (``A_i v == 0`` on some non-forced facet) — i.e. the directions
     the old strict sampler could never reach.  This is the "actual relevance" of the fix.
"""
import numpy as np
from tests.pocs.benchmark_3f2_samplers import _extract_shards
from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler as PT

QUOTA = 60


def main():
    shards = _extract_shards()
    print(f"Extracted {len(shards)} real shards from pFq(3, 2, 1/2).\n")

    geo_rows = []
    full_dim = 0
    tot_face = tot_rays = 0
    face_per_shard = []

    for i, A in enumerate(shards):
        A = np.asarray(A)
        d_orig = A.shape[1]
        cond = HyperSpaceConditioner(np.asarray(A, dtype=np.float64), max_beta=25)
        E, Bo = cond._extract_constraints()
        Z = cond._compute_integer_basis(E)
        d_flat = Z.shape[1]
        n_forced, n_nonforced = len(E), len(Bo)
        # A cone is full-dimensional (no hidden corridor) iff there is a strict interior,
        # i.e. the only forced rows are the structural equalities that define d_flat and the
        # non-forced facets span a full-dim cone within flatland.  Practically: full-dim iff
        # there is >=1 non-forced facet and the conditioner found a strict seed.
        is_full = n_nonforced > 0
        full_dim += int(is_full)
        geo_rows.append((i, d_orig, d_flat, n_forced, n_nonforced))

        # ---- impact: harvest (now closed cone) and count face-parallel directions ----
        try:
            rays = np.asarray(PT(A, rng_seed=12345).harvest(QUOTA), dtype=np.int64)
        except Exception as exc:
            face_per_shard.append((i, None, str(type(exc).__name__)))
            continue
        if len(rays) == 0 or Bo.shape[0] == 0:
            face_per_shard.append((i, 0.0, len(rays)))
            continue
        vals = rays @ Bo.T.astype(np.int64)          # exact: integer rows x integer rays
        on_face = np.any(vals == 0, axis=1)
        nf = int(on_face.sum())
        tot_face += nf
        tot_rays += len(rays)
        face_per_shard.append((i, nf / len(rays), len(rays)))

    print("=== GEOMETRY (per shard) ===")
    print(" shard  d_orig  d_flat  forced  nonforced")
    for i, do, df, nf, nb in geo_rows:
        print(f"  {i:3d}     {do:3d}     {df:3d}     {nf:3d}      {nb:3d}")
    print(f"\nFull-dimensional recession cones: {full_dim}/{len(shards)}")

    print("\n=== IMPACT (face-parallel harvested directions, closed cone) ===")
    nonzero = [f for (_, f, _) in face_per_shard if f]
    print(f"  shards with >0% face-parallel harvest: "
          f"{sum(1 for (_, f, _) in face_per_shard if f and f > 0)}/{len(shards)}")
    if tot_rays:
        print(f"  overall face-parallel fraction: {100*tot_face/tot_rays:.1f}%  "
              f"({tot_face}/{tot_rays} harvested rays)")
    per = [f for (_, f, _) in face_per_shard if f is not None]
    if per:
        print(f"  per-shard face-parallel %: min={100*min(per):.1f}  "
              f"mean={100*np.mean(per):.1f}  max={100*max(per):.1f}")


if __name__ == "__main__":
    main()
