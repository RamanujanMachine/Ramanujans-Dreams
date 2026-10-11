"""Uniformity check: does the closed-cone sampler over-represent faces, or sample
them in proportion to their true prevalence among short directions?

For one real shard we brute-force enumerate EVERY primitive closed-cone direction
(``B z <= 0``, ``gcd == 1``) with ``||v|| <= R`` and classify each as
  * face-parallel  (some facet exactly tight,  min_i B_i z == 0), or
  * strict interior (all B_i z < 0).
The TRUE face-parallel fraction is the ground truth.  We then harvest with the
production PT sampler (now closed) and compute its harvested face fraction over the
same norm band.  If sampler ~= truth -> unbiased (uniformity preserved).  If
sampler >> truth -> the walker dwells on faces (bias).
"""
import numpy as np
from tests.pocs.benchmark_3f2_samplers import _extract_shards
from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler as PT

R = 15.0
TOL = 1e-6
SHARD = 0


def gcd_rows(M):
    g = np.abs(M[:, 0]).astype(np.int64)
    for k in range(1, M.shape[1]):
        g = np.gcd(g, np.abs(M[:, k]).astype(np.int64))
    return g


def main():
    A = np.asarray(_extract_shards()[SHARD], dtype=np.float64)
    Z, B, _ = HyperSpaceConditioner(A, max_beta=25).process()
    Z = Z.astype(np.int64)
    d = Z.shape[1]
    sigma_min = np.linalg.svd(Z.astype(np.float64), compute_uv=False).min()
    step = int(np.ceil(R / sigma_min)) + 1
    print(f"shard {SHARD}: d_flat={d}, facets={B.shape[0]}, box=±{step} -> {(2*step+1)**d:.2e} points")

    grid = np.arange(-step, step + 1)
    n_dirs = 0
    n_face = 0
    # enumerate by looping the outer coordinate to bound memory
    for z0 in grid:
        rest = np.array(np.meshgrid(*([grid] * (d - 1)), indexing="ij")).reshape(d - 1, -1).T
        Zfull = np.empty((rest.shape[0], d), dtype=np.int64)
        Zfull[:, 0] = z0
        Zfull[:, 1:] = rest
        Bz = Zfull @ B.T                               # (n, m)
        inside = np.all(Bz <= TOL, axis=1)             # closed cone
        Zin = Zfull[inside]
        if not len(Zin):
            continue
        v = Zin @ Z.T                                  # original space
        norms = np.linalg.norm(v, axis=1)
        keep = (norms <= R) & (norms > 1e-9)
        Zin, v = Zin[keep], v[keep]
        if not len(Zin):
            continue
        prim = gcd_rows(v) == 1
        Zin = Zin[prim]
        if not len(Zin):
            continue
        Bz_in = Zin @ B.T
        face = np.any(np.abs(Bz_in) <= TOL, axis=1)    # some facet exactly tight
        n_dirs += len(Zin)
        n_face += int(face.sum())

    true_frac = n_face / n_dirs if n_dirs else 0.0
    print(f"TRUTH : primitive closed-cone dirs <=R: {n_dirs}, face-parallel {n_face} -> {100*true_frac:.1f}%")

    # sampler harvest over the same norm band
    rays = np.asarray(PT(A, rng_seed=12345, max_useful_norm=R).harvest(200), dtype=np.int64)
    nb = np.linalg.norm(rays, axis=1)
    rays = rays[nb <= R + 1e-9]
    Bo = HyperSpaceConditioner(A, max_beta=25)._extract_constraints()[1].astype(np.int64)
    sf = np.any((rays @ Bo.T) == 0, axis=1)
    samp_frac = sf.mean() if len(rays) else 0.0
    print(f"SAMPLER: harvested dirs <=R: {len(rays)}, face-parallel {int(sf.sum())} -> {100*samp_frac:.1f}%")
    print(f"\nratio sampler/truth = {samp_frac/true_frac:.2f}x  "
          f"({'unbiased' if abs(samp_frac-true_frac) < 0.12 else 'POSSIBLE BIAS'})")


if __name__ == "__main__":
    main()
