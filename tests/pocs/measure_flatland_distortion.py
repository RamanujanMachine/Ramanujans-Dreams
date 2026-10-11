"""Does the flatland basis distort local search?  (REAL shards, pFq(3,2,1/2))

The deep-search optimisers (Small Angle, SA, Gradient, GA) all define
"locality"/"angle" in the *flatland* integer basis, where a flatland vector ``z``
maps to a real trajectory direction by the skew linear map ``v = Z_reduced @ z``.
The LLL/BKZ conditioner only drives the orthogonality defect below
``defect_tolerance = 5.0`` (not 1.0), so meaningful skew is *allowed*.

This POC measures, on the 60 real recession cones of ``pFq(3, 2, 1/2)``, how
faithfully flatland geometry matches the real shard-space geometry the objective
δ actually lives in:

  Metric 1  Global basis distortion — singular values / condition number κ of
            ``Z_reduced``, the orthogonality defect, and how uneven the per-axis
            real step sizes ``||Z_reduced @ e_i||`` are.
  Metric 2  Angle preservation — flatland angle ∠(z1,z2) vs real angle
            ∠(Z z1, Z z2) over many in-cone pairs (correlation + discrepancy).
  Metric 3  Neighbourhood faithfulness (operator-level) — for in-cone z, the
            real geometric angle of each ±1 flatland neighbour; per-z max/min
            ratio.  Tests the SA / Small-Angle assumption that a ±1 step is a
            *small, consistent* geometric move.
  Metric 4  Real-near -> flatland-near (the user's exact phrasing) — among pairs
            that are geometrically near (real angle < theta), how far apart are
            they in flatland?  Plus the converse (flat-near -> real angle).

Two synthetic sanity checks bracket the run: an identity geometry (must report
κ = 1, defect = 1, perfect preservation) and a deliberately skewed basis (must
report κ >> 1 and visible angle decorrelation) — so the metric code is trusted
before the real-shard numbers are read.

Run from the repo root (imports ``examples.benchmark_3f2_samplers``):

    python -m tests.pocs.measure_flatland_distortion [--shards N] [--seed S]
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from dreamer.search.methods.flatland.geometry import FlatlandGeometry

SEED = 12345
SMALL_ANGLE_DEG = 5.0  # "near" threshold for Metric 4


# ----------------------------------------------------------------------
# Duck-typed shards: FlatlandGeometry only needs .A, .is_whole_space, .symbols
# ----------------------------------------------------------------------
class _AShard:
    """Minimal shard exposing just what :class:`FlatlandGeometry` reads."""

    def __init__(self, A: np.ndarray):
        """:param A: constraint matrix (rows = facet normals, cols = d_orig)."""
        self.A = np.asarray(A, dtype=np.float64)
        self.is_whole_space = False
        self.symbols = list(sp.symbols(f"x0:{self.A.shape[1]}"))


class _WholeShard:
    """Unconstrained shard -> FlatlandGeometry gives ``Z_reduced = I`` (identity)."""

    def __init__(self, d: int):
        """:param d: ambient dimension."""
        self.A = None
        self.is_whole_space = True
        self.symbols = list(sp.symbols(f"x0:{d}"))


# ----------------------------------------------------------------------
# Pure-numpy metric helpers (take Z_reduced directly so they are testable)
# ----------------------------------------------------------------------
def ortho_defect(Z: np.ndarray) -> float:
    """Orthogonality defect ``(prod ||z_i||) / sqrt(det(Z^T Z))`` (cols = basis)."""
    norms = np.linalg.norm(Z, axis=0)
    det = np.sqrt(abs(np.linalg.det(Z.T @ Z)))
    return float("inf") if det < 1e-9 else float(np.prod(norms) / det)


def basis_distortion(Z: np.ndarray) -> dict:
    """Metric 1: singular values / κ / defect / per-axis real step spread of ``Z``.

    :param Z: ``Z_reduced`` (shape ``(d_orig, d_flat)``).
    :return: dict with ``kappa``, ``defect``, ``col_norm_ratio`` (max/min of the
        per-axis real step sizes ``||Z e_i||``), and the singular values.
    """
    sv = np.linalg.svd(Z.astype(np.float64), compute_uv=False)
    col_norms = np.linalg.norm(Z, axis=0)
    return {
        "kappa": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "defect": ortho_defect(Z),
        "col_norm_ratio": float(col_norms.max() / col_norms.min()) if col_norms.min() > 0 else float("inf"),
        "sv": sv,
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geometric angle (degrees) between two vectors; NaN if either is zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _row_pair_angles(M: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Angles (degrees) between rows ``M[i]`` and ``M[j]`` (vectorised)."""
    num = np.sum(M[i] * M[j], axis=1)
    den = np.linalg.norm(M[i], axis=1) * np.linalg.norm(M[j], axis=1)
    den[den == 0] = 1.0
    return np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))


def pair_angles(
    Zset: np.ndarray, V: np.ndarray, rng: np.random.Generator, n_pairs: int = 6000
) -> Tuple[np.ndarray, np.ndarray]:
    """Metric 2: flatland vs real angle over random in-cone pairs.

    :param Zset: in-cone flatland vectors (shape ``(n, d_flat)``).
    :param V: their real images ``Z_reduced @ z`` (shape ``(n, d_orig)``).
    :return: ``(flat_angles_deg, real_angles_deg)`` aligned arrays.
    """
    n = len(Zset)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    flat = _row_pair_angles(Zset.astype(np.float64), i, j)
    real = _row_pair_angles(V, i, j)
    return flat, real


def neighbour_metrics(
    geom: FlatlandGeometry, Zset: np.ndarray, rng: np.random.Generator, n_z: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Metric 3: real angle of each in-cone ±1 flatland neighbour, per-z max/min.

    :return: ``(ratios, all_neighbour_angles_deg)`` — ``ratios[k]`` is
        ``max/min`` of the in-cone neighbour angles around the k-th sampled z.
    """
    Z = geom.Z_reduced.astype(np.float64)
    if len(Zset) > n_z:
        Zset = Zset[rng.choice(len(Zset), n_z, replace=False)]
    ratios: List[float] = []
    all_ang: List[float] = []
    for z in Zset:
        v = Z @ z
        angs = []
        for nb in geom.perturbations(z, reduce=False):
            if not geom.is_inside(nb):
                continue
            a = _angle_deg(v, Z @ nb)
            if not np.isnan(a) and a > 0:
                angs.append(a)
        if len(angs) >= 2:
            ratios.append(max(angs) / min(angs))
            all_ang.extend(angs)
    return np.array(ratios), np.array(all_ang)


# ----------------------------------------------------------------------
# In-cone flatland sampling
# ----------------------------------------------------------------------
def sample_in_cone_z(
    geom: FlatlandGeometry, rng: np.random.Generator, n_target: int = 600, box: int = 8
) -> np.ndarray:
    """Random integer flatland vectors filtered to the (non-zero) recession cone."""
    d = geom.d_flat
    found: List[np.ndarray] = []
    for _ in range(60):
        cand = rng.integers(-box, box + 1, size=(20000, d))
        cand = cand[geom.is_inside_many(cand)]
        if len(cand):
            found.append(cand)
        if sum(len(c) for c in found) >= n_target * 3:
            break
    if not found:
        return np.zeros((0, d), dtype=np.int64)
    arr = np.unique(np.concatenate(found, axis=0), axis=0)
    if len(arr) > n_target:
        arr = arr[rng.choice(len(arr), n_target, replace=False)]
    return arr.astype(np.int64)


# ----------------------------------------------------------------------
# Per-geometry report
# ----------------------------------------------------------------------
def analyse_geometry(geom: FlatlandGeometry, rng: np.random.Generator) -> Optional[dict]:
    """Run all four metrics on one geometry; ``None`` if no in-cone sample found."""
    Z = geom.Z_reduced.astype(np.float64)
    dist = basis_distortion(Z)

    Zset = sample_in_cone_z(geom, rng)
    if len(Zset) < 20:
        return {"distortion": dist, "n_incone": len(Zset), "sparse": True}
    V = (Z @ Zset.T).T

    flat, real = pair_angles(Zset, V, rng)
    disc = np.abs(real - flat)
    corr = float(np.corrcoef(flat, real)[0, 1]) if len(flat) > 2 else float("nan")

    # Metric 4: real-near pairs -> how spread in flatland (and the converse).
    real_near = real < SMALL_ANGLE_DEG
    flat_near = flat < SMALL_ANGLE_DEG
    real_near_flat = flat[real_near]   # geometrically near: their flatland angle
    flat_near_real = real[flat_near]   # flatland near (±1-ish): their real angle

    ratios, nb_ang = neighbour_metrics(geom, Zset, rng)

    return {
        "distortion": dist,
        "n_incone": len(Zset),
        "sparse": False,
        "corr": corr,
        "disc_med": float(np.median(disc)),
        "disc_p90": float(np.percentile(disc, 90)),
        "disc_max": float(disc.max()),
        "real_near_flat_med": float(np.median(real_near_flat)) if real_near_flat.size else float("nan"),
        "real_near_flat_p90": float(np.percentile(real_near_flat, 90)) if real_near_flat.size else float("nan"),
        "real_near_flat_max": float(real_near_flat.max()) if real_near_flat.size else float("nan"),
        "flat_near_real_p90": float(np.percentile(flat_near_real, 90)) if flat_near_real.size else float("nan"),
        "nb_ratio_med": float(np.median(ratios)) if ratios.size else float("nan"),
        "nb_ratio_p90": float(np.percentile(ratios, 90)) if ratios.size else float("nan"),
        "nb_ratio_max": float(ratios.max()) if ratios.size else float("nan"),
        "nb_ang_med": float(np.median(nb_ang)) if nb_ang.size else float("nan"),
        "nb_ang_max": float(nb_ang.max()) if nb_ang.size else float("nan"),
    }


def _print_report(tag: str, r: dict) -> None:
    """Pretty-print one geometry's metric dict."""
    d = r["distortion"]
    sv = ", ".join(f"{x:.3g}" for x in d["sv"])
    print(f"\n[{tag}]  d_flat={len(d['sv'])}  in-cone sample={r['n_incone']}")
    print(f"  M1 basis   : kappa={d['kappa']:.2f}  defect={d['defect']:.2f}  "
          f"col-norm max/min={d['col_norm_ratio']:.2f}  sv=[{sv}]")
    if r.get("sparse"):
        print("  (too few in-cone samples — angle metrics skipped)")
        return
    print(f"  M2 angles  : corr(flat,real)={r['corr']:.3f}  "
          f"|real-flat| med={r['disc_med']:.1f} deg  p90={r['disc_p90']:.1f}  max={r['disc_max']:.1f}")
    print(f"  M3 ±1 nbrs : real-angle med={r['nb_ang_med']:.1f} deg  max={r['nb_ang_max']:.1f}  "
          f"| per-z max/min ratio med={r['nb_ratio_med']:.2f}  p90={r['nb_ratio_p90']:.2f}  max={r['nb_ratio_max']:.2f}")
    print(f"  M4 near    : real<{SMALL_ANGLE_DEG:.0f}deg -> flat angle med={r['real_near_flat_med']:.1f}  "
          f"p90={r['real_near_flat_p90']:.1f}  max={r['real_near_flat_max']:.1f}  "
          f"| flat<{SMALL_ANGLE_DEG:.0f}deg -> real p90={r['flat_near_real_p90']:.1f}")


def _agg(rows: List[dict], key: str) -> Tuple[float, float, float]:
    """min / mean / max of ``key`` over the non-sparse rows (NaNs ignored)."""
    vals = np.array([r[key] for r in rows if not r.get("sparse") and np.isfinite(r.get(key, np.nan))])
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(vals.min()), float(vals.mean()), float(vals.max())


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------
def sanity_checks(rng: np.random.Generator) -> None:
    """Identity geometry (κ≈1, faithful) and a skewed basis (κ≫1, decorrelated)."""
    print("=" * 78)
    print("SANITY CHECKS (synthetic)")
    print("=" * 78)

    ident = analyse_geometry(FlatlandGeometry(_WholeShard(5)), rng)
    _print_report("identity Z=I (expect kappa=1, corr=1.0, ratio=1)", ident)

    # Skewed full-space basis: keep Z_reduced = I geometry but inject a skew map
    # by post-multiplying via a custom geometry.  Build an _AShard whose A forces
    # no equalities, then overwrite Z_reduced with a deliberately skewed integer
    # basis to exercise the metric code on a known-bad map.
    geom = FlatlandGeometry(_WholeShard(3))
    skew = np.array([[1, 0, 0], [0, 1, 0], [9, 9, 1]], dtype=np.int64)  # κ large
    geom.Z_reduced = skew
    geom._M = None  # whole space: every non-zero z is "in-cone" for the test
    skew_r = analyse_geometry(geom, rng)
    _print_report("skewed Z (expect kappa>>1, corr<1, ratio>>1)", skew_r)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    """Run sanity checks + the real-shard distortion census."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=0, help="limit number of real shards (0 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sanity_checks(rng)

    print("\n" + "=" * 78)
    print("REAL SHARDS — pFq(3, 2, 1/2)")
    print("=" * 78)
    from tests.pocs.benchmark_3f2_samplers import _extract_shards

    shards = _extract_shards()
    if args.shards:
        shards = shards[: args.shards]
    print(f"Extracted {len(shards)} real shards with constraints.")

    rows: List[dict] = []
    for idx, A in enumerate(shards):
        geom = FlatlandGeometry(_AShard(A))
        r = analyse_geometry(geom, rng)
        if r is None:
            continue
        rows.append(r)
        _print_report(f"shard {idx}", r)

    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(rows)} shards  (min / mean / max)")
    print("=" * 78)
    for label, key in [
        ("M1 kappa (condition number)", "kappa"),
        ("M1 orthogonality defect", "defect"),
        ("M1 col-norm max/min ratio", "col_norm_ratio"),
        ("M2 corr(flat,real)", "corr"),
        ("M2 |real-flat| p90 (deg)", "disc_p90"),
        ("M3 ±1 nbr per-z max/min ratio (med)", "nb_ratio_med"),
        ("M3 ±1 nbr per-z max/min ratio (max)", "nb_ratio_max"),
        ("M3 ±1 nbr real angle max (deg)", "nb_ang_max"),
        ("M4 real-near -> flat angle p90 (deg)", "real_near_flat_p90"),
        ("M4 flat-near -> real angle p90 (deg)", "flat_near_real_p90"),
    ]:
        if key in ("kappa", "defect", "col_norm_ratio"):
            src = [{"sparse": False, key: r["distortion"][key]} for r in rows]
        else:
            src = rows
        lo, mean, hi = _agg(src, key)
        print(f"  {label:42s}: {lo:8.3f} / {mean:8.3f} / {hi:8.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
