r"""
Numeric "bumpiness" metrics for the δ field of a shard.

Two complementary notions of how *non-smooth* the irrationality measure δ is,
both pure-numeric (no plotting) so they are unit-testable in isolation:

**(B) Spatial roughness — empirical semivariogram (density-robust).**
    How different is δ between trajectories that are a given *angular distance*
    apart in direction space.  We bin every trajectory **pair** by the angle
    between their direction vectors ``h`` and average the squared δ-difference
    per bin::

        γ(h) = ½ · mean_{pairs at lag h} (δ_i − δ_j)²

    A smooth field has γ rising gently from ~0; a needle/bumpy field has a large
    **nugget** (γ at the smallest lag already large).  Because it aggregates over
    *all pairs in a distance bin* rather than per-point neighbours, it is robust
    to non-uniform sampling (e.g. gradient ascent clustering trajectories near
    optima) — unlike a k-NN estimate, where dense clusters dominate.

    The headline dimensionless number is the **relative nugget**
    ``nugget / sill`` (sill ≈ the sample variance of δ): ≈1 → pure noise / needle
    (no spatial structure), ≈0 → smooth.

**(A) Convergence smoothness — total variation of a δ-sequence.**
    Per trajectory, how much δ wobbles as the walk deepens:
    ``TV = Σ_k |δ_{k+1} − δ_k|``.  Independent of spatial sampling.  Aggregated
    to a shard by the median over trajectories.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# (B) Spatial roughness — empirical semivariogram
# ---------------------------------------------------------------------------

def angular_distance(u: Sequence[float], v: Sequence[float]) -> Optional[float]:
    """Angle (radians, in ``[0, π]``) between two direction vectors.

    Returns ``None`` if either vector is zero (no defined direction).
    """
    au = np.asarray(u, dtype=float)
    av = np.asarray(v, dtype=float)
    nu = float(np.linalg.norm(au))
    nv = float(np.linalg.norm(av))
    if nu == 0.0 or nv == 0.0:
        return None
    cos = float(np.dot(au, av) / (nu * nv))
    return math.acos(max(-1.0, min(1.0, cos)))


def empirical_semivariogram(
    directions: Sequence[Sequence[float]],
    deltas: Sequence[float],
    *,
    lag_bins: int = 15,
    max_pairs: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Compute the empirical semivariogram of δ over angular direction-distance.

    :param directions: one direction vector per trajectory.
    :param deltas: matching δ values (non-finite entries are dropped with their
        direction).
    :param lag_bins: number of equal-width angular lag bins over ``[0, max_h]``.
    :param max_pairs: cap on the number of pairs used; above it pairs are
        subsampled uniformly (bounds the O(M²) cost on dense shards).
    :param rng: optional NumPy ``Generator`` for reproducible subsampling.
    :return: dict with ``lag_centers``, ``gamma``, ``counts`` (per bin) and the
        summary scalars ``nugget``, ``sill``, ``relative_nugget``,
        ``initial_slope``, ``n_points``, ``n_pairs``.  All-NaN summary when there
        are fewer than 2 usable points or no non-degenerate pairs.
    """
    dirs: List[np.ndarray] = []
    dels: List[float] = []
    for d, delta in zip(directions, deltas):
        if delta is None or not np.isfinite(delta):
            continue
        vec = np.asarray(d, dtype=float)
        if vec.size == 0 or float(np.linalg.norm(vec)) == 0.0:
            continue
        dirs.append(vec)
        dels.append(float(delta))

    n = len(dirs)
    empty = {
        "lag_centers": np.array([]),
        "gamma": np.array([]),
        "counts": np.array([]),
        "nugget": float("nan"),
        "sill": float("nan"),
        "relative_nugget": float("nan"),
        "initial_slope": float("nan"),
        "n_points": n,
        "n_pairs": 0,
    }
    if n < 2:
        return empty

    dels_arr = np.array(dels)
    sill = float(np.var(dels_arr))  # sample variance ≈ the semivariogram sill

    # Enumerate (or subsample) the unordered pairs.
    i_idx, j_idx = np.triu_indices(n, k=1)
    total_pairs = i_idx.size
    if total_pairs > max_pairs:
        gen = rng if rng is not None else np.random.default_rng(0)
        sel = gen.choice(total_pairs, size=max_pairs, replace=False)
        i_idx, j_idx = i_idx[sel], j_idx[sel]

    hs: List[float] = []
    sq: List[float] = []
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        h = angular_distance(dirs[i], dirs[j])
        if h is None:
            continue
        hs.append(h)
        sq.append((dels[i] - dels[j]) ** 2)

    if not hs:
        return {**empty, "sill": sill}

    hs_arr = np.array(hs)
    sq_arr = np.array(sq)
    max_h = float(hs_arr.max())
    if max_h <= 0.0:
        # All directions parallel — no spatial resolution; report the raw nugget.
        nugget = float(0.5 * sq_arr.mean())
        return {
            **empty,
            "gamma": np.array([nugget]),
            "lag_centers": np.array([0.0]),
            "counts": np.array([sq_arr.size]),
            "nugget": nugget,
            "sill": sill,
            "relative_nugget": float(nugget / sill) if sill > 0 else float("nan"),
            "initial_slope": float("nan"),
            "n_points": n,
            "n_pairs": int(sq_arr.size),
        }

    edges = np.linspace(0.0, max_h, lag_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # np.digitize: bin index in 1..lag_bins; clip the rightmost edge into the last bin.
    bin_idx = np.clip(np.digitize(hs_arr, edges[1:-1], right=False), 0, lag_bins - 1)

    gamma = np.full(lag_bins, np.nan)
    counts = np.zeros(lag_bins, dtype=int)
    for b in range(lag_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            gamma[b] = 0.5 * float(sq_arr[mask].mean())

    # Nugget = γ of the first populated bin (closest to h→0).
    populated = np.where(counts > 0)[0]
    nugget = float(gamma[populated[0]]) if populated.size else float("nan")

    initial_slope = _initial_slope(centers, gamma, counts)
    relative_nugget = float(nugget / sill) if (sill and sill > 0 and np.isfinite(nugget)) else float("nan")

    return {
        "lag_centers": centers,
        "gamma": gamma,
        "counts": counts,
        "nugget": nugget,
        "sill": sill,
        "relative_nugget": relative_nugget,
        "initial_slope": initial_slope,
        "n_points": n,
        "n_pairs": int(sq_arr.size),
    }


def _initial_slope(
    centers: np.ndarray, gamma: np.ndarray, counts: np.ndarray, n_bins: int = 4
) -> float:
    """Least-squares slope of γ vs lag over the first ``n_bins`` populated bins."""
    populated = np.where(counts > 0)[0]
    use = populated[:n_bins]
    if use.size < 2:
        return float("nan")
    x = centers[use]
    y = gamma[use]
    good = np.isfinite(y)
    if good.sum() < 2:
        return float("nan")
    slope = np.polyfit(x[good], y[good], 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# (A) Convergence smoothness — total variation of a δ-sequence
# ---------------------------------------------------------------------------

def total_variation(sequence: Sequence[float]) -> float:
    """``Σ_k |s_{k+1} − s_k|`` over the finite entries of *sequence*.

    Non-finite entries (e.g. the ``-inf`` non-convergence sentinel) are dropped
    before differencing.  Returns ``nan`` for fewer than 2 finite entries.
    """
    vals = [float(s) for s in sequence if s is not None and np.isfinite(s)]
    if len(vals) < 2:
        return float("nan")
    arr = np.array(vals)
    return float(np.abs(np.diff(arr)).sum())


def median_total_variation(sequences: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """Median total variation over a set of δ-sequences.

    :return: ``(median_TV, n_sequences_used)`` — sequences with fewer than 2
        finite points are skipped; ``nan`` median when none qualify.
    """
    tvs = [tv for tv in (total_variation(s) for s in sequences) if np.isfinite(tv)]
    if not tvs:
        return float("nan"), 0
    return float(np.median(tvs)), len(tvs)
