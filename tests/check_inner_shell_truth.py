"""Absolute ground truth: brute-force count of primitive lattice points in a shard's inner shell.

Given a shard constraint matrix ``A`` and a norm ceiling ``R`` (the
``max_useful_norm`` the samplers target — usually ``MAX_TRAJECTORY_LENGTH``),
this script enumerates **every** integer point of the conditioned solution
lattice whose original-space image is strictly inside the cone and has L2 norm
``<= R``, then reduces to primitive (``gcd == 1``) directions.  The result is the
exact number of useful trajectories that physically exist in that shard's inner
shell, plus their average norm — the denominator the MCMC samplers' yields are
measured against.

Method (parallelised brute force)
---------------------------------
1. :class:`HyperSpaceConditioner` reduces ``A`` to an integer null-space basis
   ``Z`` (``d_orig x d_flat``) and facet normals ``B`` (``m x d_flat``); a point
   ``z`` is strictly interior iff ``B z < 0`` and its original-space image is
   ``v = Z z``.
2. **Search box.**  For a vector with ``||v|| = ||Z z|| <= R`` every coordinate
   obeys ``|z_k| <= R / sigma_min(Z)`` where ``sigma_min`` is the smallest
   singular value of ``Z`` (the worst-case contraction), so the complete search
   box is ``z in [-step, step]^d_flat`` with ``step = ceil(R / sigma_min) + 1``.
   When ``Z`` is orthonormal (the usual conditioned case) ``sigma_min`` equals
   the smallest column norm, so this reproduces the simple
   ``int(R / min_col_norm) + 1`` bound while staying rigorous on skewed bases.
3. **Vectorised mixed-radix enumeration.**  The box has up to ``(2*step+1)^d``
   points (10^11 for a 5D, R=80 shard), so a literal ``itertools.product``
   yielding Python tuples is infeasible.  Instead each integer in ``[0, total)``
   is decoded to its mixed-radix digits with NumPy in bulk; the linear range is
   split into fixed-size chunks dispatched across a process pool, and each
   worker filters its chunk with vectorised array ops (norm first — it rejects
   the empty box corners cheaply — then the cone test, then ``gcd``).
4. **Cores.**  ``multiprocessing.Pool`` with ``cpu_count() - 1`` workers.

Note on the interior test: the cone constraint is ``B @ z < 0`` (``B`` is
expressed in the *flatland* basis, so it multiplies ``z``, not the original
space image ``v``).

CLI
---
``python -m tests.check_inner_shell_truth [--max-norm R] [--cores N]
[--chunk C] [--pq 3 2] [--z 0.5]`` runs the count on the first shard of the
pFq(p, q, z) CMF as a standalone demo.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Optional, Tuple

import numpy as np

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.utils.logger import Logger


# --- Per-worker read-only state (populated once per process by the initializer). ---
_ZT_INT: Optional[np.ndarray] = None   # Z.T as int64, shape (d_flat, d_orig)
_BT: Optional[np.ndarray] = None       # B.T as float64, shape (d_flat, m)
_D: int = 0                            # d_flat
_BASE: int = 0                         # 2 * step + 1
_STEP: int = 0                         # box half-width
_TOL: float = 1e-9
_MAXN2: float = 0.0                    # max_norm ** 2


def _init_worker(ZT_int, BT, d, base, step, tol, maxn2):
    """Initializer: bind the shared, read-only enumeration state in each worker.

    :param ZT_int: ``Z.T`` (``d_flat x d_orig``) as int64.
    :param BT: ``B.T`` (``d_flat x m``) as float64 (empty array if no facets).
    :param d: flatland dimension ``d_flat``.
    :param base: mixed-radix base ``2*step + 1``.
    :param step: box half-width.
    :param tol: strict-interior tolerance (a point is interior iff ``B z < -tol``).
    :param maxn2: squared norm ceiling ``R**2``.
    """
    global _ZT_INT, _BT, _D, _BASE, _STEP, _TOL, _MAXN2
    _ZT_INT, _BT, _D, _BASE, _STEP, _TOL, _MAXN2 = ZT_int, BT, d, base, step, tol, maxn2


def _count_chunk(bounds: Tuple[int, int]) -> Tuple[int, float]:
    """Count primitive interior in-shell points whose linear index lies in ``[lo, hi)``.

    Decodes the linear index range to integer ``z`` vectors (mixed radix), maps
    them to original space ``v = Z z``, and applies the three filters in cheap-to-
    expensive order: norm ``<= R`` (rejects the bulk of the empty box), strict
    interior ``B z < -tol``, and primitivity ``gcd(v) == 1``.

    :param bounds: ``(lo, hi)`` half-open linear-index range into the box.
    :return: ``(count, norm_sum)`` for the primitive interior in-shell points found.
    """
    lo, hi = bounds
    n = hi - lo
    t = np.arange(lo, hi, dtype=np.int64)
    z = np.empty((n, _D), dtype=np.int64)
    for k in range(_D):
        z[:, k] = (t % _BASE) - _STEP
        t //= _BASE

    v = z @ _ZT_INT                                  # (n, d_orig) exact int64
    sq = (v.astype(np.float64) ** 2).sum(axis=1)     # squared norms
    keep = sq <= _MAXN2
    if not keep.any():
        return 0, 0.0
    z = z[keep]
    v = v[keep]

    if _BT.shape[1] > 0:                             # strict interior: B z < -tol
        cone = np.all(z @ _BT < -_TOL, axis=1)
        if not cone.any():
            return 0, 0.0
        v = v[cone]

    g = np.gcd.reduce(np.abs(v), axis=1)             # primitive directions only
    v = v[g == 1]
    if v.shape[0] == 0:
        return 0, 0.0

    norms = np.sqrt((v.astype(np.float64) ** 2).sum(axis=1))
    return int(v.shape[0]), float(norms.sum())


def _chunk_ranges(total: int, chunk: int):
    """Yield ``(lo, hi)`` half-open ranges tiling ``[0, total)`` in ``chunk``-sized blocks."""
    lo = 0
    while lo < total:
        hi = min(lo + chunk, total)
        yield (lo, hi)
        lo = hi


def search_bounds(Z: np.ndarray, max_norm: float) -> Tuple[int, int, int]:
    """Compute the complete (rigorous) integer search box for ``||Z z|| <= max_norm``.

    Uses ``step = ceil(max_norm / sigma_min(Z)) + 1`` (smallest singular value =
    worst-case contraction), which covers every in-shell point and reduces to the
    spec's ``int(max_norm / min_col_norm) + 1`` when ``Z`` is orthonormal.

    :param Z: ``(d_orig, d_flat)`` integer null-space basis.
    :param max_norm: original-space norm ceiling ``R``.
    :return: ``(step, base, total)`` — box half-width, mixed-radix base, point count.
    """
    d = Z.shape[1]
    sigma_min = float(np.linalg.svd(Z, compute_uv=False).min())
    min_col = float(np.linalg.norm(Z, axis=0).min())
    bound = min(sigma_min, min_col)
    step = int(np.ceil(max_norm / bound)) + 1
    base = 2 * step + 1
    total = base ** d
    return step, base, total


def count_inner_shell(
    A,
    max_norm: float = 80.0,
    *,
    tol: float = 1e-9,
    num_cores: Optional[int] = None,
    chunk_size: int = 1_000_000,
    verbose: bool = True,
) -> Tuple[int, float, dict]:
    """Brute-force the exact count of primitive interior trajectories with ``norm <= max_norm``.

    :param A: ``(rows, d_orig)`` shard constraint matrix.
    :param max_norm: original-space L2 norm ceiling (the samplers' ``max_useful_norm``).
    :param tol: strict-interior tolerance (interior iff ``B z < -tol``).
    :param num_cores: worker process count; defaults to ``cpu_count() - 1``.
    :param chunk_size: integer points decoded per dispatched task.
    :param verbose: log the box size, ETA, and progress.
    :return: ``(count, avg_norm, info)`` where ``info`` holds ``d_flat``, ``step``,
        ``total`` box size, and wall-clock ``seconds``.  ``avg_norm`` is 0.0 when
        ``count == 0``.
    """
    A = np.asarray(A, dtype=np.float64)
    Z, B, _ = HyperSpaceConditioner(A, max_beta=25, defect_tolerance=5.0).process()
    Z = np.asarray(Z, dtype=np.int64)
    B = np.asarray(B, dtype=np.float64)
    d = Z.shape[1]

    info = {"d_flat": d, "step": 0, "base": 0, "total": 0, "seconds": 0.0}
    if d == 0:
        if verbose:
            Logger("Ground truth: 0-dimensional flatland; no interior points.",
                   Logger.Levels.debug).log()
        return 0, 0.0, info

    step, base, total = search_bounds(Z, max_norm)
    info.update(step=step, base=base, total=total)

    if num_cores is None:
        num_cores = max(1, mp.cpu_count() - 1)

    if verbose:
        Logger(
            f"Ground truth: d_flat={d}, box=[-{step},{step}]^{d} = {total:.3e} points, "
            f"norm<= {max_norm:.0f}, {num_cores} cores, chunk={chunk_size:.0e}.",
            Logger.Levels.info,
        ).log()

    ZT_int = np.ascontiguousarray(Z.T)
    BT = np.ascontiguousarray(B.T) if B.shape[0] else np.zeros((d, 0), dtype=np.float64)
    maxn2 = float(max_norm) ** 2

    t0 = time.perf_counter()
    total_count = 0
    norm_sum = 0.0
    n_chunks = (total + chunk_size - 1) // chunk_size
    done_chunks = 0

    with mp.Pool(
        processes=num_cores,
        initializer=_init_worker,
        initargs=(ZT_int, BT, d, base, step, tol, maxn2),
    ) as pool:
        for cnt, nsum in pool.imap_unordered(
            _count_chunk, _chunk_ranges(total, chunk_size), chunksize=4
        ):
            total_count += cnt
            norm_sum += nsum
            done_chunks += 1
            if verbose and (done_chunks % 2000 == 0 or done_chunks == n_chunks):
                frac = done_chunks / n_chunks
                elapsed = time.perf_counter() - t0
                eta = elapsed / frac - elapsed if frac > 0 else 0.0
                Logger(
                    f"  ground-truth progress {frac * 100:5.1f}%  "
                    f"found={total_count}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    Logger.Levels.info,
                ).log()

    seconds = time.perf_counter() - t0
    info["seconds"] = seconds
    avg_norm = norm_sum / total_count if total_count else 0.0
    if verbose:
        Logger(
            f"Ground truth complete: {total_count} primitive points <= {max_norm:.0f}, "
            f"avg norm {avg_norm:.2f}, in {seconds:.0f}s.",
            Logger.Levels.info,
        ).log()
    return total_count, avg_norm, info


def _demo(p: int, q: int, z, max_norm: float, num_cores: Optional[int], chunk: int) -> int:
    """Run the count on the first shard of pFq(p, q, z) as a standalone demonstration."""
    import sympy as sp
    from dreamer.configs import config

    config.configure(
        extraction={"STRATEGY": "heuristic", "IGNORE_DUPLICATE_SEARCHABLES": True,
                    "LOAD_SHARD_CACHE": False},
        logging={"GENERATE_LOGS": False},
    )
    from dreamer.loading import pFq
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer import log

    c = log(2)
    z_sym = sp.Rational(z).limit_denominator(10 ** 6) if not isinstance(z, sp.Basic) else z
    cmf = pFq(c, p, q, z_sym).to_cmf()
    shards = [s for s in ShardExtractor(c, cmf).extract() if s.A is not None]
    Logger(f"pFq({p},{q},{z}) -> {len(shards)} shards with constraints; "
           f"counting shard 0.", Logger.Levels.info).log()
    count, avg, info = count_inner_shell(
        shards[0].A, max_norm=max_norm, num_cores=num_cores, chunk_size=chunk
    )
    print(f"\nGROUND TRUTH  pFq({p},{q},{z}) shard 0:")
    print(f"  d_flat={info['d_flat']} box={info['total']:.3e} "
          f"points<= {max_norm:.0f}: {count}  avg_norm={avg:.2f}  ({info['seconds']:.0f}s)")
    return 0


def main() -> int:
    """CLI entry point: brute-force ground truth on a single pFq shard."""
    ap = argparse.ArgumentParser(description="Brute-force inner-shell ground truth for a shard.")
    ap.add_argument("--max-norm", type=float, default=80.0, help="L2 norm ceiling R.")
    ap.add_argument("--cores", type=int, default=None, help="Worker processes (default cpu-1).")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="Points per dispatched task.")
    ap.add_argument("--pq", type=int, nargs=2, default=(3, 2), help="pFq orders p q.")
    ap.add_argument("--z", type=float, default=0.5, help="pFq argument z.")
    args = ap.parse_args()
    return _demo(args.pq[0], args.pq[1], args.z, args.max_norm, args.cores, args.chunk)


if __name__ == "__main__":
    raise SystemExit(main())
