"""3F2 Shard Showdown — benchmark the production MCMC samplers against brute-force ground truth.

Extracts the shards of the 5D ``pFq(3, 2, 1/2)`` CMF (60 shards), selects a
spread of them, and for each runs the trajectory samplers plus the exact
brute-force enumerator, then prints a per-shard comparison:

* **Ground truth** — exact primitive interior points with ``norm <= gt_norm``
  (:func:`tests.check_inner_shell_truth.count_inner_shell`) and their avg norm.
* **Discrete MCMC** (:class:`DiscreteMCMCSampler`) — yield, avg norm, acceptance.
* **Linear PT MCMC** (:class:`ParallelTemperingSampler`) — yield, avg norm, acceptance.

(The experimental ``HarmonicPTSampler`` was benchmarked here and lost decisively —
longest norms, lowest yield, 3/15 zero-yield shards — so it was removed; see
``SAMPLING_MATH.md`` §14.3 for the recorded negative result.)

Both samplers target ``quota`` primitive directions with
``max_useful_norm = useful_norm`` (default 80, matching ``MAX_TRAJECTORY_LENGTH``)
and ``exact=True`` so the requested count is taken literally (no volume scaling).

Ground-truth cost note
----------------------
The shards are full-dimensional 5D cones, so the brute-force box is
``(2*ceil(R/sigma_min)+1)^5`` — ~1e11 points at ``R = 80`` (~tens of minutes per
shard on a typical workstation).  The lowest few hundred points (the only ones
the quota-200 samplers compete for) sit far inside this shell, so ``--gt-norm``
defaults to a value that still contains *far* more than the quota while keeping
the run tractable; raise it to 80 for the full (slow) count.

CLI
---
``python -m examples.benchmark_3f2_samplers [--shards 15] [--quota 200]
[--useful-norm 80] [--gt-norm 30] [--cores N] [--chunk C] [--seed S]``
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import numpy as np
import sympy as sp

from dreamer.configs import config


def _extract_shards() -> List[np.ndarray]:
    """Extract the constraint matrices of the pFq(3, 2, 1/2) shards.

    :return: list of ``(rows, 5)`` constraint matrices, one per shard with constraints.
    """
    config.configure(
        extraction={"STRATEGY": "heuristic", "IGNORE_DUPLICATE_SEARCHABLES": True,
                    "LOAD_SHARD_CACHE": False},
        logging={"GENERATE_LOGS": False},
    )
    from dreamer.loading import pFq
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer import log

    c = log(2)
    cmf = pFq(c, 3, 2, sp.Rational(1, 2)).to_cmf()
    shards = ShardExtractor(c, cmf).extract()
    return [np.asarray(s.A, dtype=np.float64) for s in shards if s.A is not None]


def _run_sampler(sampler_cls, A, quota, useful_norm, seed) -> Tuple[int, float, float]:
    """Harvest ``quota`` directions with one sampler and report (yield, avg_norm, accept).

    :return: ``(n_found, avg_norm, accept_rate)``; ``avg_norm`` is 0.0 if nothing found.
    """
    sampler = sampler_cls(A, max_useful_norm=float(useful_norm), rng_seed=seed)
    out = sampler.harvest(int(quota), exact=True)
    n = int(out.shape[0])
    avg = float(np.linalg.norm(out, axis=1).mean()) if n else 0.0
    return n, avg, float(getattr(sampler, "last_accept_rate", 0.0))


def main() -> int:
    """CLI entry point: run the 3F2 sampler showdown and print the comparison table."""
    ap = argparse.ArgumentParser(description="3F2 shard sampler showdown vs ground truth.")
    ap.add_argument("--shards", type=int, default=15, help="Number of shards to test.")
    ap.add_argument("--quota", type=int, default=200, help="Target primitive directions per sampler.")
    ap.add_argument("--useful-norm", type=float, default=80.0, help="Sampler max_useful_norm.")
    ap.add_argument("--gt-norm", type=float, default=30.0,
                    help="Ground-truth norm ceiling (80 = full/slow; default contains >> quota).")
    ap.add_argument("--cores", type=int, default=None, help="Ground-truth worker processes.")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="Ground-truth points per task.")
    ap.add_argument("--seed", type=int, default=0, help="Sampler RNG seed.")
    args = ap.parse_args()

    # Imported here so extraction config is set first and import cost is paid once.
    from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
    from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler
    from tests.check_inner_shell_truth import count_inner_shell

    all_shards = _extract_shards()
    n_total = len(all_shards)
    k = min(args.shards, n_total)
    # Even spread across the 60 shards (deterministic) for geometric diversity.
    idx = np.unique(np.linspace(0, n_total - 1, k).astype(int))
    selected = [(int(i), all_shards[int(i)]) for i in idx]

    print(f"\n{'='*100}")
    print(f"3F2(0.5) SHARD SHOWDOWN — {len(selected)} of {n_total} shards | "
          f"quota={args.quota} | sampler max_norm={args.useful_norm:.0f} | "
          f"ground-truth norm<= {args.gt_norm:.0f}")
    print(f"{'='*100}\n")

    header = (f"{'shard':>5} | {'TRUTH<=R':>9} {'avgN':>6} | "
              f"{'Disc.yld':>8} {'avgN':>6} {'acc%':>5} | "
              f"{'LinPT.yld':>9} {'avgN':>6} {'acc%':>5}")
    print(header)
    print("-" * len(header))

    rows = []
    t_start = time.perf_counter()
    for shard_i, A in selected:
        d_yield, d_avg, d_acc = _run_sampler(DiscreteMCMCSampler, A, args.quota, args.useful_norm, args.seed)
        l_yield, l_avg, l_acc = _run_sampler(ParallelTemperingSampler, A, args.quota, args.useful_norm, args.seed)
        truth, truth_avg, info = count_inner_shell(
            A, max_norm=args.gt_norm, num_cores=args.cores, chunk_size=args.chunk, verbose=False,
        )
        rows.append((shard_i, truth, truth_avg, d_yield, d_avg, d_acc, l_yield, l_avg, l_acc))
        print(f"{shard_i:>5} | {truth:>9} {truth_avg:>6.1f} | "
              f"{d_yield:>8} {d_avg:>6.1f} {d_acc*100:>5.0f} | "
              f"{l_yield:>9} {l_avg:>6.1f} {l_acc*100:>5.0f}", flush=True)

    arr = np.array([[r[3], r[4], r[6], r[7]] for r in rows], dtype=float)
    print("-" * len(header))
    print(f"{'MEAN':>5} | {'':>9} {'':>6} | "
          f"{arr[:,0].mean():>8.1f} {arr[:,1].mean():>6.1f} {'':>5} | "
          f"{arr[:,2].mean():>9.1f} {arr[:,3].mean():>6.1f} {'':>5}")
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.0f}s")
    print("Lower avg norm = shorter (better) trajectories; higher yield = more of the quota found.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
