"""Render the 3F2 shard-showdown benchmark as two professional PNG graphs.

Parses the text table written by ``examples/benchmark_3f2_samplers.py`` (default
``temp/benchmark_3f2_results.txt``) and produces:

* **Graph 1 — Average Norm Comparison** (grouped bar chart): per shard, the avg
  L2 norm of Ground Truth, Discrete MCMC, and Linear PT MCMC side by side, with a
  horizontal dashed line at the overall ground-truth average (the space's baseline
  point length) so the samplers' performance is read against capacity.
* **Graph 2 — Yield Efficiency** (line/scatter): absolute yield of Linear PT vs
  Discrete MCMC across the shards.

The parser tolerates both the current 3-sampler table and the older 4-column
table (with the removed Harmonic PT) by reading only the first three ``|``-
delimited groups (truth, discrete, linear-PT).

CLI: ``python tests/plot_3f2_benchmark.py [--results PATH] [--outdir DIR]``.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PNG
import matplotlib.pyplot as plt
import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_results(path: str) -> List[Tuple[int, float, int, float, int, float]]:
    """Parse the benchmark table into per-shard records.

    :param path: path to the benchmark results text file.
    :return: list of ``(shard, truth_avg, disc_yield, disc_avg, lin_yield, lin_avg)``.
    :raises ValueError: if no data rows are found.
    """
    rows: List[Tuple[int, float, int, float, int, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = _ANSI.sub("", raw).strip()
            parts = line.split("|")
            if len(parts) < 4:
                continue
            head = parts[0].strip()
            if not head.lstrip("-").isdigit():   # skip header / MEAN / separators / warnings
                continue
            shard = int(head)
            truth = parts[1].split()             # [count, avgN]
            disc = parts[2].split()              # [yield, avgN, acc]
            lin = parts[3].split()               # [yield, avgN, acc]
            rows.append((shard, float(truth[1]),
                         int(disc[0]), float(disc[1]),
                         int(lin[0]), float(lin[1])))
    if not rows:
        raise ValueError(f"No data rows parsed from {path!r}.")
    return rows


def plot_avg_norm(rows, out_path: str) -> None:
    """Graph 1 — grouped bar chart of avg L2 norm (Ground Truth / Discrete / Linear PT)."""
    shards = [r[0] for r in rows]
    truth_avg = np.array([r[1] for r in rows])
    disc_avg = np.array([r[3] for r in rows])
    lin_avg = np.array([r[5] for r in rows])

    x = np.arange(len(shards))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, truth_avg, w, label="Ground Truth (all pts ≤ R)", color="#9aa0a6")
    ax.bar(x, disc_avg, w, label="Discrete MCMC", color="#e8843c")
    ax.bar(x + w, lin_avg, w, label="Linear PT MCMC", color="#2e7d32")

    gt_mean = float(truth_avg.mean())
    ax.axhline(gt_mean, ls="--", lw=1.6, color="#555555",
               label=f"Ground-truth avg ≈ {gt_mean:.1f}")

    ax.set_xlabel("Shard ID")
    ax.set_ylabel("Average L2 norm  (lower = shorter / better)")
    ax.set_title("3F2(0.5) Shard Showdown — Average Trajectory Norm vs Ground Truth")
    ax.set_xticks(x)
    ax.set_xticklabels(shards)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_yield(rows, out_path: str) -> None:
    """Graph 2 — yield efficiency of Linear PT vs Discrete MCMC across shards."""
    shards = [r[0] for r in rows]
    disc_y = [r[2] for r in rows]
    lin_y = [r[4] for r in rows]
    x = np.arange(len(shards))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, disc_y, "-o", color="#e8843c", label="Discrete MCMC")
    ax.plot(x, lin_y, "-s", color="#2e7d32", label="Linear PT MCMC")
    ax.set_xlabel("Shard ID")
    ax.set_ylabel("Yield (primitive directions found, quota 200)")
    ax.set_title("3F2(0.5) Shard Showdown — Yield Efficiency: Linear PT vs Discrete MCMC")
    ax.set_xticks(x)
    ax.set_xticklabels(shards)
    ax.set_ylim(0, 210)
    ax.legend(frameon=False)
    ax.grid(ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    """CLI entry point: parse the results file and render both graphs."""
    ap = argparse.ArgumentParser(description="Plot the 3F2 sampler-showdown benchmark.")
    ap.add_argument("--results", default=os.path.join("temp", "benchmark_3f2_results.txt"),
                    help="Path to the benchmark results text file.")
    ap.add_argument("--outdir", default="temp", help="Directory for the output PNGs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = parse_results(args.results)
    print(f"Parsed {len(rows)} shards from {args.results}.")
    plot_avg_norm(rows, os.path.join(args.outdir, "3f2_avg_norm.png"))
    plot_yield(rows, os.path.join(args.outdir, "3f2_yield.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
