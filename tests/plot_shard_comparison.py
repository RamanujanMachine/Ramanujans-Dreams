"""Per-shard comparison: base raycast sampler vs production Linear-PT sampler.

For each of the 15 selected 3F2(0.5) shards (the same even spread used by
``examples/benchmark_3f2_samplers.py``) this runs both trajectory samplers at the
same norm ceiling and quota and renders a 4-pane comparison PNG into a dedicated
directory so each shard can be inspected one by one:

* **Base raycast** (:class:`RaycastPipelineSampler`) — the original guide-ray +
  raycast production engine.  Its norm cap is ``search_config.MAX_TRAJECTORY_LENGTH``,
  set to ``--useful-norm`` here for a fair comparison.
* **Linear PT** (:class:`ParallelTemperingSampler`) — the current production engine
  (``max_useful_norm = --useful-norm``).

Per shard the 4 panes are:
  1. **Norm distribution** — overlaid histograms of harvested L2 norms (+ mean lines),
     with the ground-truth average (parsed from the benchmark file) as a reference.
  2. **Angular spread** — CDF of each ray's nearest-neighbour angle (further right =
     more uniform / better separated directions).
  3. **Summary bars** — mean / median norm and mean NN-angle, side by side.
  4. **Stats text** — yield, mean/median norm, mean NN-angle, PT acceptance, and the
     ground-truth count/avg for context.

Diagnostics reuse the helpers developed for the discrete harness
(``_unit_pca_spectrum`` and the cosine NN-angle computation); ground truth is reused
from ``temp/benchmark_3f2_results.txt`` (shard IDs match) rather than recomputed.

CLI: ``python tests/plot_shard_comparison.py [--shards 15] [--quota 200]
[--useful-norm 80] [--outdir temp/shard_comparisons] [--seed 0]``.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BASE_COLOR = "#e8843c"   # base raycast (orange)
_PT_COLOR = "#2e7d32"     # linear PT (green)
_GT_COLOR = "#555555"     # ground-truth reference


def _nn_angles_deg(rays: np.ndarray) -> np.ndarray:
    """Nearest-neighbour angle (degrees) of each ray to its closest other direction.

    :param rays: ``(n, d)`` harvested integer vectors (n >= 2).
    :return: ``(n,)`` array of nearest-neighbour angles in degrees.
    """
    u = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    cos = np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(cos, -1.0)               # ignore self
    return np.degrees(np.arccos(cos.max(axis=1)))


def _metrics(rays: np.ndarray) -> Dict[str, float]:
    """Summary metrics for one harvested set.

    :param rays: ``(n, d)`` harvested vectors.
    :return: dict with yield, mean/median norm, and mean NN-angle (0 if too few points).
    """
    n = int(rays.shape[0])
    if n == 0:
        return {"yield": 0, "norm_mean": 0.0, "norm_median": 0.0, "nn_mean": 0.0}
    lengths = np.linalg.norm(rays, axis=1)
    nn_mean = float(_nn_angles_deg(rays).mean()) if n >= 2 else 0.0
    return {
        "yield": n,
        "norm_mean": float(lengths.mean()),
        "norm_median": float(np.median(lengths)),
        "nn_mean": nn_mean,
    }


def _parse_truth(results_path: str) -> Dict[int, Tuple[int, float]]:
    """Parse ground-truth ``{shard: (count, avg_norm)}`` from a benchmark results file.

    :param results_path: path to ``benchmark_3f2_results.txt`` (may be absent).
    :return: mapping shard id -> (truth_count, truth_avg_norm); empty if unavailable.
    """
    import re
    truth: Dict[int, Tuple[int, float]] = {}
    if not os.path.exists(results_path):
        return truth
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    with open(results_path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            parts = ansi.sub("", raw).strip().split("|")            # shard | TRUTH avgN | ...
            if len(parts) < 4 or not parts[0].strip().lstrip("-").isdigit():
                continue
            tfields = parts[1].split()                              # [count, avgN]
            if len(tfields) >= 2:
                truth[int(parts[0].strip())] = (int(tfields[0]), float(tfields[1]))
    return truth


def _render_shard(shard_i, base_rays, pt_rays, pt_accept, truth, out_path):
    """Render the 4-pane base-vs-PT comparison figure for one shard.

    :param shard_i: shard id (for titles / filename).
    :param base_rays: ``(n, d)`` raycast harvest.
    :param pt_rays: ``(n, d)`` Linear-PT harvest.
    :param pt_accept: PT acceptance rate (fraction).
    :param truth: ``(count, avg_norm)`` ground-truth tuple or ``None``.
    :param out_path: PNG output path.
    """
    bm = _metrics(base_rays)
    pm = _metrics(pt_rays)
    gt_avg = truth[1] if truth else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"3F2(0.5) shard {shard_i} — Base Raycast vs Linear PT   "
        f"(yield: base {bm['yield']} vs PT {pm['yield']} / quota)",
        fontsize=14, weight="bold")

    # --- Pane 1: norm distribution ---
    ax = axes[0, 0]
    bl = np.linalg.norm(base_rays, axis=1) if bm["yield"] else np.array([])
    pl = np.linalg.norm(pt_rays, axis=1) if pm["yield"] else np.array([])
    hi = max([bl.max(initial=1.0), pl.max(initial=1.0)])
    bins = np.linspace(0, hi, 40)
    if bl.size:
        ax.hist(bl, bins=bins, alpha=0.55, color=_BASE_COLOR,
                label=f"Base raycast (n={bm['yield']})")
        ax.axvline(bm["norm_mean"], color=_BASE_COLOR, ls="-", lw=2)
    if pl.size:
        ax.hist(pl, bins=bins, alpha=0.55, color=_PT_COLOR,
                label=f"Linear PT (n={pm['yield']})")
        ax.axvline(pm["norm_mean"], color=_PT_COLOR, ls="-", lw=2)
    if gt_avg:
        ax.axvline(gt_avg, color=_GT_COLOR, ls="--", lw=1.6, label=f"GT avg ≈ {gt_avg:.1f}")
    ax.set_title("Trajectory L2-norm distribution (lower = shorter)")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("count")
    ax.legend(frameon=False)

    # --- Pane 2: angular spread (NN-angle CDF) ---
    ax = axes[0, 1]
    for rays, color, label, m in ((base_rays, _BASE_COLOR, "Base raycast", bm),
                                  (pt_rays, _PT_COLOR, "Linear PT", pm)):
        if m["yield"] >= 2:
            nn = np.sort(_nn_angles_deg(rays))
            ax.plot(nn, np.linspace(0, 1, nn.size), color=color, lw=2,
                    label=f"{label} (mean {m['nn_mean']:.1f}°)")
    ax.set_title("Nearest-neighbour angle CDF (further right = more uniform)")
    ax.set_xlabel("NN angle (degrees)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False)
    ax.grid(ls=":", alpha=0.5)

    # --- Pane 3: summary bars ---
    ax = axes[1, 0]
    labels = ["mean norm", "median norm", "mean NN-angle°"]
    base_vals = [bm["norm_mean"], bm["norm_median"], bm["nn_mean"]]
    pt_vals = [pm["norm_mean"], pm["norm_median"], pm["nn_mean"]]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, base_vals, w, color=_BASE_COLOR, label="Base raycast")
    ax.bar(x + w / 2, pt_vals, w, color=_PT_COLOR, label="Linear PT")
    for xi, (bv, pv) in enumerate(zip(base_vals, pt_vals)):
        ax.text(xi - w / 2, bv, f"{bv:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, pv, f"{pv:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Summary metrics")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # --- Pane 4: stats text ---
    ax = axes[1, 1]
    ax.axis("off")
    gt_txt = (f"count ≤ GT-R: {truth[0]:,}\navg norm: {truth[1]:.1f}"
              if truth else "(not available)")
    lines = [
        "BASE RAYCAST",
        f"  yield:       {bm['yield']} / {pm['yield'] if False else ''}".rstrip(" /"),
        f"  mean norm:   {bm['norm_mean']:.2f}",
        f"  median norm: {bm['norm_median']:.2f}",
        f"  mean NN ang: {bm['nn_mean']:.2f}°",
        "",
        "LINEAR PT (production)",
        f"  yield:       {pm['yield']}",
        f"  mean norm:   {pm['norm_mean']:.2f}",
        f"  median norm: {pm['norm_median']:.2f}",
        f"  mean NN ang: {pm['nn_mean']:.2f}°",
        f"  accept rate: {pt_accept * 100:.1f}%",
        "",
        "GROUND TRUTH",
        f"  {gt_txt}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    """CLI entry point: run both samplers on each shard and render comparison PNGs."""
    ap = argparse.ArgumentParser(description="Per-shard base-raycast vs Linear-PT comparison.")
    ap.add_argument("--shards", type=int, default=15, help="Number of shards to compare.")
    ap.add_argument("--quota", type=int, default=200, help="Target directions per sampler.")
    ap.add_argument("--useful-norm", type=float, default=80.0, help="Shared norm ceiling.")
    ap.add_argument("--outdir", default=os.path.join("temp", "shard_comparisons"),
                    help="Directory for the per-shard PNGs.")
    ap.add_argument("--results", default=os.path.join("temp", "benchmark_3f2_results.txt"),
                    help="Benchmark file to reuse ground truth from.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = ap.parse_args()

    from dreamer.configs import config
    # Set the raycast norm cap to match the PT ceiling BEFORE importing the samplers'
    # config-bound modules so both engines compete under the same norm budget.
    config.configure(
        extraction={"STRATEGY": "heuristic", "IGNORE_DUPLICATE_SEARCHABLES": True,
                    "LOAD_SHARD_CACHE": False},
        search={"MAX_TRAJECTORY_LENGTH": int(args.useful_norm)},
        logging={"GENERATE_LOGS": False},
    )
    import sympy as sp
    from dreamer.loading import pFq
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer import log
    from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
    from dreamer.extraction.samplers.parallel_tempering_raycaster import ParallelTemperingSampler

    os.makedirs(args.outdir, exist_ok=True)
    truth_map = _parse_truth(args.results)

    c = log(2)
    cmf = pFq(c, 3, 2, sp.Rational(1, 2)).to_cmf()
    all_shards = [np.asarray(s.A, dtype=np.float64)
                  for s in ShardExtractor(c, cmf).extract() if s.A is not None]
    n_total = len(all_shards)
    k = min(args.shards, n_total)
    idx = np.unique(np.linspace(0, n_total - 1, k).astype(int))

    summary_rows = []
    for shard_i in idx:
        shard_i = int(shard_i)
        A = all_shards[shard_i]

        np.random.seed(args.seed)
        base_rays = np.asarray(RaycastPipelineSampler(A).harvest(int(args.quota), exact=True))

        pt = ParallelTemperingSampler(A, max_useful_norm=float(args.useful_norm), rng_seed=args.seed)
        pt_rays = np.asarray(pt.harvest(int(args.quota), exact=True))
        pt_accept = float(pt.last_accept_rate)

        truth = truth_map.get(shard_i)
        out_path = os.path.join(args.outdir, f"shard_{shard_i:02d}.png")
        _render_shard(shard_i, base_rays, pt_rays, pt_accept, truth, out_path)

        bm, pm = _metrics(base_rays), _metrics(pt_rays)
        summary_rows.append({
            "shard": shard_i,
            "base_yield": bm["yield"], "base_norm_mean": round(bm["norm_mean"], 2),
            "base_nn_deg": round(bm["nn_mean"], 2),
            "pt_yield": pm["yield"], "pt_norm_mean": round(pm["norm_mean"], 2),
            "pt_nn_deg": round(pm["nn_mean"], 2),
            "pt_accept_pct": round(pt_accept * 100, 1),
            "gt_count": truth[0] if truth else "", "gt_avg_norm": round(truth[1], 2) if truth else "",
        })
        print(f"shard {shard_i:>2}: base yld={bm['yield']:>3} normμ={bm['norm_mean']:6.2f} "
              f"nn={bm['nn_mean']:5.1f}°  |  PT yld={pm['yield']:>3} normμ={pm['norm_mean']:6.2f} "
              f"nn={pm['nn_mean']:5.1f}° acc={pt_accept*100:4.0f}%  -> {out_path}", flush=True)

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    arr = np.array([[r["base_norm_mean"], r["base_nn_deg"], r["pt_norm_mean"], r["pt_nn_deg"]]
                    for r in summary_rows], dtype=float)
    print("\n" + "=" * 70)
    print(f"MEANS over {len(summary_rows)} shards:")
    print(f"  Base raycast: norm {arr[:,0].mean():.2f}, NN-angle {arr[:,1].mean():.1f}°")
    print(f"  Linear PT:    norm {arr[:,2].mean():.2f}, NN-angle {arr[:,3].mean():.1f}°")
    print(f"Per-shard PNGs + summary.csv in: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
