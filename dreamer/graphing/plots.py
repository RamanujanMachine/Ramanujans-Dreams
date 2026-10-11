"""
Thin matplotlib wrappers for the post-process graphing stage.

A non-interactive ``Agg`` backend is forced so the stage runs headless (WSL /
CI / inside the pipeline) without a display.  Each function writes one figure to
*out_path* and closes it (no global figure-state leakage across many shards).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def plot_delta_sequence(
    deltas: Sequence[float],
    out_path: str,
    *,
    title: str,
) -> None:
    """Line plot of a δ-sequence (δ vs walk step), saved to *out_path*."""
    steps = range(1, len(deltas) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, deltas, lw=1.0)
    ax.set_xlabel("walk step")
    ax.set_ylabel("δ (irrationality measure)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_histogram(
    values: Sequence[float],
    out_path: str,
    *,
    bins: int,
    title: str,
    xlabel: str = "δ (irrationality measure)",
) -> None:
    """Histogram of *values*, saved to *out_path*."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(list(values), bins=bins, color="#4060c0", alpha=0.85, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trajectory count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
