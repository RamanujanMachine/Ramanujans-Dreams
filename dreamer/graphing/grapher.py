"""
``Grapher`` — the post-process graphing stage.

Invoked at the end of the Tier-3 post-process stage (or standalone) to turn the
per-shard JSONL search results into figures and tables under
``sys_config.EXPORT_GRAPHS``.  Driven entirely by :class:`GraphConfig`; when no
graph kind is enabled it does nothing (no directory created, no files read).

Three graph kinds (see :class:`GraphConfig`):

  1. **Best-trajectory δ-sequence** — one line plot per ``(CMF, constant)`` of δ
     over the first ``DELTA_SEQUENCE_DEPTH`` walk steps of the best-δ trajectory.
     The only kind that walks a trajectory (just the single best one per group).
  2. **δ histograms** — one per ``(constant, shard)`` and one aggregated per
     ``(constant, CMF)``, from the stored scalar ``delta`` column (cheap).
  3. **Bumpiness table** — per-shard CSV + markdown with the semivariogram-based
     spatial roughness and the median δ-sequence total variation
     (see :mod:`dreamer.graphing.bumpiness`).
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.rand import derive_rng
from dreamer.utils.storage import Formats
from dreamer.utils.storage.handler_reconstruction import (
    build_cmf_lookup_from_priorities,
    reconstruct_positions,
)
from dreamer.utils.storage.record_metrics import delta_metric
from dreamer.utils.storage.optimization_objectives import (
    objective_display_label,
    record_raw_value,
    score_record,
)
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    derive_cmf_and_shard_ids,
)
from dreamer.graphing.bumpiness import (
    empirical_semivariogram,
    median_total_variation,
)

graph_config = config.graph


def _safe(name: str) -> str:
    """Filesystem-safe slug for a CMF / constant / encoding string."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or "x"


class Grapher:
    """Generate the configured post-process graphs from the search-results JSONL."""

    def __init__(self, priorities):
        """:param priorities: ``{Constant: [Searchable, ...]}`` from the search stage."""
        self.priorities = priorities
        self._cmf_lookup = build_cmf_lookup_from_priorities(priorities)
        self._dir = sys_config.EXPORT_SEARCH_RESULTS
        self._suffix = "." + Formats.JSONL.value
        self._out = sys_config.EXPORT_GRAPHS

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self) -> None:
        if not graph_config.any_enabled():
            return
        if not os.path.isdir(self._dir):
            Logger(
                f"No search results directory at {self._dir}; skipping graphing.",
                Logger.Levels.warning,
            ).log()
            return
        os.makedirs(self._out, exist_ok=True)

        Logger(
            Logger.buffer_print(
                sys_config.LOGGING_BUFFER_SIZE, "Post-process: graphs", "="
            ),
            Logger.Levels.message,
        ).log()

        # Group shard files by (constant, cmf).  ``delta`` is per-constant, so a
        # shard searched under multiple constants is graphed once per constant.
        groups = self._group()

        if graph_config.PLOT_DELTA_HISTOGRAMS:
            self._delta_histograms(groups)
        if graph_config.PLOT_BEST_DELTA_SEQUENCE:
            self._best_delta_sequences(groups)
        if graph_config.WRITE_BUMPINESS_TABLE:
            self._bumpiness_table(groups)

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group(self) -> Dict[Tuple[str, str], Dict[str, object]]:
        """Return ``{(const_name, cmf_id): {constant, cmf_id, shards:[(enc, path)]}}``."""
        groups: Dict[Tuple[str, str], Dict[str, object]] = {}
        for constant, shards in self.priorities.items():
            const_name = getattr(constant, "name", None)
            for shard in shards:
                try:
                    cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
                except Exception:
                    continue
                path = os.path.join(self._dir, shard_id + self._suffix)
                if not os.path.isfile(path):
                    continue
                key = (const_name, cmf_id)
                grp = groups.setdefault(
                    key, {"constant": constant, "cmf_id": cmf_id, "shards": []}
                )
                grp["shards"].append((enc, path))
        return groups

    # ------------------------------------------------------------------
    # (2) δ histograms
    # ------------------------------------------------------------------

    def _delta_histograms(self, groups) -> None:
        from dreamer.graphing.plots import plot_histogram

        bins = graph_config.HISTOGRAM_BINS
        for (const_name, cmf_id), grp in groups.items():
            cmf_deltas: List[float] = []
            for enc, path in grp["shards"]:
                deltas = self._shard_deltas(path, const_name)
                cmf_deltas.extend(deltas)
                if deltas:
                    out = os.path.join(
                        self._out,
                        f"hist_delta__{_safe(cmf_id)}__{_safe(const_name)}__shard_{_safe(enc)}.png",
                    )
                    plot_histogram(
                        deltas, out, bins=bins,
                        title=f"δ histogram — {cmf_id} [{const_name}] shard {enc}",
                    )
            if cmf_deltas:
                out = os.path.join(
                    self._out, f"hist_delta__{_safe(cmf_id)}__{_safe(const_name)}__CMF.png"
                )
                plot_histogram(
                    cmf_deltas, out, bins=bins,
                    title=f"δ histogram — {cmf_id} [{const_name}] (whole CMF)",
                )

    # ------------------------------------------------------------------
    # (1) Best-trajectory δ-sequence
    # ------------------------------------------------------------------

    def _best_delta_sequences(self, groups) -> None:
        from dreamer.graphing.plots import plot_delta_sequence

        depth = graph_config.DELTA_SEQUENCE_DEPTH
        label = objective_display_label(config.system.OPTIMIZATION_OBJECTIVE)
        for (const_name, cmf_id), grp in groups.items():
            best = self._best_record(grp["shards"], const_name)
            if best is None:
                continue
            best_value, record = best
            cmf = self._cmf_lookup.get(cmf_id)
            if cmf is None:
                Logger(
                    f"Graphing skip δ-seq: no CMF for {cmf_id!r}", Logger.Levels.warning
                ).log()
                continue
            constant = grp["constant"]
            constant_sympy = getattr(constant, "value_sympy", None)
            try:
                start, direction = reconstruct_positions(cmf, record)
                handler = TrajectoryAttributesHandler.from_cmf(
                    cmf, direction, start, constant=constant_sympy, walk_depth=depth,
                )
                deltas = handler.delta_sequence(depth)
            except Exception as e:
                Logger(
                    f"Graphing δ-seq error for {cmf_id} [{const_name}]: {e}",
                    Logger.Levels.warning,
                ).log()
                continue
            if not deltas:
                continue
            out = os.path.join(
                self._out, f"best_delta_seq__{_safe(cmf_id)}__{_safe(const_name)}.png"
            )
            plot_delta_sequence(
                deltas, out,
                title=(
                    f"Best-{label} trajectory δ-sequence — {cmf_id} [{const_name}] "
                    f"({label}≈{best_value:.4f}, first {len(deltas)} steps)"
                ),
            )

    # ------------------------------------------------------------------
    # (3) Bumpiness table
    # ------------------------------------------------------------------

    def _bumpiness_table(self, groups) -> None:
        rows: List[dict] = []
        for (const_name, cmf_id), grp in groups.items():
            for enc, path in grp["shards"]:
                records = load_seen_trajectories(path)  # {tid: {const: record}}
                directions: List[list] = []
                deltas: List[float] = []
                seqs: List[list] = []
                for by_const in records.values():
                    rec = by_const.get(const_name)
                    if rec is None:
                        continue
                    d = delta_metric(rec, const_name)
                    direction = rec.get("direction")
                    if d is not None and direction:
                        directions.append(direction)
                        deltas.append(d)
                    seq = rec.get("delta_sequence")
                    if isinstance(seq, (list, tuple)) and len(seq) >= 2:
                        seqs.append(seq)

                rng = derive_rng(cmf_id, enc, "variogram")
                vario = empirical_semivariogram(
                    directions, deltas,
                    lag_bins=graph_config.VARIOGRAM_LAG_BINS,
                    max_pairs=graph_config.VARIOGRAM_MAX_PAIRS,
                    rng=rng,
                )
                median_tv, n_tv = median_total_variation(seqs)
                rows.append({
                    "cmf": cmf_id,
                    "constant": const_name,
                    "shard": enc,
                    "n_trajectories": vario["n_points"],
                    "relative_nugget": vario["relative_nugget"],
                    "nugget": vario["nugget"],
                    "sill": vario["sill"],
                    "initial_slope": vario["initial_slope"],
                    "median_delta_seq_TV": median_tv,
                    "n_TV": n_tv,
                })

        if not rows:
            return
        self._write_bumpiness_csv(rows)
        self._write_bumpiness_md(rows)

    def _write_bumpiness_csv(self, rows: List[dict]) -> None:
        out = os.path.join(self._out, "bumpiness.csv")
        fields = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_bumpiness_md(self, rows: List[dict]) -> None:
        out = os.path.join(self._out, "bumpiness.md")
        fields = list(rows[0].keys())

        def fmt(v):
            if isinstance(v, float):
                return "nan" if v != v else f"{v:.4g}"
            return str(v)

        lines = [
            "# Shard bumpiness (δ non-smoothness)",
            "",
            "* **relative_nugget** = nugget / sill of the empirical semivariogram of "
            "δ over angular direction-distance — ≈1 → needle/bumpy (no spatial "
            "structure), ≈0 → smooth.  Density-robust (pairs binned by distance).",
            "* **median_delta_seq_TV** = median over trajectories of the total "
            "variation of the stored δ-sequence (convergence wobble); needs "
            "`delta_sequence` in TIER3_ATTRIBUTES, else nan.",
            "",
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
        ]
        for r in rows:
            lines.append("| " + " | ".join(fmt(r[k]) for k in fields) + " |")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    def _shard_deltas(self, path: str, const_name: Optional[str]) -> List[float]:
        records = load_seen_trajectories(path)  # {tid: {const: record}}
        out: List[float] = []
        for by_const in records.values():
            rec = by_const.get(const_name)
            if rec is None:
                continue
            d = delta_metric(rec, const_name)
            if d is not None:
                out.append(d)
        return out

    def _best_record(
        self, shards, const_name: Optional[str]
    ) -> Optional[Tuple[float, dict]]:
        """Return ``(best_objective_value, record)`` over all shards of one
        ``(constant, CMF)``.

        "Best" is chosen by the active optimisation objective's signed score (so it
        is correct for both larger- and smaller-is-better objectives); the returned
        value is that objective's raw value for display.
        """
        objective_name = config.system.OPTIMIZATION_OBJECTIVE
        best_score = -float("inf")
        best: Optional[Tuple[float, dict]] = None
        for _enc, path in shards:
            for by_const in load_seen_trajectories(path).values():
                rec = by_const.get(const_name)
                if rec is None:
                    continue
                scored = score_record(rec, objective_name)
                if scored is None:
                    continue
                score, _identified = scored
                if score != score or score == -float("inf"):  # NaN / worst sentinel
                    continue
                if best is None or score > best_score:
                    best_score = score
                    raw = record_raw_value(rec, objective_name)
                    best = (raw if raw is not None else score, rec)
        return best
