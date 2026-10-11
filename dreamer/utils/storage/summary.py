"""
Markdown summary writer.

Reads the JSONL outputs of a completed pipeline run — the per-shard
``<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl`` trajectory store (flat
directory, one file per shard), plus the optional
``<EXPORT_CMFS>/<constant>/cmfs.jsonl`` metadata file — and renders a
single ``summary.md`` describing what the run scanned, what was found,
and where the best trajectories live.

Each JSONL record is a **flat row for one ``(trajectory, constant)`` pair**
(``delta``, ``identified``, etc. are scalar columns; ``constant`` names the
target); a shard's file therefore holds one row per constant per trajectory
and so still covers all constants searched in that shard.

This-run filtering
==================

Past runs may leave orphan JSONL files in the search-results directory
(e.g. extraction sampling discovered a shard last time but not this
time — the file persists because the writer always appends).  Callers
that want the summary to describe **this run only** should pass
``this_run_shards`` (a mapping from constant name to the set of shard
ids the current pipeline knows about).  Files whose shard_id isn't in
that set are silently skipped.  Passing ``None`` (the default) keeps
the legacy behaviour of summarising every file on disk.

Designed to be safe to call after a partial / failed run: missing
directories, empty files, and non-finite deltas are all tolerated and
produce a report that still makes sense.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Set, Tuple

from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.storage.optimization_objectives import (
    objective_display_label,
    record_raw_value,
    score_record,
)


# ---------------------------------------------------------------------------
# Per-shard / per-cmf aggregation
# ---------------------------------------------------------------------------

class _ShardStats:
    """Aggregate trajectory-level statistics for one shard.

    The "best" trajectory is selected by the active optimisation objective
    (``best_score``, the signed higher-is-better value), while ``best_value`` is
    that objective's raw value for display.  ``positive_delta`` stays a
    δ-specific count (δ is always meaningful — the irrationality measure).
    """

    __slots__ = (
        "shard_id", "cmf_id", "constant", "objective_name", "trajectories",
        "identified", "positive_delta", "best_score", "best_value",
        "best_trajectory_id", "best_start", "best_direction", "interior_point",
    )

    def __init__(self, shard_id: str, cmf_id: str, constant: str, objective_name: str):
        self.shard_id = shard_id
        self.cmf_id = cmf_id
        self.constant = constant
        self.objective_name = objective_name
        self.trajectories = 0
        self.identified = 0
        self.positive_delta = 0
        self.best_score: Optional[float] = None   # signed objective, for selection
        self.best_value: Optional[float] = None   # raw objective, for display
        self.best_trajectory_id: Optional[str] = None
        self.best_start: Optional[list] = None
        self.best_direction: Optional[list] = None
        # Representative integer point inside the shard, as returned by
        # the extractor (legacy lattice scan or v2 MILP).  Filled in from
        # ``ShardDTO.interior_point`` after the trajectory walk if the
        # ``EXPORT_CMFS`` sidecar is available.
        self.interior_point: Optional[Tuple[int, ...]] = None

    def add(self, record: dict) -> None:
        """Accumulate one flat ``(trajectory, constant)`` *record* for this constant.

        The best trajectory is chosen by the active objective's signed score; the
        δ-positive count is tracked separately (δ is always meaningful).
        """
        self.trajectories += 1
        if bool(record.get("identified")):
            self.identified += 1

        delta = _finite_float(record.get("delta"))
        if delta is not None and delta > 0:
            self.positive_delta += 1

        # Best trajectory by the active objective's signed score.
        scored = score_record(record, self.objective_name)
        if scored is None:
            return
        score, _identified = scored
        if not math.isfinite(score):
            return
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_value = _finite_float(
                record_raw_value(record, self.objective_name)
            )
            self.best_trajectory_id = record.get("trajectory_id")
            self.best_start = record.get("start_point")
            self.best_direction = record.get("direction")


def _finite_float(v) -> Optional[float]:
    """Return ``float(v)`` if it's a finite real number, else ``None``.

    The pipeline encodes "did not converge" as ``-inf``; treat that (and any
    ``NaN`` / parse failure) as "no data" so it doesn't poison max/sum stats.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# CMF metadata sidecar
# ---------------------------------------------------------------------------

def _load_cmf_metadata(export_cmfs_root: Optional[str], constant_name: str) -> Dict[str, dict]:
    """Return ``{cmf_id: cmf_dict}`` from the flat ``<root>/cmfs.jsonl``.

    Empty when the file or directory is missing — the summary degrades to
    showing only the per-shard data with no family / shift annotations.
    The *constant_name* parameter is kept for API stability but is no longer
    used to scope the file path (all CMFs are stored together in one file).
    """
    if not export_cmfs_root:
        return {}
    path = os.path.join(export_cmfs_root, "cmfs.jsonl")
    if not os.path.isfile(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cmf_id = record.get("cmf_id")
            if cmf_id:
                out[cmf_id] = record
    return out


def _load_shard_metadata(
    export_cmfs_root: Optional[str], constant_name: str
) -> Dict[str, dict]:
    """Return ``{shard_id: shard_dict}`` from every ``*__shards.jsonl``
    under ``<root>/<safe_const>/`` and, as a fallback, any other constant
    subdirectory under *export_cmfs_root*.

    With multi-constant shards, shard records may only be written under the
    primary constant's directory.  If the requested constant's directory
    yields nothing, we search the other subdirectories so that the
    ``interior_point`` annotation still appears in the summary.
    Missing directories / files degrade silently.
    """
    if not export_cmfs_root or not os.path.isdir(export_cmfs_root):
        return {}

    def _read_dir(const_dir: str) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        if not os.path.isdir(const_dir):
            return result
        for fname in sorted(os.listdir(const_dir)):
            if not fname.endswith("__shards.jsonl"):
                continue
            path = os.path.join(const_dir, fname)
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    shard_id = record.get("shard_id")
                    if shard_id:
                        result[shard_id] = record
        return result

    safe_const = "".join(c for c in constant_name if c.isalnum() or c in ("-", "_"))
    out = _read_dir(os.path.join(export_cmfs_root, safe_const))

    if not out:
        # Fallback: search all constant subdirectories.
        for entry in sorted(os.listdir(export_cmfs_root)):
            if entry == safe_const:
                continue
            candidate = _read_dir(os.path.join(export_cmfs_root, entry))
            out.update(candidate)

    return out


# ---------------------------------------------------------------------------
# Walk EXPORT_SEARCH_RESULTS and build the data model
# ---------------------------------------------------------------------------

def _collect_shard_stats(
    search_results_root: str,
    objective_name: str,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> Dict[str, Dict[str, List[_ShardStats]]]:
    """Walk the flat search-results directory and aggregate per-shard stats.

    JSONL files now live directly in *search_results_root* (one file per
    shard, no constant subdirectory).  Each record is a flat row for one
    ``(trajectory, constant)`` pair, with ``delta`` and ``identified`` as
    scalar columns and ``constant`` naming the target.

    When ``this_run_shards`` is provided it must map constant name to a set
    of shard ids the current run touched.  JSONL files not covered by the
    filter are skipped (orphan protection).

    Returns ``{constant_name: {cmf_id: [shard_stats, ...]}}``.
    """
    out: Dict[str, Dict[str, List[_ShardStats]]] = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(search_results_root):
        return out

    # Build the allowed-shard-ids superset (union across all constants).
    all_allowed: Optional[Set[str]] = None
    if this_run_shards is not None:
        all_allowed = set()
        for ids in this_run_shards.values():
            all_allowed.update(ids)

    for fname in sorted(os.listdir(search_results_root)):
        if not fname.endswith(".jsonl"):
            continue
        # Skip non-shard files (e.g. summary.md itself, though it's .md).
        shard_id = fname[: -len(".jsonl")]
        if all_allowed is not None and shard_id not in all_allowed:
            continue

        cmf_id = shard_id.rsplit("__", 1)[0] if "__" in shard_id else shard_id
        # merged: {trajectory_id: {constant: flat_record}}
        merged = load_seen_trajectories(os.path.join(search_results_root, fname))
        if not merged:
            continue

        # Determine which constants appear in this shard's rows.
        const_names_in_file: Set[str] = set()
        for by_const in merged.values():
            const_names_in_file.update(by_const.keys())

        # If this_run_shards is given, restrict to its constant keys.
        if this_run_shards is not None:
            const_names_in_file = const_names_in_file & set(this_run_shards.keys())
            # Also include constants for which shard_id is listed even if the
            # JSONL has no row yet for that constant (e.g. analyzed but threshold
            # not met — will be backfilled as empty below).
            for c_name, ids in this_run_shards.items():
                if shard_id in ids:
                    const_names_in_file.add(c_name)

        for const_name in sorted(const_names_in_file):
            if this_run_shards is not None and shard_id not in this_run_shards.get(const_name, set()):
                continue
            stats = _ShardStats(shard_id, cmf_id, const_name, objective_name)
            for by_const in merged.values():
                rec = by_const.get(const_name)
                if rec is not None:
                    stats.add(rec)
            out[const_name][cmf_id].append(stats)

    # Backfill empty rows for extracted-but-not-yet-searched shards.
    if this_run_shards is not None:
        for const_name, allowed in this_run_shards.items():
            seen_ids = {
                s.shard_id
                for shards in out[const_name].values()
                for s in shards
            }
            for shard_id in sorted(allowed):
                if shard_id not in seen_ids:
                    cmf_id = shard_id.rsplit("__", 1)[0] if "__" in shard_id else shard_id
                    out[const_name][cmf_id].append(
                        _ShardStats(shard_id, cmf_id, const_name, objective_name)
                    )

    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_metric(value: Optional[float], *, precision: int = 4) -> str:
    """Render a metric value or ``"—"`` for missing data."""
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def _fmt_shard_label(shard_id: str, width: int = 18) -> str:
    """Trim the shard id to its trailing hash for tabular display."""
    tail = shard_id.rsplit("__", 1)[-1]
    return f"`…{tail[:width]}`"


def _fmt_traj_cell(s: _ShardStats) -> str:
    """Render the per-shard "best trajectory" cell.

    Carries both the start→direction (the math handle the user actually
    reads) and the trajectory id's trailing hash (the handle for
    ``examples/search_data.py show-trajectory``).
    """
    if not s.best_trajectory_id:
        return "—"
    tail = s.best_trajectory_id.rsplit("__", 1)[-1]
    start = list(s.best_start) if s.best_start is not None else "?"
    direction = list(s.best_direction) if s.best_direction is not None else "?"
    return f"`{start}` → `{direction}`  (`…{tail[:18]}`)"


def _render_overview(
    data: Dict[str, Dict[str, List[_ShardStats]]],
    label: str,
) -> Tuple[str, Optional[Tuple[str, _ShardStats]]]:
    """Render the top-level per-constant overview table.

    Returns ``(markdown, overall_best)`` where ``overall_best`` is
    ``(constant, shard_stats)`` for the best objective score across the whole run,
    or ``None`` when no finite score was recorded.  *label* is the objective's
    display name (``"δ"`` for the default objective).
    """
    lines = [
        "## Run overview",
        "",
        f"| Constant | CMFs | Shards | Trajectories | Identified | Positive δ | Best {label} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    overall_best: Optional[Tuple[str, _ShardStats]] = None
    for const_name in sorted(data.keys()):
        cmf_shards = data[const_name]
        n_cmfs = len(cmf_shards)
        all_shards = [s for shards in cmf_shards.values() for s in shards]
        n_shards = len(all_shards)
        n_traj = sum(s.trajectories for s in all_shards)
        n_ident = sum(s.identified for s in all_shards)
        n_pos = sum(s.positive_delta for s in all_shards)
        best_for_const = max(
            (s for s in all_shards if s.best_score is not None),
            key=lambda s: s.best_score,
            default=None,
        )
        best_str = _fmt_metric(best_for_const.best_value) if best_for_const else "—"
        lines.append(
            f"| `{const_name}` | {n_cmfs} | {n_shards} | {n_traj} | "
            f"{n_ident} | {n_pos} | **{best_str}** |"
        )
        if best_for_const is not None and (
            overall_best is None
            or best_for_const.best_score > overall_best[1].best_score
        ):
            overall_best = (const_name, best_for_const)

    if overall_best is None:
        lines += ["", "_No converging trajectory recorded in this run._"]
    else:
        const, s = overall_best
        start = list(s.best_start) if s.best_start is not None else "?"
        direction = list(s.best_direction) if s.best_direction is not None else "?"
        lines += [
            "",
            f"**Overall best {label}**: {_fmt_metric(s.best_value)}  ",
            f"- constant: `{const}`  ",
            f"- start point: `{start}`  ",
            f"- direction: `{direction}`  ",
            f"- cmf: `{s.cmf_id}`  ",
            f"- shard: `{s.shard_id}`  ",
            f"- trajectory: `{s.best_trajectory_id}`",
        ]
    return "\n".join(lines), overall_best


def _fmt_interior_point(pt: Optional[Tuple[int, ...]]) -> str:
    """Format a shard's interior witness point for the markdown table."""
    if pt is None:
        return "—"
    return f"`{list(pt)}`"


def _render_per_constant_section(
    const_name: str,
    cmf_shards: Dict[str, List[_ShardStats]],
    cmf_metadata: Dict[str, dict],
    label: str,
) -> str:
    lines = [f"## {const_name}", ""]
    for cmf_id in sorted(cmf_shards.keys()):
        shards = sorted(
            cmf_shards[cmf_id],
            key=lambda s: (s.best_score is None, -(s.best_score or 0.0)),
        )
        meta = cmf_metadata.get(cmf_id, {})
        family = meta.get("family_id")
        shift = meta.get("coordinate_shift")
        n_traj = sum(s.trajectories for s in shards)
        n_ident = sum(s.identified for s in shards)
        n_pos = sum(s.positive_delta for s in shards)
        best_for_cmf = max(
            (s for s in shards if s.best_score is not None),
            key=lambda s: s.best_score,
            default=None,
        )
        best_str = _fmt_metric(best_for_cmf.best_value) if best_for_cmf else "—"

        lines.append(f"### CMF `{cmf_id}`")
        bullets: List[str] = []
        if family:
            bullets.append(f"- Family: `{family}`")
        if shift is not None:
            bullets.append(f"- Coordinate shift: `{list(shift)}`")
        bullets.append(f"- Shards: **{len(shards)}**")
        bullets.append(
            f"- Trajectories: **{n_traj}**  "
            f"(identified: **{n_ident}**, positive δ: **{n_pos}**)"
        )
        bullets.append(f"- Best {label} within CMF: **{best_str}**")
        lines += bullets + [""]

        lines += [
            f"| Shard | Start point | Trajectories | Identified | Positive δ | Best {label} | Best trajectory (start → direction) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for s in shards:
            lines.append(
                f"| {_fmt_shard_label(s.shard_id)} | "
                f"{_fmt_interior_point(s.interior_point)} | "
                f"{s.trajectories} | {s.identified} | {s.positive_delta} | "
                f"{_fmt_metric(s.best_value)} | {_fmt_traj_cell(s)} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_summary_markdown(
    *,
    search_results_root: str,
    export_cmfs_root: Optional[str] = None,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> str:
    """Build the full ``summary.md`` content for a pipeline run.

    Parameters
    ----------
    search_results_root:
        Root directory containing per-constant subdirectories of per-shard
        JSONL files (``sys_config.EXPORT_SEARCH_RESULTS``).
    export_cmfs_root:
        Optional sidecar root where CMF DTOs live
        (``sys_config.EXPORT_CMFS``) — used only to annotate per-CMF rows
        with family / coordinate-shift info.  Missing files are tolerated.
    this_run_shards:
        Optional ``{constant_name: {shard_id, ...}}`` mapping describing
        which shards the *current* pipeline run actually touched.  When
        provided, any JSONL whose shard id is not in the set is treated
        as a stale leftover from a past run and dropped.  ``None``
        (default) summarises every file on disk.
    """
    from dreamer.configs import config
    objective_name = config.system.OPTIMIZATION_OBJECTIVE
    label = objective_display_label(objective_name)

    data = _collect_shard_stats(search_results_root, objective_name, this_run_shards)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = [
        "# Pipeline Summary",
        "",
        f"_Generated: {timestamp}_",
        "",
        f"Search-results root: `{search_results_root}`",
        f"Optimisation objective: `{objective_name}`",
    ]
    if export_cmfs_root:
        header.append(f"CMF-metadata root: `{export_cmfs_root}`")
    if this_run_shards is not None:
        # Unique shard IDs (a shard may appear under multiple constants).
        unique_shards = {sid for ids in this_run_shards.values() for sid in ids}
        header.append(
            f"Scope: this run only ({len(unique_shards)} shard(s) across "
            f"{len(this_run_shards)} constant(s))"
        )
    header.append("")

    if not data:
        if this_run_shards is not None:
            header += ["_Extraction produced no shards in this run._", ""]
        else:
            header += [
                "_No per-shard JSONL files were found — nothing to summarise._",
                "",
            ]
        return "\n".join(header)

    # Backfill each shard's interior witness point from the
    # ``<EXPORT_CMFS>/<const>/<cmf>__shards.jsonl`` sidecar so the
    # per-shard table can show where the search actually started.
    for const_name, cmf_shards in data.items():
        shard_meta = _load_shard_metadata(export_cmfs_root, const_name)
        if not shard_meta:
            continue
        for shards in cmf_shards.values():
            for s in shards:
                rec = shard_meta.get(s.shard_id)
                if not rec:
                    continue
                pt = rec.get("interior_point")
                if pt is not None:
                    try:
                        s.interior_point = tuple(int(v) for v in pt)
                    except (TypeError, ValueError):
                        s.interior_point = tuple(pt)

    overview_md, _overall_best = _render_overview(data, label)

    sections: List[str] = []
    for const_name in sorted(data.keys()):
        cmf_metadata = _load_cmf_metadata(export_cmfs_root, const_name)
        sections.append(
            _render_per_constant_section(
                const_name, data[const_name], cmf_metadata, label
            )
        )

    return "\n".join(header + [overview_md, "", "---", ""] + sections)


def write_summary(
    *,
    search_results_root: str,
    export_cmfs_root: Optional[str] = None,
    output_path: Optional[str] = None,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> Optional[str]:
    """Render and write ``summary.md`` for a pipeline run.

    The output defaults to ``<search_results_root>/summary.md``.  Returns
    the path written, or ``None`` when ``search_results_root`` itself is
    missing (caller should treat that as "nothing to summarise").

    See :func:`build_summary_markdown` for the meaning of
    ``this_run_shards``.
    """
    if not os.path.isdir(search_results_root):
        return None
    if output_path is None:
        output_path = os.path.join(search_results_root, "summary.md")
    content = build_summary_markdown(
        search_results_root=search_results_root,
        export_cmfs_root=export_cmfs_root,
        this_run_shards=this_run_shards,
    )
    with open(output_path, "w") as f:
        f.write(content)
    return output_path
