"""
Freshness report of stored trajectory attributes.

Answers "what would the next run reuse, and what would it recompute?" for a
results directory under the **current** config and ``system.REUSE_POLICY`` —
per attribute: rows where it is fresh, stale (stored with another fingerprint)
or missing.  Useful before a long rerun, after changing a knob, or to check that
a stage did what was expected.

Example::

    from dreamer.utils.storage.attribute_report import attribute_report, format_attribute_report
    print(format_attribute_report(attribute_report("./search results")))
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from typing import Dict, Iterable, Optional

from dreamer.configs import config
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.storage.attribute_fingerprints import (
    TIER1,
    attribute_fingerprint,
    reuse_policy,
    tier1_record_fresh,
)
from dreamer.utils.storage.attribute_registry import attribute_name


def attribute_report(
    results_dir: str,
    attributes: Optional[Iterable[str]] = None,
) -> Dict[str, Counter]:
    """Count fresh / stale / missing values per attribute in *results_dir*.

    :param results_dir: Directory of per-shard ``*.jsonl`` files.
    :param attributes: Attribute names to report (default: the configured Tier-2
        and Tier-3 lists).  The Tier-1 core is always reported as ``"tier1"``.
    :return: ``{attribute: Counter(fresh=…, stale=…, missing=…)}`` over all
        ``(trajectory, constant)`` rows.
    """
    if attributes is None:
        attributes = [attribute_name(s) for s in config.search.TIER2_ATTRIBUTES]
        attributes += [attribute_name(s) for s in config.post_process.TIER3_ATTRIBUTES]
    attributes = list(dict.fromkeys(attributes))
    keep = reuse_policy() == "keep"
    report: Dict[str, Counter] = {TIER1: Counter(), **{a: Counter() for a in attributes}}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        for by_const in load_seen_trajectories(path).values():
            for rec in by_const.values():
                depth = rec.get("walk_depth")
                t1 = tier1_record_fresh(rec, attribute_fingerprint(TIER1, depth))
                report[TIER1]["fresh" if t1 else "stale"] += 1
                fps = rec.get("fingerprints") or {}
                for a in attributes:
                    if a not in rec and f"{a}_error" not in rec:
                        report[a]["missing"] += 1
                    elif keep or fps.get(a) == attribute_fingerprint(a, depth):
                        report[a]["fresh"] += 1
                    else:
                        report[a]["stale"] += 1
    return report


def format_attribute_report(report: Dict[str, Counter]) -> str:
    """Render :func:`attribute_report` as an aligned text table."""
    lines = [f"{'attribute':<30} {'fresh':>8} {'stale':>8} {'missing':>8}"]
    for name, c in report.items():
        lines.append(f"{name:<30} {c['fresh']:>8} {c['stale']:>8} {c['missing']:>8}")
    return "\n".join(lines)
