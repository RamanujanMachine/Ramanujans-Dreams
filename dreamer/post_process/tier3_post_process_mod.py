"""
Tier3PostProcessModV1 — post-process stage.

Runs once, after the Search stage finishes.  For every constant in the
search priorities, scans its per-shard JSONL files and computes the
configured Tier-3 attributes (e.g. ``asymptotics``, ``kamidelta``) for
any trajectory that lacks them.  Results are appended as patch records;
the merge-on-read reader transparently folds them into the base records
on the next load.

Process model — identical pattern to the Search stage:

  Producer (main thread)
    For each constant → for each shard JSONL:
      1. Merge existing records (so we know which Tier-3 attrs are present).
      2. For each trajectory missing some Tier-3 attrs:
           - Look up the CMF by ``cmf_id`` in the in-memory ``cmf_lookup``
             (fallback: load from disk under ``sys_config.EXPORT_CMFS``).
           - Reconstruct ``Position`` objects for start / direction.
           - Build a ``TrajectoryAttributesHandler``.
           - Push ``(trajectory_matrix, patch_dict)`` to the worker pool.

  Workers / Writer (via ``worker_pool``)
    ``compute_tier3_for_item`` fills in the missing attributes, and
    ``write_jsonl_line`` appends the patch as one JSON line in the same
    file the searcher produced.

The whole stage short-circuits when ``post_process.TIER3_ATTRIBUTES`` is
empty — no JSONL is read, no subprocesses created.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import (
    load_seen_trajectories,
    worker_pool,
    write_jsonl_line,
)
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.schemes.post_process_scheme import PostProcessModScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage import Formats
from dreamer.utils.storage.attribute_registry import attribute_name, compute_attributes
from dreamer.utils.storage.handler_reconstruction import (
    build_cmf_lookup_from_priorities,
    reconstruct_positions,
)
from dreamer.utils.storage.predicate_specs import (
    TopNSelector,
    iter_top_n_selectors,
    parse_predicate_spec,
)
from dreamer.utils.storage.record_metrics import METRIC_EXTRACTORS
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    derive_cmf_and_shard_ids,
)
from dreamer.utils.ui.tqdm_config import SmartTQDM

post_process_config = config.post_process


# ---------------------------------------------------------------------------
# Per-item worker  (module-level so it pickles to subprocesses)
# ---------------------------------------------------------------------------

def compute_tier3_for_item(item):
    """Per-item worker for the post-process stage.

    *item* is ``(trajectory_matrix, constant, patch_dict, context)`` where
    *constant* is the sympy expression for the target constant (e.g.
    ``sp.log(2)``).  Constant context is required by attributes that compare
    against the limit (``delta_sequence``, ``limit``); pass ``None`` when none
    is available — those attributes will then be skipped with an error entry.
    *context* is the shard/CMF predicate context (``{"trajectory_id": ...,
    "top_n_sets": {...}}``) threaded into ``compute_attributes`` so shard-level
    gates (``top N highest <metric> in shard|cmf``) fire; ``None`` for the
    legacy no-context path.

    Reads ``post_process.TIER3_ATTRIBUTES`` from the (subprocess-local) config and
    computes every entry not already present as a **top-level** key on the flat
    *patch* record.  Per-attribute failures are stored as ``<name>_error``; a
    fatal handler failure is recorded under ``worker_error``.  The patch (a flat
    dict) is returned for the writer.
    """
    from dreamer.configs import config

    traj_matrix, constant, patch, context = item
    attrs_to_compute = config.post_process.TIER3_ATTRIBUTES
    # Specs may be bare strings or ``(name, predicate)`` tuples; filter by
    # resolved name so predicates still fire inside ``compute_attributes``.
    missing = [
        spec for spec in attrs_to_compute
        if attribute_name(spec) not in patch
    ]

    if missing and traj_matrix is not None:
        try:
            handler = TrajectoryAttributesHandler(traj_matrix, constant=constant)
            patch.update(
                compute_attributes(handler, missing, on_error="store", context=context)
            )
        except Exception as e:
            tid = patch.get("trajectory_id", "?")
            Logger(
                f"compute_tier3_for_item error on trajectory {tid}: {e}",
                Logger.Levels.warning,
            ).log()
            patch["worker_error"] = str(e)
    return patch


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class Tier3PostProcessModV1(PostProcessModScheme):
    """Default post-process implementation — patches existing JSONL files
    with Tier-3 attributes.

    See module docstring for the full data flow.
    """

    def __init__(
        self,
        priorities: Dict[Constant, List[Searchable]],
    ):
        """
        :param priorities: Search-stage priorities; provides the in-memory
            CMF lookup keyed by ``cmf_name``.
        """
        super().__init__(
            priorities,
            description='Tier-3 post-process — fills expensive attributes via patch records',
            version='1.0.0',
        )
        self._cmf_lookup: Dict[str, object] = self._build_cmf_lookup(priorities)
        self._shard_lookup: Dict[str, Tuple[Searchable, Constant]] = (
            self._build_shard_lookup(priorities)
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=False)
    def execute(self) -> None:
        # Two independent sub-stages: Tier-3 attribute computation and graphing.
        # Either can be enabled alone (e.g. graphs over already-computed JSONL
        # with no new attributes).  Compute first so graphs see fresh metrics.
        if post_process_config.TIER3_ATTRIBUTES:
            self._compute_attributes_stage()
        self._graphing_stage()

    def _compute_attributes_stage(self) -> None:
        Logger(
            Logger.buffer_print(
                sys_config.LOGGING_BUFFER_SIZE,
                'Post-process: Tier-3 attributes',
                '=',
            ),
            Logger.Levels.message,
        ).log()

        num_workers = sys_config.NUM_BACKGROUND_WORKERS
        config_overrides = config.export_configurations()

        dir_path = sys_config.EXPORT_SEARCH_RESULTS
        if not os.path.isdir(dir_path):
            Logger(
                f"No search results directory at {dir_path}; skipping post-process.",
                Logger.Levels.warning,
            ).log()
            return

        # Restrict to shards that survived analysis (i.e. live in
        # ``self.priorities``).  Extraction may have produced JSONLs for
        # shards that didn't pass the identification threshold, but Tier-3
        # attributes on those are wasted work — we only want to spend the
        # compute budget on shards downstream stages actually care about.
        # Duplicate shards (same id under multiple constants) are scanned
        # once; the JSONL records carry per-constant attributes so all
        # constants are still covered.
        shard_paths: List[str] = []
        seen_ids: set = set()
        suffix = '.' + Formats.JSONL.value
        for shards in self.priorities.values():
            for shard in shards:
                try:
                    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
                except Exception as e:
                    Logger(
                        f"Skipping shard with unresolvable id during post-process: {e}",
                        Logger.Levels.warning,
                    ).log()
                    continue
                if shard_id in seen_ids:
                    continue
                seen_ids.add(shard_id)
                path = os.path.join(dir_path, shard_id + suffix)
                if not os.path.isfile(path):
                    # Analyzer didn't write a JSONL for this shard — nothing
                    # to patch.  Quiet skip (a missing file isn't an error).
                    continue
                shard_paths.append(path)

        if not shard_paths:
            return

        # Phase 1 — ranking.  Precompute, for every "top N … in shard|cmf"
        # selector referenced in the config, the set of trajectory ids that
        # qualify.  Reads only stored JSONL metric values (no re-walk); skipped
        # entirely when no top-N selector is configured.
        selectors = iter_top_n_selectors(post_process_config.TIER3_ATTRIBUTES)
        top_n_sets = self._compute_top_n_sets(selectors) if selectors else {}

        # Phase 2 — compute.  One pass per shard JSONL, gating each trajectory's
        # attributes with the precomputed top-N membership via ``context``.
        for shard_jsonl in SmartTQDM(
            shard_paths,
            desc='Post-process shards: ',
            **sys_config.TQDM_CONFIG,
        ):
            self._run_jsonl(
                shard_jsonl,
                num_workers=num_workers,
                config_overrides=config_overrides,
                top_n_sets=top_n_sets,
            )

    def _graphing_stage(self) -> None:
        """Run the configured post-process graphs (no-op when none enabled)."""
        if not config.graph.any_enabled():
            return
        # Imported lazily so matplotlib is only loaded when graphing is on.
        from dreamer.graphing import Grapher
        Grapher(self.priorities).generate()

    # ------------------------------------------------------------------
    # Per-file pipeline
    # ------------------------------------------------------------------

    def _run_jsonl(
        self,
        jsonl_path: str,
        *,
        num_workers: int,
        config_overrides: dict,
        top_n_sets: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Run the producer → worker_pool pipeline for one shard JSONL.

        Skips entirely if every trajectory is already fully covered (no
        ``push`` calls → no ``worker_pool`` is created, no subprocess spawn).

        :param top_n_sets: precomputed ``{selector.key: {trajectory_id}}`` from
            Phase 1; used to gate (and pre-skip) top-N attributes per trajectory.
        """
        merged = load_seen_trajectories(jsonl_path)
        if not merged:
            return

        desired = {attribute_name(s) for s in post_process_config.TIER3_ATTRIBUTES}
        # Quickly scan first — if nothing is missing on any (traj, const) row, skip
        # spawning the worker pool entirely.  Rows are flat, so a row's computed
        # attributes are its top-level keys.
        if not any(
            desired - set(rec.keys())
            for by_const in merged.values() for rec in by_const.values()
        ):
            return

        with worker_pool(
            num_workers=num_workers,
            worker_fn=compute_tier3_for_item,
            writer_fn=write_jsonl_line,
            output_path=jsonl_path,
            config_overrides=config_overrides,
        ) as push:
            self._produce(merged, desired, push, top_n_sets or {})

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _produce(
        self,
        merged: Dict[str, dict],
        desired: set,
        sink,
        top_n_sets: Dict[str, Set[str]],
    ) -> None:
        """Emit ``(traj_matrix, constant, patch, context)`` for every trajectory
        missing Tier-3 attrs *that can actually be computed*.

        Trajectories whose CMF cannot be resolved are logged and skipped.
        Trajectories whose only-missing attributes are all gated off by a
        ``top N … in shard|cmf`` selector (this trajectory is not in the
        qualifying set) are **pre-skipped** — context alone settles them, so we
        avoid building a handler for the (usually large) non-selected majority.

        The loop is wrapped in a tqdm bar — Tier-3 attributes (asymptotics,
        delta_sequence, …) can be minutes-per-trajectory, so without a
        progress indicator a long shard looks indistinguishable from a hang.
        """
        resolved_specs = self._resolved_specs()
        # One item per (trajectory, constant) row that is missing a Tier-3 attr.
        items = [
            (tid, record)
            for by_const in merged.values()
            for record in by_const.values()
            for tid in (record.get("trajectory_id"),)
            if desired - set(record.keys())
        ]
        for tid, record in SmartTQDM(
            items,
            desc='Tier-3 trajectories: ',
            **sys_config.TQDM_CONFIG,
        ):
            missing = desired - set(record.keys())
            if not missing:
                continue

            # Cheap pre-skip: if every missing attribute is gated off for this
            # trajectory by a top-N selector, nothing will be computed — don't
            # build the handler.  Handler-only / unconditional specs always
            # survive here (they need the handler to decide).
            if not self._survives_top_n(missing, tid, top_n_sets, resolved_specs):
                continue

            cmf_name = record.get("cmf_id")
            cmf = self._cmf_lookup.get(cmf_name)
            if cmf is None:
                Logger(
                    f"Tier-3 skip: no CMF found for cmf_id={cmf_name!r} "
                    f"(trajectory {tid[:8]}…)",
                    Logger.Levels.warning,
                ).log()
                continue

            shard_entry = self._shard_lookup.get(record.get("shard_id"))
            shard, constant = (shard_entry if shard_entry is not None else (None, None))

            # Resolve the sympy constant for this trajectory.  Prefer the
            # typed object from the shard lookup; fall back to parsing the
            # record's ``constant`` string (populated by the searcher) so
            # post-process can run standalone — without it, attributes that
            # need the limit (``delta_sequence``) silently error out.
            constant_sympy = constant.value_sympy if constant is not None else None
            if constant_sympy is None:
                const_str = record.get("constant")
                if const_str:
                    try:
                        constant_sympy = sp.sympify(const_str)
                    except (sp.SympifyError, SyntaxError, TypeError):
                        constant_sympy = None

            try:
                start, direction = self._reconstruct_positions(cmf, record)
                handler = TrajectoryAttributesHandler.from_cmf(
                    cmf, direction, start,
                    constant=constant_sympy,
                    searchable=shard,
                )
            except Exception as e:
                Logger(
                    f"Tier-3 handler error for trajectory {tid[:8]}…: {e}",
                    Logger.Levels.warning,
                ).log()
                continue

            patch = {"trajectory_id": tid, "constant": record.get("constant")}
            context = self._build_context(tid, top_n_sets, resolved_specs)
            sink((handler.trajectory_matrix, constant_sympy, patch, context))

    # ------------------------------------------------------------------
    # Ranking (Phase 1) + predicate context
    # ------------------------------------------------------------------

    @staticmethod
    def _resolved_specs() -> List[Tuple[str, object]]:
        """Resolve each configured Tier-3 spec to ``(attr_name, predicate)``.

        ``predicate`` is ``None`` for an unconditional (bare-string) attribute,
        a :class:`TopNSelector` for a shard/CMF gate, or another callable for a
        handler-only predicate.  Resolution failures are treated as ``None`` so
        the worker (not the producer) surfaces a misconfigured predicate.
        """
        out: List[Tuple[str, object]] = []
        for spec in post_process_config.TIER3_ATTRIBUTES:
            if isinstance(spec, str):
                out.append((spec, None))
                continue
            name, pred_ref = spec
            try:
                out.append((name, parse_predicate_spec(pred_ref)))
            except (KeyError, ValueError, TypeError):
                out.append((name, None))
        return out

    @staticmethod
    def _survives_top_n(
        missing: Set[str],
        tid: str,
        top_n_sets: Dict[str, Set[str]],
        resolved_specs: List[Tuple[str, object]],
    ) -> bool:
        """Return True if at least one *missing* attribute can still be computed
        for ``tid`` after applying the context-only top-N gates."""
        for name, predicate in resolved_specs:
            if name not in missing:
                continue
            if isinstance(predicate, TopNSelector):
                if tid in top_n_sets.get(predicate.key, set()):
                    return True
            else:
                # Unconditional or handler-only predicate — can't decide here.
                return True
        return False

    @staticmethod
    def _build_context(
        tid: str,
        top_n_sets: Dict[str, Set[str]],
        resolved_specs: List[Tuple[str, object]],
    ) -> dict:
        """Build the per-trajectory predicate context for the worker.

        Only carries the membership the worker needs: for each referenced
        top-N selector, a singleton ``{tid}`` when the trajectory qualifies
        else an empty set — so the (possibly large) full id-set is never
        shipped to subprocesses.
        """
        per_selector: Dict[str, Set[str]] = {}
        for _name, predicate in resolved_specs:
            if isinstance(predicate, TopNSelector):
                qualifies = tid in top_n_sets.get(predicate.key, set())
                per_selector[predicate.key] = {tid} if qualifies else set()
        return {"trajectory_id": tid, "top_n_sets": per_selector}

    def _compute_top_n_sets(
        self,
        selectors: List[TopNSelector],
    ) -> Dict[str, Set[str]]:
        """Phase-1 ranking: ``{selector.key: {trajectory_id}}`` over the
        configured top-N selectors.

        Rankings are computed per ``(constant, CMF)`` group (the metric for
        ``delta`` is per-constant), reading only stored JSONL values via
        :data:`record_metrics.METRIC_EXTRACTORS`.  A ``shard``-scoped selector
        ranks within each shard; a ``cmf``-scoped one pools every shard of the
        CMF.  Qualifying ids are unioned across constants, so a trajectory that
        is top-N under any searched constant passes the gate.  Peak memory is
        bounded to one CMF's records (loaded, ranked, then discarded).
        """
        sets: Dict[str, Set[str]] = {sel.key: set() for sel in selectors}
        shard_scope = [s for s in selectors if s.scope == "shard"]
        cmf_scope = [s for s in selectors if s.scope == "cmf"]
        suffix = '.' + Formats.JSONL.value
        dir_path = sys_config.EXPORT_SEARCH_RESULTS

        for constant, shards in self.priorities.items():
            const_name = getattr(constant, "name", None)
            by_cmf: Dict[str, List[str]] = defaultdict(list)
            for shard in shards:
                try:
                    cmf_id, shard_id, _ = derive_cmf_and_shard_ids(shard)
                except Exception:
                    continue
                path = os.path.join(dir_path, shard_id + suffix)
                if os.path.isfile(path):
                    by_cmf[cmf_id].append(path)

            for paths in by_cmf.values():
                shard_records = {p: load_seen_trajectories(p) for p in paths}
                for sel in shard_scope:
                    for recs in shard_records.values():
                        self._rank_into(sets[sel.key], recs, sel, const_name)
                if cmf_scope:
                    pooled: Dict[str, dict] = {}
                    for recs in shard_records.values():
                        pooled.update(recs)
                    for sel in cmf_scope:
                        self._rank_into(sets[sel.key], pooled, sel, const_name)
        return sets

    @staticmethod
    def _rank_into(
        target: Set[str],
        records: Dict[str, dict],
        selector: TopNSelector,
        const_name: Optional[str],
    ) -> None:
        """Rank *records* by the selector's metric and add the top-N ids to *target*.

        *records* is the nested ``{trajectory_id: {constant: row}}`` map; ranking is
        per *const_name*, so each trajectory contributes its row for that constant.
        """
        extractor = METRIC_EXTRACTORS.get(selector.metric)
        if extractor is None:
            return
        scored: List[Tuple[float, str]] = []
        for tid, by_const in records.items():
            rec = by_const.get(const_name)
            if rec is None:
                continue
            value = extractor(rec, const_name)
            if value is None:
                continue
            scored.append((value, tid))
        if not scored:
            return
        scored.sort(key=lambda vt: vt[0], reverse=selector.highest)
        for _value, tid in scored[: selector.n]:
            target.add(tid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cmf_lookup(
        priorities: Dict[Constant, List[Searchable]],
    ) -> Dict[str, object]:
        """Return ``{cmf_name: CMF}`` from in-memory priorities (disk fallback).

        Thin wrapper over the shared
        :func:`dreamer.utils.storage.handler_reconstruction.build_cmf_lookup_from_priorities`.
        """
        return build_cmf_lookup_from_priorities(priorities)

    @staticmethod
    def _build_shard_lookup(
        priorities: Dict[Constant, List[Searchable]],
    ) -> Dict[str, Tuple[Searchable, Constant]]:
        """Return ``{shard_id: (shard, constant)}`` from in-memory priorities.

        Empty when post-process runs standalone (no priorities); callers must
        tolerate a missing entry and fall back to ``constant=None,
        searchable=None`` (acceptable since Tier-3 attrs don't require them).
        """
        lookup: Dict[str, Tuple[Searchable, Constant]] = {}
        for constant, shards in priorities.items():
            for shard in shards:
                try:
                    _, sid, _ = derive_cmf_and_shard_ids(shard)
                except Exception:
                    continue
                lookup.setdefault(sid, (shard, constant))
        return lookup

    @staticmethod
    def _reconstruct_positions(cmf, record: dict):
        """Rebuild ``(start, direction)`` ``Position`` objects from JSONL fields.

        Thin wrapper over the shared
        :func:`dreamer.utils.storage.handler_reconstruction.reconstruct_positions`.
        """
        return reconstruct_positions(cmf, record)
