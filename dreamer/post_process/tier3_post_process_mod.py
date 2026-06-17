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
from typing import Dict, List, Optional, Tuple

import sympy as sp
from ramanujantools import Position

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
from dreamer.utils.storage import Formats, Importer
from dreamer.utils.storage.attribute_registry import attribute_name, compute_attributes
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

    *item* is ``(trajectory_matrix, constant, patch_dict)`` where *constant*
    is the sympy expression for the target constant (e.g. ``sp.log(2)``).
    Constant context is required by attributes that compare against the
    limit (``delta_sequence``, ``limit``); pass ``None`` when none is
    available — those attributes will then be skipped with an error entry.

    Reads ``post_process.TIER3_ATTRIBUTES`` from the (subprocess-local)
    config and computes every entry not already present in
    ``patch_dict['extended_metrics']``.  Per-attribute failures are stored as
    ``<name>_error``; a fatal handler failure is recorded under
    ``worker_error``.  The patch is returned for the writer.
    """
    from dreamer.configs import config

    traj_matrix, constant, patch = item
    attrs_to_compute = config.post_process.TIER3_ATTRIBUTES
    extended_metrics = patch.setdefault("extended_metrics", {})
    # Specs may be bare strings or ``(name, predicate)`` tuples; filter by
    # resolved name so predicates still fire inside ``compute_attributes``.
    missing = [
        spec for spec in attrs_to_compute
        if attribute_name(spec) not in extended_metrics
    ]

    if missing and traj_matrix is not None:
        try:
            handler = TrajectoryAttributesHandler(traj_matrix, constant=constant)
            extended_metrics.update(
                compute_attributes(handler, missing, on_error="store")
            )
        except Exception as e:
            tid = patch.get("trajectory_id", "?")
            Logger(
                f"compute_tier3_for_item error on trajectory {tid}: {e}",
                Logger.Levels.warning,
            ).log()
            extended_metrics["worker_error"] = str(e)
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
        if not post_process_config.TIER3_ATTRIBUTES:
            # No work configured — stay completely silent and skip.
            return

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

        for shard_jsonl in SmartTQDM(
            shard_paths,
            desc='Post-process shards: ',
            **sys_config.TQDM_CONFIG,
        ):
            self._run_jsonl(
                shard_jsonl,
                num_workers=num_workers,
                config_overrides=config_overrides,
            )

    # ------------------------------------------------------------------
    # Per-file pipeline
    # ------------------------------------------------------------------

    def _run_jsonl(
        self,
        jsonl_path: str,
        *,
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Run the producer → worker_pool pipeline for one shard JSONL.
        
        Skips entirely if every trajectory is already fully covered (no
        ``push`` calls → no ``worker_pool`` is created, no subprocess spawn).
        """
        merged = load_seen_trajectories(jsonl_path)
        if not merged:
            return

        desired = {attribute_name(s) for s in post_process_config.TIER3_ATTRIBUTES}
        # Quickly scan first — if nothing is missing anywhere, skip
        # spawning the worker pool entirely.
        if not any(
            desired - set((r.get("extended_metrics") or {}).keys())
            for r in merged.values()
        ):
            return

        with worker_pool(
            num_workers=num_workers,
            worker_fn=compute_tier3_for_item,
            writer_fn=write_jsonl_line,
            output_path=jsonl_path,
            config_overrides=config_overrides,
        ) as push:
            self._produce(merged, desired, push)

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _produce(
        self,
        merged: Dict[str, dict],
        desired: set,
        sink,
    ) -> None:
        """Emit ``(traj_matrix, patch)`` for every trajectory missing Tier-3 attrs.

        Trajectories whose CMF cannot be resolved are logged and skipped.

        The loop is wrapped in a tqdm bar — Tier-3 attributes (asymptotics,
        delta_sequence, …) can be minutes-per-trajectory, so without a
        progress indicator a long shard looks indistinguishable from a hang.
        """
        items = [
            (tid, record)
            for tid, record in merged.items()
            if desired - set((record.get("extended_metrics") or {}).keys())
        ]
        for tid, record in SmartTQDM(
            items,
            desc='Tier-3 trajectories: ',
            **sys_config.TQDM_CONFIG,
        ):
            existing = set((record.get("extended_metrics") or {}).keys())
            missing = desired - existing
            if not missing:
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

            patch = {"trajectory_id": tid, "extended_metrics": {}}
            sink((handler.trajectory_matrix(), constant_sympy, patch))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cmf_lookup(
        priorities: Dict[Constant, List[Searchable]],
    ) -> Dict[str, object]:
        """Return ``{cmf_name: CMF}`` from in-memory priorities.

        If priorities is empty (e.g. post-process is being run standalone),
        fall back to loading CMFs from ``sys_config.EXPORT_CMFS`` if set.
        """
        lookup: Dict[str, object] = {}
        for searchables in priorities.values():
            for s in searchables:
                cmf_name = getattr(s, 'cmf_name', None)
                cmf = getattr(s, 'cmf', None)
                if cmf_name and cmf is not None and cmf_name not in lookup:
                    lookup[cmf_name] = cmf

        if lookup:
            return lookup

        return Tier3PostProcessModV1._load_cmfs_from_disk()

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
    def _load_cmfs_from_disk() -> Dict[str, object]:
        """Best-effort load of CMFs from ``sys_config.EXPORT_CMFS``.

        Returns ``{}`` if the path is not configured or no usable files are
        found.  Producer-side errors will be logged on a per-trajectory
        basis when the lookup fails to resolve a name.
        """
        root = sys_config.EXPORT_CMFS
        if not root or not os.path.isdir(root):
            return {}

        lookup: Dict[str, object] = {}
        for const_dir in os.listdir(root):
            const_path = os.path.join(root, const_dir)
            if not os.path.isdir(const_path):
                continue
            for f_name in os.listdir(const_path):
                file_path = os.path.join(const_path, f_name)
                try:
                    data = Importer.imprt(file_path)
                except Exception:
                    continue
                for item in Tier3PostProcessModV1._iter_cmf_data(data):
                    cmf_name = getattr(item, 'cmf_name', None)
                    cmf = getattr(item, 'cmf', None)
                    if cmf_name and cmf is not None:
                        lookup.setdefault(cmf_name, cmf)
        return lookup

    @staticmethod
    def _iter_cmf_data(data):
        """Yield CMFData-shaped objects from a (possibly nested) imported payload."""
        if data is None:
            return
        if hasattr(data, 'cmf') and hasattr(data, 'cmf_name'):
            yield data
            return
        if isinstance(data, dict):
            for v in data.values():
                yield from Tier3PostProcessModV1._iter_cmf_data(v)
        elif isinstance(data, (list, tuple, set)):
            for v in data:
                yield from Tier3PostProcessModV1._iter_cmf_data(v)

    @staticmethod
    def _reconstruct_positions(cmf, record: dict):
        """Rebuild ``(start, direction)`` ``Position`` objects from JSONL fields.

        Tuples in the record are stored in ``cmf.matrices.keys()`` order,
        matching ``_position_to_tuple`` at write time.  Integers stored as
        Python ``int`` are wrapped back into ``sp.Integer``; any non-integer
        slot (rare — only when ``_position_to_tuple`` fell back to ``str``)
        is parsed via ``sp.sympify`` so symbolic shifts survive the round-trip.
        """
        symbols = list(cmf.matrices.keys())

        def _to_position(values) -> Position:
            if values is None or len(values) != len(symbols):
                raise ValueError(
                    f"Position has {len(values) if values is not None else 0} entries; "
                    f"CMF expects {len(symbols)}."
                )
            mapping: Dict[object, object] = {}
            for sym, v in zip(symbols, values):
                if isinstance(v, int):
                    mapping[sym] = sp.Integer(v)
                else:
                    mapping[sym] = sp.sympify(v)
            return Position(mapping)

        return _to_position(record.get("start_point")), _to_position(record.get("direction"))
