"""
SearcherModV1 — search-stage module.

Replaces the old ``SerialSearcher.search()`` / ``DataManager`` flow with a
handler-based pipeline:

  Producer (main thread)
    For each unique shard (deduplicated by shard_id across all constants):
      For each (trajectory, start) pair sampled from the shard, per searched
      constant (``attribute_fingerprints`` decide, under ``REUSE_POLICY``):
        * no valid Tier-1 row → queue Tier-1 **and** the configured Tier-2
          attributes, computed in one task on one handler (nothing twice);
        * valid Tier-1 row but some Tier-2 attributes missing/stale → queue
          only those, on a handler seeded with the stored δ / identified / p,q
          (no LIReC, no δ recompute);
        * otherwise fully covered → skip.
      The queued tasks run on one **stage-wide** process pool
      (:class:`~dreamer.utils.stage_pool.StagePool`, created once for all
      shards, every core); results come back in task order and the main
      process — the single writer — appends rows and patches to the shard file.

Output files:
    ``sys_config.EXPORT_SEARCH_RESULTS / <shard_id>.jsonl``
where ``<shard_id>`` encodes the parent CMF (``<cmf_id>__<encoding_hash>``).

Only constants that were **identified** for a given shard during the
analysis stage are searched: ``priorities[const]`` lists only shards for
which ``const`` passed the threshold, so the intersection of constants in
``shard.consts`` and the keys under which the shard appears gives exactly
the set to compute delta for.
"""

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.logger import Logger
from dreamer.utils.storage.attribute_registry import attribute_name
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dtos,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.utils.multi_processing import (
    load_seen_trajectories_for_search,
    search_worker_budget,
    write_jsonl_line,
)
from dreamer.utils.stage_pool import (
    BufferedJsonlWriter,
    StagePool,
    TaskError,
    Tier1Task,
    run_tier1_task,
)
from dreamer.utils.storage.attribute_fingerprints import stale_attributes, tier1_is_fresh

search_config = config.search


class SearcherModV1(SearcherModScheme):
    """Search module — deep trajectory search with optional asynchronous Tier-2
    attribute computation.

    Receives the full ``priorities`` dict (``{Constant: [Shard, ...]}``) so
    it can determine, per shard, which subset of its constants were
    identified during analysis.  Only identified constants are searched;
    the trajectory walk is shared across constants for each trajectory.
    """

    def __init__(self, priorities, use_LIReC: bool):
        """
        :param priorities: ``Dict[Constant, List[Shard]]`` — shards that passed
            analysis for each constant.
        :param use_LIReC: Whether to use LIReC for constant identification.
        """
        super().__init__(
            priorities,
            use_LIReC,
            description='Search module — deep search with Tier-1 DTO output',
            version='2.0.0',
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """Run the search pipeline over all unique shards."""
        if not self.searchables:
            return

        os.makedirs(sys_config.EXPORT_SEARCH_RESULTS, exist_ok=True)

        config_overrides = config.export_configurations()

        # Build shard_id → (Shard, Set[Constant_identified]) mapping.
        # A shard can be identified for a subset of its consts; only those
        # are computed during the deep search.
        shard_identified: Dict[str, Set[Constant]] = defaultdict(set)
        shard_by_id: Dict[str, Shard] = {}
        for const, shards in self.priorities.items():
            for shard in shards:
                _, shard_id, _ = derive_cmf_and_shard_ids(shard)
                shard_by_id[shard_id] = shard
                shard_identified[shard_id].add(const)

        # One persistent pool for the whole stage (shards handed to the workers
        # once), on every core: Tier-1 and Tier-2 both run in its tasks.
        shard_idx = {sid: i for i, sid in enumerate(shard_by_id)}
        with StagePool(list(shard_by_id.values()), workers=search_worker_budget()) as pool:
            for shard_id, shard in SmartTQDM(
                shard_by_id.items(),
                desc='Searching in shards: ',
                **sys_config.TQDM_CONFIG,
            ):
                identified_consts = list(shard_identified[shard_id])
                self._run_shard(
                    shard, identified_consts, 0, config_overrides,
                    pool=pool, shard_idx=shard_idx[shard_id],
                )

    # ------------------------------------------------------------------
    # Per-shard pipeline
    # ------------------------------------------------------------------

    def _run_shard(
        self,
        shard: Shard,
        identified_consts: List[Constant],
        num_workers: int,
        config_overrides: dict,
        pool: Optional[StagePool] = None,
        shard_idx: int = 0,
    ) -> None:
        """Run the search for a single shard using only *identified_consts*.

        :param num_workers: Unused (kept for interface stability; the former
            separate Tier-2 queue is gone — Tier-2 runs in the pool tasks).
        :param config_overrides: Unused (the pool propagates the config).
        :param pool: The stage-wide Tier-1 pool (``None`` → serial in-process).
        :param shard_idx: This shard's index in *pool*'s shard list.
        """
        cmf_id, shard_id, shard_encoding_str = derive_cmf_and_shard_ids(shard)
        Logger(
            f"Starting deep search on shard {shard_id} (cmf={cmf_id})",
            Logger.Levels.debug,
        ).log()
        output_path = os.path.join(
            sys_config.EXPORT_SEARCH_RESULTS, f"{shard_id}.jsonl"
        )
        seen_trajectories = load_seen_trajectories_for_search(output_path, shard_id)

        # The main process is the single writer of the shard file.
        with BufferedJsonlWriter() as writer:
            out = writer.sink(output_path)
            self._produce(
                shard=shard,
                identified_consts=identified_consts,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=lambda item: write_jsonl_line(item[2], out),
                seen_trajectories=seen_trajectories,
                pool=pool,
                shard_idx=shard_idx,
            )

        Logger(
            f"Finished deep search on shard {shard_id}",
            Logger.Levels.debug,
        ).log()

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _produce(
        self,
        shard: Shard,
        identified_consts: List[Constant],
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
        pool: Optional[StagePool] = None,
        shard_idx: int = 0,
    ) -> None:
        """Iterate over trajectory pairs and hand work to *sink*.

        Classification (cache lookup) runs in the main process; the handler
        builds / walks of every trajectory that needs work run on *pool* (the
        stage-wide :class:`StagePool`, in which this shard sits at *shard_idx*).
        With ``pool=None`` a serial in-process pool over just *shard* is used.

        Three cases per trajectory:

        1. **Complete** — trajectory_id already in *seen_trajectories* with
           every configured Tier-2 attribute present.  Skip.
        2. **Partial** — trajectory_id known but some Tier-2 attrs missing.
           Emit a patch dict.
        3. **New** — build the full multi-constant Tier-1 DTO and emit it.
        """
        primary_const = shard.consts[0]
        searcher = SerialSearcher(shard, primary_const, use_LIReC=self.use_LIReC)

        try:
            pairs = searcher.sample_pairs(
                trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM,
            )
        except ValueError as e:
            Logger(
                f"Skipping shard {shard_id}: {e}",
                Logger.Levels.warning,
            ).log()
            return

        tier2_specs = tuple(search_config.TIER2_ATTRIBUTES)
        tier2_names = [attribute_name(spec) for spec in tier2_specs]

        # Plan (main process): classify every sampled trajectory against the
        # cache and queue one task per trajectory that needs work.
        planned: List[tuple] = []
        tasks: List[Tier1Task] = []
        counts = {"reused": 0, "tier2_only": 0, "new": 0}
        for traj, start in pairs:
            start_t = _position_to_tuple(start)
            dir_t = _position_to_tuple(traj)
            trajectory_id = derive_trajectory_id(
                shard_id, shard.cmf_name, shard_encoding_str, start_t, dir_t,
            )
            walk_depth = walk_depth_for(shard.cmf, traj)
            bucket = seen_trajectories.get(trajectory_id, {})

            # Per searched constant (see attribute_fingerprints / REUSE_POLICY):
            #   'new'    — no valid Tier-1 row → Tier-1 + Tier-2 in one task;
            #   'seeded' — valid Tier-1 row, some Tier-2 attrs missing or stale →
            #              only those, on a handler seeded with the stored values;
            #   otherwise fully covered → skip.
            new_consts: List[Constant] = []
            seeded: List[tuple] = []
            for c in identified_consts:
                rec = bucket.get(c.name)
                if not tier1_is_fresh(rec, walk_depth):
                    new_consts.append(c)
                    continue
                stale = set(stale_attributes(rec, tier2_names, walk_depth))
                if stale:
                    seeded.append((c, rec, tuple(sp_ for sp_ in tier2_specs if attribute_name(sp_) in stale)))
            if not new_consts and not seeded:
                counts["reused"] += 1
                continue
            counts["new" if new_consts else "tier2_only"] += 1
            planned.append((trajectory_id, new_consts, seeded))
            tasks.append(Tier1Task(
                key=trajectory_id, shard_idx=shard_idx, trajectory=traj, start=start,
                constants=tuple(new_consts), cmf_id=cmf_id, shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                extra_metrics=tier2_specs, seeded=tuple(seeded),
            ))

        Logger(
            f"Shard {shard_id[:24]}…: {len(pairs)} trajectories → "
            f"{counts['reused']} reused · {counts['tier2_only']} Tier-2 only · {counts['new']} new",
            Logger.Levels.info,
        ).log()
        if not tasks:
            return

        # Compute on the stage pool; results arrive in task order and the main
        # process forwards them to the sink (rows first, then patches).
        own_pool = None
        if pool is None:
            own_pool = StagePool([shard], workers=1)
            pool = own_pool.__enter__()
        try:
            results = pool.imap(_search_walk_task(), tasks)
            for (trajectory_id, new_consts, seeded), task, res in zip(
                planned,
                tasks,
                SmartTQDM(
                    results,
                    total=len(tasks),
                    desc=f"  Shard {shard_id[:8]}… trajectories",
                    leave=False,
                    **sys_config.TQDM_CONFIG,
                ),
            ):
                if isinstance(res, TaskError):
                    Logger(
                        f"Handler error — shard {shard_id}, traj={task.trajectory}, "
                        f"start={task.start}: {res.message}",
                        Logger.Levels.warning,
                    ).log()
                    continue
                bucket = seen_trajectories.setdefault(trajectory_id, {})
                for c, dto in zip(new_consts, res.dtos):
                    sink((None, c.value_sympy, dto))
                    bucket[c.name] = json.loads(dto.to_json_line())
                for (c, _rec, _specs), patch in zip(seeded, res.patches):
                    sink((None, c.value_sympy, patch))
                    merged = bucket.setdefault(c.name, {})
                    fps = {**merged.get("fingerprints", {}), **patch.get("fingerprints", {})}
                    merged.update(patch)
                    merged["fingerprints"] = fps
        finally:
            if own_pool is not None:
                own_pool.__exit__(None, None, None)

def _search_walk_task():
    """Stage-pool task callable: rows (Tier-1 + Tier-2) and seeded patches.

    A ``functools.partial`` of :func:`run_tier1_task` bound to this module's
    handler / DTO-builder names (resolved now, so test monkeypatches apply); it
    pickles into the light ``stage_pool`` / ``trajectory_attributes`` modules, so
    workers do not import the search package.
    """
    import functools
    return functools.partial(
        run_tier1_task,
        handler_cls=TrajectoryAttributesHandler,
        build_dtos=build_trajectory_dtos,
    )
