"""
SearcherModV1 — search-stage module.

Replaces the old ``SerialSearcher.search()`` / ``DataManager`` flow with a
handler-based pipeline:

  Producer (main thread)
    For each unique shard (deduplicated by shard_id across all constants):
      For each (trajectory, start) pair sampled from the shard:
        1. Compute ``trajectory_id`` from (cmf_name, encoding, start, direction)
           — cheap; no symbolic work, no trajectory walk.
        2. If the id is already in the JSONL and every configured Tier-2
           attribute is present, skip immediately.
        3. Otherwise queue a Tier-1 task (handler build + walk + DTOs).
      The queued tasks run on one **stage-wide** process pool
      (:class:`~dreamer.utils.stage_pool.StagePool`, created once for all
      shards); results come back in task order and are pushed to the sink.

  push(item) — provided by the generic ``worker_pool`` context manager:
    * ``TIER2_ATTRIBUTES`` empty (default) → ``push`` is a synchronous
      writer; the JSONL is written from the main thread, no subprocesses.
    * ``TIER2_ATTRIBUTES`` non-empty → ``push`` enqueues to a worker pool.

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
    compute_tier2_for_item,
    load_seen_trajectories_for_search,
    search_worker_budget,
    worker_pool,
    write_jsonl_line,
)
from dreamer.utils.stage_pool import StagePool, TaskError, Tier1Task, run_tier1_task

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

        num_workers = sys_config.NUM_BACKGROUND_WORKERS
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

        # One persistent Tier-1 pool for the whole stage (shards handed to the
        # workers once).  Sized by the search budget, which reserves cores for the
        # Tier-2 queue + writer when Tier-2 is active.
        shard_idx = {sid: i for i, sid in enumerate(shard_by_id)}
        with StagePool(list(shard_by_id.values()), workers=search_worker_budget()) as pool:
            for shard_id, shard in SmartTQDM(
                shard_by_id.items(),
                desc='Searching in shards: ',
                **sys_config.TQDM_CONFIG,
            ):
                identified_consts = list(shard_identified[shard_id])
                self._run_shard(
                    shard, identified_consts, num_workers, config_overrides,
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

        :param num_workers: Tier-2 background workers (when Tier-2 is active).
        :param config_overrides: Config propagated to the Tier-2 workers.
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

        with worker_pool(
            num_workers=num_workers,
            worker_fn=compute_tier2_for_item,
            writer_fn=write_jsonl_line,
            output_path=output_path,
            config_overrides=config_overrides,
            parallel=bool(search_config.TIER2_ATTRIBUTES),
        ) as push:
            self._produce(
                shard=shard,
                identified_consts=identified_consts,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=push,
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

        desired = {attribute_name(s) for s in search_config.TIER2_ATTRIBUTES}

        # Plan (main process): classify every sampled trajectory against the
        # cache and queue one Tier-1 task per trajectory that needs work.
        planned: List[tuple] = []
        tasks: List[Tier1Task] = []
        for traj, start in pairs:
            start_t = _position_to_tuple(start)
            dir_t = _position_to_tuple(traj)
            trajectory_id = derive_trajectory_id(
                shard_id, shard.cmf_name, shard_encoding_str, start_t, dir_t,
            )
            current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, traj))
            bucket = seen_trajectories.get(trajectory_id, {})

            # Classify each searched constant against its cached (traj, const) row:
            #   'new'      — no fresh row (needs the full Tier-1 DTO), or
            #   'partial'  — fresh row but missing some Tier-2 attributes (patch), or
            #   'complete' — fresh row with every desired attribute (skip).
            new_consts: List[Constant] = []
            partial_consts: List[Constant] = []
            for c in identified_consts:
                rec = bucket.get(c.name)
                if rec is None or rec.get("config_fingerprint") != current_fp:
                    new_consts.append(c)
                elif desired - set(rec.keys()):
                    partial_consts.append(c)
            if not new_consts and not partial_consts:
                continue  # every constant fully covered

            planned.append((trajectory_id, dir_t, start_t, new_consts, partial_consts))
            tasks.append(Tier1Task(
                key=trajectory_id, shard_idx=shard_idx, trajectory=traj, start=start,
                constants=tuple(new_consts), cmf_id=cmf_id, shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
            ))

        if not tasks:
            return

        # Compute: the shared handler (walk + new-constant DTOs) for each queued
        # trajectory runs on the stage pool; results arrive in task order and the
        # main process forwards them to the sink.  The symbolic matrix is shipped
        # back only when the Tier-2 queue will use it.
        task_fn = _search_walk_task_with_matrix if desired else _search_walk_task
        own_pool = None
        if pool is None:
            own_pool = StagePool([shard], workers=1)
            pool = own_pool.__enter__()
        try:
            results = pool.imap(task_fn, tasks)
            for (trajectory_id, dir_t, start_t, new_consts, partial_consts), task, res in zip(
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
                matrix = res.matrix

                # New constants — emit the full flat DTO (Tier-2 filled by the worker).
                for c, dto in zip(new_consts, res.dtos):
                    sink((matrix, c.value_sympy, dto))
                    seen_trajectories.setdefault(trajectory_id, {})[c.name] = (
                        json.loads(dto.to_json_line())
                    )
                # Partial constants — emit a patch; the worker fills missing Tier-2.
                # Carry the direction / start so the worker can compute
                # direction-normalised attributes (e.g. convergence_rate).
                for c in partial_consts:
                    patch = {
                        "trajectory_id": trajectory_id, "constant": c.name,
                        "direction": dir_t, "start_point": start_t,
                    }
                    sink((matrix, c.value_sympy, patch))
        finally:
            if own_pool is not None:
                own_pool.__exit__(None, None, None)


def _search_walk_task(task: Tier1Task):
    """Stage-pool task: handler + new-constant Tier-1 DTOs for one trajectory.

    Module-level (picklable); uses this module's ``TrajectoryAttributesHandler``
    and ``build_trajectory_dtos`` names, resolved at call time.
    """
    return run_tier1_task(
        task,
        handler_cls=TrajectoryAttributesHandler,
        build_dtos=build_trajectory_dtos,
    )


def _search_walk_task_with_matrix(task: Tier1Task):
    """:func:`_search_walk_task` that also returns the trajectory matrix (Tier-2 input)."""
    return run_tier1_task(
        task,
        handler_cls=TrajectoryAttributesHandler,
        build_dtos=build_trajectory_dtos,
        want_matrix=True,
    )
