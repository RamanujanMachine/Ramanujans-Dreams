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
        3. Otherwise build the handler and push per-constant Tier-1 data.

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
from dreamer.configs.logging import logging_config
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
    worker_pool,
    write_jsonl_line,
)

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

        for shard_id, shard in SmartTQDM(
            shard_by_id.items(),
            desc='Searching in shards: ',
            **sys_config.TQDM_CONFIG,
        ):
            identified_consts = list(shard_identified[shard_id])
            self._run_shard(shard, identified_consts, num_workers, config_overrides)

    # ------------------------------------------------------------------
    # Per-shard pipeline
    # ------------------------------------------------------------------

    def _run_shard(
        self,
        shard: Shard,
        identified_consts: List[Constant],
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Run the search for a single shard using only *identified_consts*."""
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
    ) -> None:
        """Iterate over trajectory pairs and hand work to *sink*.

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

        for traj, start in SmartTQDM(
            pairs,
            desc=f"  Shard {shard_id[:8]}… trajectories",
            leave=False,
            **sys_config.TQDM_CONFIG,
        ):
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

            # Build the (shared) handler once for whichever constants need work.
            try:
                with Logger.watchdog(
                    f"Tier-1 trajectory compute (shard {shard_id})",
                    logging_config.WATCHDOG_TRAJECTORY_SECONDS,
                    detail=lambda: f"traj_id={trajectory_id} start={start_t} direction={dir_t}",
                ):
                    handler = TrajectoryAttributesHandler.from_cmf(
                        shard.cmf, traj, start, constant=None, searchable=shard,
                    )
                    dtos = build_trajectory_dtos(
                        handler,
                        cmf_id=cmf_id,
                        shard_id=shard_id,
                        cmf_name=shard.cmf_name,
                        shard_encoding_str=shard_encoding_str,
                        start=start,
                        direction=traj,
                        constants=new_consts,
                    ) if new_consts else []
            except Exception as e:
                Logger(
                    f"Handler error — shard {shard_id}, traj={traj}, start={start}: {e}",
                    Logger.Levels.warning,
                ).log()
                continue

            # New constants — emit the full flat DTO (Tier-2 filled by the worker).
            for c, dto in zip(new_consts, dtos):
                sink((handler.trajectory_matrix, c.value_sympy, dto))
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
                sink((handler.trajectory_matrix, c.value_sympy, patch))
