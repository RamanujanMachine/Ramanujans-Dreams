"""
GradientAscentMod — search-stage module driving :class:`GradientAscentSearch`.

Modelled on :class:`SimulatedAnnealingMod` (the modern DTO/JSONL pipeline).
For each unique shard (deduplicated by ``shard_id`` across all constants) it
opens a ``worker_pool`` writing ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl`` and
runs gradient ascent **once per identified constant**.  A constant whose
reservoir produces no initial identification, or whose ascent stalls on an
unidentified region, is logged and skipped.
"""

import os
from collections import defaultdict
from typing import Dict, List, Set

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.parallel_eval import make_shared_eval_pool
from dreamer.search.methods.gradient_ascent import (
    GradientAscentSearch,
    NoInitialIdentification,
    SearchStalled,
)
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.multi_processing import (
    compute_tier2_for_item,
    load_seen_trajectories,
    worker_pool,
    write_jsonl_line,
)

search_config = config.search


class GradientAscentMod(SearcherModScheme):
    """Search module — per-shard, per-constant gradient ascent."""

    def __init__(self, priorities, use_LIReC: bool = True):
        """
        :param priorities: ``Dict[Constant, List[Shard]]`` — shards that passed
            analysis for each constant.
        :param use_LIReC: Whether to use LIReC for constant identification.
        """
        super().__init__(
            priorities,
            use_LIReC,
            name="GradientAscent",
            description="Search module — gradient ascent with Tier-1 DTO output",
            version="1.0.0",
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """Run the search over all unique shards."""
        if not self.searchables:
            return

        os.makedirs(sys_config.EXPORT_SEARCH_RESULTS, exist_ok=True)

        num_workers = sys_config.NUM_BACKGROUND_WORKERS
        config_overrides = config.export_configurations()

        shard_identified: Dict[str, Set[Constant]] = defaultdict(set)
        shard_by_id: Dict[str, Shard] = {}
        for const, shards in self.priorities.items():
            for shard in shards:
                _, shard_id, _ = derive_cmf_and_shard_ids(shard)
                shard_by_id[shard_id] = shard
                shard_identified[shard_id].add(const)

        for shard_id, shard in SmartTQDM(
            shard_by_id.items(),
            desc="Gradient ascent in shards: ",
            **sys_config.TQDM_CONFIG,
        ):
            identified_consts = list(shard_identified[shard_id])
            self._run_shard(shard, identified_consts, num_workers, config_overrides)

    def _run_shard(
        self,
        shard: Shard,
        identified_consts: List[Constant],
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Run gradient ascent for each identified constant of a single shard."""
        cmf_id, shard_id, shard_encoding_str = derive_cmf_and_shard_ids(shard)
        output_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, f"{shard_id}.jsonl")
        seen_trajectories = load_seen_trajectories(output_path)

        handler_cache: dict = {}

        # Flatland geometry (LLL/BKZ) + interior start built once per shard
        # (constant-independent).  The persistent per-shard process pool walks
        # each step's forward-difference gradient probe batch; reused across all
        # constants and steps.
        geom = FlatlandGeometry(shard)
        start = shard.get_interior_point()
        eval_pool, pq_manager = make_shared_eval_pool(
            shard, start, search_config.GRAD_NUM_EVAL_WORKERS
        )

        try:
            with worker_pool(
                num_workers=num_workers,
                worker_fn=compute_tier2_for_item,
                writer_fn=write_jsonl_line,
                output_path=output_path,
                config_overrides=config_overrides,
                parallel=bool(search_config.TIER2_ATTRIBUTES),
            ) as push:
                for const in identified_consts:
                    method = GradientAscentSearch(shard, const, use_LIReC=self.use_LIReC)
                    try:
                        method.run(
                            constant=const,
                            cmf_id=cmf_id,
                            shard_id=shard_id,
                            shard_encoding_str=shard_encoding_str,
                            sink=push,
                            seen_trajectories=seen_trajectories,
                            handler_cache=handler_cache,
                            geom=geom,
                            start=start,
                            pool=eval_pool,
                        )
                    except (NoInitialIdentification, SearchStalled) as e:
                        Logger(str(e), Logger.Levels.warning).log()
                        continue
        finally:
            if eval_pool is not None:
                eval_pool.close()
                eval_pool.join()
            if pq_manager is not None:
                pq_manager.shutdown()
