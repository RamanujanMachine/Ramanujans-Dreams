"""
Template: add your own **search** stage to the pipeline.

Copy this file, rename the classes, and fill in the ``TODO``s. A search stage
is two classes (keep them in separate files when contributing — see
CONTRIBUTING.md):

  * ``MySearchMethod`` — the *internal logic*: how your algorithm picks
    trajectory directions inside one shard and evaluates them.
  * ``MySearchMod`` (a ``SearcherModScheme``) — the *external interface*: how the
    method is driven over all the prioritised shards and how results are stored.

Wire your module into the pipeline with::

    from dreamer import System
    System(function_sources=[...], searcher=MySearchMod).run(constants=[...])

``System`` constructs the module as ``MySearchMod(priorities, use_LIReC)`` and
calls ``.execute()``; ``execute`` writes its results to disk and returns nothing.

------------------------------------------------------------------------------
HOW RESULTS ARE STORED (read this once)
------------------------------------------------------------------------------
There is **one canonical store**: one **JSONL** file per shard,
``<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl``, with **one trajectory per line**.
(No pickles — the run summary, best-δ reporting, and cross-run de-duplication
all read this JSONL store.)

You do not write the file by hand. Instead:

  1. Open a ``worker_pool`` for the shard — it gives you a ``push`` callable and
     a background writer that appends each record as a line.
  2. For each trajectory, build a ``TrajectoryAttributesHandler`` (the walk) and
     a ``TrajectoryDTO`` (its Tier-1 attributes: δ, identified, limit, p/q, …).
  3. ``push((trajectory_matrix, constant_sympy, dto))`` — the pool serialises the
     DTO to a JSONL line (and, if ``config.search.TIER2_ATTRIBUTES`` is set,
     computes those extra attributes in the background first).

This template mirrors the production default,
``dreamer.search.searchers.hedgehog_scan_mod.SearcherModV1`` (and the optimiser
searchers, e.g. ``small_angle_mod.py``). Read ``SearcherModV1`` for the full
re-run optimisation (config-fingerprint staleness checks, partial Tier-2
patches); this template keeps the loop minimal on purpose.
"""

import os
from collections import defaultdict
from typing import Callable, Dict, List, Set

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
)
from dreamer.utils.multi_processing import (
    compute_tier2_for_item,
    load_seen_trajectories_for_search,
    worker_pool,
    write_jsonl_line,
)

search_config = config.search


class MySearchMethod:
    """Your search algorithm, scoped to one shard and one constant.

    Emits one JSONL record per trajectory it evaluates by pushing a
    ``(trajectory_matrix, constant_sympy, dto)`` tuple into ``sink``.
    """

    def __init__(self, shard: Shard, constant: Constant, use_LIReC: bool = True):
        """
        :param shard: The shard to search within.
        :param constant: The constant to look for (identified for this shard
            during analysis).
        :param use_LIReC: Identify constants with LIReC while searching.
        """
        self.shard = shard
        self.constant = constant
        self.use_LIReC = use_LIReC
        # TODO: store any algorithm-specific arguments.

    def run(
        self,
        *,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
    ) -> None:
        """
        Search ``self.shard`` for ``self.constant`` and emit one record per
        trajectory to ``sink``.

        :param cmf_id, shard_id, shard_encoding_str: stable identifiers for the
            shard (from ``derive_cmf_and_shard_ids``); needed to build records.
        :param sink: push callable from the ``worker_pool`` — call it with
            ``(trajectory_matrix, constant_sympy, dto)`` to persist a trajectory.
        :param seen_trajectories: trajectories already on disk for this shard
            (keyed by ``trajectory_id``); use it to avoid re-walking.
        """
        # 1) DECIDE WHICH TRAJECTORIES TO EVALUATE. This is the only part that
        #    is really specific to your algorithm. Here we simply *sample* a
        #    batch (like the default searcher); a real custom method would
        #    hill-climb / mutate / anneal instead and yield directions as it goes.
        sampler = SerialSearcher(self.shard, self.constant, use_LIReC=self.use_LIReC)
        pairs = sampler.sample_pairs(
            trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM,
        )  # -> iterable of (trajectory_direction, start_point)

        for traj, start in SmartTQDM(
            pairs, desc=f"  shard {shard_id[:8]}… ({self.constant.name})",
            leave=False, **sys_config.TQDM_CONFIG,
        ):
            trajectory_id = derive_trajectory_id(
                shard_id, self.shard.cmf_name, shard_encoding_str,
                _position_to_tuple(start), _position_to_tuple(traj),
            )
            if trajectory_id in seen_trajectories:
                continue  # already evaluated (this or a previous run) — skip.

            # 2) EVALUATE: walk the trajectory and compute its Tier-1 attributes.
            try:
                handler = TrajectoryAttributesHandler.from_cmf(
                    self.shard.cmf, traj, start,
                    constant=None,             # per-constant δ injected below
                    searchable=self.shard,
                )
                dto = build_trajectory_dto(
                    handler,
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    cmf_name=self.shard.cmf_name,
                    shard_encoding_str=shard_encoding_str,
                    start=start,
                    direction=traj,
                    constants=[self.constant],  # Constant objects → keys are c.name
                )
            except Exception as e:  # a single bad trajectory must not kill the run
                Logger(
                    f"Handler error — shard {shard_id}, traj={traj}: {e}",
                    Logger.Levels.warning,
                ).log()
                continue

            # 3) PERSIST: one JSONL line per trajectory, via the pool's writer.
            seen_trajectories[trajectory_id] = {"trajectory_id": trajectory_id}
            sink((handler.trajectory_matrix, self.constant.value_sympy, dto))


class MySearchMod(SearcherModScheme):
    """The stage module — drives ``MySearchMethod`` over the prioritised shards.

    ``System`` instantiates this as ``MySearchMod(priorities, use_LIReC)``.
    ``priorities`` is the ``Dict[Constant, List[Shard]]`` produced by analysis
    (each constant mapped to the shards that identified it); the base class also
    exposes a deduplicated flat list as ``self.searchables``.
    """

    def __init__(
            self,
            priorities,                          # Dict[Constant, List[Shard]]
            use_LIReC: bool = True,
            # TODO: <your arguments here>
            #   If you add constructor arguments, wire the module in with
            #   functools.partial so System can still construct it with just
            #   (priorities, use_LIReC):  searcher=partial(MySearchMod, my_arg=...)
    ):
        super().__init__(
            priorities, use_LIReC,
            name='A very witty name',
            description='My super cool and smart search module',
            version='your version here :)',
        )
        # TODO: store your own arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """Run the search over every unique shard (writes JSONL, returns nothing)."""
        if not self.searchables:
            return

        os.makedirs(sys_config.EXPORT_SEARCH_RESULTS, exist_ok=True)
        num_workers = sys_config.NUM_BACKGROUND_WORKERS
        config_overrides = config.export_configurations()

        # Deduplicate shards by shard_id (one shard can appear under several
        # constants) and remember which constants each shard was identified for —
        # only those are worth searching.
        shard_by_id: Dict[str, Shard] = {}
        shard_identified: Dict[str, Set[Constant]] = defaultdict(set)
        for const, shards in self.priorities.items():
            for shard in shards:
                _, shard_id, _ = derive_cmf_and_shard_ids(shard)
                shard_by_id[shard_id] = shard
                shard_identified[shard_id].add(const)

        for shard_id, shard in SmartTQDM(
                shard_by_id.items(), desc='Searching shards: ', **sys_config.TQDM_CONFIG,
        ):
            self._run_shard(
                shard, list(shard_identified[shard_id]), num_workers, config_overrides,
            )

    def _run_shard(
        self,
        shard: Shard,
        identified_consts: List[Constant],
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Open the shard's JSONL store and run the method for each constant."""
        cmf_id, shard_id, shard_encoding_str = derive_cmf_and_shard_ids(shard)
        output_path = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, f"{shard_id}.jsonl")

        # Records already on disk (from analysis or a previous run) so we don't
        # recompute them. Returns a dict keyed by trajectory_id.
        seen_trajectories = load_seen_trajectories_for_search(output_path, shard_id)

        # The worker_pool gives us `push`: it serialises each DTO to a JSONL line
        # (and computes Tier-2 attributes in the background when configured).
        # With an empty TIER2_ATTRIBUTES no subprocess is spawned — writes are inline.
        with worker_pool(
            num_workers=num_workers,
            worker_fn=compute_tier2_for_item,
            writer_fn=write_jsonl_line,
            output_path=output_path,
            config_overrides=config_overrides,
            parallel=bool(search_config.TIER2_ATTRIBUTES),
        ) as push:
            for const in identified_consts:
                MySearchMethod(shard, const, use_LIReC=self.use_LIReC).run(
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    shard_encoding_str=shard_encoding_str,
                    sink=push,
                    seen_trajectories=seen_trajectories,
                )
