"""
Template: add your own **analysis** stage to the pipeline.

Copy this file, rename the classes, and fill in the ``TODO``s. An analysis
stage is two classes (keep them in separate files when contributing — see
CONTRIBUTING.md):

  * ``MyAnalyzer`` — the *internal logic*: cheaply probe one shard (sample a few
    trajectories, compute their Tier-1 attributes) and report the best δ per
    constant.
  * ``MyAnalyzerMod`` (an ``AnalyzerModScheme``) — the *external interface*: run
    the analyzer over every shard, then filter + rank the shards the search
    stage will consume.

Wire your module into the pipeline with::

    from dreamer import System
    System(function_sources=[...], analyzers=[MyAnalyzerMod]).run(constants=[...])

``System`` constructs the module as ``MyAnalyzerMod(cmf_data)`` and calls
``.execute()``. Analysis is triage: keep the shards that actually produce the
constant (above ``config.analysis.IDENTIFY_THRESHOLD``) and rank them best-first
so the expensive search stage spends its budget where it matters.

------------------------------------------------------------------------------
HOW RESULTS ARE STORED (read this once)
------------------------------------------------------------------------------
Analysis writes the **same per-shard JSONL** the search stage reads — one
trajectory per line in ``<shard_id>.jsonl`` (no pickles). By default that lives
under ``config.system.EXPORT_SEARCH_RESULTS`` so search re-uses the records
directly; when ``config.analysis.STORE_TRAJECTORIES_SEPARATELY`` is on, analysis
writes to ``config.system.EXPORT_ANALYSIS_RESULTS`` instead (search still seeds
its cache from there).

This template mirrors the production default,
``dreamer.analysis.analyzers.serial_scan.analyzer_mod.AnalyzerModV1`` — read it
for the full re-run optimisation (config-fingerprint staleness, cached-δ reuse);
this template keeps the loop minimal on purpose.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
)
from dreamer.utils.multi_processing import load_seen_trajectories

analysis_config = config.analysis


class MyAnalyzer:
    """Probe a single shard: sample trajectories, write their Tier-1 JSONL
    records, and report the best δ found for each of the shard's constants.
    """

    def __init__(self, shard: Shard, use_LIReC: bool = True):
        """
        :param shard: The shard to analyse (carries its own constants in
            ``shard.consts``).
        :param use_LIReC: Identify constants with LIReC.
        """
        self.shard = shard
        self.use_LIReC = use_LIReC

    def analyze(
        self,
        *,
        cmf_id: str,
        shard_id: str,
        encoding_str: str,
        jsonl_path: str,
        seen_trajectories: dict,
    ) -> Dict[Constant, Optional[float]]:
        """
        Sample + evaluate trajectories in the shard and write one JSONL line per
        trajectory.

        :return: ``{Constant: best_delta}`` for every constant of the shard that
            passed ``IDENTIFY_THRESHOLD`` (others are omitted). The trajectory
            walk is computed once and scored against **all** the shard's
            constants at once.
        """
        # Trajectory sampling is constant-independent; the first constant is only
        # used to drive the sampler.
        sampler = SerialSearcher(self.shard, self.shard.consts[0], use_LIReC=self.use_LIReC)
        try:
            pairs = sampler.sample_pairs(
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
                sampling_method=analysis_config.SAMPLING_METHOD,
            )
        except ValueError as e:
            Logger(f"Skipping shard {shard_id}: {e}", Logger.Levels.warning).log()
            return {}

        total = 0
        identified_count: Dict[str, int] = defaultdict(int)
        best_delta: Dict[str, Optional[float]] = {c.name: None for c in self.shard.consts}

        with open(jsonl_path, "a") as fout:
            for traj, start in SmartTQDM(
                pairs, desc=f"  shard {shard_id[:8]}…", leave=False, **sys_config.TQDM_CONFIG,
            ):
                trajectory_id = derive_trajectory_id(
                    shard_id, self.shard.cmf_name, encoding_str,
                    _position_to_tuple(start), _position_to_tuple(traj),
                )
                if trajectory_id in seen_trajectories:
                    continue  # already on disk — skip (AnalyzerModV1 reuses its δ)

                try:
                    handler = TrajectoryAttributesHandler.from_cmf(
                        self.shard.cmf, traj, start, constant=None, searchable=self.shard,
                    )
                    dto = build_trajectory_dto(
                        handler,
                        cmf_id=cmf_id,
                        shard_id=shard_id,
                        cmf_name=self.shard.cmf_name,
                        shard_encoding_str=encoding_str,
                        start=start,
                        direction=traj,
                        constants=self.shard.consts,  # score all constants at once
                    )
                except Exception as e:
                    Logger(
                        f"Handler error — shard {shard_id}, traj={traj}: {e}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                fout.write(dto.to_json_line() + "\n")
                fout.flush()
                seen_trajectories[trajectory_id] = {"trajectory_id": trajectory_id}
                total += 1

                for c in self.shard.consts:
                    if bool((dto.identified or {}).get(c.name, False)):
                        identified_count[c.name] += 1
                        delta_val = dto.delta_estimate.get(c.name)
                        if delta_val is not None and (
                            best_delta[c.name] is None or delta_val > best_delta[c.name]
                        ):
                            best_delta[c.name] = delta_val

        # Keep a constant only if enough of its trajectories identified it.
        result: Dict[Constant, Optional[float]] = {}
        for c in self.shard.consts:
            ident_pct = identified_count[c.name] / total if total else 0.0
            if ident_pct >= analysis_config.IDENTIFY_THRESHOLD and best_delta[c.name] is not None:
                result[c] = best_delta[c.name]
        return result


class MyAnalyzerMod(AnalyzerModScheme):
    """The stage module — analyses every shard and returns them filtered + ranked.

    ``System`` instantiates this as ``MyAnalyzerMod(cmf_data)``, where
    ``cmf_data`` is ``Dict[Constant, List[Searchable]]`` from extraction.
    """

    def __init__(
            self,
            cmf_data: Dict[Constant, List[Searchable]],
            # TODO: <your arguments here>
            #   If you add constructor arguments, wire the module in with
            #   functools.partial so System can still construct it with just
            #   cmf_data:  analyzers=[partial(MyAnalyzerMod, my_arg=...)]
    ):
        super().__init__(
            cmf_data,
            name='A very witty name',
            desc='My super cool and smart analysis module',
            version='your version here :)',
        )
        # TODO: store your own arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Analyse every constant's shards and return them filtered + ranked.

        :return: Mapping from each constant to its shards, best-δ first (with
            lower dimension as a tie-break). Only constants that passed the
            threshold for at least one shard appear.
        """
        out_dir = (
            sys_config.EXPORT_ANALYSIS_RESULTS
            if analysis_config.STORE_TRAJECTORIES_SEPARATELY
            else sys_config.EXPORT_SEARCH_RESULTS
        )
        os.makedirs(out_dir, exist_ok=True)

        # shard_id → {Constant: best_delta} and shard_id → Shard, deduplicating
        # shards that appear under several constants.
        shard_const_best: Dict[str, Dict[Constant, float]] = {}
        shard_by_id: Dict[str, Shard] = {}
        seen_shard_ids: Set[str] = set()

        for constant, shards in SmartTQDM(
                self.cmf_data.items(), desc='Analyzing constants and their CMFs',
                **sys_config.TQDM_CONFIG,
        ):
            Logger(
                Logger.buffer_print(
                    sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant.name}', '='
                ), Logger.Levels.message,
            ).log()

            for shard in shards:
                cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)
                if shard_id in seen_shard_ids:
                    continue  # analyse each unique shard once
                seen_shard_ids.add(shard_id)
                shard_by_id[shard_id] = shard

                jsonl_path = os.path.join(out_dir, f"{shard_id}.jsonl")
                shard_const_best[shard_id] = MyAnalyzer(shard, use_LIReC=True).analyze(
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    encoding_str=encoding_str,
                    jsonl_path=jsonl_path,
                    seen_trajectories=load_seen_trajectories(jsonl_path),
                )

        # Build the ranked per-constant priority lists.
        result: Dict[Constant, List[Searchable]] = {c: [] for c in self.cmf_data.keys()}
        for const in self.cmf_data.keys():
            passing = [
                shard_by_id[sid] for sid, best in shard_const_best.items()
                if best.get(const) is not None
            ]
            result[const] = sorted(
                passing,
                key=lambda s: (
                    -shard_const_best[derive_cmf_and_shard_ids(s)[1]][const],  # best δ first
                    s.dim,                                                     # then smaller dim
                ),
            )
        return result
