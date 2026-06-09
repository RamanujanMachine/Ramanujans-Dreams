"""
AnalyzerModV1 — analysis-stage module.

For each shard, samples trajectories and records Tier-1 attributes
(``delta``, ``identified``, plus the rest of the ``TrajectoryDTO`` core
fields) as one JSONL line per trajectory.  The trajectory walk is computed
*once* per (trajectory, shard) and evaluated against **all** constants
bound to the shard; per-constant attributes (``delta_estimate``,
``p_vector``, ``q_vector``, ``identified``) are stored as dicts keyed by
constant name.

**JSONL layout** — one file per shard (no constant subdirectory):
    ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl``

**Per-shard deduplication** — each unique shard (by shard_id) is processed
exactly once even if it appears under several constants in the input dict.

**Analysis threshold** — a shard is kept for constant C if C's
``identified_pct`` meets ``IDENTIFY_THRESHOLD``.  The shard is placed in
the output dict under *every* constant for which it passes; a shard that
passes for none of its constants is discarded entirely.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.configs.system import sys_config
from dreamer.configs import config
from dreamer.extraction.shard import Shard
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.search.methods.hedgehog_scan import SerialSearcher

analysis_config = config.analysis


class AnalyzerModV1(AnalyzerModScheme):
    """Analysis module: filters and ranks shards by Tier-1 trajectory attributes.

    For each unique shard, samples trajectories and computes ``delta`` +
    ``identified`` for every constant in ``shard.consts``.  Records are
    appended to a single per-shard JSONL (shared by the searcher) at
    ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl``.

    Shards passing the identified-percentage threshold for at least one
    constant are kept and sorted by best observed delta; they are placed
    in the result dict under every constant for which they pass.
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        """
        :param cmf_data: Mapping from each constant to its list of shards.
        """
        super().__init__(
            cmf_data,
            desc='Analysis module — per-trajectory dedup, ranks shards by best delta',
            version='4',
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """Filter and rank shards for every constant.

        Returns a mapping from constant → shards sorted by best delta
        (descending), then by dimension (ascending, as a tie-breaker).
        Only constants whose shards are identified above threshold appear.
        """
        os.makedirs(sys_config.EXPORT_SEARCH_RESULTS, exist_ok=True)

        result: Dict[Constant, List[Searchable]] = {c: [] for c in self.cmf_data.keys()}

        # Collect the superset of all constants we need to analyse.
        all_constants: Set[Constant] = set(self.cmf_data.keys())

        # Deduplicate shards — the same Shard object may appear under several
        # constants.  Process each unique shard_id exactly once.
        seen_shard_ids: Set[str] = set()

        # shard_id → {const: best_delta}  (None = not identified / no walk)
        shard_const_best: Dict[str, Dict[Constant, Optional[float]]] = {}
        # shard_id → Shard object (to build the sorted result later)
        shard_objects: Dict[str, Shard] = {}

        # Iterate in a deterministic order: all constants, then their shards.
        for constant, shards in SmartTQDM(
            self.cmf_data.items(),
            desc='Analyzing constants and their CMFs',
            **sys_config.TQDM_CONFIG,
        ):
            Logger(
                Logger.buffer_print(
                    sys_config.LOGGING_BUFFER_SIZE,
                    f'Analyzing for {constant.name}',
                    '=',
                ),
                Logger.Levels.message,
            ).log()

            for i, shard in enumerate(shards):
                cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)

                if shard_id in seen_shard_ids:
                    # Already analysed — skip.
                    continue
                seen_shard_ids.add(shard_id)
                shard_objects[shard_id] = shard

                shard_jsonl_path = os.path.join(
                    sys_config.EXPORT_SEARCH_RESULTS, f"{shard_id}.jsonl"
                )
                seen_trajectories = load_seen_trajectories(shard_jsonl_path)

                per_const_best = self._analyze_shard(
                    shard,
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    encoding_str=encoding_str,
                    jsonl_path=shard_jsonl_path,
                    seen_trajectories=seen_trajectories,
                )
                shard_const_best[shard_id] = per_const_best

                if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                    for c in shard.consts:
                        if c in per_const_best:
                            bd = per_const_best[c]
                            bd_str = f'{bd:.4f}' if bd is not None else 'N/A'
                            Logger(
                                f"Shard {i+1} in {cmf_id} - searching {c.name}: best_delta={bd_str}",
                                Logger.Levels.info,
                            ).log()
                        else:
                            Logger(
                                f"Shard {i+1} in {cmf_id} - searching {c.name}: not identified",
                                Logger.Levels.info,
                            ).log()

        # Build per-constant priority lists from the analysis results.
        for const in all_constants:
            # Gather (shard, best_delta) pairs where this constant was identified.
            passing: List[Tuple[Shard, float]] = []
            for shard_id, per_const_best in shard_const_best.items():
                bd = per_const_best.get(const)
                if bd is not None:
                    passing.append((shard_objects[shard_id], bd))

            result[const] = sorted(
                [s for s, _ in passing],
                key=lambda s: (
                    -(shard_const_best[derive_cmf_and_shard_ids(s)[1]].get(const, -float('inf')) or -float('inf')),
                    s.dim,
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Per-shard analysis
    # ------------------------------------------------------------------

    def _analyze_shard(
        self,
        shard: Shard,
        *,
        cmf_id: str,
        shard_id: str,
        encoding_str: str,
        jsonl_path: str,
        seen_trajectories: dict,
    ) -> Dict[Constant, Optional[float]]:
        """Sample trajectories in *shard* and aggregate Tier-1 stats for all constants.

        Returns ``{Constant: best_delta_or_None}`` for each constant in
        ``shard.consts`` that passed the identified-percentage threshold.
        Constants that did not reach the threshold map to ``None`` (excluded
        from the result dict entirely so the caller can distinguish "failed"
        from "constant not in shard").

        The trajectory walk is computed once per trajectory and evaluated
        against every constant via ``build_trajectory_dto(..., constants=...)``.
        """
        # Use the first constant just to drive the SerialSearcher for pair sampling
        # (trajectory sampling is constant-independent).
        primary_const = shard.consts[0]
        searcher = SerialSearcher(shard, primary_const, use_LIReC=False)
        try:
            pairs = searcher.sample_pairs(
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
                sampling_method=analysis_config.SAMPLING_METHOD,
            )
        except ValueError as e:
            Logger(
                f"Skipping shard {shard_id}: {e}",
                Logger.Levels.warning,
            ).log()
            return {}

        # Per-constant accumulators.
        total = 0
        identified_count: Dict[str, int] = defaultdict(int)
        best_delta: Dict[str, Optional[float]] = {c.name: None for c in shard.consts}

        with open(jsonl_path, "a") as fout:
            for traj, start in SmartTQDM(
                pairs,
                desc=f"  Shard {shard_id[:8]}… trajectories",
                leave=False,
                **sys_config.TQDM_CONFIG,
            ):
                start_t = _position_to_tuple(start)
                dir_t = _position_to_tuple(traj)
                tid = derive_trajectory_id(
                    shard_id, shard.cmf_name, encoding_str, start_t, dir_t,
                )

                # Reuse a cached record only when it was computed under the same
                # configuration — a changed walk depth / walk type / identification
                # tolerance makes the stored δ stale and forces recomputation.
                current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, traj))

                cached = seen_trajectories.get(tid)
                if (
                    cached is not None
                    and cached.get("config_fingerprint") == current_fp
                    and "delta_estimate" in cached
                    and isinstance(cached["delta_estimate"], dict)
                    and "identified" in cached
                    and isinstance(cached["identified"], dict)
                    # All shard constants must be covered in the cached record.
                    and all(c.name in cached["delta_estimate"] for c in shard.consts)
                ):
                    # Reuse cached record — no handler, no walk.
                    for c in shard.consts:
                        delta_val = cached["delta_estimate"].get(c.name)
                        identified_val = bool(cached["identified"].get(c.name, False))
                        if identified_val:
                            identified_count[c.name] += 1
                            if delta_val is not None:
                                cur = best_delta.get(c.name)
                                if cur is None or delta_val > cur:
                                    best_delta[c.name] = delta_val
                    total += 1
                    continue

                try:
                    handler = TrajectoryAttributesHandler.from_cmf(
                        shard.cmf, traj, start,
                        constant=None,  # constant injected per-constant in build_trajectory_dto
                        searchable=shard,
                    )
                    dto = build_trajectory_dto(
                        handler,
                        cmf_id=cmf_id,
                        shard_id=shard_id,
                        cmf_name=shard.cmf_name,
                        shard_encoding_str=encoding_str,
                        start=start,
                        direction=traj,
                        constants=shard.consts,  # Constant objects → keys are c.name
                    )
                except Exception as e:
                    Logger(
                        f"Handler error — shard {shard_id}, "
                        f"traj={traj}, start={start}: {e}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                line = dto.to_json_line()
                fout.write(line + "\n")
                fout.flush()
                seen_trajectories[tid] = json.loads(line)

                for c in shard.consts:
                    delta_val = dto.delta_estimate.get(c.name)
                    identified_val = bool((dto.identified or {}).get(c.name, False))
                    if identified_val:
                        identified_count[c.name] += 1
                        if delta_val is not None:
                            cur = best_delta.get(c.name)
                            if cur is None or delta_val > cur:
                                best_delta[c.name] = delta_val

                total += 1

        # Build the final result: only include constants that passed the threshold.
        result: Dict[Constant, Optional[float]] = {}
        for c in shard.consts:
            ident_pct = identified_count[c.name] / total if total else 0.0
            if (
                ident_pct >= analysis_config.IDENTIFY_THRESHOLD
                and best_delta.get(c.name) is not None
            ):
                result[c] = best_delta[c.name]

        return result
