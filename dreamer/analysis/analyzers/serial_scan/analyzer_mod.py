"""
AnalyzerModV1 — analysis-stage module.

For each shard, samples trajectories and records Tier-1 attributes
(``delta``, ``identified``, plus the rest of the ``TrajectoryDTO`` core
fields) as one JSONL line per trajectory.  The trajectory walk is computed
*once* per (trajectory, shard) and evaluated against **all** constants
bound to the shard; the per-constant attributes (``delta``, ``p_vector``,
``q_vector``, ``identified``) are then written as one **flat row per
``(trajectory, constant)`` pair** (one JSONL line each).

**JSONL layout** — one file per shard (no constant subdirectory):
    ``EXPORT_SEARCH_RESULTS/<shard_id>.jsonl`` by default, shared with the
    search stage.  When ``analysis.STORE_TRAJECTORIES_SEPARATELY`` is enabled the
    records are written to a separate per-shard store,
    ``EXPORT_ANALYSIS_RESULTS/<shard_id>.jsonl``, instead; the search stage then
    seeds its cache from that store (see
    ``multi_processing.load_seen_trajectories_for_search``).

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
from dreamer.configs.logging import logging_config
from dreamer.extraction.shard import Shard
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dtos,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.storage.atlas_writer import update_shard_found_constants
from dreamer.utils.storage.optimization_objectives import score_record
from dreamer.search.methods.hedgehog_scan import SerialSearcher
import math

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

        Returns a mapping from constant → shards sorted by the best observed
        value of the active optimisation objective (``system.OPTIMIZATION_OBJECTIVE``
        — δ by default) descending, then by dimension (ascending, as a
        tie-breaker).  Ranking follows the objective, but a shard is kept only when
        its identified percentage meets ``IDENTIFY_THRESHOLD`` — identification is
        a prerequisite we care about regardless of which objective is optimised.
        """
        # Trajectory records normally share the search-results dir so the search
        # stage reuses them directly.  When STORE_TRAJECTORIES_SEPARATELY is set,
        # the analysis stage keeps its own per-shard store under
        # EXPORT_ANALYSIS_RESULTS instead (the search stage still seeds its cache
        # from it — see load_seen_trajectories_for_search).
        out_dir = (
            sys_config.EXPORT_ANALYSIS_RESULTS
            if analysis_config.STORE_TRAJECTORIES_SEPARATELY
            else sys_config.EXPORT_SEARCH_RESULTS
        )
        os.makedirs(out_dir, exist_ok=True)

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

        # cmf_name → {shard_id: [identified constant names]} — populated by
        # _analyze_shard, flushed to the shard JSONL after the loop so a constant
        # is recorded as found in a shard only when a trajectory identified it.
        self._found_constants_by_shard: Dict[str, Dict[str, List[str]]] = {}

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

            shard_width = int(math.log10(len(shards))) + 1

            for i, shard in enumerate(shards):
                cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)

                if shard_id in seen_shard_ids:
                    # Already analysed — skip.
                    continue
                seen_shard_ids.add(shard_id)
                shard_objects[shard_id] = shard

                shard_jsonl_path = os.path.join(out_dir, f"{shard_id}.jsonl")
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
                    objective_name = config.system.OPTIMIZATION_OBJECTIVE
                    for c in shard.consts:
                        if c in per_const_best:
                            bd = per_const_best[c]
                            bd_str = f'{bd:.4f}' if bd is not None else 'N/A'
                            Logger(
                                f"Shard {i+1:0{shard_width}d} in {cmf_id} - searching {c.name}: best {objective_name}={bd_str} [identified: ✅]",
                                Logger.Levels.info,
                            ).log()
                        else:
                            Logger(
                                f"Shard {i+1:0{shard_width}d} in {cmf_id} - searching {c.name}: {' ':<17} [identified: ❌]",
                                Logger.Levels.info,
                            ).log()

        # Persist the actually-found constants into the per-CMF shard JSONL so the
        # cached records (and any no-extractor rerun) reflect only constants that an
        # identified trajectory converged to — not every candidate constant.
        if sys_config.EXPORT_CMFS:
            for cmf_name, found_by_shard in self._found_constants_by_shard.items():
                update_shard_found_constants(
                    sys_config.EXPORT_CMFS, cmf_name, found_by_shard
                )

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

        Returns ``{Constant: best_objective_score}`` for each constant in
        ``shard.consts`` that passed the identified-percentage threshold, where the
        score is the best (signed, higher-is-better) value of the active
        optimisation objective (δ by default).  Constants that did not reach the
        threshold are excluded from the result dict entirely, so the caller can
        distinguish "failed" from "constant not in shard".

        The trajectory walk is computed once per trajectory and evaluated
        against every constant via ``build_trajectory_dto(..., constants=...)``.
        """
        Logger(
            f"Starting analysis on shard {shard_id} "
            f"(cmf={cmf_id}, encoding={encoding_str})",
            Logger.Levels.debug,
        ).log()

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

        # Seed the user-supplied trajectory into the analysis set (first) so its δ is
        # computed and recorded alongside the sampled ones.  It is paired with the
        # shard's interior point, exactly like ``sample_pairs``.  Per-run dedup below
        # avoids walking it twice if the sampler happened to draw the same direction.
        if getattr(shard, "selected_trajectory", None) is not None:
            pairs = [(shard.selected_trajectory, shard.get_interior_point())] + list(pairs)

        # Per-constant accumulators.  ``best_score`` tracks the best (signed,
        # higher-is-better) value of the active optimisation objective — δ by
        # default, but e.g. convergence_rate when configured — for each constant.
        objective_name = config.system.OPTIMIZATION_OBJECTIVE
        total = 0
        identified_count: Dict[str, int] = defaultdict(int)
        best_score: Dict[str, Optional[float]] = {c.name: None for c in shard.consts}
        processed_tids: set = set()

        def _accumulate(record: dict, const_name: str) -> None:
            """Fold one ``(trajectory, constant)`` *record* into the counts + best.

            Identification is counted independently of the objective (it is a hard
            prerequisite we care about regardless); ranking uses the active
            objective's signed score via the shared ``score_record``.
            """
            scored = score_record(record, objective_name)
            if scored is None:
                return
            sc, identified_val = scored
            if not identified_val:
                return
            identified_count[const_name] += 1
            if math.isfinite(sc):
                cur = best_score.get(const_name)
                if cur is None or sc > cur:
                    best_score[const_name] = sc

        def _reusable(rec: Optional[dict], fp: str) -> bool:
            """A cached per-constant row is reusable when it is fresh (matching
            config fingerprint) *and* carries the active objective's column —
            ``score_record`` returns ``None`` only when that column is absent."""
            return (
                rec is not None
                and rec.get("config_fingerprint") == fp
                and score_record(rec, objective_name) is not None
            )

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

                # Skip a trajectory already handled in this run (e.g. the injected
                # seed also drawn by the sampler) — count + write it only once.
                if tid in processed_tids:
                    continue
                processed_tids.add(tid)

                current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, traj))
                bucket = seen_trajectories.get(tid, {})

                # Reuse only when *every* shard constant has a fresh, objective-
                # covered row — else recompute (a changed walk depth / walk type /
                # identification tolerance, or a switched objective column).
                if all(_reusable(bucket.get(c.name), current_fp) for c in shard.consts):
                    for c in shard.consts:
                        _accumulate(bucket[c.name], c.name)
                    total += 1
                    continue

                try:
                    with Logger.watchdog(
                        f"Tier-1 trajectory compute (shard {shard_id})",
                        logging_config.WATCHDOG_TRAJECTORY_SECONDS,
                        detail=lambda: f"traj_id={tid} start={start_t} direction={dir_t}",
                    ):
                        handler = TrajectoryAttributesHandler.from_cmf(
                            shard.cmf, traj, start,
                            constant=None,  # injected per-constant in build_trajectory_dtos
                            searchable=shard,
                        )
                        dtos = build_trajectory_dtos(
                            handler,
                            cmf_id=cmf_id,
                            shard_id=shard_id,
                            cmf_name=shard.cmf_name,
                            shard_encoding_str=encoding_str,
                            start=start,
                            direction=traj,
                            constants=shard.consts,  # one flat row per constant
                        )
                except Exception as e:
                    Logger(
                        f"Handler error — shard {shard_id}, "
                        f"traj={traj}, start={start}: {e}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                for c, dto in zip(shard.consts, dtos):
                    line = dto.to_json_line()
                    fout.write(line + "\n")
                    record = json.loads(line)
                    seen_trajectories.setdefault(tid, {})[c.name] = record
                    _accumulate(record, c.name)
                fout.flush()
                total += 1

        Logger(
            f"Finished analysis on shard {shard_id}: {total} trajectories",
            Logger.Levels.debug,
        ).log()

        # User-facing summary: which start point we searched from and how many of the
        # sampled trajectories identified each constant (LIReC).
        start_disp = _position_to_tuple(shard.get_interior_point())
        pct_summary = ", ".join(
            f"{c.name} {100.0 * identified_count[c.name] / total:.2f}% "
            f"({identified_count[c.name]}/{total})"
            if total else f"{c.name} N/A"
            for c in shard.consts
        )
        Logger(
            f"Shard {shard_id} — start point {start_disp}; "
            f"identified trajectories: {pct_summary}",
            Logger.Levels.info,
        ).log()

        # Build the final result: only include constants that passed the threshold.
        # A constant is "found" in this shard iff at least one trajectory identified
        # it (LIReC), independent of the prioritisation threshold — recorded so the
        # shard JSONL only lists genuinely-found constants.
        self._found_constants_by_shard.setdefault(shard.cmf_name, {})[shard_id] = [
            c.name for c in shard.consts if identified_count[c.name] > 0
        ]

        result: Dict[Constant, Optional[float]] = {}
        for c in shard.consts:
            ident_pct = identified_count[c.name] / total if total else 0.0
            if (
                ident_pct >= analysis_config.IDENTIFY_THRESHOLD
                and best_score.get(c.name) is not None
            ):
                result[c] = best_score[c.name]

        return result
