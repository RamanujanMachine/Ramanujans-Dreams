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

**Parallelism** — sampling and cache reuse run in the main process; every
trajectory that needs a walk (across *all* shards) is computed on one
stage-wide :class:`~dreamer.utils.stage_pool.StagePool`, and the main process
is the single writer of the JSONL (buffered appends).  See :meth:`execute`.

**Analysis threshold** — a shard is kept for constant C if C's
``identified_pct`` meets ``IDENTIFY_THRESHOLD``.  The shard is placed in
the output dict under *every* constant for which it passes; a shard that
passes for none of its constants is discarded entirely.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
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
    build_trajectory_dtos,
    derive_cmf_and_shard_ids,
    derive_trajectory_id,
    tier1_config_fingerprint,
    walk_depth_for,
)
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.utils.stage_pool import (
    BufferedJsonlWriter,
    StagePool,
    TaskError,
    Tier1Task,
    run_tier1_task,
    total_core_budget,
)
from dreamer.utils.storage.atlas_writer import update_shard_found_constants
from dreamer.utils.storage.attribute_fingerprints import tier1_record_fresh
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

        Runs in three phases so every core is busy while walks are computed:

        1. **Plan** (main process): for every unique shard, load its JSONL once,
           sample the trajectory pairs, reuse fresh cached rows, and queue a
           Tier-1 task for every trajectory that needs a walk.
        2. **Compute**: all queued walks — across *all* shards — stream through
           one :class:`~dreamer.utils.stage_pool.StagePool` (``chunksize`` 1,
           dynamic load balancing).  The main process is the single writer: it
           appends each returned row through a buffered writer and folds it into
           the shard's statistics.
        3. **Finish** (main process): per-shard summaries, found-constants
           bookkeeping, threshold filtering and ranking.
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

        # cmf_name → {shard_id: [identified constant names]} — populated by
        # _finish_shard, flushed to the shard JSONL after the loop so a constant
        # is recorded as found in a shard only when a trajectory identified it.
        self._found_constants_by_shard: Dict[str, Dict[str, List[str]]] = {}

        # Deduplicate shards — the same Shard object may appear under several
        # constants.  Each unique shard_id is analysed exactly once; ``listing``
        # keeps the (constant, position) order for the per-shard report.
        runs: List[_ShardRun] = []
        run_by_id: Dict[str, _ShardRun] = {}
        listing: List[Tuple[Constant, int, int, _ShardRun]] = []
        for constant, shards in self.cmf_data.items():
            shard_width = int(math.log10(len(shards))) + 1 if shards else 1
            for i, shard in enumerate(shards):
                cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)
                run = run_by_id.get(shard_id)
                if run is None:
                    run = _ShardRun(
                        index=len(runs), shard=shard, cmf_id=cmf_id,
                        shard_id=shard_id, encoding_str=encoding_str,
                        jsonl_path=os.path.join(out_dir, f"{shard_id}.jsonl"),
                    )
                    runs.append(run)
                    run_by_id[shard_id] = run
                listing.append((constant, i, shard_width, run))

        # Phase 1 — plan every shard (sampling + cache reuse) in the main process.
        tasks: List[Tier1Task] = []
        for run in SmartTQDM(runs, desc='Sampling shards', **sys_config.TQDM_CONFIG):
            tasks.extend(self._plan_shard(run))

        # Phase 2 — compute every queued walk on one stage-wide pool; the main
        # process is the single writer.
        if tasks:
            workers = min(total_core_budget(), len(tasks))
            with StagePool([r.shard for r in runs], workers=workers) as pool, \
                    BufferedJsonlWriter() as writer:
                results = pool.imap(_analysis_walk_task(), tasks)
                for task, res in zip(
                    tasks,
                    SmartTQDM(results, total=len(tasks), desc='Analyzing trajectories',
                              **sys_config.TQDM_CONFIG),
                ):
                    self._absorb_result(runs[task.shard_idx], task, res, writer)

        # Phase 3 — per-shard summaries + threshold filtering.
        shard_const_best: Dict[str, Dict[Constant, Optional[float]]] = {
            run.shard_id: self._finish_shard(run) for run in runs
        }
        shard_objects: Dict[str, Shard] = {run.shard_id: run.shard for run in runs}

        self._report(listing, shard_const_best)

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
    # Per-shard phases
    # ------------------------------------------------------------------

    def _plan_shard(self, run: "_ShardRun") -> List[Tier1Task]:
        """Sample *run*'s trajectories, reuse cached rows, return the walks to compute.

        Loads the shard JSONL once.  A trajectory is reused (no walk) only when
        **every** shard constant has a fresh row — matching ``config_fingerprint``
        — that carries the active objective's column; its rows are folded into the
        shard statistics right away.  Every other trajectory becomes one
        :class:`Tier1Task` that walks once and emits one row per shard constant.

        :param run: The shard's mutable analysis state.
        :return: Tier-1 tasks for this shard (possibly empty).
        """
        shard = run.shard
        Logger(
            f"Starting analysis on shard {run.shard_id} "
            f"(cmf={run.cmf_id}, encoding={run.encoding_str})",
            Logger.Levels.debug,
        ).log()
        run.seen = load_seen_trajectories(run.jsonl_path)

        # Use the first constant just to drive the SerialSearcher for pair sampling
        # (trajectory sampling is constant-independent).
        searcher = SerialSearcher(shard, shard.consts[0], use_LIReC=False)
        try:
            pairs = searcher.sample_pairs(
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
                sampling_method=analysis_config.SAMPLING_METHOD,
            )
        except ValueError as e:
            Logger(
                f"Skipping shard {run.shard_id}: {e}",
                Logger.Levels.warning,
            ).log()
            run.skipped = True
            return []

        # Seed the user-supplied trajectory into the analysis set (first) so its δ is
        # computed and recorded alongside the sampled ones.  It is paired with the
        # shard's interior point, exactly like ``sample_pairs``.  Per-run dedup below
        # avoids walking it twice if the sampler happened to draw the same direction.
        if getattr(shard, "selected_trajectory", None) is not None:
            pairs = [(shard.selected_trajectory, shard.get_interior_point())] + list(pairs)

        objective_name = config.system.OPTIMIZATION_OBJECTIVE
        consts = tuple(shard.consts)
        processed_tids: set = set()
        tasks: List[Tier1Task] = []
        for traj, start in pairs:
            start_t = _position_to_tuple(start)
            dir_t = _position_to_tuple(traj)
            tid = derive_trajectory_id(
                run.shard_id, shard.cmf_name, run.encoding_str, start_t, dir_t,
            )
            # Skip a trajectory already handled in this run (e.g. the injected
            # seed also drawn by the sampler) — count + write it only once.
            if tid in processed_tids:
                continue
            processed_tids.add(tid)

            current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, traj))
            bucket = run.seen.get(tid, {})
            # Reuse only when *every* shard constant has a fresh, objective-
            # covered row — else recompute (a changed walk depth / walk type /
            # identification tolerance, or a switched objective column).
            if all(_reusable(bucket.get(c.name), current_fp, objective_name) for c in consts):
                for c in consts:
                    run.accumulate(bucket[c.name], c.name, objective_name)
                run.total += 1
                continue

            tasks.append(Tier1Task(
                key=tid, shard_idx=run.index, trajectory=traj, start=start,
                constants=consts, cmf_id=run.cmf_id, shard_id=run.shard_id,
                shard_encoding_str=run.encoding_str,
            ))
        Logger(
            f"Shard {run.shard_id[:24]}…: {len(processed_tids)} trajectories → "
            f"{len(processed_tids) - len(tasks)} reused · {len(tasks)} new",
            Logger.Levels.info,
        ).log()
        return tasks

    @staticmethod
    def _absorb_result(run: "_ShardRun", task: Tier1Task, res, writer: BufferedJsonlWriter) -> None:
        """Write one computed trajectory's rows and fold them into *run*'s stats.

        :param run: The owning shard's analysis state.
        :param task: The task that produced *res* (for error context).
        :param res: :class:`Tier1Result` or :class:`TaskError`.
        :param writer: The stage's single buffered writer.
        """
        if isinstance(res, TaskError):
            Logger(
                f"Handler error — shard {run.shard_id}, "
                f"traj={task.trajectory}, start={task.start}: {res.message}",
                Logger.Levels.warning,
            ).log()
            return
        objective_name = config.system.OPTIMIZATION_OBJECTIVE
        for c, dto in zip(run.shard.consts, res.dtos):
            line = dto.to_json_line()
            writer.write_line(run.jsonl_path, line)
            record = json.loads(line)
            run.seen.setdefault(task.key, {})[c.name] = record
            run.accumulate(record, c.name, objective_name)
        run.total += 1

    def _finish_shard(self, run: "_ShardRun") -> Dict[Constant, Optional[float]]:
        """Summarise *run* and return ``{Constant: best_objective_score}``.

        Only constants that passed ``IDENTIFY_THRESHOLD`` (and have a finite best
        score) are included, so the caller can distinguish "failed" from
        "constant not in shard".  A skipped shard (sampling failed) returns ``{}``.
        """
        if run.skipped:
            return {}
        shard = run.shard
        total = run.total
        Logger(
            f"Finished analysis on shard {run.shard_id}: {total} trajectories",
            Logger.Levels.debug,
        ).log()

        # User-facing summary: which start point we searched from and how many of the
        # sampled trajectories identified each constant (LIReC).
        start_disp = _position_to_tuple(shard.get_interior_point())
        pct_summary = ", ".join(
            f"{c.name} {100.0 * run.identified_count[c.name] / total:.2f}% "
            f"({run.identified_count[c.name]}/{total})"
            if total else f"{c.name} N/A"
            for c in shard.consts
        )
        Logger(
            f"Shard {run.shard_id} — start point {start_disp}; "
            f"identified trajectories: {pct_summary}",
            Logger.Levels.info,
        ).log()

        # A constant is "found" in this shard iff at least one trajectory identified
        # it (LIReC), independent of the prioritisation threshold — recorded so the
        # shard JSONL only lists genuinely-found constants.
        self._found_constants_by_shard.setdefault(shard.cmf_name, {})[run.shard_id] = [
            c.name for c in shard.consts if run.identified_count[c.name] > 0
        ]

        out: Dict[Constant, Optional[float]] = {}
        for c in shard.consts:
            ident_pct = run.identified_count[c.name] / total if total else 0.0
            if (
                ident_pct >= analysis_config.IDENTIFY_THRESHOLD
                and run.best_score.get(c.name) is not None
            ):
                out[c] = run.best_score[c.name]
        return out

    @staticmethod
    def _report(listing, shard_const_best) -> None:
        """Per-constant, per-shard user report (``PRINT_FOR_EVERY_SEARCHABLE``)."""
        current = None
        for constant, i, shard_width, run in listing:
            if constant is not current:
                current = constant
                Logger(
                    Logger.buffer_print(
                        sys_config.LOGGING_BUFFER_SIZE,
                        f'Analyzing for {constant.name}',
                        '=',
                    ),
                    Logger.Levels.message,
                ).log()
            if not analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                continue
            per_const_best = shard_const_best.get(run.shard_id, {})
            objective_name = config.system.OPTIMIZATION_OBJECTIVE
            for c in run.shard.consts:
                if c in per_const_best:
                    bd = per_const_best[c]
                    bd_str = f'{bd:.4f}' if bd is not None else 'N/A'
                    Logger(
                        f"Shard {i+1:0{shard_width}d} in {run.cmf_id} - searching {c.name}: best {objective_name}={bd_str} [identified: ✅]",
                        Logger.Levels.info,
                    ).log()
                else:
                    Logger(
                        f"Shard {i+1:0{shard_width}d} in {run.cmf_id} - searching {c.name}: {' ':<17} [identified: ❌]",
                        Logger.Levels.info,
                    ).log()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reusable(rec: Optional[dict], fp: str, objective_name: str) -> bool:
    """A cached per-constant row is reusable when its Tier-1 core is fresh under
    ``REUSE_POLICY`` (:func:`tier1_record_fresh`) *and* it carries the active
    objective's column — ``score_record`` returns ``None`` only when that column
    is absent."""
    return tier1_record_fresh(rec, fp) and score_record(rec, objective_name) is not None


@dataclass
class _ShardRun:
    """Mutable per-shard analysis state shared by the plan / compute / finish phases."""

    index: int
    shard: Shard
    cmf_id: str
    shard_id: str
    encoding_str: str
    jsonl_path: str
    seen: dict = field(default_factory=dict)
    total: int = 0
    skipped: bool = False
    identified_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    best_score: Dict[str, Optional[float]] = field(default_factory=dict)

    def accumulate(self, record: dict, const_name: str, objective_name: str) -> None:
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
        self.identified_count[const_name] += 1
        if math.isfinite(sc):
            cur = self.best_score.get(const_name)
            if cur is None or sc > cur:
                self.best_score[const_name] = sc


def _analysis_walk_task():
    """Stage-pool task callable: one trajectory walk → one Tier-1 row per constant.

    A ``functools.partial`` of :func:`run_tier1_task` bound to this module's
    ``TrajectoryAttributesHandler`` / ``build_trajectory_dtos`` (resolved now, so
    test monkeypatches apply).  It pickles as references into the light
    ``stage_pool`` / ``trajectory_attributes`` modules, so a worker unpickling it
    does **not** import this module (and the samplers / numba behind it).
    """
    import functools
    return functools.partial(
        run_tier1_task,
        handler_cls=TrajectoryAttributesHandler,
        build_dtos=build_trajectory_dtos,
    )
