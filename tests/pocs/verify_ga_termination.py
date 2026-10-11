"""Behavioral check: GA termination fix (no GRAD_GRAD_TOL) on 2F1(-1)/log(2).

Drives the real load->extract->analyze pipeline (temp output dir) to obtain the
identified shards of ``pFq(log(2),2,1,-1)``, then runs **only** Gradient Ascent on
each and reports best δ + number of δ-evaluations emitted (the sphere-plot point
count). The point is to confirm GA now climbs past the δ ≈ -1 basin (it used to
quit there via the removed gradient-magnitude stop) and that per-shard counts are
no longer wildly inconsistent.

Run:  python -m tests.pocs.verify_ga_termination
"""

import tempfile
import time
from typing import List

from dreamer import log
from dreamer import extraction, analysis
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.discrete_local_max import (
    orthogonal_neighbours,
    evaluate_neighbours,
)
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.gradient_ascent.grad_ascent_scan import (
    GradientAscentSearch,
    NoInitialIdentification,
    SearchStalled,
)

search_config = config.search
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.system import System
from dreamer.loading import pFq


class _GAChecker(SearcherModScheme):
    """Throw-away searcher: captures real shards and runs GA-only per shard."""

    def __init__(self, priorities, use_LIReC: bool = True):
        super().__init__(priorities, use_LIReC, name="GA-check", description="", version="0")

    def execute(self) -> None:
        for const, shards in self.priorities.items():
            unique: List[Shard] = []
            seen = set()
            for s in shards:
                if id(s) not in seen:
                    seen.add(id(s))
                    unique.append(s)
            print(f"\n### {const.name}: {len(unique)} shard(s) ###")
            print(f"{'shard':<8} {'d_flat':>6} {'seedδ':>9} {'#nbr':>5} {'best-nbrδ':>10} "
                  f"{'GAbestδ':>9} {'#ev':>5} {'secs':>6}")
            print("-" * 70)
            for i, shard in enumerate(unique):
                geom = FlatlandGeometry(shard)
                start = shard.get_interior_point()
                enc = ",".join(str(x) for x in shard.encoding)
                ctx = dict(
                    geom=geom, shard=shard, start=start, constant=const,
                    cmf_id=shard.cmf_name, shard_id=f"shard{i}", shard_encoding_str=enc,
                    sink=lambda _it: None, seen_trajectories={}, handler_cache={},
                )

                # --- Seed + its ±1-neighbourhood (the certificate's reachable set) ---
                seed_delta = nbr_best = float("nan")
                n_nbr = 0
                method = GradientAscentSearch(shard, const, use_LIReC=self.use_LIReC)
                try:
                    seed_z = method._select_seed(geom, ctx, f"shard{i}", const)
                    seed_delta, _ = evaluate_in_flatland(seed_z, **ctx)
                    nbrs = orthogonal_neighbours(
                        seed_z, geom, search_config.SEARCH_MAX_TRAJ_LEN,
                        search_config.SEARCH_TRAJ_NORM,
                    )
                    n_nbr = len(nbrs)
                    nbr_deltas = [d for d, ident in evaluate_neighbours(nbrs, ctx) if ident]
                    nbr_best = max(nbr_deltas) if nbr_deltas else float("nan")
                except NoInitialIdentification:
                    print(f"shard{i:<3} {geom.d_flat:>6}   (no identifying seed)")
                    continue

                # --- Full GA run (fresh state) ---
                emitted = [0]
                method = GradientAscentSearch(shard, const, use_LIReC=self.use_LIReC)
                t0 = time.perf_counter()
                try:
                    method.run(
                        constant=const, cmf_id=shard.cmf_name, shard_id=f"shard{i}",
                        shard_encoding_str=enc,
                        sink=lambda _it: emitted.__setitem__(0, emitted[0] + 1),
                        seen_trajectories={}, handler_cache={}, geom=geom, start=start,
                        pool=None,
                    )
                    best = f"{method.best_delta:.4f}"
                except (NoInitialIdentification, SearchStalled) as exc:
                    best = type(exc).__name__
                secs = time.perf_counter() - t0
                print(f"shard{i:<3} {geom.d_flat:>6} {seed_delta:>9.4f} {n_nbr:>5} "
                      f"{nbr_best:>10.4f} {best:>9} {emitted[0]:>5} {secs:>6.1f}")


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="ga_check_")
    overrides = dict(
        EXPORT_SEARCH_RESULTS=tmp, EXPORT_ANALYSIS_PRIORITIES=tmp,
        EXPORT_CMFS=tmp, EXPORT_ANALYSIS_RESULTS=tmp,
        NUM_BACKGROUND_WORKERS=0, USE_LIReC=True,
    )
    prev = {k: getattr(sys_config, k, None) for k in overrides}
    for k, v in overrides.items():
        setattr(sys_config, k, v)
    print(f"GA termination check on 2F1(-1)/log(2)  (temp dir: {tmp})")
    try:
        System(
            function_sources=[pFq(log(2), 2, 1, -1)],
            extractor=extraction.extractor.ShardExtractorMod,
            analyzers=[analysis.AnalyzerModV1],
            searcher=_GAChecker,
            post_processor=None,
        ).run(constants=[log(2)])
    finally:
        for k, v in prev.items():
            setattr(sys_config, k, v)


if __name__ == "__main__":
    main()
