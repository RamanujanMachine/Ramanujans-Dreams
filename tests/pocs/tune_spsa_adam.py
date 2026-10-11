"""Hyperparameter tuning + baseline comparison for the Hybrid SPSA + Adam search.

Runs the *real* pipeline (load -> extract -> analyze) on a small CMF — by default
2F1(-1) searching for log(2), i.e. ``pFq(log(2), 2, 1, -1)`` — to obtain genuinely
identified shards, then for each shard:

  * runs plain Gradient Ascent (the existing method) as a **baseline**, and
  * sweeps a small grid of SPSA hyperparameters (c0 / learning rate / gamma),

measuring, per (method, config, shard): best δ reached, number of δ-evaluations
emitted (a cost proxy — each is one symbolic walk in the worst case), wall-clock
seconds, and (for SPSA) whether the discrete fallback fired and after how many
macro steps.

The comparison is captured inside a throw-away "searcher" handed to ``System`` so
the full extraction/analysis machinery produces the shards; nothing is written to
the production result dirs (everything is redirected to a temp folder).

Run:  python -m tests.pocs.tune_spsa_adam
"""

import itertools
import statistics
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import sympy as sp

from dreamer import log, pi, e  # noqa: F401  (constants available for variants)
from dreamer import extraction, analysis
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.gradient_ascent.grad_ascent_scan import (
    GradientAscentSearch,
    NoInitialIdentification as GradNoInit,
    SearchStalled,
)
from dreamer.search.methods.gradient_ascent.spsa_adam_ascent import (
    HybridSPSASearch,
    NoInitialIdentification as SPSANoInit,
)
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.system import System
from dreamer.loading import pFq


# --------------------------------------------------------------------------
# Configuration of the experiment
# --------------------------------------------------------------------------

#: Cap on shards swept (keeps the run to minutes; raise for a fuller picture).
MAX_SHARDS = 4

#: SPSA macro-step budget during tuning (smaller than production for speed).
TUNE_SPSA_MAX_STEPS = 30

#: Hyperparameter grid.  Kept small and one-factor-ish around the defaults.
GRID_C0 = [0.1, 0.2, 0.4]
GRID_LR = [0.3, 0.5, 1.0]
GRID_GAMMA = [0.101]  # decay exponent; 0.0 => constant c_k (add to explore)


def _grid() -> List[Dict[str, float]]:
    return [
        {"SPSA_C0": c0, "SPSA_LR": lr, "SPSA_GAMMA": g}
        for c0, lr, g in itertools.product(GRID_C0, GRID_LR, GRID_GAMMA)
    ]


# --------------------------------------------------------------------------
# Per-run measurement
# --------------------------------------------------------------------------

def _measure(method, *, constant: Constant, shard: Shard, geom, start) -> Tuple:
    """Run one search method on one shard with fresh caches; return metrics.

    :return: ``(best_delta, n_emitted, secs, fell_back, macro_steps)`` or ``None``
        if the shard could not be seeded (no identifying reservoir trajectory).
    """
    emitted = [0]

    def sink(_item):
        emitted[0] += 1

    seen: dict = {}
    handler_cache: dict = {}
    cmf_id = getattr(shard, "cmf_name", "cmf")
    shard_enc = ",".join(str(x) for x in shard.encoding)

    t0 = time.perf_counter()
    try:
        method.run(
            constant=constant,
            cmf_id=cmf_id,
            shard_id=cmf_id,
            shard_encoding_str=shard_enc,
            sink=sink,
            seen_trajectories=seen,
            handler_cache=handler_cache,
            geom=geom,
            start=start,
            pool=None,
        )
    except (GradNoInit, SPSANoInit):
        return None
    except SearchStalled:
        # Gradient ascent abandoned the shard — record as a failed (−inf) run.
        secs = time.perf_counter() - t0
        return (float("-inf"), emitted[0], secs, True, 0)

    secs = time.perf_counter() - t0
    fell_back = bool(getattr(method, "used_discrete_fallback", False))
    macro = int(getattr(method, "macro_steps", 0))
    return (float(method.best_delta), emitted[0], secs, fell_back, macro)


class _TuningSearcher(SearcherModScheme):
    """Throw-away searcher: captures real shards and runs the comparison sweep."""

    def __init__(self, priorities, use_LIReC: bool = True):
        super().__init__(priorities, use_LIReC, name="SPSA-Tuner",
                         description="tuning harness", version="0")

    def execute(self) -> None:
        for const, shards in self.priorities.items():
            unique: List[Shard] = []
            seen_ids = set()
            for s in shards:
                if id(s) not in seen_ids:
                    seen_ids.add(id(s))
                    unique.append(s)
            shards = unique[:MAX_SHARDS]
            print(f"\n############ Constant {const.name}: "
                  f"{len(shards)} shard(s) (of {len(unique)}) ############")
            self._sweep_constant(const, shards)

    def _sweep_constant(self, const: Constant, shards: List[Shard]) -> None:
        # rows: config-label -> list of (best_delta, n_emitted, secs, fell_back, macro)
        results: Dict[str, List[Tuple]] = {}

        for si, shard in enumerate(shards):
            geom = FlatlandGeometry(shard)
            start = shard.get_interior_point()

            # --- Baseline: plain gradient ascent ---
            grad = GradientAscentSearch(shard, const, use_LIReC=self.use_LIReC)
            base = _measure(grad, constant=const, shard=shard, geom=geom, start=start)
            results.setdefault("BASELINE grad-ascent", []).append(base)

            # --- SPSA grid ---
            old_max = config.search.SPSA_MAX_STEPS
            config.search.SPSA_MAX_STEPS = TUNE_SPSA_MAX_STEPS
            try:
                for combo in _grid():
                    label = (f"SPSA c0={combo['SPSA_C0']} lr={combo['SPSA_LR']} "
                             f"g={combo['SPSA_GAMMA']}")
                    prev = {k: getattr(config.search, k) for k in combo}
                    for k, v in combo.items():
                        setattr(config.search, k, v)
                    try:
                        m = HybridSPSASearch(shard, const, use_LIReC=self.use_LIReC)
                        res = _measure(m, constant=const, shard=shard, geom=geom, start=start)
                    finally:
                        for k, v in prev.items():
                            setattr(config.search, k, v)
                    results.setdefault(label, []).append(res)
            finally:
                config.search.SPSA_MAX_STEPS = old_max

            print(f"  shard {si} swept.")

        self._report(const, results)

    @staticmethod
    def _report(const: Constant, results: Dict[str, List[Tuple]]) -> None:
        print(f"\n=== RESULTS for {const.name} "
              f"(mean over shards; None = unseeded shard skipped) ===")
        header = f"{'config':<34} {'mean δ':>10} {'mean #ev':>9} {'mean s':>8} {'fallback%':>9}"
        print(header)
        print("-" * len(header))

        def agg(rows: List[Tuple]):
            real = [r for r in rows if r is not None]
            if not real:
                return None
            deltas = [r[0] for r in real if r[0] != float("-inf")]
            mean_delta = statistics.mean(deltas) if deltas else float("-inf")
            mean_ev = statistics.mean(r[1] for r in real)
            mean_s = statistics.mean(r[2] for r in real)
            fb = 100.0 * sum(1 for r in real if r[3]) / len(real)
            return mean_delta, mean_ev, mean_s, fb

        # Baseline first, then SPSA configs sorted by mean δ desc.
        ordered = sorted(
            results.items(),
            key=lambda kv: (kv[0] != "BASELINE grad-ascent",
                            -(agg(kv[1])[0] if agg(kv[1]) else -1e9)),
        )
        for label, rows in ordered:
            a = agg(rows)
            if a is None:
                print(f"{label:<34} {'(no seeded shard)':>40}")
                continue
            md, mev, ms, fb = a
            print(f"{label:<34} {md:>10.4f} {mev:>9.1f} {ms:>8.2f} {fb:>8.0f}%")


def main(function_source=None, constant: Optional[Constant] = None) -> None:
    """Run the tuning experiment.

    :param function_source: A CMF / Formatter to search (default ``pFq(log(2),2,1,-1)``).
    :param constant: The constant to search for (default ``log(2)``).
    """
    if constant is None:
        constant = log(2)
    if function_source is None:
        function_source = pFq(constant, 2, 1, -1)

    tmp = tempfile.mkdtemp(prefix="spsa_tune_")
    overrides = dict(
        EXPORT_SEARCH_RESULTS=tmp,
        EXPORT_ANALYSIS_PRIORITIES=tmp,
        EXPORT_CMFS=tmp,
        EXPORT_ANALYSIS_RESULTS=tmp,
        NUM_BACKGROUND_WORKERS=0,
        USE_LIReC=True,
    )
    prev = {k: getattr(sys_config, k, None) for k in overrides}
    for k, v in overrides.items():
        setattr(sys_config, k, v)

    print(f"Tuning Hybrid SPSA on {function_source.cmf_name} for {constant.name}.")
    print(f"(temp output dir: {tmp})")
    try:
        System(
            function_sources=[function_source],
            extractor=extraction.extractor.ShardExtractorMod,
            analyzers=[analysis.AnalyzerModV1],
            searcher=_TuningSearcher,
            post_processor=None,
        ).run(constants=[constant])
    finally:
        for k, v in prev.items():
            setattr(sys_config, k, v)


if __name__ == "__main__":
    main()
