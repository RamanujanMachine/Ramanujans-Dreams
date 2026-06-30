"""
Micro-benchmark: does parallel neighbour evaluation actually speed up search?

The Simulated Annealing / Genetic search methods evaluate a *batch* of
neighbours per step via a :class:`concurrent.futures.ThreadPoolExecutor`
(``ANNEAL_NUM_EVAL_WORKERS`` / ``GA_NUM_EVAL_WORKERS``).  Because the heavy
per-neighbour cost is symbolic (sympy walk) + numeric (mpmath) + LIReC
identification — much of which holds the CPython GIL — threading only helps if
that work releases the GIL.  This script measures the wall-clock of one
shard's SA climb at ``workers = 1`` vs ``workers = N`` so the speed-up (if any)
is observed rather than assumed.

Run (WSL ``rama`` env)::

    python examples/benchmark_search_parallelism.py

It is intentionally small (few iterations / one shard) so it finishes quickly;
treat the ratio, not the absolute time, as the signal.
"""

import time

from dreamer import log
from dreamer.loading import pFq
from dreamer.configs import config
from dreamer.extraction.extractor import ShardExtractorMod
from dreamer.utils.constants.constant import Constant
from dreamer.search.methods.annealing.annealing_scan import SimulatedAnnealingSearch


def _first_shard():
    """Extract one shard from the pFq(log(2), 2, 1, -1) CMF for benchmarking."""
    formatter = pFq(log(2), 2, 1, -1)
    cmf_data = formatter.to_cmf()
    const = Constant.get_constant(formatter.consts[0])
    shards_by_const = ShardExtractorMod({const: [cmf_data]}).execute()
    shards = shards_by_const.get(const, [])
    if not shards:
        raise SystemExit("No shards extracted — nothing to benchmark.")
    return shards[0], const


def _time_climb(shard, const, workers: int) -> float:
    """Run one SA climb on *shard* with *workers* eval threads; return seconds."""
    config.search.ANNEAL_NUM_EVAL_WORKERS = workers
    # Keep the run short and deterministic-ish.
    config.search.ANNEAL_MAX_ITERS = 20
    config.search.ANNEAL_MAX_TOTAL_STEPS = 200

    method = SimulatedAnnealingSearch(shard, const, use_LIReC=True)
    t0 = time.perf_counter()
    method.search()
    return time.perf_counter() - t0


def main():
    """Benchmark Simulated Annealing search wall-time across worker counts."""
    shard, const = _first_shard()
    print(f"Benchmarking SA on shard {getattr(shard, 'cmf_name', '?')} …\n")

    results = {}
    for workers in (1, config.search.ANNEAL_NUM_EVAL_WORKERS or 6):
        # Two runs each; report the faster (warm caches / JIT noise).
        best = min(_time_climb(shard, const, workers) for _ in range(2))
        results[workers] = best
        print(f"  workers={workers:>2}:  {best:6.2f} s")

    ws = sorted(results)
    if len(ws) == 2 and results[ws[0]] > 0:
        speedup = results[ws[0]] / results[ws[1]]
        print(f"\nSpeed-up ({ws[0]}→{ws[1]} workers): {speedup:.2f}×")
        if speedup < 1.15:
            print("→ Threading gives little/no benefit here (GIL-bound work). "
                  "Consider process-based parallelism or leaving workers=1.")


if __name__ == "__main__":
    main()
