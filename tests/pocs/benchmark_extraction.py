"""
Isolated benchmark for the four shard-extraction strategies on a real
CMF, with no Dreams pipeline overhead.

Run via::

    conda activate rama
    python examples/benchmark_extraction.py

For each strategy in ``["legacy", "auto", "exact", "heuristic"]`` we

1. build the CMF + hyperplanes (done once, outside the timing loop)
2. set ``extraction_config.STRATEGY = strategy``
3. call :meth:`ShardExtractor.extract` and time it
4. report runtime + shard count + the set of sign-encodings discovered

The default CMF below matches the one used in
``examples/main_example.py`` (pFq(log(2), p=2, q=1, z=-1)) so results
are directly comparable.  Edit ``build_cmf_data()`` to benchmark a
different system.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# Make sure repo root is importable when running the script directly.
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sympy as sp

from dreamer import config, log, calegary, zeta
from dreamer.configs import extraction_config
from dreamer.extraction.extractor import ShardExtractor
from dreamer.extraction.v2 import lrs_io
from dreamer.loading import pFq
from dreamer.utils.types import CMFData


@dataclass
class BenchResult:
    """Holds the outcome of one strategy run."""

    strategy: str
    seconds: float
    num_shards: int
    encodings: Set[Tuple[int, ...]]
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"  {self.strategy:<10} FAILED: {self.error}"
        return (
            f"  {self.strategy:<10} {self.seconds:>8.3f}s  "
            f"shards={self.num_shards}"
        )


def build_cmf_data() -> Tuple[CMFData, sp.Expr]:
    """
    Construct the CMF that the benchmark runs against.

    Default: pFq(log(2), p=2, q=1, z=-1) -- the same one main_example
    uses.  Edit here to benchmark something else (e.g. larger p/q for
    higher-dimensional arrangements).
    """
    # constant = log(2)
    # formatter = pFq(constant, 2, 1, -1)
    # constant = calegary
    # formatter = pFq(constant, 3, 2, sp.Rational(1, 4), shifts=5 * [sp.Rational(1, 2)])
    # constant = calegary
    # formatter = pFq(constant, 3, 2, 1)
    constant = zeta(2)
    formatter = pFq(constant, 6, 5, 1)
    return formatter.to_cmf(), constant


def run_strategy(strategy: str, cmf_data: CMFData, constant) -> BenchResult:
    """Set the config, run :class:`ShardExtractor`, time it."""
    extraction_config.STRATEGY = strategy
    extractor = ShardExtractor(constant, cmf_data)
    t0 = time.perf_counter()
    try:
        shards = extractor.extract()
    except Exception as exc:  # noqa: BLE001 - report any failure
        elapsed = time.perf_counter() - t0
        return BenchResult(
            strategy=strategy,
            seconds=elapsed,
            num_shards=0,
            encodings=set(),
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - t0
    encodings = {tuple(s.encoding) for s in shards if s.encoding}
    return BenchResult(
        strategy=strategy,
        seconds=elapsed,
        num_shards=len(shards),
        encodings=encodings,
    )


def main() -> int:
    # Minimal config -- we don't run the pipeline, so most knobs don't
    # matter, but a couple touch the extractor directly.
    config.configure(
        extraction={
            'INIT_POINT_MAX_COORD': 2,
            'IGNORE_DUPLICATE_SEARCHABLES': True,
            # Under 'auto': exact gets EXACT_TIMEOUT_SECONDS, then heuristic
            # gets HEURISTIC_TIMEOUT_SECONDS independently.
            'EXACT_TIMEOUT_SECONDS': 60,
            'HEURISTIC_TIMEOUT_SECONDS': 100,
            # Good-Turing stop: cease a phase when <HEURISTIC_MISSING_MASS
            # fraction of samples would land in a new cell.
            'HEURISTIC_MISSING_MASS': 1e-4,    # lower => more coverage / longer
            # 'HEURISTIC_NUM_RAYS': None,      # None (default) = unlimited
            # Face-aligned phase: reach tube/slab cells generic rays miss.
            'HEURISTIC_FACE_ALIGNED': True,
        },
        logging={'GENERATE_LOGS': False},
    )

    cmf_data, constant = build_cmf_data()
    print(f"CMF: {cmf_data.cmf_name} (dim={cmf_data.cmf.dim()})")
    print(f"lrs binary available: {lrs_io.lrs_available()}")
    print()

    strategies: List[str] = ["legacy", "heuristic"] #, "exact"]
    results: List[BenchResult] = []
    for strategy in strategies:
        print(f"-> running strategy={strategy!r} ...", flush=True)
        try:
            results.append(run_strategy(strategy, cmf_data, constant))
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001
            print(traceback.format_exc())
            results.append(BenchResult(strategy, 0.0, 0, set(),
                                       error="harness error (see traceback)"))

    print()
    print("Results")
    print("-------")
    for r in results:
        print(r.summary())

    # Compare encoding sets across strategies that succeeded.
    ok = [r for r in results if r.error is None]
    if len(ok) >= 2:
        print()
        print("Encoding overlap (Jaccard) vs first successful strategy:")
        base = ok[0]
        for r in ok[1:]:
            if not base.encodings and not r.encodings:
                j = 1.0
            elif not base.encodings or not r.encodings:
                j = 0.0
            else:
                inter = len(base.encodings & r.encodings)
                union = len(base.encodings | r.encodings)
                j = inter / union
            print(f"  {base.strategy} vs {r.strategy:<10} = {j:.3f} "
                  f"(|A|={len(base.encodings)}, |B|={len(r.encodings)}, "
                  f"|A & B|={len(base.encodings & r.encodings)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
