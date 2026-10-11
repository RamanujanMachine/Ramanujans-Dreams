"""
Measure-first benchmark: is process-based δ-evaluation worth it for the GA?

The Genetic Algorithm evaluates a *batch* of trajectory directions (the new
children) per generation.  Today that batch is walked either serially or via a
``ThreadPoolExecutor`` (GIL-bound, ~1.21× measured).  Before committing to a
process-pool refactor we must answer one question with numbers, not assumption:

    Does a persistent per-shard ``multiprocessing.Pool`` beat serial for the
    GA's δ-evaluation, **once the pickle cost is paid**?

The honest design this script measures (and the one a real refactor would use):

* The shard is handed to the workers **once** at pool creation (via the
  ``initializer``).  On Linux/WSL (fork start method, the production env) this
  is copy-on-write — effectively free — so the per-task wire cost is only the
  ``direction`` sent in and the ``(trajectory_matrix, value_sympy, dto)`` tuple
  sent back (exactly what the real ``sink`` receives).
* Each task does the same Case-C work as ``evaluate_in_flatland``:
  ``TrajectoryAttributesHandler.from_cmf`` → ``build_trajectory_dto`` →
  ``trajectory_matrix()``.

It prints: mean serial walk time, the result-tuple pickle round-trip cost, and
the serial-vs-parallel wall-clock ratio for one batch, plus a go/no-go verdict.

Run (WSL ``rama`` env)::

    python examples/benchmark_ga_parallel_eval.py
"""

import pickle
import time
from multiprocessing import Pool

from dreamer import log
from dreamer.loading import pFq
from dreamer.configs import config
from dreamer.extraction.extractor import ShardExtractorMod
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.utils.constants.constant import Constant
from dreamer.search.methods.flatland.geometry import FlatlandGeometry


# --- batch size: a representative GA child batch (pop ~ 20 + 2*dim) ----------
N_DIRECTIONS = 10


# ---------------------------------------------------------------------------
# Worker side: shard handed in once via the initializer; tasks send a direction
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _init_worker(config_overrides, shard, start, constant):
    """Pool initializer: re-apply config and stash the per-shard context."""
    config.configure(**config_overrides)
    _WORKER_STATE["shard"] = shard
    _WORKER_STATE["start"] = start
    _WORKER_STATE["constant"] = constant


def _walk_one(direction):
    """Case-C walk for one direction; returns the tuple the real sink gets."""
    from dreamer.utils.storage.trajectory_attributes import (
        TrajectoryAttributesHandler,
        build_trajectory_dto,
    )

    shard = _WORKER_STATE["shard"]
    start = _WORKER_STATE["start"]
    constant = _WORKER_STATE["constant"]
    handler = TrajectoryAttributesHandler.from_cmf(
        shard.cmf, direction, start, constant=None, searchable=shard
    )
    dto = build_trajectory_dto(
        handler,
        cmf_id="",
        shard_id="s",
        cmf_name=shard.cmf_name,
        shard_encoding_str="",
        start=start,
        direction=direction,
        constants=[constant],
    )
    return (handler.trajectory_matrix, constant.value_sympy, dto)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _first_shard():
    """Extract one shard from pFq(log(2), 2, 1, -1) for benchmarking."""
    formatter = pFq(log(2), 2, 1, -1)
    cmf_data = formatter.to_cmf()
    const = Constant.get_constant(formatter.consts[0])
    shards_by_const = ShardExtractorMod({const: [cmf_data]}).execute()
    shards = shards_by_const.get(const, [])
    if not shards:
        raise SystemExit("No shards extracted — nothing to benchmark.")
    return shards[0], const


def _distinct_directions(shard, geom, n):
    """Sample ``n`` distinct in-cone primitive directions (real-space Positions)."""
    orch = ShardSamplingOrchestrator(shard)
    raw = list(orch.sample_trajectories(max(n * 4, 20)))
    out, seen = [], set()
    for t in raw:
        z = geom.to_flatland(t)
        if not geom.is_inside(z):
            continue
        direction = geom.to_real_primitive(z)
        key = tuple(int(direction[s]) for s in shard.symbols)
        if key in seen or not any(key):
            continue
        seen.add(key)
        out.append(direction)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _serial(directions, start, constant):
    """Walk every direction in-process; return (seconds, results)."""
    t0 = time.perf_counter()
    results = []
    _init_worker(config.export_configurations(), _SHARD, start, constant)
    for d in directions:
        results.append(_walk_one(d))
    return time.perf_counter() - t0, results


def _parallel(directions, start, constant, workers):
    """Walk every direction via a persistent Pool; return (seconds, pool_setup_s)."""
    overrides = config.export_configurations()
    t_pool0 = time.perf_counter()
    pool = Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(overrides, _SHARD, start, constant),
    )
    pool_setup = time.perf_counter() - t_pool0
    try:
        t0 = time.perf_counter()
        pool.map(_walk_one, directions)
        elapsed = time.perf_counter() - t0
    finally:
        pool.close()
        pool.join()
    return elapsed, pool_setup


def _pickle_roundtrip(obj):
    """Return (bytes, dumps+loads seconds) for one object."""
    t0 = time.perf_counter()
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.loads(blob)
    return len(blob), time.perf_counter() - t0


_SHARD = None  # set in main (must be module-global so fork workers inherit it)


def main():
    """Run the GA parallel-eval feasibility benchmark and print a verdict."""
    global _SHARD
    _SHARD, const = _first_shard()
    geom = FlatlandGeometry(_SHARD)
    start = _SHARD.get_interior_point()
    directions = _distinct_directions(_SHARD, geom, N_DIRECTIONS)
    if len(directions) < 2:
        raise SystemExit("Could not sample enough distinct directions.")

    print(f"Shard {getattr(_SHARD, 'cmf_name', '?')}, {len(directions)} directions\n")

    # 1) Picklability + cost of the per-shard init payload (free under fork, but
    #    a spawn-based env would pay this once per worker).
    try:
        shard_bytes, shard_s = _pickle_roundtrip(_SHARD)
        print(f"shard pickle:  {shard_bytes/1024:8.1f} KiB   {shard_s*1e3:6.1f} ms "
              f"(once per pool; ~free under Linux fork)")
    except Exception as exc:  # noqa: BLE001
        print(f"shard pickle:  FAILED — {exc}\n"
              f"→ process pool infeasible without a picklable shard/cmf.")
        return

    # 2) Serial batch (also gives us a result tuple to weigh).
    serial_s, results = _serial(directions, start, const)
    per_walk = serial_s / len(directions)
    res_bytes, res_s = _pickle_roundtrip(results[0])
    print(f"result pickle: {res_bytes/1024:8.1f} KiB   {res_s*1e3:6.1f} ms "
          f"(per task, sent back from worker)")
    print(f"\nmean serial walk: {per_walk*1e3:7.1f} ms/direction")
    print(f"serial batch:     {serial_s:7.2f} s  ({len(directions)} directions)")

    # 3) Parallel batch with a persistent pool.
    workers = min(len(directions), config.search.GA_NUM_EVAL_WORKERS or 6)
    par_s, pool_setup = _parallel(directions, start, const, workers)
    print(f"pool setup:       {pool_setup:7.2f} s  (once per shard)")
    print(f"parallel batch:   {par_s:7.2f} s  (workers={workers}, excl. setup)")

    # --- Verdict -----------------------------------------------------------
    speedup = serial_s / par_s if par_s > 0 else 0.0
    print(f"\nbatch speed-up (serial / parallel, excl. pool setup): {speedup:.2f}×")
    pickle_frac = res_s / per_walk if per_walk > 0 else float("inf")
    print(f"result-pickle as fraction of one walk: {pickle_frac*100:.1f}%")

    print("\nVerdict:")
    if per_walk < 0.05:
        print("  • Each walk is very cheap — pool/pickle overhead likely dominates; "
              "processes unlikely to pay off.")
    if speedup >= 1.5:
        print("  • Parallel batch is meaningfully faster → process pool is worth it.")
    elif speedup >= 1.15:
        print("  • Modest gain — worth it only if pool setup is amortised across "
              "many generations.")
    else:
        print("  • Little/no gain — keep serial/threads; record finding, skip refactor.")


if __name__ == "__main__":
    main()
