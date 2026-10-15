"""
A simulated run for trying the monitor without a real workload.

It has the shape of a Ramanujan's Dreams run — loading → extraction →
analysis → search (simulated annealing per shard) → postprocess — and uses
the real machinery: a pool of worker processes burning CPU, worker event
files, a hang watchdog that fires in a worker, structured events (pool
created / shrunk, a "worker died"), simulated-annealing progress from
:func:`runmon.estimators.sa_estimator`, and a short fake walltime so the
countdown, warnings and grace-period flush are visible.

    python -m runmon demo            # run it and watch it in this terminal
    python -m runmon demo --no-watch --dir /tmp/demo   # just run (watch elsewhere)
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import runmon
from runmon.estimators import sa_estimator, schedule_temperature


def _burn(seconds: float) -> float:
    end, x = time.perf_counter() + seconds, 0.0
    while time.perf_counter() < end:
        x += math.sqrt(x + 1.0)
    return x


def _worker_init(run_dir: str) -> None:
    runmon.init_worker(run_dir)


def _trajectory(args):
    """A fake per-trajectory computation in a worker (some fail, one hangs a bit)."""
    shard, i, cost, hang = args
    if hang:
        with runmon.watch(f"trajectory {shard}:{i}", hang, detail=f"shard {shard}, trajectory {i}"):
            _burn(hang * 2.5)                 # overruns: the worker's watchdog fires twice
    else:
        _burn(cost)
    if shard < 10 and random.Random(shard * 1000 + i).random() < 0.05:
        runmon.event("task_failed", f"trajectory {shard}:{i} raised ZeroDivisionError (simulated)",
                     "WARNING", shard=shard, trajectory=i)
        return None
    return random.Random(i).uniform(-0.8, 0.1)


class _Pool:
    """Process pool exposing ``monitor_info()`` like the project's pools do."""

    def __init__(self, name, workers, run_dir):
        self.name, self.requested, self.workers = name, workers, workers
        self.completed = self.deaths = self.shrinks = 0
        self.in_flight = 0
        self._ex = ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                       initializer=_worker_init, initargs=(run_dir,))
        self._ex.submit(_burn, 0.0).result()   # start the workers
        runmon.register_pool(self)
        runmon.event("pool_created", f"{name} pool with {workers} workers", name=name, workers=workers)

    def monitor_info(self):
        return {"name": self.name, "workers": self.workers, "requested": self.requested,
                "in_flight": self.in_flight, "completed": self.completed, "deaths": self.deaths,
                "shrinks": self.shrinks, "grows": 0, "pids": list(getattr(self._ex, "_processes", {}) or {})}

    def map(self, fn, items):
        items = list(items)
        self.in_flight = len(items)
        for r in self._ex.map(fn, items):
            self.completed += 1
            self.in_flight -= 1
            yield r

    def simulate_death(self):
        self.deaths += 1
        self.shrinks += 1
        self.workers = max(1, self.workers // 2)
        runmon.event("worker_died", f"[{self.name}] a worker process died (simulated OOM); restarting with "
                     f"{self.workers} worker(s) and resubmitting 3 unfinished task(s)", "WARNING",
                     pool=self.name, workers=self.workers)

    def close(self):
        self._ex.shutdown()
        runmon.event("pool_closed", f"{self.name} pool closed", name=self.name)


def run(run_dir: str, speed: float = 1.0, workers: int = 4) -> None:
    """Run the simulation (``speed`` > 1 is faster)."""
    unit = 1.0 / speed
    # A short fake walltime: its warnings and grace period land in the search stage.
    os.environ["RUNMON_WALLTIME_END"] = str(time.time() + 110 * unit)
    runmon.start(run_dir, run_id=f"demo-{int(time.time())}", cores=workers, memory_mb=4096,
                 walltime_warn_s=(60 * unit, 40 * unit, 20 * unit), walltime_grace_s=10 * unit,
                 walltime_check_s=0.5,
                 stall_s=60, meta={"demo": True, "constant": "zeta(3)"})
    rng = random.Random(7)
    try:
        with runmon.task("loading", total=2, kind="stage") as st:
            for name in ("pFq(3,2,-1)", "MeijerG(5,5)"):
                time.sleep(1.5 * unit)
                runmon.event("cmf_loaded", f"loaded {name}", cmf=name)
                st.advance()

        shards = [f"shard {i}" for i in range(6)]
        with runmon.task("extraction", total=len(shards), kind="stage") as st:
            for sh in shards:
                with runmon.task(sh, total=20, kind="shard") as node:
                    for _ in range(20):
                        time.sleep(0.12 * unit)
                        node.advance()
                    node.metric(hyperplanes=rng.randint(30, 60))
                st.advance()
            runmon.event("extraction_done", f"{len(shards)} shards found", shards=len(shards))

        pool = _Pool("stage", workers, run_dir)
        with runmon.task("analysis", total=4, kind="stage") as st:
            for s in range(4):
                n = 24
                with runmon.task(f"shard {s}", total=n, kind="shard") as node:
                    found = 0
                    tasks = [(s, i, 0.6 * unit, 2.0 * unit if (s, i) == (1, 7) else 0) for i in range(n)]
                    for r in pool.map(_trajectory, tasks):
                        node.advance()
                        if r is not None and r > -0.2:
                            found += 1
                        node.metric(identified=found)
                st.advance()

        with runmon.task("search", total=3, kind="stage") as st:
            with runmon.task("annealing", total=3, kind="method") as method:
                for s in range(3):
                    T0, Tmin, max_iters, max_steps = 1.0, 1e-4, 50, 400
                    with runmon.task(f"shard {s} / zeta(3)", kind="shard") as node:
                        k = steps = 0
                        best = -1.0
                        while k < max_iters and steps < max_steps:
                            steps += 1
                            vals = [v for v in pool.map(_trajectory, [(10 + s, steps * 10 + j, 0.2 * unit, 0)
                                                                      for j in range(workers)]) if v is not None]
                            if vals and (max(vals) > best or rng.random() < 0.3):
                                k += 1
                                best = max(best, max(vals))
                            T = schedule_temperature(k, T0, "log")
                            est = sa_estimator(k, steps, T0=T0, Tmin=Tmin, max_iters=max_iters,
                                               max_total_steps=max_steps, schedule="log")
                            node.estimate(est.fraction(), T=round(T, 4), best_delta=round(best, 4),
                                          accepted=k, clock=est.dominant())
                            if s == 1 and steps == 20:
                                pool.simulate_death()
                    method.advance()
                    st.advance()

        with runmon.task("postprocess", total=30, kind="stage") as st:
            for _ in pool.map(_trajectory, [(99, i, 0.5 * unit, 0) for i in range(30)]):
                st.advance()
        pool.close()
        runmon.event("run_summary", "best delta 0.0168 (simulated)", best=0.0168)
    finally:
        runmon.stop()
