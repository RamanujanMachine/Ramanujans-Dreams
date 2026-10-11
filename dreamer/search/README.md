# Search stage — `dreamer/search/`

> **Goal:** search the prioritised shards more deeply for high-δ trajectories,
> recording rich per-trajectory attributes.
> **Input:** `Dict[Constant, List[Shard]]` (the ranked analysis priorities).
> **Output:** per-shard `<shard_id>.jsonl` trajectory records (Tier-1 + optional
> Tier-2 attributes).

For where this stage sits in the whole pipeline, see the
[main README](../../README.md#system-structure).

---

## What this stage does

Search is where the budget goes. For each shard that survived analysis, it looks
much harder for trajectories with a high irrationality measure **δ**, and stores
not just Tier-1 values but optionally heavier **Tier-2** attributes
(eigenvalues, spectral gap, convergence rate, …).

There are **two families** of searcher, wired interchangeably via
`System(searcher=...)`:

### 1. The default "hedgehog" searcher — `SearcherModV1`

Re-samples many trajectory directions in the shard (same samplers as analysis,
but with the search config), computes Tier-1 for each, and computes any
configured Tier-2 attributes in the same task as each trajectory's walk. It
re-uses records already on disk (from analysis or earlier runs): a stored row
gets only its missing / out-of-date attributes, computed from the stored values
without re-identifying, so nothing is walked or identified twice. All of it runs
on one **stage-wide process pool** created once for all shards (every core,
capped by `config.system.STAGE_POOL_WORKERS`); `config.system.REUSE_POLICY`
controls how stored values are trusted.

### 2. The direction-space optimisers

Instead of sampling broadly, these *iteratively climb* δ in **flatland** — the
reduced integer direction space of the shard, where a trajectory direction is a
point and δ is the landscape:

| Searcher module | Method | Idea |
|-----------------|--------|------|
| `SmallAngleSearchMod` | small-angle hill-climb | nudge the direction by the smallest resolvable angle. |
| `GeneticSearchModV2` | genetic algorithm | evolve a population of directions. |
| `SimulatedAnnealingMod` | simulated annealing | temperature-controlled random walk over directions; idle eval workers pre-walk the next step's candidates (`ANNEAL_LOOKAHEAD`) without changing the search path. |
| `GradientAscentMod` | gradient ascent (Adam) | follow a finite-difference δ-gradient. |
| `HybridSPSAMod` | SPSA + Adam + discrete fallback | 2-evaluation stochastic gradient, good at high dimension. |

All five share the **flatland** geometry/lattice utilities and end with the same
**discrete micro-hill-climb finalization** (an optional resolution-doubling
endgame, `config.search.ENABLE_MICRO_HILL_CLIMB`) that certifies the reported
best trajectory is a true lattice local maximum. They are reproducible from
`config.search.GLOBAL_SEED`.

### Process pools: stage pool vs eval pool

Every process pool in the package is a `ResilientPool` (a dead worker does not
hang the run; worker counts follow cores and memory). Two kinds do the
trajectory work, and they differ in **what one task is** and **who waits for it**:

| | **Stage pool** (`StagePool`) | **Eval pool** (`make_eval_pool`) |
|---|---|---|
| Used by | analysis, the hedgehog searcher, Tier-3 post-process | the five optimisers above and the micro-climb |
| Lifetime | one per stage, for **all** shards | one per shard, for all constants of that shard |
| Workers hold | every CMF (each task carries a light copy of its shard) | this shard and its start point |
| One task | one trajectory (Tier-1 rows + Tier-2/3 for its constants) | one candidate direction of an optimiser step |
| Scheduling | the whole stage's trajectories stream through it | a batch per step; the optimiser waits for the batch (SA may pre-walk the next step) |

The writer that appends rows to `<shard_id>.jsonl` always runs in the main
process (`BufferedJsonlWriter` for stage pools, `worker_pool(...)` for the
optimisers — despite its name, a plain in-process writer).

### Reuse counts

Every searcher logs, per shard (and per constant for the optimisers), how its
trajectories were served — the same vocabulary as analysis:

```
Shard pFq_2_1_…: 120 trajectories → 80 reused · 10 Tier-2 only · 30 new
Simulated Annealing — shard pFq_2_1_…, log-2: 412 evaluations → 150 reused · 190 revisited · 72 new
```

`reused` = served from a stored row of an earlier run; `revisited` = computed
earlier in this run (an optimiser came back to the same ray); `new` = walked now
(`replacing stale rows` when a stored row's configuration fingerprint no longer
matches); also `on a cached walk`, `failed`, `outside the cone` when non-zero.

### Attribute tiers

- **Tier-1** — always computed (δ, identified, limit, recurrence, p/q).
- **Tier-2** — `config.search.TIER2_ATTRIBUTES`, computed during search together with
  Tier-1 (or from a stored row's Tier-1 values), on every core.
- **Tier-3** — the most expensive attributes, computed afterwards by the
  optional [post-process stage](../post_process/README.md).

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `searchers/` | The **stage modules** (one per method) — the external interface wired into `System`: `hedgehog_scan_mod.py`, `small_angle_mod.py`, `genetic_search_mod.py`, `annealing_mod.py`, `gradient_ascent_mod.py`, `spsa_adam_mod.py`, plus the shared `micro_climb_finalize.py`. |
| `methods/` | The **algorithms** themselves. `hedgehog_scan.py` (the `SerialSearcher` reused by analysis), and one sub-package per optimiser: `small_angle/`, `genetic_search/`, `annealing/`, `gradient_ascent/`. |
| `methods/flatland/` | Shared direction-space machinery: `geometry.py`, `lattice.py`, `seed.py`, `discrete_local_max.py`, `evaluator.py`, `parallel_eval.py`. |
| `errors.py` | Search-specific exceptions (e.g. `SearchStalled`). |

> Note: `methods/genetic.py` + `searchers/genetic_mod.py` are a legacy (V1)
> genetic searcher kept for reference; the pipeline uses `GeneticSearchModV2`.

---

## Key configurations

`config.search.*` is the largest config category: the shared δ-evaluation knobs
(`NUM_TRAJECTORIES_FROM_DIM`, `DEPTH_FROM_TRAJECTORY_LEN`, `SEARCH_MAX_TRAJ_LEN`,
`SAMPLING_METHOD`, `GLOBAL_SEED`, `TIER2_ATTRIBUTES`, `ENABLE_MICRO_HILL_CLIMB`)
plus per-method blocks (`GA_*`, `SA_*` small-angle, `ANNEAL_*`, `GRAD_*`,
`SPSA_*`). See the [configuration index](../configs/README.md#search).

---

## Extending this stage

Adding a search method is **two classes** (kept in separate files):

1. An **algorithm class** — picks which trajectory directions to evaluate inside
   a shard and emits a `TrajectoryDTO` per trajectory. Put it in
   `methods/<your_method>/<your_method>_scan.py`.
2. A **module** — subclass `SearcherModScheme`, implement `execute()` (returns
   `None`): dedup shards, open the per-shard JSONL `worker_pool`, and drive the
   algorithm. Put it in `searchers/<your_method>_mod.py`.

The default `SearcherModV1` and the optimiser mods show this JSONL/DTO pattern;
`examples/search.py` is a minimal copy-paste skeleton of both classes.

Wire it as `System(searcher=MySearchMod)`. Start from the ready-to-copy skeleton
in [`examples/search.py`](../../examples/search.py); the `*_scan.py` / `*_mod.py`
pairs already here are the production-grade reference implementations.
