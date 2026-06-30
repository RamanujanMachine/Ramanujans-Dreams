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
configured Tier-2 attributes asynchronously in background workers. It re-uses
records already on disk from analysis, so nothing is walked twice. With an empty
`config.search.TIER2_ATTRIBUTES` it does no extra work beyond Tier-1 and spawns
no subprocesses.

### 2. The direction-space optimisers

Instead of sampling broadly, these *iteratively climb* δ in **flatland** — the
reduced integer direction space of the shard, where a trajectory direction is a
point and δ is the landscape:

| Searcher module | Method | Idea |
|-----------------|--------|------|
| `SmallAngleSearchMod` | small-angle hill-climb | nudge the direction by the smallest resolvable angle. |
| `GeneticSearchModV2` | genetic algorithm | evolve a population of directions. |
| `SimulatedAnnealingMod` | simulated annealing | temperature-controlled random walk over directions. |
| `GradientAscentMod` | gradient ascent (Adam) | follow a finite-difference δ-gradient. |
| `HybridSPSAMod` | SPSA + Adam + discrete fallback | 2-evaluation stochastic gradient, good at high dimension. |

All five share the **flatland** geometry/lattice utilities and end with the same
**discrete micro-hill-climb finalization** (an optional resolution-doubling
endgame, `config.search.ENABLE_MICRO_HILL_CLIMB`) that certifies the reported
best trajectory is a true lattice local maximum. They are reproducible from
`config.search.GLOBAL_SEED`.

### Attribute tiers

- **Tier-1** — always computed (δ, identified, limit, recurrence, p/q).
- **Tier-2** — `config.search.TIER2_ATTRIBUTES`, computed in background workers
  during search; land in each record's `extended_metrics`.
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
