# Extraction stage — `dreamer/extraction/`

> **Goal:** cut each CMF's integer lattice into bounded convex cells called
> **shards**, and provide the machinery to **sample integer trajectory
> directions** inside a shard.
> **Input:** `Dict[Constant, List[CMFData]]` (from loading).
> **Output:** `Dict[Constant, List[Shard]]`.

For where this stage sits in the whole pipeline, see the
[main README](../../README.md#system-structure).

---

## What this stage does

A CMF defines a family of **trajectories** (straight integer directions through
its lattice), and each trajectory yields one candidate formula. But the CMF's
matrices have singularities — **hyperplanes** where entries blow up — that carve
the lattice into regions. Inside one region every trajectory behaves
consistently; across a hyperplane the behaviour changes. Those regions are the
**shards**.

So extraction does two things:

1. **Find the hyperplanes** (`hyperplanes.py`): the canonical set of singular
   hyperplanes of the CMF (zeros of the matrix determinant + poles of entries),
   restricted to those that actually touch the integer lattice, and **sorted
   deterministically** so that a shard's sign pattern over the hyperplanes means
   the same thing across runs.

2. **Locate the shards**: each shard is a convex cell `A·x < b` (an intersection
   of hyperplane half-spaces) together with an **interior integer point** to
   search from. A CMF can produce many shards; symmetry can be exploited to
   avoid extracting equivalent ones twice.

Once a shard is known, this stage also knows how to **sample trajectory
directions** inside it — the recession cone of valid integer directions — which
the analysis and search stages consume.

### Extraction strategies

`config.extraction.STRATEGY` selects how shards are found:

- **`exact`** — `lrs` reverse-search enumeration + MILP witnesses (authoritative,
  but can be infeasible in high dimension).
- **`heuristic`** — ray-shooting from the origin (+ an optional face-aligned
  phase for thin/unbounded cells). Scales to high-dimensional CMFs.
- **`auto`** (default) — try `exact` within a time budget, fall back to
  `heuristic` on timeout.
- **`legacy`** — brute-force lattice scan (fallback only).

### Trajectory direction constraints (optional)

`config.extraction.TRAJECTORY_CONSTRAINTS` (default `None`) pins the **direction** of
every sampled trajectory, e.g. `{"x0": 12, "y1": 28}` means "always take 12 steps in
`x0` for every 28 in `y1`". Keys are CMF variable names; values are integers.

Because downstream trajectories may be length-scaled, only the **scale-invariant** part
is enforced: the reduced **ratio** between the fixed coordinates (`12:28 = 3:7`) and their
**sign**. So the constraint is a *homogeneous sub-cone* of the shard's recession cone,
which means:

- **Shards are filtered** — a shard whose cone admits no such direction is dropped at
  extraction (you only ever search shards that can contain a constrained trajectory).
- **Every sampler honours it** (folded into the flatland conditioner), and **search stays
  on the sub-cone** too — so all stages respect the constraint.
- A **single** fixed coordinate degenerates to a sign constraint (`x0 > 0`); you need
  **≥ 2** fixed coordinates for a real ratio. A fixed value of `0` means that coordinate
  is exactly `0`. The literal numbers survive only through their ratio/sign (a harvested
  direction for `{x0:12,y1:28}` may print as `(3,…,7)`).

### Trajectory samplers

The trajectory-direction samplers live in this package (`samplers/`,
`sampling_orchestrators/`), but they are **driven by the analysis and search
stages**, not by extraction itself — extraction only locates the shards. The
engine is chosen per stage by `config.analysis.SAMPLING_METHOD` /
`config.search.SAMPLING_METHOD` (there is no `extraction.SAMPLING_METHOD`):

- **`pt`** — Parallel-Tempering replica-exchange lattice walk (default; best in
  tightly constrained cones).
- **`discrete`** — a repulsive / PID-annealed discrete lattice walk.
- **`raycast`** — continuous guide-ray + raycast harvesting pipeline.

All randomness is reproducible from `config.search.GLOBAL_SEED` and runs in the
main process only (workers do deterministic work).

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `extractor.py` | `ShardExtractorMod` — the stage orchestrator (groups CMFs, runs the strategy, exports `ShardDTO`s). |
| `hyperplanes.py` | `Hyperplane` + `extract_cmf_hyperplanes` — the canonical, deterministically-sorted singular hyperplane set. |
| `shard.py` | `Shard` — a CMF + constants + convex region (`A·x < b`) + interior point; trajectory validity tests. |
| `v2/` | The modern extractor: `lrs` I/O, MILP witnesses, reverse-search cells, ray-shooting, symmetry, and the `ExtractionManager` that routes the strategies. |
| `samplers/` | Trajectory-direction samplers (parallel tempering, discrete MCMC, raycaster, sphere, CHRR) + the flatland `conditioner`. |
| `sampling_orchestrators/` | `ShardSamplingOrchestrator` — picks + seeds the sampler per shard. |
| `utils/` | `initial_points.py` (legacy interior-point scan), `fast_gcd.py`. |

---

## Key configurations

`config.extraction.*` — `STRATEGY`, the `EXACT_*` / `HEURISTIC_*` budgets and
tuning knobs, `LOAD_SHARD_CACHE`, `IGNORE_DUPLICATE_SEARCHABLES`,
`TRAJECTORY_CONSTRAINTS` (the direction filter described above). (The sampler
engine is set on the analysis/search configs, not here — see above.) See the
[configuration index](../configs/README.md#extraction) for the full, annotated list.

---

## Extending this stage

**Add a new extraction module** → subclass `ExtractionModScheme` and implement
`execute()` returning `Dict[Constant, List[Shard]]`; wire it as
`System(extractor=MyExtractorMod)`. `ShardExtractorMod` is the reference.

**Add a new sampler** → implement a sampler in `samplers/` conforming to the
sampler interface and register it in the orchestrator's `SAMPLING_METHOD` switch.
