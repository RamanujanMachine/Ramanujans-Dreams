# Configuration — `dreamer/configs/`

This directory holds **all** of the system's tunable settings, grouped into
**categories** (one per pipeline concern). Every category is a flat dataclass of
named, documented fields, and all of them are reached through a single global
manager.

For what each pipeline stage actually does, see the per-stage READMEs:
[loading](../loading/README.md) · [extraction](../extraction/README.md) ·
[analysis](../analysis/README.md) · [search](../search/README.md) ·
[post-process](../post_process/README.md) · [graphing](../graphing/README.md).
For the big picture, see the [main README](../../README.md).

---

## How configuration works

```python
from dreamer import config

# Read a setting
config.search.SEARCH_MAX_TRAJ_LEN
config.extraction.STRATEGY

# Change settings (any number of categories / fields at once)
config.configure(
    extraction={'STRATEGY': 'heuristic'},
    search={'MAX_TRAJECTORY_LENGTH': 15, 'SAMPLING_METHOD': 'pt'},
    analysis={'IDENTIFY_THRESHOLD': 1e-3},
)

# List a category's fields with their built-in descriptions (in the terminal)
config.search.display()
```

Every field carries a one-line `description`; **`config.<category>.display()` is
the authoritative, always-current reference** — the tables below are an orienting
summary of the knobs you are most likely to touch, not an exhaustive dump.

> **Reproducibility:** `config.search.GLOBAL_SEED` (default `42`) seeds *all*
> samplers and search methods. Set it to `None` for nondeterministic runs.

### The categories

| Category | File | Drives |
|----------|------|--------|
| `config.system` | `system.py` | paths, export directories, core/worker budget |
| `config.database` | `database.py` | DB source retrieve/store mode |
| `config.extraction` | `extraction.py` | shard-extraction strategy + trajectory sampling |
| `config.analysis` | `analysis.py` | shard filtering / prioritisation |
| `config.search` | `search.py` | deep search + Tier-2 attributes (largest category) |
| `config.post_process` | `post_process.py` | Tier-3 attributes |
| `config.graph` | `graph.py` | post-process graphing |
| `config.logging` | `logging.py` | logging / profiling / watchdogs |

The plumbing — `config_manager.py` (the `ConfigManager` facade) and
`configurable.py` (the `Configurable` base giving every category `display()` /
export / metadata) — is shared and rarely touched.

---

## `system`

Mostly **where things go** and **how many cores to use**.

| Field | Meaning |
|-------|---------|
| `EXPORT_CMFS` | Directory loaded CMFs are exported to (enables extractor-free reloads; `None` = off). |
| `EXPORT_ANALYSIS_PRIORITIES` | Directory the ranked shard priorities are persisted to. |
| `EXPORT_SEARCH_RESULTS` | The canonical per-shard trajectory store (`<shard_id>.jsonl`), shared by analysis + search. |
| `EXPORT_ANALYSIS_RESULTS` | Separate analysis store, used only when `analysis.STORE_TRAJECTORIES_SEPARATELY` is on. |
| `EXPORT_GRAPHS` | Where the graphing stage writes figures/tables. |
| `*_FORMAT` | `'pkl'` or `'json'` for the corresponding export. |
| `TOTAL_CORES` / `NUM_BACKGROUND_WORKERS` | The core budget (`None` ⇒ `os.cpu_count()`); workers reserved for Tier-2. |
| `CONSTANTS` | Default constants to search when `run()` is given none. |
| `OPTIMIZATION_OBJECTIVE` | The numeric trajectory attribute the **whole pipeline** optimises for — both the analysis-stage shard ranking and the search-stage optimisers. `'delta'` (default) reproduces the historical behaviour exactly. Must be a registered objective (see below); each objective knows whether its optimum is the highest or lowest value, so "smaller is better" metrics work without inverting the search. Identification (LIReC) is always required regardless of the objective. |

**Optimisation objectives.** `OPTIMIZATION_OBJECTIVE` must name an entry in
`dreamer.utils.storage.optimization_objectives.OBJECTIVES` — the registry that
gates which attributes are optimisable (numeric, with a known optimal direction)
and stores that direction. Shipped objectives: `delta` (max) and
`convergence_rate` (max, the length-normalised spectral rate
`approximated_digits_per_step / ‖direction‖₂`). Since a stored result is one flat
row per `(trajectory, constant)`, the objective is simply a **column** on that row
(`delta` is the core field; `convergence_rate` etc. are their own key). Ranking
reads it via `optimization_objectives.score_record`; the objective is **not** part
of the config hash, so switching it never invalidates δ. To add an objective: add
a handler method + an `ATTRIBUTE_REGISTRY` entry, then register it in `OBJECTIVES`
with its direction.

## `database`

| Field | Meaning |
|-------|---------|
| `USAGE` | `RETRIEVE_DATA` / `STORE_DATA` / `STORE_THEN_RETRIEVE` for DB sources. |

## `extraction`

| Field | Meaning |
|-------|---------|
| `STRATEGY` | `'auto'` / `'exact'` / `'heuristic'` / `'legacy'` — how shards are found (see [extraction README](../extraction/README.md)). |
| `EXACT_TIMEOUT_SECONDS` / `HEURISTIC_TIMEOUT_SECONDS` | Per-strategy wall-clock budgets (used by `'auto'` to decide fallback). |
| `LOAD_SHARD_CACHE` | Reuse a previously-extracted `…__shards.jsonl` instead of re-extracting. |
| `IGNORE_DUPLICATE_SEARCHABLES` | Drop symmetry-equivalent shards. |
| `TRAJECTORY_CONSTRAINTS` | Optional `{var: int}` direction constraint, e.g. `{'x0': 12, 'y1': 28}` — pins the scale-invariant **ratio + sign** of every sampled trajectory and **drops shards** whose cone admits no such direction. Honoured by all samplers + search (see [extraction README](../extraction/README.md)). `None` = unconstrained. |
| `HEURISTIC_*`, `EXACT_*` | Fine-tuning of the ray-shooter / reverse-search (see `display()`). |

## `analysis`

| Field | Meaning |
|-------|---------|
| `NUM_TRAJECTORIES_FROM_DIM` | `lambda dim: int(...)` — how many trajectories to sample per shard. |
| `IDENTIFY_THRESHOLD` | Minimum identified-fraction to keep a shard (`-1` disables filtering). Independent of the objective — identification is always a prerequisite; the shard is then *ranked* by `system.OPTIMIZATION_OBJECTIVE`. |
| `SAMPLING_METHOD` | `'pt'` / `'discrete'` / `'raycast'` — sampler for analysis (independent of the search sampler). |
| `STORE_TRAJECTORIES_SEPARATELY` | Write analysis records to their own store instead of the shared one. |
| `USE_LIReC` | Use LIReC for identification. |

## `search`

The largest category. Shared knobs plus one block per optimiser
(see the [search README](../search/README.md)).

| Field | Meaning |
|-------|---------|
| `GLOBAL_SEED` | Master RNG seed for all samplers + search methods. |
| `NUM_TRAJECTORIES_FROM_DIM` | Trajectory count per shard during search. |
| `DEPTH_FROM_TRAJECTORY_LEN` | Walk depth as a function of trajectory length + dimension. |
| `SEARCH_MAX_TRAJ_LEN` / `SEARCH_TRAJ_NORM` | Cap on trajectory length (bounds symbolic cost) and the norm it's measured in. |
| `SAMPLING_METHOD` | Sampler for the default hedgehog searcher. |
| `TIER2_ATTRIBUTES` | Background-worker attributes computed during search (empty ⇒ no subprocesses). |
| `ENABLE_MICRO_HILL_CLIMB` | The shared resolution-doubling finalization endgame after every method. |
| `GA_*` | Genetic-algorithm knobs. |
| `SA_*` | Small-Angle search knobs. |
| `ANNEAL_*` | Simulated-Annealing knobs. |
| `GRAD_*` | Gradient-Ascent knobs. |
| `SPSA_*` | Hybrid SPSA + Adam knobs. |

## `post_process`

| Field | Meaning |
|-------|---------|
| `TIER3_ATTRIBUTES` | Tuple of expensive attributes to compute after search, each optionally gated by a predicate (grammar below). Empty ⇒ stage is a no-op. |

### The `TIER3_ATTRIBUTES` grammar

Each entry is **either** a bare attribute name (always computed) **or** an
`(attribute, predicate)` tuple (computed only for trajectories the predicate
accepts — this is how you avoid paying for an expensive attribute everywhere):

```python
config.configure(post_process={'TIER3_ATTRIBUTES': (
    'precision_at',                                       # always
    ('asymptotics', 'if_identified'),                     # only identified trajectories
    ('delta_sequence', 'top 10 highest delta in shard'),  # only the 10 best-δ per shard
    ('relation',     'max_degree below 4'),               # only low-degree recurrences
    ('kamidelta',    'top 3 highest convergence_rate in cmf')),
})
```

| Predicate | Meaning |
|-----------|---------|
| `if_identified` | the trajectory identified the constant |
| `if_has_degree_2` | the recurrence has a degree-2 coefficient |
| `max_degree below N` / `max_degree above N` | recurrence polynomial degree (max over coefficients) `< N` / `> N` |
| `top N highest/lowest <metric> in shard` | the N best/worst `<metric>` **within the trajectory's shard** |
| `top N highest/lowest <metric> in cmf` | …ranked across the **whole CMF** instead |

> **General template:** `top N highest|lowest <metric> in shard|cmf`.

The `<metric>` for a `top N …` selector **must already be stored** in the JSONL
(the ranking pass only *reads* values — it never re-walks). Available metrics:
`delta` (always present), `convergence_rate` (in the default `TIER2_ATTRIBUTES`),
`approximated_digits_per_step`, `spectral_gap`, `gcd_slope`, `precision_at`. Each
reads the handler-computed value straight from the record — `convergence_rate` is
the single system-wide definition (`approximated_digits_per_step / ‖direction‖₂`),
not recomputed here. To rank on one that isn't stored yet, add it to
`search.TIER2_ATTRIBUTES` (or as a bare Tier-3 attribute) first.

## `graph`

| Field | Meaning |
|-------|---------|
| `PLOT_BEST_DELTA_SEQUENCE` | δ-sequence plot of the best trajectory per `(CMF, constant)`. |
| `PLOT_DELTA_HISTOGRAMS` | δ histograms per shard and per CMF. |
| `WRITE_BUMPINESS_TABLE` | Per-shard δ-roughness table (semivariogram + δ-sequence total variation). |
| `DELTA_SEQUENCE_DEPTH` / `HISTOGRAM_BINS` / `VARIOGRAM_*` | Parameters for the above. |

See the [graphing README](../graphing/README.md) for what each artefact means.

## `logging`

| Field | Meaning |
|-------|---------|
| `GENERATE_LOGS` / `LOG_FILENAME` | Enable + locate the debug log file. |
| `PROFILE` / `PROFILE_SUMMARY` | Per-timer profiling output. |
| `WATCHDOG_ENABLED` / `WATCHDOG_*_SECONDS` | Hang-detection warnings for long-running trajectory/attribute computations. |
| `EXCEPTION_SHOW_TRACE` / `DEBUG_SHOW_TRACE` | Include stack traces in logs. |
