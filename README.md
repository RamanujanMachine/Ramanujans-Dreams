# Ramanujan's Dreams
Ramanujan's Dreams is a modular system for advanced search in CMFs.

## Installation
* This project is supported fully only on Mac-OS and Linux.  
If you are a Windows user, it is recommended to use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).
* Install via:
    ```bash
    pip install git+https://github.com/UriKH/RamanujansDreams.git
    ```

## Usage
Interaction with the system is via the System class (`from dreamer import System`) and using the config files.

[//]: # (Common usage example with detailed instructions in [colab]&#40;https://colab.research.google.com/drive/1t6qo0LBBHTHTQyojXH566cNJRBhziN_3?usp=sharing&#41;.  )
[//]: # (**Note**: The Colab might be slow and unstable as it's running online. For stable run download the colab as a Jupyter notebook.)

### Structure:
The system is composed of 5 stages:
1. Loading - storing and retrieving mapping from a constant to the inspiration functions.
2. Extraction - extraction of the searchables from the CMF of the inspiration functions.
3. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
4. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.
5. Post-process (optional) - computes expensive per-trajectory attributes for the already-found trajectories, and (optionally) renders graphs/tables. See [Post-processing configuration](#post-processing-configuration).

[//]: # (**Note:** each module could be executed independently of the others. In its current version, the system only wraps the modules together and connects them. )

### Configuration
Configuration management is done using distinct configuration **categories**, all accessed via a single global configuration manager. Each category is a flat group of named settings; changing a setting never requires importing the category object directly.

```python
from dreamer import config

# Access different categories of configurations
config.system.<CONFIG>        # paths, core budget, export directories
config.extraction.<CONFIG>    # shard extraction strategy / sampling
config.analysis.<CONFIG>      # shard filtering / prioritization
config.search.<CONFIG>        # deep search + Tier-2 attributes
config.post_process.<CONFIG>  # Tier-3 attributes (see below)
config.graph.<CONFIG>         # post-process graphing (see below)
config.logging.<CONFIG>
config.database.<CONFIG>

# change specific configurations
config.configure(
    <CATEGORY> = {<CONFIG>: <VALUE>, ...},
    <CATEGORY> = {<CONFIG>: <VALUE>, ...},
    ...
)

# Checkout possible configurations (with descriptions) using the terminal
config.<CATEGORY>.display()
```

There are a few important configurations you might want to change:
- `config.search.NUM_TRAJECTORIES_FROM_DIM` - a lambda function of the form `lambda dim: int(...)` which computes the number of trajectories to be generated from a given dimension.
- `config.analysis.NUM_TRAJECTORIES_FROM_DIM` - same configuration as above but for analysis stage.
- `config.analysis.IDENTIFY_THRESHOLD` - "what fraction of the shard was identified as containing the constant?"

> **How attributes are stored.** Every searched trajectory is one JSON line in
> `<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl`. Cheap **Tier-1** values
> (`delta`, `identified`, `limit`, …) are always written. Heavier **Tier-2**
> attributes (`eigenvalues`, `spectral_gap`, `convergence_class`, …) are
> computed during Search when listed in `config.search.TIER2_ATTRIBUTES`, and
> land in each record's open `extended_metrics` dict. The optional **post-process
> stage** below adds the most expensive **Tier-3** attributes afterwards.

### Post-processing configuration

The post-process stage (`post_process.Tier3PostProcessModV1`, passed as
`post_processor=` to `System`) runs **once after Search** and has two
independent jobs, each off by default:

1. **Tier-3 attributes** — `config.post_process.TIER3_ATTRIBUTES`
2. **Graphing** — `config.graph` (writes under `config.system.EXPORT_GRAPHS`)

It reads the existing JSONL, computes only what's missing, and appends *patch*
records (it never rewrites your data). An empty `TIER3_ATTRIBUTES` **and** a
disabled `graph` config make the whole stage a no-op.

#### `TIER3_ATTRIBUTES` — what to compute, and for which trajectories

This is a tuple where **each entry is either**:

- a **bare attribute name** — always computed, e.g. `'asymptotics'`; or
- a **`(attribute, predicate)` tuple** — the attribute is computed **only for
  trajectories the predicate accepts** (this is how you avoid paying for an
  expensive attribute on every trajectory).

```python
config.configure(
    post_process={
        'TIER3_ATTRIBUTES': (
            'precision_at',                                       # always
            ('asymptotics', 'if_identified'),                     # only identified trajectories
            ('delta_sequence', 'top 10 highest delta in shard'),  # only the 10 best-δ per shard
            ('relation',     'max_degree below 4'),               # only low-degree recurrences
            ('kamidelta',    'top 3 highest convergence_rate in cmf'),
        ),
    },
)
```

**The predicate** can be:

| Predicate | Meaning |
|-----------|---------|
| `'if_identified'` | the trajectory identified the constant |
| `'if_has_degree_2'` | the recurrence has a degree-2 coefficient |
| `'max_degree below N'` / `'max_degree above N'` | recurrence polynomial degree (max over coefficients) is `< N` / `> N` |
| `'top N highest <metric> in shard'` | among the `N` largest `<metric>` **within the trajectory's shard** |
| `'top N lowest  <metric> in shard'` | among the `N` smallest, within the shard |
| `'top N highest/lowest <metric> in cmf'` | …ranked across the **whole CMF** instead of a single shard |

> **General template:** `[top N highest|lowest] <metric> in <shard|cmf>`.

**`<metric>` for the `top N …` selectors must already be stored in the JSONL**
(the ranking pass only *reads* values — it never re-walks a trajectory).
Available metrics:

| metric | source | notes |
|--------|--------|-------|
| `delta` | Tier-1 (always present) | per-constant irrationality measure |
| `convergence_rate` | needs `eigenvalues` in Tier-2 | normalised eigenvalue-error gap `(log\|λ₁\|−log\|λ₂\|)/‖v‖`; larger = faster |
| `asymptotic_digits_per_step` | needs it in Tier-2/Tier-3 | mean new digits per step (tail) |
| `spectral_gap`, `gcd_slope`, `precision_at` | needs the matching attribute stored | |

> ⚠️ **Common gotcha:** to rank by a metric other than `delta`, make sure that
> metric is computed for *every* trajectory first — e.g. add `'eigenvalues'` to
> `config.search.TIER2_ATTRIBUTES` before using `convergence_rate`, or add
> `'asymptotic_digits_per_step'` as a **bare** (unconditional) Tier-3 attribute
> before ranking on it. Trajectories whose metric is missing are simply excluded
> from the ranking.

Notes on semantics:
- `top N … in shard` ranks within each shard; `in cmf` pools all shards of a CMF.
- A trajectory rejected by its predicate is reprocessed cheaply on each run
  (the gate is settled before any walk), so re-running is safe and idempotent.
- Discover everything available with `config.post_process.display()`.

#### Graphing

Enable any of the three graph kinds in `config.graph`; output goes to
`config.system.EXPORT_GRAPHS`:

```python
config.configure(
    system={'EXPORT_GRAPHS': './graphs'},
    graph={
        'PLOT_BEST_DELTA_SEQUENCE': True,  # δ vs step for the best trajectory of each (CMF, constant)
        'PLOT_DELTA_HISTOGRAMS':   True,   # δ histogram per shard and per CMF
        'WRITE_BUMPINESS_TABLE':   True,   # per-shard "how non-smooth is δ" table (CSV + markdown)
        'DELTA_SEQUENCE_DEPTH':    1000,   # steps for the best-δ-sequence plot
    },
)
```

The **bumpiness table** quantifies how non-smooth the δ field of each shard is,
with two columns: a density-robust **spatial roughness** (empirical
semivariogram of δ over direction space — `relative_nugget` ≈ 1 → needle/bumpy,
≈ 0 → smooth) and the median per-trajectory **δ-sequence total variation**
(convergence wobble; needs `delta_sequence` stored as a Tier-3 attribute).
See [`context/algorithms/05_bumpiness_metrics.md`](context/algorithms/05_bumpiness_metrics.md)
for the math.

[//]: # (Each `<X>_config` contains the configurations for this section. You can access those directly in order to view the current values.  )
[//]: # (In order to change them you can use: `<X>_config.<property> = <new-value>`  )
[//]: # (Or, by using the global configuration manager: `config.configure&#40;<X> = {<property> : <new-value> }&#41;`  )
[//]: # (The latter allows the **addition of new configurations**.)

### Run
A classic run would look something like this:

```python
from dreamer import System, config, log
from dreamer import analysis, search, extraction, loading, post_process

# Optional reconfigure
config.configure(...)

my_system = System(
    function_sources=[loading.pFq(log(2), 2, 1, -1)],  # Set up the loading stage - provide inspiration functions
    extractor=extraction.extractor.ShardExtractorMod,  # Choose an extraction module
    analyzers=[analysis.AnalyzerModV1],  # Choose an analysis module(s)
    searcher=search.SearcherModV1,  # Choose the search module
    post_processor=post_process.Tier3PostProcessModV1,  # Optional: Tier-3 attributes + graphs (see Post-processing configuration)
)

my_system.run(constants=[log(2)])
```

Advanced options are:
* Using a database as on of the inspiration functions source.
* Using pickled inspiration function objects from past runs as inspiration functions source.
* Using pickled past analysis results as input to the analysis stage.

### Terminal Setup

If you are a PyCharm user, the output might look a bit off due to `tqdm` default configurations.  
To make sure the output console looks right:
1. Enter: `Run > Edit Configurations > Modify Options`
2. Select: `Emulate terminal in output console`

[//]: # (#### Notes: )
[//]: # (- When loading inspiration functions, you can use formerly computed CMFs using pickle files &#40;might be unstable&#41;, maunally list the inspiration functions or using a DB &#40;instructions below&#41;.)
[//]: # (- Changing configurations could be done in two ways:)
[//]: # (  1. Using `config.configure&#40;<config_section> = {<configuration-name> : <new value>}&#41;` - that way new configurations could be added to newly developed modules.)
[//]: # (  2. Using each section's private configuration e.g. `db_config.USAGE = DBUsage.RETRIEVE_DATA`.)
[//]: # (  3. If you are a PyCharm user, your terminal might be a bit off due to `tqdm` defualt configurations.  )
[//]: # (   To make sure the terminal looks right set: `Run > Edit Configurations > Emulate terminal in output console`)
[//]: # (### Loading using a DB)
[//]: # (1. You can add to the DB manually &#40;i.e. by using its interface&#41; or by loading via a json file)
[//]: # (2. To create a loadable json file run the following &#40;with your inspiration functions listed&#41;:)
[//]: # (    ```)
[//]: # (    dreamer.loading.DBModScheme.export_future_append_to_json&#40;)
[//]: # (        [ <your inspiration functions> ],)
[//]: # (        path='my_append_instruction')
[//]: # (    &#41;)
[//]: # (    ```)
[//]: # (3. On system creation, insert the inspiration functions sources as `if_srcs=[BasicDBMod&#40;json_path='my_append_instruction.json'&#41;]`  )
[//]: # (   When reading this file, the system will execute the `append` command and will try to add the inspiration function ${}_2F_1&#40;0.5&#41;$ to set of inpiration funcitons for $\pi$ with the shift in start point as $x=0,~y=0,~z=\text{sp.Rational&#40;1,2&#41;}$.)

## License
This project is licensed under the terms of the [MIT License](LICENSE).

## Contribution
* Please open an issue for any bug or error you encounter.
* For further details see [instructions](CONTRIBUTING.md).
