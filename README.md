# Ramanujan's Dreams
Ramanujan's Dreams is a modular system for advanced search in CMFs.

## Table of Contents

1. [Installation](#installation) - Download and setup your runtime environment.
2. [Structure](#structure)  
    2.1. [System](#system-structure) - Overview of the system's design.  
    2.2. [Project](#project-structure) - Structure of the repository. 
3. [Usage](#usage)  
    3.1. [Configuration](#configuration) - How to configure the system and explore options.  
    3.2. [Run](#run) - Running the system, simple example.
4. [Contribution](#contribution) - How to customize the system and add your own modules.
5. [License](#license)

## Installation
* This project is supported only on Mac-OS and Linux.  
If you are a Windows user, it is recommended to use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).
* Install the package via:
    ```bash
    pip install git+https://github.com/UriKH/RamanujansDreams.git
    ```

**Note:** If you are developing using an IDE the output might look a bit off due to terminal default configurations.  
If you are a PyCharm user, an easy fix is:
1. Select: `Run -> Edit Configurations -> Modify Options`
2. And then: `Emulate terminal in output console`

## Structure

### System Structure
The system is a pipeline composed of 5 stages:
1. Loading - storing and retrieving mapping from a constant to the inspiration functions.
2. Extraction - extraction of the searchables from the CMF of the inspiration functions.
3. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
4. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.
5. Post-process (optional) - computes expensive per-trajectory attributes for the already-found trajectories, and (optionally) renders graphs/tables. See [Post-processing configuration](#post-processing-configuration).

### Project Structure

```
dreamer    --> The system itself
examples   --> Examples of how to run and templates for customized modules
data_utils --> Results exploration tools
graphs     --> Utility scripts for analysis of 3D CMFs and statistics
tests      --> System tests
```

#### Where to read more

Each pipeline stage has its own README explaining what it does and what its
directory contains. Start with the one matching what you're working on:

| Stage | README |
|-------|--------|
| Loading | [`dreamer/loading/`](dreamer/loading/README.md) |
| Extraction | [`dreamer/extraction/`](dreamer/extraction/README.md) |
| Analysis | [`dreamer/analysis/`](dreamer/analysis/README.md) |
| Search | [`dreamer/search/`](dreamer/search/README.md) |
| Post-process | [`dreamer/post_process/`](dreamer/post_process/README.md) |
| Graphing (post-process renderer) | [`dreamer/graphing/`](dreamer/graphing/README.md) |
| **All configuration** | [`dreamer/configs/`](dreamer/configs/README.md) |

To extend the system with your own modules, see [CONTRIBUTING.md](CONTRIBUTING.md).


## Usage
Interaction with the system is via the System class (`from dreamer import System`) and using the config files.

[//]: # (Common usage example with detailed instructions in [colab]&#40;https://colab.research.google.com/drive/1t6qo0LBBHTHTQyojXH566cNJRBhziN_3?usp=sharing&#41;.  )
[//]: # (**Note**: The Colab might be slow and unstable as it's running online. For stable run download the colab as a Jupyter notebook.)

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

> For the **full, annotated list of every configuration category and field**,
> see the [configuration README](dreamer/configs/README.md). In a running
> session, `config.<category>.display()` prints the live values and descriptions.

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

1. **Tier-3 attributes** — `config.post_process.TIER3_ATTRIBUTES`: the most
   expensive per-trajectory attributes, each optionally restricted to a subset
   of trajectories via a predicate (e.g. `if_identified`, or
   `top 10 highest delta in shard`).
2. **Graphing** — `config.graph`: δ-sequence plots, δ histograms, and a per-shard
   δ-roughness ("bumpiness") table, written under `config.system.EXPORT_GRAPHS`.

It reads the existing JSONL, computes only what's missing, and appends *patch*
records (it never rewrites your data). An empty `TIER3_ATTRIBUTES` **and** a
disabled `graph` config make the whole stage a no-op.

The full attribute/predicate grammar, the rankable metrics, and the graph
parameters are documented in the
[configuration README](dreamer/configs/README.md#post_process) and the
[post-process](dreamer/post_process/README.md) / [graphing](dreamer/graphing/README.md)
stage READMEs.

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

System(
    function_sources=[loading.pFq(log(2), 2, 1, -1)],    # Set up the loading stage - provide inspiration functions
    extractor=extraction.extractor.ShardExtractorMod,    # Choose an extraction module
    analyzers=[analysis.AnalyzerModV1],                  # Choose an analysis module(s)
    searcher=search.SearcherModV1,                       # Choose the search module
    post_processor=post_process.Tier3PostProcessModV1,   # Optional: Tier-3 attributes + graphs (see Post-processing configuration)
).run(constants=[log(2)])
```

Advanced options are:
* Using a database as one of the inspiration-function sources.
* Re-using CMFs **exported from a past run** as an inspiration-function source
  (point `function_sources` at the export directory — see
  `config.system.EXPORT_CMFS`).
* Re-using a past run's **analysis priorities** to skip straight to search (run
  without an `analyzers=` argument and the priorities are reloaded from
  `config.system.EXPORT_ANALYSIS_PRIORITIES`).

> **Where results live.** Every searched/analysed trajectory is one line in a
> per-shard **JSONL** file under `config.system.EXPORT_SEARCH_RESULTS`
> (`<shard_id>.jsonl`). That JSONL store is the canonical record the run summary
> and best-δ reporting read back — see the
> [search](dreamer/search/README.md) and [analysis](dreamer/analysis/README.md)
> stage READMEs.


[//]: # (#### Notes: )
[//]: # (- When loading inspiration functions, you can re-use formerly exported CMFs, manually list the inspiration functions, or use a DB &#40;instructions below&#41;.)
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
