# Contributing to Ramanujan's Dreams

This guide explains **how the library is laid out**, **what each folder
contains**, and **how to add your own modules** to the pipeline. If you only
want to *run* the system, start with the [README](README.md); come back here
when you want to extend it.

A working mental model first: the system is a **5-stage pipeline** over CMFs.
Each stage is *substitutable* — it is defined by an abstract **scheme** (a base
class), and any class implementing that scheme can be swapped in via the
`System(...)` constructor without touching the rest of the system. Adding a
feature almost always means **writing a new module that conforms to a stage's
scheme**, not editing the core.

- [Reporting issues](#reporting-issues)
- [Repository layout](#repository-layout)
- [The `dreamer` package](#the-dreamer-package)
- [How to add a module](#how-to-add-a-module)
- [Development standards](#development-standards)

---

## Reporting issues

Please open a GitHub issue for any bug or unexpected behaviour you encounter,
with a minimal reproduction (the `function_sources`, the wired modules, and the
`config.configure(...)` you used) where possible.

---

## Repository layout

```
dreamer       --> The system itself (the installable package; configs live inside it)
examples      --> Runnable examples and configuration templates
data_utils    --> Tools for exploring results after a run
graphs        --> Standalone analysis/plotting scripts (3D CMFs, statistics)
tests         --> Test suite (pytest)
experiments   --> Ad-hoc experiment scripts (not part of the package)
```

The package itself, `dreamer/`, is organised **one directory per pipeline
stage**, plus shared infrastructure. Each stage directory has its **own
README** describing what it accomplishes and what's inside — those are the best
entry points when working on a stage.

---

## The `dreamer` package

```
dreamer/
├── loading/        Stage 1 — build CMFs from inspiration functions / DBs / exports
├── extraction/     Stage 2 — slice each CMF into shards; sample trajectory directions
├── analysis/       Stage 3 — cheap triage: filter + rank shards by best δ
├── search/         Stage 4 — deep search for high-δ trajectories (5 methods)
├── post_process/   Stage 5 — optional: expensive Tier-3 attributes (+ graphing)
├── graphing/       Renderer used by post-process (δ plots, histograms, bumpiness)
├── configs/        All tunable settings, grouped into categories
├── system/         The System orchestrator that wires the stages together
└── utils/          Shared infrastructure (schemes, storage, geometry, RNG, …)
```

| Stage | Directory | Default module | What it does | README |
|-------|-----------|----------------|--------------|--------|
| 1. Loading | `dreamer/loading/` | formatters / DB modules | Turn each constant into CMFs to search. | [loading](dreamer/loading/README.md) |
| 2. Extraction | `dreamer/extraction/` | `ShardExtractorMod` | Cut each CMF into convex **shards**. | [extraction](dreamer/extraction/README.md) |
| 3. Analysis | `dreamer/analysis/` | `AnalyzerModV1` | Filter + rank shards (cheap Tier-1 pass). | [analysis](dreamer/analysis/README.md) |
| 4. Search | `dreamer/search/` | `SearcherModV1` (+ 5 optimisers) | Search shards deeply for high-δ formulas. | [search](dreamer/search/README.md) |
| 5. Post-process | `dreamer/post_process/` | `Tier3PostProcessModV1` | Optional Tier-3 attributes + graphs. | [post-process](dreamer/post_process/README.md) · [graphing](dreamer/graphing/README.md) |

Cross-cutting:

| Directory | Contents |
|-----------|----------|
| `dreamer/configs/` | One dataclass per config **category** + the global `ConfigManager`. See the [configuration README](dreamer/configs/README.md). |
| `dreamer/system/` | `System` — the orchestrator that runs the stages in order and threads data between them. |
| `dreamer/utils/schemes/` | The **abstract base classes** that define each stage's contract (this is what you subclass to add a module). |
| `dreamer/utils/storage/` | The per-trajectory storage layer: DTOs, the attribute handler, the JSONL "atlas", stable IDs. |
| `dreamer/utils/` (rest) | `geometry`, `constants`, the logger, RNG seeding (`rand.py`), multiprocessing helpers, type annotations. |

---

## How to add a module

Every stage is defined by a scheme in `dreamer/utils/schemes/`. To extend a
stage you **subclass its scheme**, implement the required method(s), and pass
your class to `System(...)`. The default module in each stage directory is the
reference implementation to copy from.

| Stage | Scheme to subclass (`dreamer/utils/schemes/`) | Implement | Wire as |
|-------|-----------------------------------------------|-----------|---------|
| Loading (function source) | `DBModScheme` (`db_scheme.py`) | `execute(constants) -> Dict[Constant, List[CMFData]]` | `function_sources=[MyMod(...)]` |
| Loading (inspiration fn) | `Formatter` (`dreamer/loading/funcs/formatter.py`) | CMF construction + JSON round-trip | `function_sources=[MyFormatter(...)]` |
| Extraction | `ExtractionModScheme` (`extraction_scheme.py`) | `execute() -> Dict[Constant, List[Searchable]]` | `extractor=MyExtractorMod` |
| Analysis | `AnalyzerModScheme` (`analysis_scheme.py`) | `execute() -> Dict[Constant, List[Searchable]]` | `analyzers=[MyAnalyzerMod]` |
| Search | `SearcherModScheme` (`searcher_scheme.py`) | `execute() -> None` (writes per-shard JSONL) | `searcher=MySearchMod` |
| Post-process | `PostProcessModScheme` (`post_process_scheme.py`) | `execute() -> None` | `post_processor=MyPostProcessMod` |

> **Storage:** the search and analysis stages persist one **JSONL** record per
> trajectory to `<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl` — the canonical store
> the run summary reads. You don't write it by hand: open a `worker_pool`, build a
> `TrajectoryAttributesHandler` + `TrajectoryDTO` per trajectory, and `push` it.
> The `examples/` templates show this end-to-end.

### The method / module split (search & analysis)

For the search and analysis stages there is a deliberate two-class split:

- The **algorithm class** holds the *internal logic* — how it picks trajectory
  directions inside a shard and evaluates them (this is the part that's genuinely
  different between, say, a hill-climb and a genetic search).
- The **module** (`SearcherModScheme` / `AnalyzerModScheme`) holds the *external
  interface* — which shards to process, opening the JSONL store, and ranking.

> The current default modules (`SearcherModV1`, `AnalyzerModV1`) and the optimiser
> searchers follow this shape. The repo also ships older inner abstract bases
> (`SearchMethod`, `AnalyzerScheme`) that predate the JSONL/DTO pipeline; the
> `examples/` templates intentionally model the **current** modules instead.

Keep them in **separate files** so they can be reused independently:

- Search method → `dreamer/search/methods/<your_method>/<name>_scan.py`
- Search module → `dreamer/search/searchers/<name>_mod.py`
- Analyzer method → `dreamer/analysis/analysis_methods/<name>.py`
- Analyzer module → `dreamer/analysis/analyzers/<name>/<name>_mod.py`

Each stage's README has an "Extending this stage" section with the specifics.

### Copy-paste templates

The [`examples/`](examples/) directory contains ready-to-copy, fully-commented
skeletons for the two stages you're most likely to extend. They model the
**current** default modules (the JSONL/DTO pipeline), match the exact constructor
signatures `System` calls, and mark every spot you need to fill in with a `TODO`:

| Template | Stage | What it skeletons |
|----------|-------|-------------------|
| [`examples/search.py`](examples/search.py) | Search | `MySearchMethod` (your algorithm) + `MySearchMod` (the module, writes per-shard JSONL). |
| [`examples/analysis.py`](examples/analysis.py) | Analysis | `MyAnalyzer` (probe one shard) + `MyAnalyzerMod` (the module, ranks shards). |

Copy the file, rename the classes, fill in the `TODO`s, and wire it in:

```python
System(
    function_sources=[...],
    searcher=MySearchMod,    # or analyzers=[MyAnalyzerMod]
).run(constants=[...])
```

If your module needs **extra constructor arguments**, wrap it with
`functools.partial` so `System` can still build it with the standard signature —
e.g. `searcher=partial(MySearchMod, my_arg=...)`. The templates note exactly
where this applies.

Each template's docstring also points at the production-grade default module for
that stage (`SearcherModV1`, `AnalyzerModV1`) as a deeper reference once the
skeleton is working.

---

## Development standards

- **Document** every new public class/function (description + `:param`/`:return`/
  `:raises`), follow PEP 8, and keep modules small and reusable.
- **Test** every new public function — normal operation, edge cases, and at
  least one known-answer check. Run the suite with `pytest tests/ -v`.
- **Verify mathematics numerically** (100+ digits with `mpmath`); never use
  Python `float` for mathematical computation.
- **Keep the docs in sync.** When you change a stage's behaviour or its config,
  update that stage's `README.md`, the [configuration README](dreamer/configs/README.md),
  and the [main README](README.md) in the **same change** — see the project's
  Definition of Done. If you change a stage **scheme** (a constructor signature
  or required method), also update the matching template in
  [`examples/`](examples/) — the templates encode those exact signatures.
- **Reproducibility:** anything stochastic must derive its RNG from
  `config.search.GLOBAL_SEED` via `dreamer.utils.rand`, and must run in the main
  process (workers do deterministic work only).
