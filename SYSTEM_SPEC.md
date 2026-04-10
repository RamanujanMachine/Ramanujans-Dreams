# System Specification — Ramanujan's Dreams

> **Canonical reference for all contributors (human and AI).**
> Before writing code, designing a module, or proposing a change — read this file.

---

## 1. Mission

Discover new mathematical formulas for fundamental constants by systematically
searching inside **Conservative Matrix Fields (CMFs)**.

A successful outcome is a **new polynomial continued fraction** (PCF) or
recurrence relation that converges to a constant like $\pi$, $e$, $\zeta(3)$,
or $\ln 2$, verified to 100+ decimal places and classified against known CMF
families.

---

## 2. What the System Does — In One Paragraph

The system takes a **mathematical constant** and one or more **inspiration
functions** (hypergeometric $_pF_q$, Meijer-G, or raw CMFs), extracts the
CMF's internal structure, partitions its parameter space into bounded
**shards**, samples **trajectories** through each shard, evaluates matrix
products along those trajectories, and identifies which trajectories produce
convergents of the target constant. The best shards are then searched more
deeply for exact polynomial continued fractions.

---

## 3. Pipeline Stages

The system is a four-stage pipeline. Every stage is **modular**: you can swap
in a new implementation without touching the others.

```
 Constant + Inspiration Functions
               │
               ▼
  ┌─────────────────────────┐
  │    1. LOADING            │  Map constants → CMFs (from DB, JSON, or code)
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │    2. EXTRACTION         │  Partition CMF space into bounded shards (Ax < b)
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │    3. ANALYSIS           │  Sample trajectories, compute δ, filter & rank shards
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │    4. SEARCH             │  Deep search in promising shards → discover PCFs
  └─────────────────────────┘
```

### 3.1 Loading

| Responsibility | Details |
|:---------------|:--------|
| Input | `Constant` objects + `Formatter` objects (pFq, MeijerG, BaseCMF) |
| Output | `Dict[Constant, List[ShiftCMF]]` — each constant mapped to its CMFs with shifts |
| Storage | SQLite database (`families_v1.db`), pickle files, or in-memory |
| Key classes | `Formatter`, `pFq`, `MeijerG`, `BaseCMF`, `DB`, `BasicDBMod` |

### 3.2 Extraction

| Responsibility | Details |
|:---------------|:--------|
| Input | `Dict[Constant, List[ShiftCMF]]` |
| Output | `Dict[Constant, List[Shard]]` — bounded convex regions of the CMF lattice |
| How | Enumerate hyperplanes (matrix zeros & poles) → sign-vector encoding → build inequality system $Ax < b$ → find interior points |
| Key classes | `ShardExtractorMod`, `ShardExtractor`, `Shard`, `Hyperplane` |

### 3.3 Analysis

| Responsibility | Details |
|:---------------|:--------|
| Input | `Dict[Constant, List[Shard]]` |
| Output | `Dict[Constant, List[Shard]]` — filtered and ranked by promise |
| How | For each shard: sample ~$10^d$ trajectories (d = dimension), walk the CMF, compute convergence δ, keep shards above `IDENTIFY_THRESHOLD` |
| Key classes | `AnalyzerModV1`, `Analyzer`, `SerialSearcher` |

### 3.4 Search

| Responsibility | Details |
|:---------------|:--------|
| Input | Prioritized shards from Analysis |
| Output | `Dict[Searchable, DataManager]` — discovered PCFs and search vectors |
| How | Deeper walks (depth up to 1500), exact rational convergent extraction, LIReC/RIES identification |
| Key classes | `SearcherModV1`, `SerialSearcher`, `DataManager` |

---

## 4. Core Mathematical Objects

### Constant
A target mathematical constant (e.g., $\pi$, $e$, $\zeta(3)$).
Represented by a sympy expression and an mpmath value.
Registry pattern: all constants are globally registered and deduplicated by name.

### CMF (Conservative Matrix Field)
The central algebraic structure. A CMF assigns a matrix $M_s$ to each symbol
$s$ in a $d$-dimensional lattice, subject to a commutativity constraint.
Walking along a path in the lattice and multiplying the corresponding matrices
produces convergents of mathematical constants.

Implementation lives in `ramanujantools`. Dreamer wraps it via `Formatter`
subclasses for serialization and shift management.

### Shard
A bounded convex region of the CMF's integer lattice, defined by linear
inequalities $Ax < b$. Each shard has an interior point and a set of sampled
trajectories. Shards are the atomic unit of work for analysis and search.

### Trajectory
An integer direction vector inside a shard. The system walks the CMF along
this direction, multiplying matrices, to compute a convergent. If the
convergent approaches the target constant, the trajectory is "identified."

### PCF (Polynomial Continued Fraction)
The end product. A continued fraction $a_0 + \cfrac{b_1}{a_1 + \cfrac{b_2}{a_2 + \cdots}}$
where $a_n$ and $b_n$ are polynomials in $n$. Extracted from promising
trajectories. Characterized by convergence rate (δ) and irrationality measure.

---

## 5. Key Invariants

These must hold at all times. Any code change that violates them is incorrect.

1. **All numerical verification uses `mpmath` at ≥ 100 digits.** Python `float` is forbidden for mathematical computation.
2. **Every `Formatter` can round-trip through JSON.** `from_json_obj(to_json_obj(x))` must reconstruct an equivalent object.
3. **Shard inequalities are strict.** A point on the boundary ($Ax = b$) is **outside** the shard.
4. **The Constant registry is the single source of truth.** Two constants with the same name are the same constant.
5. **Stage outputs are stage inputs.** Loading → Extraction → Analysis → Search. Each stage takes the previous stage's output without transformation.
6. **Modules are substitutable.** Any `AnalyzerModScheme` subclass can replace `AnalyzerModV1` without changing `System`.

---

## 6. Current Limitations & Known Gaps

Be aware of these when contributing. They are opportunities, not just problems.

| Area | Limitation | Impact |
|:-----|:-----------|:-------|
| **Windows / LIReC** | LIReC has installation issues on Windows; `cysignals`/`fpylll` are Linux-only | Tests and runs may need WSL or Linux |
| **Parallelism** | Extraction parallelism is commented out; search has basic chunking | Single-threaded bottleneck on large CMFs |
| **Trajectory sampling** | `EndToEndSamplingEngine` works but uniformity degrades in high dimensions | Thin cones may be undersampled |
| **Database** | SQLite is local-only; no shared/remote DB | Multi-user workflows need manual merging |
| **Search depth** | Max depth capped at 1500 | Some PCFs need deeper walks to converge |
| **Ascent logic** | Mentioned in README but not implemented | Cannot yet climb from a PCF to a higher-level identity |
| **Proof generation** | System finds formulas but does not prove them | Discovered PCFs are conjectures until proven |

---

## 7. Development Priorities

Ordered by impact. When choosing what to work on, prefer items higher on this list.

### Tier 1 — Reliability
- [ ] **Fix all critical bugs** before adding features. See open bug-fix branches.
- [ ] **Increase test coverage** for core modules (Constant, Formatter, Shard, DB, System).
- [ ] **CI pipeline**: automated `pytest` on every push; block merges on failure.
- [ ] **Cross-platform installation**: resolve LIReC/Windows issues or provide a Docker image.

### Tier 2 — Depth of Search
- [ ] **Adaptive depth control**: automatically increase walk depth when δ is improving.
- [ ] **Improved trajectory sampling**: better coverage of thin cones in high dimensions.
- [ ] **New CMF families**: integrate additional inspiration function types beyond pFq and MeijerG.
- [ ] **Shard prioritization heuristics**: ML-based ranking of shards by likely discovery yield.

### Tier 3 — Scale
- [ ] **Parallel extraction and search**: multi-process/distributed shard evaluation.
- [ ] **Remote database**: shared SQLite → PostgreSQL or similar for team-wide deduplication.
- [ ] **Result deduplication**: detect coboundary-equivalent PCFs automatically.

### Tier 4 — Scientific Output
- [ ] **Ascent logic**: given a PCF, find the CMF it belongs to and generate related formulas.
- [ ] **Automated proof sketches**: generate symbolic proofs or proof obligations for discovered formulas.
- [ ] **Paper-ready output**: auto-generate LaTeX summaries of discovered formulas with full verification.

---

## 8. Code Conventions

### File layout
```
dreamer/
├── configs/        # Dataclass-based configuration (one file per stage)
├── loading/        # Stage 1: DB, formatters, JSON serialization
├── extraction/     # Stage 2: hyperplanes, shards, samplers
├── analysis/       # Stage 3: analyzers, prioritization
├── search/         # Stage 4: searchers, search methods
├── system/         # System orchestrator (System.run)
└── utils/          # Shared: constants, schemes, storage, logging, types
```

### Naming
- **Modules** (`*_mod.py`): orchestrate a stage, implement `execute()`.
- **Methods** (e.g., `serial_searcher.py`): the algorithmic core, called by modules.
- **Schemes** (`*_scheme.py`): abstract base classes defining the interface.
- **Formatters** (`*_fmt.py`): JSON-serializable wrappers around `ramanujantools` objects.

### Adding a new module
1. Create a class inheriting from the relevant `*Scheme` (e.g., `SearcherModScheme`).
2. Implement the required methods (`execute()`, etc.).
3. Place the method in `dreamer/<stage>/methods/<name>/` and the module in `dreamer/<stage>/searchers/<name>/`.
4. Add tests in `tests/test_<name>.py`.
5. Register it by importing it in the stage's `__init__.py`.

### Testing rules
- Framework: `pytest`. Tests live in `tests/`.
- Every public function must have at least one test (see `COVERAGE_POLICY.md`).
- Mathematical tests must use `mpmath` with ≥ 100 digits of precision.
- Run: `pytest tests/ -v`

---

## 9. Relationship to Other Repos

| Repo | Role | How Dreamer Uses It |
|:-----|:-----|:--------------------|
| [`ramanujantools`](https://github.com/RamanujanMachine/ramanujantools) | Core math library: CMF, PCF, Matrix, Position, Limit | Primary dependency — all CMF operations go through it |
| [`LIReC`](https://github.com/RamanujanMachine/LIReC) | Library of Integer Relations and Constants | Used for constant identification in analysis/search |
| [`RamanujanMachine`](https://github.com/RamanujanMachine/RamanujanMachine) | Original discovery algorithms (MITM-RF, ESMA) | Reference implementations; not a direct dependency |
| [`euler2ai`](https://github.com/RamanujanMachine/euler2ai) | Formula harvesting from arXiv | Future integration for seeding inspiration functions |

---

## 10. Decision Log

Record important architectural decisions here so future contributors understand
**why**, not just **what**.

| Date | Decision | Rationale |
|:-----|:---------|:----------|
| 2024 | Use `ramanujantools` as the CMF engine, not a custom implementation | Avoid duplication; leverage tested group library |
| 2024 | SQLite for local DB | Simplicity; no server needed for single-user runs |
| 2024 | Modular stage architecture with abstract schemes | Allow independent development of new analyzers/searchers |
| 2025 | `IDENTIFY_THRESHOLD = -1` means "accept all shards" | Analysis stage filters shards; -1 disables filtering for exploratory runs |
| 2026-04 | Fix `Constant.value_mpmath` infinite recursion | `@cached_property` was calling itself; use `_explicit_mpmath` backing field |
| 2026-04 | Fix `MeijerG.__init__` super() call argument order | `use_inv_t` was being passed as `shifts` positional arg |
| 2026-04 | Fix `SearcherModV1.execute()` empty return | `dms` dict was never populated with search results |

---

## 11. How to Use This Document

- **Before starting any task**: re-read sections 3–6 to understand the pipeline, objects, invariants, and limitations.
- **Before proposing an architecture change**: check section 10 (Decision Log) — someone may have already considered and rejected your idea.
- **After completing a significant change**: update sections 6, 7, or 10 as appropriate.
- **When unsure what to work on**: consult section 7 (Development Priorities).

This document is **living**. If something is wrong or missing, fix it in the
same PR as the related code change.


## 12. Architecture Overview
This is a modular pipeline system for discovering polynomial continued fractions (PCFs) of mathematical constants via Conservative Matrix Fields (CMFs). The four-stage pipeline (Loading → Extraction → Analysis → Search) processes constants through CMF parameter spaces, partitioning into "shards" for parallelizable search. See `SYSTEM_SPEC.md` for full details.

Key components:
- `dreamer/system/system.py`: Orchestrates the pipeline via `System.run()`.
- `dreamer/configs/`: Dataclass-based configs for each stage (e.g., `analysis.py` for `IDENTIFY_THRESHOLD`).
- `dreamer/loading/`, `extraction/`, `analysis/`, `search/`: Modular stages with swappable implementations.

Data flows: `Constant` → `Dict[Constant, List[ShiftCMF]]` → `Dict[Constant, List[Shard]]` → prioritized shards → `Dict[Searchable, DataManager]` with discovered PCFs.

Reference: `SYSTEM_SPEC.md` (canonical), `README.md` (usage), `pyproject.toml` (deps), `DEFINITION_OF_DONE.md` (completion criteria), `COVERAGE_POLICY.md` (testing).
