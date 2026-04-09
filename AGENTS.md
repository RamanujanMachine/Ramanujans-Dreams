# AGENTS.md - AI Coding Agent Guidelines for Ramanujan's Dreams

## Architecture Overview
This is a modular pipeline system for discovering polynomial continued fractions (PCFs) of mathematical constants via Conservative Matrix Fields (CMFs). The four-stage pipeline (Loading → Extraction → Analysis → Search) processes constants through CMF parameter spaces, partitioning into "shards" for parallelizable search. See `SYSTEM_SPEC.md` for full details.

Key components:
- `dreamer/system/system.py`: Orchestrates the pipeline via `System.run()`.
- `dreamer/configs/`: Dataclass-based configs for each stage (e.g., `analysis.py` for `IDENTIFY_THRESHOLD`).
- `dreamer/loading/`, `extraction/`, `analysis/`, `search/`: Modular stages with swappable implementations.

Data flows: `Constant` → `Dict[Constant, List[ShiftCMF]]` → `Dict[Constant, List[Shard]]` → prioritized shards → `Dict[Searchable, DataManager]` with discovered PCFs.

## Critical Workflows
- **Running the system**: Instantiate `System` with stage modules, call `run(constants=[mpmath.log(2)])`. Config via `from dreamer import config; config.configure(analysis={'IDENTIFY_THRESHOLD': -1})`.
- **Testing**: `pytest tests/ -v` (requires `mpmath.mp.dps = 150+` for math tests). Every public function needs tests per `COVERAGE_POLICY.md`.
- **Precision handling**: Always use `mpmath.mpf` for computations; set `mpmath.mp.dps = 2 * desired_digits`. Never use Python `float`.
- **PR delivery**: Push branches (e.g., `feat/new-searcher`) to remote, create PRs. No local-only commits. See `DEFINITION_OF_DONE.md` §6.

## Project-Specific Patterns
- **Module naming**: Orchestrators end in `_mod.py` (e.g., `SearcherModV1`), methods in `methods/` (e.g., `serial_searcher.py`), schemes in `*_scheme.py` for interfaces.
- **Formatter pattern**: JSON-serializable wrappers (e.g., `pFq`, `MeijerG`) around `ramanujantools` objects; must implement `from_json_obj`/`to_json_obj`.
- **Shard boundaries**: Strict inequalities (`Ax < b`); points on boundary are outside. Use `Hyperplane` for zero/pole enumeration.
- **Configuration access**: `config.<category>.<key>` (e.g., `config.search.NUM_TRAJECTORIES_FROM_DIM` as `lambda dim: 10**dim`).
- **Logging**: Use `dreamer.utils.logger` for structured output.

## Integration Points
- **ramanujantools**: Primary CMF/PCF library; all math ops via `import ramanujantools as rmt`.
- **LIReC**: For constant identification; Windows users need WSL due to `cysignals`/`fpylll` dependencies.
- **Database**: SQLite via `peewee` models in `dreamer.loading.databases`; pickle files for caching CMFs/shards.
- **External tools**: Mathematica/RISC for proofs; access via team request.

## Conventions Differing from Standards
- **No float**: Math code uses `mpmath` exclusively; guard with `assert mpmath.mp.dps >= 100`.
- **Registry pattern**: Constants deduplicated globally; import from `dreamer.constants` or define via `Constant(sympy_expr, mpmath_value)`.
- **Modular substitution**: Stages are interfaces (`*Scheme` subclasses); swap e.g., `AnalyzerModV1` without pipeline changes.
- **Trajectory sampling**: `lambda dim: 10**dim` trajectories; high-dim uniformity degrades—bias toward thin cones.
- **Verification threshold**: `IDENTIFY_THRESHOLD = -1` disables filtering for exploration; positive values filter shards by convergence fraction.

Reference: `SYSTEM_SPEC.md` (canonical), `README.md` (usage), `pyproject.toml` (deps), `DEFINITION_OF_DONE.md` (completion criteria).
