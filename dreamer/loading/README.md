# Loading stage — `dreamer/loading/`

> **Goal:** turn each target constant into one or more **CMFs to search**.
> **Input:** the `function_sources` you pass to `System` (formatters, database
> modules, or export paths) + the run's constants.
> **Output:** `Dict[Constant, List[CMFData]]` — for every constant, the list of
> CMFs (each a `ramanujantools` CMF plus an integer coordinate *shift*) in which
> the pipeline will look for formulas.

For where this stage sits in the whole pipeline, see the
[main README](../../README.md#system-structure).

---

## What this stage does

A **CMF** (Continued Matrix Field) is the mathematical object this whole system
searches. The loading stage is the only place that *creates* CMFs; every later
stage just consumes them. You never build a CMF matrix by hand — instead you
name an **inspiration function** (e.g. a hypergeometric ${}_pF_q$), and a
**formatter** turns it into the corresponding CMF.

Three kinds of source can be mixed freely in `function_sources`:

| Source kind | What it is | Example |
|-------------|------------|---------|
| **Formatter** | A live inspiration function turned into a CMF. | `pFq(log(2), 2, 1, -1)` |
| **Database module** | A `DBModScheme` that retrieves/stores CMF descriptions in a DB. | `BasicDBMod(json_path=...)` |
| **Path string** | A directory of previously-exported CMFs to reload (no pickles). | `"./CMFs/pi"` |

A `CMFData` bundles the CMF, an integer (or exact `sp.Rational`) **shift** that
moves the search's start point, and metadata. The same `CMFData` can be attached
to several constants at once, so a CMF shared by `pi` and `log(2)` is loaded once.

### Formatters and the no-pickle contract

`Formatter` (in `funcs/formatter.py`) is the bridge between a live CMF and a
**JSON-serialisable** description of it. Every formatter:

- normalises the constant(s) and validates the shifts (a `float` or irrational
  shift is rejected — shifts must be exact integers or `sp.Rational`);
- builds a deterministic `cmf_name` that **excludes the constant**, so the same
  CMF keeps a stable id even when you later search it for a different constant;
- can round-trip to/from JSON, which is what lets the pipeline export CMFs and
  reload them later **without pickle files**.

Concrete formatters live in `funcs/` — currently `pFq` (hypergeometric),
`MeijerG`, and `BaseCMF` (wrap an arbitrary `ramanujantools` CMF directly).

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `funcs/formatter.py` | `Formatter` ABC — the CMF ⇄ JSON bridge and self-registry. |
| `funcs/pFq_fmt.py` | `pFq` formatter — hypergeometric ${}_pF_q$ inspiration functions. |
| `funcs/meijerG_fmt.py` | `MeijerG` formatter — Meijer-G inspiration functions. |
| `funcs/base_cmf.py` | `BaseCMF` — wrap an existing `ramanujantools` CMF with no special structure. |
| `databases/db_v1/` | A reference database module: `db.py` (the store), `db_mod.py` (`BasicDBMod` / `DBModScheme`), `config.py`. |
| `config.py` | Loading-stage annotation constants (DB command/data tags). |
| `errors.py` | Loading-specific exceptions. |

---

## Key configurations

Loading has no dedicated config *category*; it is driven by **what you pass to
`System(function_sources=...)`** and by the database config:

- `config.database.USAGE` — retrieve / store / store-then-retrieve mode for DB sources.
- `config.system.EXPORT_CMFS` — directory the loaded CMFs are exported to (the
  JSON artefact that later enables extractor-free reloads).

See the [configuration index](../configs/README.md) for the full list.

---

## Extending this stage

**Add a new inspiration function** → subclass `Formatter` in a new file under
`funcs/` and implement its CMF construction + JSON round-trip; subclasses
auto-register, so `Formatter.from_json_obj` can rebuild them. Use
`funcs/pFq_fmt.py` as the reference.

**Add a new CMF source** (e.g. a different database backend) → subclass
`DBModScheme` and implement `execute(constants)` to return
`Dict[Constant, List[CMFData]]`. Pass an instance in `function_sources`.
