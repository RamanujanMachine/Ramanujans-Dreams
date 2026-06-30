# Post-process stage — `dreamer/post_process/`

> **Goal:** after search finishes, add the **most expensive (Tier-3)
> attributes** to selected trajectories, and optionally render graphs/tables.
> **Input:** the search priorities + the per-shard JSONL files search wrote.
> **Output:** appended *patch* records (never a rewrite) + optional figures.

This stage is **optional and off by default** — wire it with
`System(post_processor=Tier3PostProcessModV1)`. For where it sits in the
pipeline, see the [main README](../../README.md#system-structure).

---

## What this stage does

Some attributes are too expensive to compute for every trajectory during search
(asymptotics, full δ-sequences, …). The post-process stage runs **once after
search**, reads the existing JSONL records, computes only what's missing for the
trajectories you select, and **appends patch records** — it never rewrites your
data, so re-running is safe and idempotent.

It has **two independent jobs**, each off until you enable it:

1. **Tier-3 attributes** — `config.post_process.TIER3_ATTRIBUTES`. A tuple where
   each entry is either a bare attribute name (always computed) or an
   `(attribute, predicate)` pair so the expensive attribute is computed *only*
   for trajectories the predicate accepts. Predicates include `if_identified`,
   `max_degree below/above N`, and the **`top N highest/lowest <metric> in
   shard|cmf`** selectors (which rank using values *already stored* in the
   JSONL — no re-walk).

2. **Graphing** — `config.graph`, implemented in
   [`dreamer/graphing/`](../graphing/README.md): δ-sequence plots, δ histograms,
   and the "bumpiness" roughness table. Output goes to
   `config.system.EXPORT_GRAPHS`.

An empty `TIER3_ATTRIBUTES` **and** a disabled `config.graph` make the whole
stage a no-op (no files read, no subprocesses spawned).

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `tier3_post_process_mod.py` | `Tier3PostProcessModV1` — the stage orchestrator: the two-phase, CMF-grouped Tier-3 producer + the call into the graphing stage. |

The actual figure/table rendering lives in the sibling
[`dreamer/graphing/`](../graphing/) package.

---

## Key configurations

- `config.post_process.TIER3_ATTRIBUTES` — what to compute and for which
  trajectories (the predicate/selector grammar).
- `config.graph.*` — which graphs to produce and their parameters.
- `config.system.EXPORT_GRAPHS` — where figures/tables are written.

See the [configuration index](../configs/README.md#post_process) for the full,
annotated grammar and metric list.

> **Gotcha:** the `top N … <metric>` selectors can only rank on a metric that is
> **already stored** in the JSONL. To rank on something other than `delta`, make
> sure that metric is computed first (e.g. add `eigenvalues` to
> `config.search.TIER2_ATTRIBUTES`, or add the metric as a bare Tier-3 attribute).

---

## Extending this stage

Subclass `PostProcessModScheme` and implement `execute()` (no return value); it
receives the search `priorities` and is free to read the JSONL, compute, and
append patch records. `Tier3PostProcessModV1` is the reference.
