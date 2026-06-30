# Analysis stage — `dreamer/analysis/`

> **Goal:** cheaply probe every shard, keep the ones that actually produce the
> constant, and rank them so the (expensive) search stage spends its budget
> where it matters.
> **Input:** `Dict[Constant, List[Shard]]` (from extraction).
> **Output:** `Dict[Constant, List[Shard]]` — the **same shards, filtered and
> ranked** (best first).

For where this stage sits in the whole pipeline, see the
[main README](../../README.md#system-structure).

---

## What this stage does

Extraction can hand over a lot of shards, and most of them may never converge to
the target constant. Analysis is the cheap triage pass:

1. **Sample** a modest number of trajectory directions in each shard
   (`config.analysis.NUM_TRAJECTORIES_FROM_DIM` controls how many, as a function
   of CMF dimension).
2. **Compute Tier-1 attributes** for each sampled trajectory — the cheap,
   always-available values: the irrationality measure **δ**, whether the
   trajectory **identified** the constant (via LIReC), the limit, and the p/q
   convergent vectors. These are computed for *every* constant bound to the shard
   in a single shared walk.
3. **Filter**: a shard is discarded if the fraction of its sampled trajectories
   that identify the constant falls below `config.analysis.IDENTIFY_THRESHOLD`
   (`-1` disables filtering).
4. **Rank** the survivors by best δ (smaller CMF dimension breaks ties), and
   return them per constant.

### Where the results are written

Every analysed trajectory is one JSON line in a per-shard `<shard_id>.jsonl`
file. By default analysis writes into the **same** store the search stage uses
(`config.system.EXPORT_SEARCH_RESULTS`), so its Tier-1 records are already on
disk and **re-used** by the time search runs — no trajectory is walked twice.
Set `config.analysis.STORE_TRAJECTORIES_SEPARATELY = True` to keep the analysis
records in their own store (`EXPORT_ANALYSIS_RESULTS`); the search stage still
seeds its cache from there, so cross-stage reuse is preserved either way.

Because analysis and search share the same per-trajectory store and attribute
machinery, the analyzer is built on top of the search stage's `SerialSearcher`
rather than a separate engine.

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `analyzers/serial_scan/analyzer_mod.py` | `AnalyzerModV1` — the stage orchestrator: dedup shards, sample, Tier-1, filter, rank. |
| `analyzers/serial_scan/config.py` | Defaults specific to the serial-scan analyzer. |
| `analysis_methods/serial_scan_analyzer.py` | The analysis method (sampling + prioritisation logic) the module wraps. |
| `errors.py` | Analysis-specific exceptions. |

---

## Key configurations

`config.analysis.*` — `NUM_TRAJECTORIES_FROM_DIM`, `IDENTIFY_THRESHOLD`,
`SAMPLING_METHOD`, `STORE_TRAJECTORIES_SEPARATELY`, `USE_LIReC`, and the print
toggles. See the [configuration index](../configs/README.md#analysis).

---

## Extending this stage

**Add a new analysis module** → subclass `AnalyzerModScheme` and implement
`execute()` returning a ranked `Dict[Constant, List[Shard]]`; wire it as
`System(analyzers=[MyAnalyzerMod])`. You can pass several analyzers — the system
merges their rankings into a consensus order. `AnalyzerModV1` is the reference: it
samples trajectories, writes Tier-1 JSONL records, and ranks shards by best δ.

Start from the ready-to-copy skeleton in
[`examples/analysis.py`](../../examples/analysis.py), which stubs the probe-a-shard
class and the module, and matches the constructor signature `System` calls.
