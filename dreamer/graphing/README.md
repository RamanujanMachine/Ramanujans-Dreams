# Graphing — `dreamer/graphing/`

> **What:** the figure/table renderer invoked at the end of the
> [post-process stage](../post_process/README.md). It is **not** a pipeline stage
> of its own — it runs only when at least one graph is enabled in `config.graph`.
> **Output:** figures and tables under `config.system.EXPORT_GRAPHS`.

For where post-processing sits in the pipeline, see the
[main README](../../README.md#system-structure).

---

## What this produces

Reading the per-shard JSONL records that search/post-process already wrote, the
grapher can emit three kinds of artefact (each off by default):

| `config.graph` toggle | Output | Cost |
|-----------------------|--------|------|
| `PLOT_BEST_DELTA_SEQUENCE` | δ vs. step for the **best** trajectory of each `(CMF, constant)`, over the first `DELTA_SEQUENCE_DEPTH` steps. | The only graph that walks a trajectory (just the single best one). |
| `PLOT_DELTA_HISTOGRAMS` | Histogram of δ across trajectories, one per shard and one per CMF. | Cheap — reads stored δ only. |
| `WRITE_BUMPINESS_TABLE` | Per-shard "how non-smooth is the δ field" table (`bumpiness.csv` + `.md`). | Cheap — reads stored values. |

### The bumpiness table

Quantifies how rough each shard's δ landscape is, with two complementary columns:

- **Spatial roughness** — a density-robust **empirical semivariogram** of δ over
  *angular* direction-distance. The headline number is `relative_nugget` ∈ [0, 1]:
  ≈ 1 means a "needle" (δ jumps between nearby directions), ≈ 0 means a smooth
  field. A semivariogram is used (rather than k-nearest-neighbours) because the
  samplers cluster trajectories non-uniformly, which would bias a k-NN estimate.
- **Convergence wobble** — the median per-trajectory **total variation** of the
  stored `delta_sequence` (how much δ oscillates as the walk deepens). This is
  `NaN` unless `delta_sequence` was stored as a Tier-3 attribute.

The semivariogram lag binning and pair-sampling cap are controlled by
`config.graph.VARIOGRAM_LAG_BINS` and `VARIOGRAM_MAX_PAIRS`.

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `grapher.py` | `Grapher` — the entry point the post-process stage calls; routes the enabled graph kinds and handles I/O. |
| `plots.py` | The matplotlib rendering of the δ-sequence plots and histograms. |
| `bumpiness.py` | The semivariogram + total-variation roughness computation behind the bumpiness table. |

---

## Key configurations

`config.graph.*` (toggles + parameters) and `config.system.EXPORT_GRAPHS`
(output directory). See the [configuration index](../configs/README.md#graph).
