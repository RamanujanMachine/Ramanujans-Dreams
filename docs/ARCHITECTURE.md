# Ramanujan's Dreams — Architecture

How a run flows through the system: the stages, the processes that do the work,
what one trajectory's computation involves, when stored work is reused, and how
the pools survive failures. The diagrams are [Mermaid](https://mermaid.js.org/):
GitHub renders them in place, and `npx -p @mermaid-js/mermaid-cli mmdc -i ARCHITECTURE.md -o arch.svg`
(or `-o arch.png`) exports them as images.

Contents:
1. [Pipeline](#1-pipeline) — stages, what each reads and writes
2. [Processes](#2-processes-and-pools) — main process, stage pools, eval pools, extraction pools
3. [One trajectory](#3-one-trajectory) — walk, identification, δ, Tier-2, rows
4. [Reuse](#4-reuse-what-is-computed-again-and-what-is-not) — fingerprints and reuse counts
5. [Search](#5-search-hedgehog-and-the-optimisers) — hedgehog vs. the optimisers
6. [Resilience and memory](#6-resilience-and-memory) — dead workers, memory-aware sizing

Terms used throughout:

| Term | Meaning |
|---|---|
| **CMF** | Conservative matrix field; the input object a constant is searched in. |
| **Shard** | One cell of a CMF's hyperplane arrangement; trajectories inside a shard share behaviour. |
| **Trajectory** | A direction + start point in a shard; walking it yields a sequence converging to a constant. |
| **Tier-1** | δ, identified, p/q vectors, projection column — computed for every trajectory. |
| **Tier-2** | Extra attributes (`search.TIER2_ATTRIBUTES`) computed during search. |
| **Tier-3** | Expensive attributes (`post_process.TIER3_ATTRIBUTES`) computed after search. |
| **Row** | One `(trajectory, constant)` record in `<shard_id>.jsonl` — the shared store. |
| **Stage pool** | A process pool per stage; one task = one trajectory. |
| **Eval pool** | A process pool per shard inside an optimiser; one task = one candidate direction. |

---

## 1. Pipeline

The stages run one after another (`System.run`); the parallelism is inside each
stage. Every stage reads and appends to the same per-shard store, so a later
stage — or a later run — reuses what an earlier one computed.

```mermaid
flowchart TB
    U(["constants + config"]) --> L

    subgraph PIPE["System.run — stages run in sequence"]
        direction TB
        L["<b>1. Loading</b><br/>constant → CMF formatters<br/>(pFq, MeijerG, …)"]
        E["<b>2. Extraction</b><br/>CMF → shards<br/>heuristic ray shooting or exact enumeration"]
        A["<b>3. Analysis</b><br/>sample trajectories per shard → Tier-1<br/>rank shards by the objective, drop unidentified"]
        S["<b>4. Search</b><br/>deep search in the ranked shards<br/>Tier-1 for new trajectories + Tier-2"]
        P["<b>5. Post-process</b> (optional)<br/>Tier-3 for selected rows<br/>+ graphs / tables"]
        M["<b>Summary</b><br/>summary.md of the run"]
        L --> E --> A --> S --> P --> M
    end

    subgraph STORE["On disk (config.system.EXPORT_*)"]
        direction TB
        C[("EXPORT_CMFS<br/>cmfs.jsonl, shards, formatters")]
        R[("EXPORT_ANALYSIS_PRIORITIES<br/>ranked shard ids per constant")]
        J[("EXPORT_SEARCH_RESULTS<br/>shard_id.jsonl — one row per<br/>(trajectory, constant)")]
        G[("EXPORT_GRAPHS<br/>figures, tables")]
    end

    E -- "write shards" --> C
    C -. "reload, skip extraction" .-> E
    A -- "append Tier-1 rows" --> J
    A -- "write ranking" --> R
    J -. "reuse rows" .-> A
    J -. "reuse rows" .-> S
    S -- "append rows / Tier-2 patches" --> J
    J -. "reuse rows" .-> P
    P -- "append Tier-3 patches" --> J
    P --> G
    J -. "read" .-> M
```

---

## 2. Processes and pools

Only the main process orchestrates and writes files. All heavy work runs in
process pools, and every pool is a `ResilientPool` (section 6). The two pools that
compute trajectories differ in what one task is and who waits for it.

```mermaid
flowchart LR
    subgraph MAIN["Main process"]
        direction TB
        ORCH["Stage orchestration<br/>plan: classify trajectories<br/>(reused / Tier-2 only / new)"]
        OPT["Optimiser loop<br/>SA · GA · Small Angle · GD · SPSA<br/>+ micro-climb"]
        WR["Writer<br/>BufferedJsonlWriter /<br/>direct sink — the only writer"]
        CACHE["Caches<br/>stored rows, handler LRU,<br/>shard p/q cache"]
    end

    subgraph SP["Stage pool — one per stage"]
        direction TB
        SPW["workers 1…N<br/>hold every CMF<br/>task = one trajectory"]
    end

    subgraph EP["Eval pool — one per shard"]
        direction TB
        EPW["workers 1…N<br/>hold this shard + start<br/>task = one candidate direction"]
    end

    subgraph XP["Extraction pools"]
        direction TB
        XPW["heuristic shooting ·<br/>witness refinement ·<br/>exact extraction · cell enumeration"]
    end

    PQ[("Shared p/q log<br/>(Manager, per shard)")]

    ORCH -- "Tier1Task (light shard copy)" --> SPW
    SPW -- "rows + patches, in task order" --> ORCH
    OPT -- "batch of directions per step" --> EPW
    EPW -- "rows (δ, Tier-2)" --> OPT
    ORCH --> WR
    OPT --> WR
    SPW -. "share identified p/q" .- PQ
    EPW -. "share identified p/q" .- PQ
    MAIN -- "extraction tasks" --> XPW
```

| | Stage pool (`StagePool`) | Eval pool (`make_eval_pool`) |
|---|---|---|
| Used by | analysis, hedgehog search, Tier-3 | the optimisers and the micro-climb |
| Lifetime | one stage, all shards | one shard, all its constants |
| One task | one trajectory: Tier-1 rows + Tier-2/3 for its constants | one candidate direction of an optimiser step |
| Waits | the stage streams all trajectories through it | the optimiser waits per step (SA may pre-walk the next step) |

Workers start with `forkserver` on Linux and `spawn` on macOS
(`system.POOL_START_METHOD`), and their number is fitted to cores and memory.

---

## 3. One trajectory

What a stage-pool task (or an eval-pool walk) does for one trajectory. The walk
products are exact rational matrices; the constant is evaluated with FLINT arb.

```mermaid
flowchart TB
    T(["trajectory = direction + start"]) --> M["Trajectory matrix M(n)<br/>cmf.trajectory_matrix — symbolic, entries polynomial in n"]
    M --> D["Walk depth D = DEPTH_FROM_TRAJECTORY_LEN<br/>(1000 – 1500 steps)"]
    D --> W1["Walk to min(IDENTIFY_DEPTH, D)<br/>stores W(d) = M(0)···M(d−1) at every depth Tier-1 needs on the way<br/>(0.9D, 0.95D, D and the step before each)"]
    W1 --> ID{"Identify p, q<br/>shard p/q cache, else LIReC<br/>(local PSLQ on 1000-digit values)"}
    ID -- "no relation" --> X["δ = −∞ · identified = false<br/>(retry once at depth D, continuing the walk)"]
    ID -- "p, q found" --> CV{"Converges?<br/>estimates at 0.9D, 0.95D, D agree<br/>(stored products — no new walk)"}
    CV -- "no" --> X
    CV -- "yes" --> DL["δ = −1 − log(err) / log(q),  err = abs(p/q − L)<br/>L = constant via arb at 10k digits, doubled until<br/>err ≥ 2⁴⁰ units in the last place of L<br/>(cap MAX_CONSTANT_RESOLUTION = 10⁶ digits)"]
    DL --> T2["Tier-2 attributes (search stage)<br/>eigenvalues, spectral gap, convergence rate, …"]
    X --> ROW
    T2 --> ROW["Row per (trajectory, constant)<br/>values + per-attribute fingerprints"]
    ROW --> WRT[("shard_id.jsonl<br/>via the main-process writer")]
    ROW -. "after a task ≥ 1 s" .-> REL["release memory<br/>clear sympy's cache, drop walk products"]
```

---

## 4. Reuse: what is computed again and what is not

Every stored value carries a **fingerprint** of the configuration it depends on
(its own knobs plus its dependencies' fingerprints). A value is reused while its
fingerprint matches; `system.REUSE_POLICY` can also keep everything (`keep`) or
ignore earlier runs (`recompute`).

```mermaid
flowchart TB
    Q(["trajectory × constant to evaluate"]) --> H{"Stored row for it?"}
    H -- "no" --> NEW["<b>new</b><br/>walk + Tier-1 (+ Tier-2)"]
    H -- "yes" --> F{"Tier-1 fingerprint matches?"}
    F -- "no (config changed)" --> STALE["<b>new</b> — replaces a stale row"]
    F -- "yes" --> T2{"All requested Tier-2/3<br/>attributes fresh?"}
    T2 -- "yes" --> RE["<b>reused</b><br/>no walk, no LIReC"]
    T2 -- "no" --> PATCH["<b>Tier-2 only</b><br/>missing attributes computed from the<br/>stored Tier-1 values (seeded), appended as a patch"]
```

Each stage logs how its work was served:

```text
Shard pFq_2_1_…: 120 trajectories → 80 reused · 10 Tier-2 only · 30 new            (analysis, hedgehog)
Simulated Annealing — shard pFq_2_1_…, log-2: 412 evaluations →
    150 reused · 190 revisited · 72 new (4 replacing stale rows)                    (optimisers, micro-climb)
```

`revisited` = computed earlier in the same run (an optimiser returned to a ray).

---

## 5. Search: hedgehog and the optimisers

```mermaid
flowchart TB
    PR(["ranked shards per constant<br/>(from analysis)"]) --> CH{"System(searcher=…)"}
    CH -- "SearcherModV1" --> HH["<b>Hedgehog</b><br/>sample many directions per shard<br/>→ plan (reused / Tier-2 only / new)<br/>→ stage pool → writer"]
    CH -- "an optimiser module" --> PS["Per shard: flatland geometry (LLL),<br/>interior start, <b>eval pool</b>, stored rows"]
    PS --> PC["Per constant: seed from a reservoir<br/>of identified directions"]
    PC --> LOOP["Optimiser step<br/>SA neighbours · GA children ·<br/>SPSA / GD probes · Small-Angle perturbations"]
    LOOP --> EB["evaluate_batch<br/>stored row → score · cached walk → recompute ·<br/>else walk on the eval pool"]
    EB --> LOOP
    EB -. "SA: idle workers pre-walk<br/>the next step's candidates" .-> LA["look-ahead"]
    LOOP -- "done" --> MC["Micro-climb finalisation<br/>certify the best ray is a lattice local maximum"]
    MC --> LOG["log reuse counts per (shard, constant)"]
```

---

## 6. Resilience and memory

A worker killed by the out-of-memory killer used to hang `multiprocessing.Pool`
forever. `ResilientPool` notices the death, recovers, and keeps memory in check.

```mermaid
stateDiagram-v2
    [*] --> Sized: workers = min(cores, (available − reserve) / per-worker)<br/>start with 2 (POOL_SLOW_START)
    Sized --> Running
    Running --> Running: task done — record its peak memory
    Running --> Grow: warm and memory left for another<br/>worker at the measured peak
    Grow --> Running: raise the worker limit in place
    Running --> Shrink: available memory below the reserve
    Shrink --> Running: drain to half the workers
    Running --> Recover: a worker died
    Recover --> Running: half the workers,<br/>resubmit unfinished tasks
    Recover --> Isolate: same task suspected twice
    Isolate --> Running: runs alone — succeeds, or its<br/>caller gets a failure result (δ = −∞)
    Running --> InitError: worker initializer raised
    InitError --> [*]: WorkerInitError with the traceback<br/>(not retried as out-of-memory)
```

Memory per long trajectory (4F3, L2 128, depth 1000): walk ~16 MB, δ ~0 MB
(arb), peak ~0.4 GB per worker; sympy's cache is cleared after tasks ≥ 1 s so
nothing accumulates across trajectories.
