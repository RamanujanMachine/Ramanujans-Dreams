# runmon — live monitoring for long multi-process Python runs

`runmon` records what a long run is doing — stage and progress, every process's CPU
and memory against the run's allocation, important events, hangs, and the batch
job's remaining walltime — **without taking a core from the computation**: all
monitoring work happens in one daemon thread of the main process that sleeps
between short jobs (milliseconds of CPU per tick).

It is generic: no imports from any host project. Ramanujan's Dreams is one user.

## What a run leaves behind

In the run directory:

| File | Content |
|---|---|
| `events.jsonl` | One JSON event per line — searchable with `grep` / `jq`, replayable. |
| `timeline.jsonl` | One line per minute: mean and max of the 1 s CPU / memory samples (see below). |
| `events.w<pid>.jsonl` | Each worker process's own events (merged into `events.jsonl` by the hub). |
| `status.json` | The latest full snapshot, replaced atomically every few seconds. |
| `crash.txt` | Tracebacks of hard crashes (segfaults), via `faulthandler`. |

Event fields: `v` (schema), `ts`, `seq`, `pid`, `role`, `level`, `kind`, `msg`,
`ctx.task` (where the run was), `f` (structured fields). Kinds include `log`,
`run_start` / `run_end`, `watchdog`, `stall`, `over_allocation`, `walltime`,
`walltime_warning`, `walltime_grace`, `heartbeat`, `memory_pressure` / `memory_recovered`
(machine memory or the run's allocation nearly used up), `disk_low` / `disk_recovered`
(free space where the run directory lives), `signal_received`, plus whatever the program
emits (`pool_created`, `worker_died`, …).

**Timeline.** The hub samples the process tree every second, but writing every sample
would be a lot of data, and one snapshot per heartbeat loses the spikes in between. So
every minute (`HubConfig.timeline_s`) it writes one record to `timeline.jsonl`: the
**mean** (the shape of the curve) and **max** (the spikes) of the window's samples for
CPU, memory, process count and pool workers, the lowest machine-available memory, the
same per role (`"eval pool worker"`, `main`, …), pool tasks finished and their run
time, processes started (worker restarts), the active stage and the free disk space.
The browser view plots it, so a reopened run shows the curve the live view showed
(`runmon web` falls back to heartbeats for runs recorded without a timeline). A 14-day
run is about 20 000 short lines.

**Peak utilization and jitter.** Each timeline record also holds `cpu_util_max` and
`mem_util_max` (the highest use in the window as a fraction of the run's allocation),
and, for CPU and memory, `*_std` and `*_jitter`. *Jitter* is the mean absolute change
between consecutive 1 s samples divided by the mean: 0 is a flat line, 0.5 means a
typical step is half the level. It measures how spiky a signal is, whereas the standard
deviation also counts slow drifts. At the end of the run a `resource_summary` event
(and the `summary` block of `status.json`, shown as a text line in the browser view)
gives the same figures for the whole run: peak and mean utilization, jitter, spread.

```bash
jq -c 'select(.level == "WARNING" or .level == "ERROR")' events.jsonl
jq -c 'select(.kind == "worker_died") | {ts, msg, pool: .f.pool}' events.jsonl
```

## Using it

```python
import runmon

runmon.start("runs/my-run", cores=16, memory_mb=64_000)   # main process

with runmon.task("search", total=len(shards), kind="stage") as stage:
    for shard in shards:
        with runmon.task(f"shard {shard}", kind="shard") as node:
            for step in loop():
                node.estimate(fraction, T=temperature)       # loops with no fixed count
        stage.advance()

runmon.event("pool_created", "eval pool with 8 workers", workers=8)
with runmon.watch("walk", 300, detail=lambda: f"trajectory {t}"):
    walk(t)                                                # warns while it overruns

runmon.stop()          # also automatic at exit / SIGTERM / SIGINT
```

Worker processes call `runmon.init_worker(run_dir)` (e.g. in a pool initializer);
their `runmon.event(...)` and `runmon.watch(...)` then go to their own event file.
Every call is a no-op when monitoring is off.

Pools report themselves with `runmon.register_pool(pool)`: any object with
`monitor_info() -> {"name", "workers", "pids", ...}` (held weakly).

### Progress

A node's completed fraction is its explicit estimate if it has one, else
`(done + open child's fraction) / total`. Updates only assign attributes, about a
microsecond each, so they are safe in tight loops. Closing a child does not
advance its parent; the caller counts.

`runmon.estimators` covers loops that end when the first of several budgets runs
out (`ClockEstimator`). It also handles simulated-annealing schedules:
`schedule_inverse`, `sa_k_end`, `sa_estimator`. The temperature is a fixed function
of the accepted-move count, so the estimate uses the budgets: accepted moves up to
`min(max_iters, k at Tmin)`, and total steps.

### Allocation, not machine

Pass `cores` / `memory_mb` (the job's request). Utilization is reported against
them, and an `over_allocation` alert fires when the run uses more. This matters on
batch nodes that don't confine jobs. For example, a 4-core PBS job on an 80-core
node can start 80 workers.

### Batch-job walltime

`runmon.walltime.detect()` finds the job's end time. It checks, in order:
`RUNMON_WALLTIME_END`, Torque `PBS_WALLTIME`, PBS Pro `qstat -f $PBS_JOBID`, then
Slurm `squeue`. The hub logs warnings 60, 15 and 5 minutes before the end. In the
grace period (default 5 minutes) it runs the `runmon.on_grace(callback)` hooks and
flushes everything to disk. PBS Pro sends no warning before a walltime kill, so the
record has to be complete *before* the deadline.

### How a run ends

Normal exit, `SIGTERM`, `SIGINT`, `SIGHUP` and `SIGUSR1` all write a final
`run_end{reason}` event and a final `status.json`. On SIGTERM the program's
`finally` blocks still run, and the exit code is 143. `SIGKILL` cannot be caught,
so events are flushed as they are written; `ERROR` events are written synchronously.

## Watching and searching a run

```bash
runmon watch runs/latest                      # local run: live server, else status.json
runmon watch ~/work/myrun/runs/latest --ssh me@login.cluster   # run in a batch job on a cluster
runmon watch node17:41237 --token …           # a server directly (token is in monitor.json)
runmon query runs/latest --level WARNING --since 2h
runmon query ~/work/myrun/runs/latest --ssh me@login.cluster --kind worker_died
runmon web runs/latest                        # the same live view in your browser (localhost)
runmon web ~/work/myrun/runs/latest --ssh me@login.cluster
runmon demo                                   # a simulated run, to try it out
```

**Browser view.** `runmon web` starts a small HTTP server on your own machine, on
`http://127.0.0.1:8765/` (or the next free port), and opens the page. It takes the
same targets as `watch`, so it also works for a cluster run over `--ssh`. The page:

- **Status:** stages, alerts, and tiles for CPU in cores against the allocation,
  memory against the allocation, walltime left and processes.
- **Progress tree:** estimates are striped and marked `~`.
- **Charts:** CPU and memory, with the allocation as a dashed line. History covers the
  whole run, from `timeline.jsonl` (one mean / max record per minute).
- **Stage markers:** a coloured vertical line with a label where each stage starts, on
  both charts. Every `kind="stage"` progress node writes `stage_start` / `stage_end`
  events.
- **Tables:** processes and pools.
- **Events:** filterable, with a "search full log" button that runs `runmon query`.

The page has no external dependencies and follows the system's light or dark theme
(override with `?theme=light` or `?theme=dark`). The server is read-only, listens on
the loopback address only, and refuses requests whose `Host` is not local.

**Live server.** Start with `server=True`. The hub thread then also serves clients over
TCP and writes `monitor.json` (mode 0600) with the host, port and a random token. A
client must send the token first. It then receives a snapshot every second and every
event as it happens, and `bye` when the run ends. The server is read-only. Each client
has a bounded buffer, so a slow client cannot stall the run. `host="auto"` binds
127.0.0.1, except inside a PBS/Slurm job, where it binds all interfaces and advertises
the node's name. Those runs need that, because nothing but the login node can reach
a compute node.

**Watching a cluster run over SSH.** `--ssh USER@LOGIN`:

1. Reads the run's `monitor.json` through the login node.
2. Opens `ssh -N -L 127.0.0.1:<free port>:<node>:<port> USER@LOGIN`. The **login node**
   forwards to the compute node, so you never SSH into the node.
3. Connects to that forwarded port.

If the run has no live server, or the tunnel fails, it streams `status.json` over the
same SSH connection instead. Your `~/.ssh/config` (keys, ProxyJump, ControlMaster)
applies.

## Status

Done: the core (events, progress, estimators, sampler, watchdog, walltime,
lifecycle, hub), the live server, the terminal client (local, `host:port`, `--ssh`),
`runmon query`, the simulated run, and the Ramanujan's Dreams wiring
(`dreamer/utils/monitoring.py`). Later: a web view, either served by the same hub or
as a client of the same protocol.
