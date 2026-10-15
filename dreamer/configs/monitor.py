from dataclasses import dataclass, field
from .configurable import Configurable


@dataclass
class MonitorConfig(Configurable):
    ENABLED: bool = field(
        default=True,
        metadata={"description": (
            "Run the live run monitor (runmon): a structured event log, per-process CPU/memory, "
            "stage progress, hang watchdogs and batch-job walltime, maintained by one sleeping "
            "thread in the main process. Set False to run without it (e.g. quick local runs)."
        )},
    )
    RUN_DIR: str = field(
        default="runs",
        metadata={"description": (
            "Directory holding one sub-directory per run (events.jsonl, status.json, worker "
            "event files); 'latest' points at the newest. Watch a run with "
            "`python -m runmon watch <RUN_DIR>/latest`."
        )},
    )
    SERVER: bool = field(
        default=True,
        metadata={"description": (
            "Serve live status to `runmon watch` clients over TCP (token-protected; details in "
            "the run's monitor.json). Without it, clients read status.json instead."
        )},
    )
    HOST: str = field(
        default="auto",
        metadata={"description": (
            "Address the monitor server binds: 'auto' = 127.0.0.1 normally, all interfaces "
            "inside a PBS/Slurm job (so the cluster login node can forward to the compute node); "
            "or an explicit address."
        )},
    )
    PORT: int = field(
        default=0,
        metadata={"description": "Monitor server port (0 = any free port; the chosen one is in monitor.json)."},
    )
    EVENT_LOG_LEVEL: str = field(
        default="INFO",
        metadata={"description": (
            "Lowest level of main-process log lines copied into events.jsonl (DEBUG / INFO / "
            "WARNING). Structured events (pools, stages, watchdog, walltime) are always kept."
        )},
    )
    WORKER_EVENT_LOG_LEVEL: str = field(
        default="WARNING",
        metadata={"description": (
            "Lowest level of worker-process log lines copied into the run's event log."
        )},
    )
    SAMPLE_SECONDS: float = field(
        default=1.0,
        metadata={"description": "Seconds between CPU / memory samples of the run's processes."},
    )
    HEARTBEAT_SECONDS: float = field(
        default=30.0,
        metadata={"description": "Seconds between compact status snapshots written into events.jsonl (for replay)."},
    )
    STALL_SECONDS: float = field(
        default=1800.0,
        metadata={"description": "Warn when no progress has been reported for this many seconds."},
    )
    WALLTIME_GRACE_SECONDS: float = field(
        default=300.0,
        metadata={"description": (
            "Inside a batch job: this many seconds before the walltime deadline, flush every "
            "log and status file to disk (PBS kills jobs without warning)."
        )},
    )


monitor_config: MonitorConfig = MonitorConfig()
