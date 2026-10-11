from .configurable import Configurable
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SystemConfig(Configurable):
    # ============================== Arguments ==============================
    CONSTANTS: List[str] | str = field(
        default_factory=list,
        metadata={"description": "Registered constants to search for when no runtime constants are provided."},
    )

    # ============================== Printing and errors ==============================
    MODULE_ERROR_SHOW_TRACE: bool = field(
        default=True,
        metadata={"description": "Whether stage-level module errors include full stack traces in logs."},
    )
    TQDM_CONFIG: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Progress-bar rendering options shared by SmartTQDM calls."},
    )
    LOGGING_BUFFER_SIZE: int = field(
        default=150,
        metadata={"description": "Width used by buffered log banners and separators."},
    )

    # ============================== constant mapping ==============================
    USE_LIReC: bool = field(
        default=True,
        metadata={"description": "Enable LIReC-backed identification when available."},
    )

    # ============================== optimization objective ==============================
    OPTIMIZATION_OBJECTIVE: str = field(
        default="delta",
        metadata={"description": (
            "The trajectory attribute the WHOLE pipeline optimises for — steers "
            "both the analysis-stage shard ranking and the search-stage optimisers "
            "(identically, hence a system-level setting, not a per-stage one). Must "
            "be a numeric attribute registered in "
            "dreamer.utils.storage.optimization_objectives.OBJECTIVES, which also "
            "records whether its optimum is the highest or lowest value. 'delta' "
            "(default) reproduces the historical behaviour exactly. Identification "
            "(LIReC) is always required regardless of the objective."
        )},
    )

    def __post_init__(self):
        self.TQDM_CONFIG = {
            'bar_format': '{desc:<40}' + ' ' * 5 + '{bar} | {elapsed} {rate_fmt} ({percentage:.2f}%)',
            'ncols': 100
        }

    DEFAULT_DIR_SUFFIX: str = field(
        default='tempdir',
        metadata={"description": "Suffix that marks temporary export directories eligible for cleanup."},
    )
    EXPORT_CMFS: Optional[str] = field(
        default=None,
        metadata={"description": "Optional directory path for exporting loaded CMFs before extraction."},
    )
    EXPORT_ANALYSIS_PRIORITIES: Optional[str] = field(
        default=None,
        metadata={"description": "Optional directory path for persisting analyzed shard priorities."},
    )
    EXPORT_GRAPHS: str = field(
        default='graphs.tempdir',
        metadata={
            "description": (
                "Directory the post-process graphing stage writes figures and "
                "tables to (delta-sequence plots, delta histograms, bumpiness "
                "tables). Only used when GraphConfig enables at least one graph."
            )
        },
    )
    EXPORT_SEARCH_RESULTS: str = field(
        default='search_results.tempdir',
        metadata={"description": "Directory used by search stage to save discovered results and metadata."},
    )
    EXPORT_SEARCH_RESULTS_FORMAT: str = field(
        default='pkl',
        metadata={"description": "Format to save the discovered results and metadata ('pkl' or 'json')."},
    )
    EXPORT_ANALYSIS_RESULTS: str = field(
        default='analysis_results.tempdir',
        metadata={
            "description": (
                "Directory for the analysis stage's separate per-trajectory store, "
                "used only when analysis.STORE_TRAJECTORIES_SEPARATELY is True. "
                "Each analysed shard is written to <shard_id>.jsonl here instead of "
                "in EXPORT_SEARCH_RESULTS. When the flag is False (default) the "
                "analyzer writes directly to EXPORT_SEARCH_RESULTS and this path is "
                "unused."
            )
        },
    )
    TOTAL_CORES: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Total CPU cores the pipeline may use. None resolves to "
                "os.cpu_count() at runtime. Single source of truth for every "
                "process pool (stage pools, eval pools, extraction); nothing is "
                "reserved. See dreamer.utils.multi_processing.search_worker_budget."
            )
        },
    )
    REUSE_POLICY: str = field(
        default="strict",
        metadata={
            "description": (
                "How stored trajectory attributes are reused across runs. 'strict' "
                "(default): reuse a stored value only while its per-attribute "
                "fingerprint (the knobs it depends on, its dependencies, its code "
                "version) matches the current config — exactly the affected "
                "attributes are recomputed. 'keep': reuse any stored value, ignoring "
                "fingerprints (fast exploratory reruns; values may be stale). "
                "'recompute': ignore stored values."
            )
        },
    )
    RUN_ID: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Identifier of the current pipeline run (set automatically by "
                "System.run). Used by REUSE_POLICY='recompute' to tell this run's "
                "rows from earlier ones. Not meant to be set by hand."
            )
        },
    )
    STAGE_POOL_WORKERS: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Worker cap for the per-stage trajectory pools (analysis, hedgehog "
                "search, Tier-3 post-process). None/0 = use the full core budget; a "
                "positive value caps at min(value, budget). A resolved count <= 1 "
                "runs every task serially in the main process (no subprocess)."
            )
        },
    )
    MEMORY_AWARE_WORKERS: bool = field(
        default=True,
        metadata={"description": (
            "Cap every process pool by available memory, not only by cores: "
            "workers = min(cores, (available - MEMORY_RESERVE_MB) / per-worker MB), and "
            "shrink a running pool when available memory drops below the reserve (before "
            "the OS kills workers). See dreamer.utils.memory_budget."
        )},
    )
    MEMORY_LIMIT_MB: Optional[int] = field(
        default=None,
        metadata={"description": (
            "Memory (MB) this run may use; None = detect (Linux MemAvailable + cgroup/SLURM "
            "limit, macOS vm_stat). Set it on shared nodes."
        )},
    )
    MEMORY_RESERVE_MB: int = field(
        default=1024,
        metadata={"description": "Memory (MB) kept free for the main process and the OS."},
    )
    WORKER_MEMORY_MB: Optional[int] = field(
        default=None,
        metadata={"description": (
            "Memory budget per worker (MB). None = measured fresh-worker RSS (~130 MB) + "
            "WORKER_TASK_HEADROOM_MB."
        )},
    )
    WORKER_TASK_HEADROOM_MB: int = field(
        default=512,
        metadata={"description": (
            "Per-worker allowance (MB) for a task's own data on top of the import baseline "
            "(walked matrices, symbolic trajectory matrix, caches). Measured warm-up: "
            "~50 MB (2F1) to ~350 MB (4F3); raise it for very large CMFs."
        )},
    )
    POOL_START_METHOD: str = field(
        default="auto",
        metadata={"description": (
            "Process start method for all pools: 'auto' = forkserver on Linux (no "
            "fork-with-threads deadlocks, workers can be added at any time), spawn on "
            "macOS; or 'spawn' / 'forkserver' / 'fork' explicitly."
        )},
    )
    POOL_SLOW_START: int = field(
        default=2,
        metadata={"description": (
            "Memory-aware pools that can grow start with at most this many workers, "
            "measure the real per-task memory peak, then grow to every core that fits. "
            "Avoids starting all workers on a task that needs gigabytes each (e.g. long "
            "trajectories on a large CMF). 0 = start at the full memory cap."
        )},
    )
    SYMPY_CACHE_CLEAR_AFTER_SECONDS: Optional[float] = field(
        default=1.0,
        metadata={"description": (
            "After a trajectory task (worker or search evaluation) that took at least "
            "this many seconds, clear sympy's global cache, which otherwise keeps the "
            "huge rationals of long walks alive (~20 MB per long trajectory, up to GBs "
            "per worker). Light tasks skip it (~5 ms re-warm cost). 0 = after every "
            "task; None or negative = never."
        )},
    )
    POOL_MAX_TASKS_PER_CHILD: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Recycle each pool worker after this many tasks (None = never). "
                "Bounds slow memory growth of long-lived workers on large CMFs, at "
                "the cost of re-spawning workers (each re-imports dreamer). Not "
                "available with the 'fork' start method (ignored there)."
            )
        },
    )
    STAGE_POOL_CHUNKSIZE: int = field(
        default=1,
        metadata={
            "description": (
                "Tasks handed to a stage-pool worker at a time (analysis, hedgehog "
                "search, Tier-3 post-process). 1 gives full dynamic load balancing: "
                "an idle worker pulls the next trajectory immediately. Raise it only "
                "when single tasks are so cheap that dispatch overhead dominates."
            )
        },
    )
    WRITER_BATCH_SIZE: int = field(
        default=50,
        metadata={
            "description": (
                "Number of records the JSONL writer accumulates before issuing "
                "an explicit fsync. Larger values reduce I/O overhead but increase "
                "the worst-case loss window on a crash."
            )
        },
    )
    WRITER_FLUSH_TIMEOUT_SECONDS: float = field(
        default=5.0,
        metadata={
            "description": (
                "Seconds of queue inactivity that trigger a flush even when the "
                "batch is not full. Bounds tail-of-run latency before data is "
                "durable on disk."
            )
        },
    )
    PATH_TO_SEARCHABLES: str = field(
        default='searchables.tempdir',
        metadata={"description": "Default import directory for precomputed searchable shards."},
    )
    EXPORT_SEARCHABLES_FORMAT: str = field(
        default='pkl',
        metadata={"description": "Format used to export/import extracted searchables ('pkl' or 'json')."},
    )
    EXPORT_ANALYSIS_PRIORITIES_FORMAT: str = field(
        default='pkl',
        metadata={"description": "Format used to export/import analysis priorities ('pkl' or 'json')."},
    )


sys_config: SystemConfig = SystemConfig()
