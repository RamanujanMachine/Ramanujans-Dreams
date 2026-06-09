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
                "Legacy directory for the analyzer's shard-level audit trail. "
                "Unused since the analyzer was switched to per-trajectory dedup "
                "and writes directly to EXPORT_SEARCH_RESULTS. Retained for "
                "backward compatibility / ad-hoc tools."
            )
        },
    )
    TOTAL_CORES: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Total CPU cores the pipeline may use. None resolves to "
                "os.cpu_count() at runtime. Single source of truth for the core "
                "budget: when Tier-2 is active, NUM_BACKGROUND_WORKERS + 1 (writer) "
                "cores are reserved for the Tier-2 queue + sink and the rest go to "
                "the search/eval pools; when Tier-2 is inactive, all cores go to "
                "search/analysis. See dreamer.utils.multi_processing.search_worker_budget."
            )
        },
    )
    NUM_BACKGROUND_WORKERS: int = field(
        default=4,
        metadata={"description": "Number of background worker processes that compute Tier-2 trajectory attributes during search. When Tier-2 is active these (plus 1 writer/sink) are the cores reserved from TOTAL_CORES."},
    )
    WRITER_BATCH_SIZE: int = field(
        default=100,
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
