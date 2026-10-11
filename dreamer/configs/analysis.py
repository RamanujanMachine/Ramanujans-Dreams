from dataclasses import dataclass, field
from .configurable import Configurable
from typing import Callable


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


@dataclass
class AnalysisConfig(Configurable):
    """
    Stage analysis configurations
    """
    # ============================= Parallelism and efficiency =============================
    USE_CACHING: bool = field(
        default=True,
        metadata={"description": "Enable LRU caches used by analysis computations."},
    )

    NUM_TRAJECTORIES_FROM_DIM: Callable = field(
        default=traj_from_dim,
        metadata={"description": "Callable that maps searchable dimension to number of sampled trajectories."},
    )
    SAMPLING_METHOD: str = field(
        default="pt",
        metadata={
            "description": (
                "Trajectory-sampling engine used during the analysis stage: "
                "'raycast' (default), 'discrete' (DiscreteMCMCSampler), or 'pt' "
                "(ParallelTemperingSampler).  Mirrors search.SAMPLING_METHOD but is "
                "independent, so analysis and search can use different samplers.  See "
                "that knob for the engine descriptions."
            )
        },
    )
    IDENTIFY_THRESHOLD: float = field(
        default=-1,
        metadata={"description": "Minimum identified-trajectory ratio required to keep a shard; -1 disables filtering."},
    )
    STORE_TRAJECTORIES_SEPARATELY: bool = field(
        default=False,
        metadata={
            "description": (
                "When True, the analysis stage writes its per-trajectory JSONL "
                "records to a separate per-shard store under "
                "sys_config.EXPORT_ANALYSIS_RESULTS (one <shard_id>.jsonl per "
                "shard) instead of co-mingling them with the search results in "
                "EXPORT_SEARCH_RESULTS. Cross-stage cache reuse is preserved: the "
                "search stage still seeds its cache from the analysis store and "
                "copies any reused record into EXPORT_SEARCH_RESULTS so the search "
                "file stays self-contained. Default False keeps the legacy shared "
                "layout."
            )
        },
    )

    # ============================= Printing and error management =============================
    PRINT_FOR_EVERY_SEARCHABLE: bool = field(
        default=True,
        metadata={"description": "Log per-searchable analysis summaries during analyzer execution."},
    )
    SHOW_START_POINT: bool = field(
        default=True,
        metadata={"description": "Include searchable start points in analysis logs."},
    )
    SHOW_SEARCHABLE: bool = field(
        default=False,
        metadata={"description": "Include full searchable object dumps in analysis logs."},
    )

    # ============================= Analysis features =============================
    USE_LIReC: bool = field(
        default=True,
        metadata={"description": "Use LIReC constant-identification routines instead of fallback heuristics."},
    )


analysis_config: AnalysisConfig = AnalysisConfig()
