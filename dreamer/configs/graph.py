from dataclasses import dataclass, field

from .configurable import Configurable


@dataclass
class GraphConfig(Configurable):
    """Configuration for the post-process graphing stage.

    Runs once, at the end of the Tier-3 post-process stage, reading the existing
    per-shard JSONL files and writing figures / tables under
    ``sys_config.EXPORT_GRAPHS``.  Every graph kind is **off by default** — the
    whole stage short-circuits when nothing is enabled (no files read, no
    figures written), exactly like an empty ``TIER3_ATTRIBUTES``.
    """

    # ---- which graphs to produce ----
    PLOT_BEST_DELTA_SEQUENCE: bool = field(
        default=False,
        metadata={
            "description": (
                "Line plot of the irrationality-measure (δ) sequence over the "
                "first DELTA_SEQUENCE_DEPTH steps of the best-δ trajectory in each "
                "(CMF, constant). One figure per CMF/constant. This is the only "
                "graph that walks a trajectory (just the single best one)."
            )
        },
    )
    PLOT_DELTA_HISTOGRAMS: bool = field(
        default=False,
        metadata={
            "description": (
                "Histogram of δ across trajectories, one per shard and one "
                "aggregated per CMF. Cheap — reads the stored `delta` column only."
            )
        },
    )
    WRITE_BUMPINESS_TABLE: bool = field(
        default=False,
        metadata={
            "description": (
                "Per-shard 'bumpiness' table (CSV + markdown) measuring how "
                "non-smooth δ is. Two metrics: (B) a density-robust spatial "
                "roughness from the empirical semivariogram of δ over angular "
                "direction-distance (nugget + initial slope), and (A) the median "
                "per-trajectory total variation of the stored delta_sequence "
                "(NaN where delta_sequence was not computed)."
            )
        },
    )

    # ---- parameters ----
    DELTA_SEQUENCE_DEPTH: int = field(
        default=1000,
        metadata={"description": "Walk depth for the best-trajectory δ-sequence plot."},
    )
    HISTOGRAM_BINS: int = field(
        default=40,
        metadata={"description": "Number of bins for the δ histograms."},
    )
    VARIOGRAM_LAG_BINS: int = field(
        default=15,
        metadata={
            "description": (
                "Number of angular-distance lag bins for the empirical "
                "semivariogram used by the bumpiness table."
            )
        },
    )
    VARIOGRAM_MAX_PAIRS: int = field(
        default=200_000,
        metadata={
            "description": (
                "Cap on trajectory pairs sampled per shard when building the "
                "semivariogram (pairs are subsampled above this to bound the "
                "O(M^2) cost on densely sampled shards)."
            )
        },
    )

    def any_enabled(self) -> bool:
        """True iff at least one graph kind is enabled (stage runs)."""
        return bool(
            self.PLOT_BEST_DELTA_SEQUENCE
            or self.PLOT_DELTA_HISTOGRAMS
            or self.WRITE_BUMPINESS_TABLE
        )


graph_config: GraphConfig = GraphConfig()
