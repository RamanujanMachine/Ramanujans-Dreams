from dataclasses import dataclass, field
from typing import Tuple

from .configurable import Configurable


@dataclass
class PostProcessConfig(Configurable):
    """Configuration for the Tier-3 post-process stage.

    Runs once after the Search stage finishes.  Reads existing per-shard
    JSONL files, reconstructs ``TrajectoryAttributesHandler`` for every
    trajectory missing the configured attributes, and appends patch records
    that the merge-on-read reader folds in transparently.

    Empty ``TIER3_ATTRIBUTES`` (the default) short-circuits the whole
    stage — no JSONL files are read, no subprocesses spawned.
    """

    TIER3_ATTRIBUTES: Tuple[str, ...] = field(
        default=(
            # ("precision_at", "if_identified"), #("delta_sequence", "if_identified"),
            # ("digits_per_step", "if_identified"), ("digits_computed", "if_identified")
        ),
        metadata={
            "description": (
                "Expensive symbolic attributes (e.g. 'asymptotics', 'kamidelta') "
                "computed in the post-process stage after Search has finished. "
                "Empty = skip the stage entirely.\n\n"
                "Each entry is a bare attribute name (always compute) or a "
                "(name, predicate) tuple. The predicate gates the computation and "
                "may be:\n"
                "  * a registered name — 'if_identified', 'if_has_degree_2', "
                "'if_top_n_delta';\n"
                "  * 'max_degree below N' / 'max_degree above N' — the recurrence's "
                "polynomial degree (max over coefficient degrees) is under/over N;\n"
                "  * 'top N highest <metric> in shard' / '... lowest ... in cmf' — "
                "keep the N trajectories with the highest/lowest stored <metric> "
                "within their shard or whole CMF. <metric> is read from already-"
                "stored values (no re-walk): delta, convergence_rate (normalised "
                "eigenvalue-error gap), approximated_digits_per_step, digits_approximation, "
                "digits_computed, avg_computed_digits_per_step, spectral_gap, "
                "gcd_slope, precision_at. To rank on a metric that isn't stored yet, "
                "add it to the TIER2/TIER3 attribute lists first.\n\n"
                "Example: ('asymptotics', 'top 3 highest convergence_rate in cmf')."
            )
        },
    )


post_process_config: PostProcessConfig = PostProcessConfig()
