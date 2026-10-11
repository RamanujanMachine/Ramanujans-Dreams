from dataclasses import dataclass, field
from typing import Dict, Optional

from .configurable import Configurable


@dataclass
class ExtractionConfig(Configurable):
    """
    Extraction stage configurations
    """
    TRAJECTORY_CONSTRAINTS: Optional[Dict[str, int]] = field(
        default=None,
        metadata={
            "description": (
                "Optional direction constraints fixing the *ratio* (and sign) of "
                "sampled trajectories, e.g. {'x0': 12, 'y1': 28} means every "
                "trajectory takes 12 steps in x0 for every 28 in y1 (scale-invariant: "
                "the literal values matter only through their reduced ratio and sign). "
                "Keys are CMF variable names; values are integers. Applied at every "
                "stage that conditions a cone (extraction/analysis sampling + search) "
                "and used to FILTER shards: a shard is dropped unless its recession "
                "cone admits such a direction. None/empty = unconstrained. A single "
                "fixed coordinate degenerates to a sign constraint; >=2 are needed for "
                "a real ratio. See context/algorithms/06_trajectory_constraints.md."
            )
        },
    )
    PARALLELIZE: bool = field(
        default=True,
        metadata={"description": "Enable parallel shard extraction routines."},
    )
    INIT_POINT_MAX_COORD: int = field(
        default=2,
        metadata={"description": "Maximum coordinate magnitude when searching for shard interior points."},
    )
    IGNORE_DUPLICATE_SEARCHABLES: bool = field(
        default=False,
        metadata={"description": "Skip duplicate searchables detected during extraction."},
    )
    STRATEGY: str = field(
        default="auto",
        metadata={
            "description": (
                "Shard-extraction strategy.  'auto' / 'exact' / 'heuristic' route "
                "through the v2 ExtractionManager (lrs + MILP, with ray-shooting "
                "fallback); 'legacy' falls back to the brute-force lattice scan "
                "in dreamer.extraction.utils.initial_points."
            )
        },
    )
    EXACT_TIMEOUT_SECONDS: float = field(
        default=3600.0,
        metadata={
            "description": (
                "Wall-clock budget (seconds) for the exact extractor. "
                "Under STRATEGY='auto': once this deadline passes the exact "
                "extractor raises ExtractionTimeout and the heuristic takes "
                "over.  Under STRATEGY='exact': the extractor raises on "
                "timeout.  Ignored under STRATEGY='heuristic'. "
                "Default 3600 (1 hour)."
            )
        },
    )
    HEURISTIC_TIMEOUT_SECONDS: float = field(
        default=3600.0,
        metadata={
            "description": (
                "Wall-clock budget (seconds) for the heuristic ray-shooter. "
                "Applies whether the heuristic runs standalone "
                "(STRATEGY='heuristic') or as the fallback after the exact "
                "method times out (STRATEGY='auto').  Ignored under "
                "STRATEGY='exact'.  Default 3600 (1 hour).  For high-D scans "
                "(e.g. 6F5/11D) where exact never finishes, this controls how "
                "long the heuristic runs."
            )
        },
    )
    LOAD_SHARD_CACHE: bool = field(
        default=False,
        metadata={
            "description": (
                "If True and a '<cmf>__shards.jsonl' file already exists for "
                "the CMF under EXPORT_CMFS, load those shards (encoding + "
                "interior point) and skip the expensive shard extraction. "
                "Newly found shards are still appended to the same file."
            )
        },
    )
    EXACT_UNBOUNDED_CHECK: str = field(
        default="lp",
        metadata={
            "description": (
                "Backend for the exact strategy's per-cell unbounded check: "
                "'lp' (default) uses an in-process recession-cone LP (~0.5ms, "
                "no external dependency); 'lrs' spawns the lrslib binary "
                "(~30x slower per cell, authoritative cross-check)."
            )
        },
    )
    EXACT_NUM_WORKERS: int = field(
        default=1,
        metadata={
            "description": (
                "Process count for the exact strategy's parallel reverse "
                "search.  >1 dispatches disjoint cell subtrees across "
                "processes (each enumerates + classifies + locates its own "
                "subtree, and salvages partial results on timeout).  1 "
                "(default) runs serially.  Forces the 'lp' unbounded check."
            )
        },
    )
    HEURISTIC_REFINE_WITNESSES: bool = field(
        default=False,
        metadata={
            "description": (
                "If True, the heuristic ray-shooter post-processes far-out "
                "shards with an MILP to return the L1-minimal "
                "(closest-to-origin) integer start point instead of the raw "
                "ray witness.  Only witnesses with L1 norm above "
                "HEURISTIC_REFINE_L1_THRESHOLD are recomputed.  Default False "
                "keeps the solver-free fast path.  Also applies to the "
                "heuristic top-up under STRATEGY='auto'."
            )
        },
    )
    HEURISTIC_REFINE_L1_THRESHOLD: float = field(
        default=50.0,
        metadata={
            "description": (
                "When HEURISTIC_REFINE_WITNESSES is on, only ray witnesses "
                "whose L1 norm (sum of |coordinates|) exceeds this are "
                "recomputed via MILP; smaller witnesses are already close "
                "enough to the origin and kept as-is.  0 refines every "
                "shard.  Default 50."
            )
        },
    )
    HEURISTIC_REFINE_WORKERS: int = field(
        default=1,
        metadata={
            "description": (
                "Process count for the heuristic's MILP witness refinement "
                "(the per-shard MILPs are independent).  >1 dispatches them "
                "across processes; 1 (default) runs serially."
            )
        },
    )
    HEURISTIC_NUM_RAYS: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Optional hard ceiling on samples processed by the heuristic "
                "(safety cap).  None (default) = unlimited; the missing-mass "
                "plateau and/or TIMEOUT_SECONDS govern instead.  Set a "
                "finite value only to bound a run regardless of saturation.  "
                "Use HEURISTIC_TIMEOUT_SECONDS to time-budget the shoot instead."
            )
        },
    )
    HEURISTIC_MISSING_MASS: float = field(
        default=5e-4,
        metadata={
            "description": (
                "Stop a heuristic phase once its Good-Turing missing-mass "
                "estimate (f1/n: fraction of samples landing in a cell seen "
                "exactly once) stays below this fraction for a few consecutive "
                "batches.  Estimates P(next sample is a new cell), so it is "
                "scale-invariant and robust to 'plateau then spike'.  Default "
                "5e-4.  Lower = more coverage / longer runs; 0 disables early "
                "stopping."
            )
        },
    )
    HEURISTIC_FACE_ALIGNED: bool = field(
        default=True,
        metadata={
            "description": (
                "If True, run a second face-aligned shooting phase after "
                "generic ray shooting to reach unbounded cells with "
                "lower-dimensional recession cones (tubes/slabs) that origin "
                "rays structurally miss.  Default True."
            )
        },
    )
    HEURISTIC_FACE_SUBSETS: int = field(
        default=200,
        metadata={
            "description": (
                "Number of random hyperplane subsets sampled in the "
                "face-aligned phase (each yields nullspace shooting "
                "directions).  Only used when HEURISTIC_FACE_ALIGNED=True. "
                "Default 200."
            )
        },
    )
    HEURISTIC_FACE_OFFSETS: int = field(
        default=50,
        metadata={
            "description": (
                "Number of random integer start offsets swept per face-aligned "
                "direction (sweeping offsets enumerates the slab cells sharing "
                "that recession direction).  Only used when "
                "HEURISTIC_FACE_ALIGNED=True.  Default 50."
            )
        },
    )


extraction_config: ExtractionConfig = ExtractionConfig()
