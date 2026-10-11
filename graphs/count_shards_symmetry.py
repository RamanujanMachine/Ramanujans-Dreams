"""
Count the shards of a CMF with the heuristic extractor, with and without
symmetry reduction.

Given a CMF described by a :class:`~dreamer.loading.funcs.formatter.Formatter`
object and a wall-clock ``timeout`` (in minutes), this script runs the heuristic
ray-shooting extractor twice on the same hyperplane arrangement:

1. **without** symmetry reduction  -> counts every shard (cell) found;
2. **with** symmetry reduction (canonical teleportation into a fundamental
   domain, e.g. the ``S_p x S_q`` symmetry of a ``pFq`` CMF) -> counts one
   representative per symmetry orbit.

Each run is bounded by the same timeout (so the total wall-clock is ~2x the
timeout).  The heuristic may also stop early once its missing-mass estimate
plateaus -- exactly as in production -- so the reported counts are "shards found
within the budget", which the timeout caps from above.

Output::

    #shards found after X minutes (timeout) = S1
    #shards after symmetry consideration found after X minutes (timeout) = S2

Run directly for the worked example, or import :func:`count_shards`.
"""

from __future__ import annotations

import argparse
from typing import Tuple

from dreamer.configs import extraction_config
from dreamer.extraction.extractor import extract_cmf_hyperplanes
from dreamer.extraction.v2 import ExtractionManager, symmetry_for_cmf
from dreamer.loading.funcs.formatter import Formatter


def _make_manager(max_seconds: float, symmetry) -> ExtractionManager:
    """Build a heuristic-only :class:`ExtractionManager` mirroring production knobs.

    The face-aligned phase / missing-mass / face-subset settings are read from
    ``extraction_config`` so the count matches the production heuristic; only the
    time budget and the symmetry strategy are overridden per call.

    :param max_seconds: Wall-clock budget for the ray shoot.
    :param symmetry: A :class:`SymmetryStrategy` (or ``None`` to disable
        symmetry reduction).
    :return: A configured ``ExtractionManager`` in ``"heuristic"`` strategy.
    """
    return ExtractionManager(
        strategy="heuristic",
        heuristic_max_seconds=max_seconds,
        heuristic_num_rays=None,
        heuristic_missing_mass=extraction_config.HEURISTIC_MISSING_MASS,
        heuristic_face_aligned=extraction_config.HEURISTIC_FACE_ALIGNED,
        heuristic_face_subsets=extraction_config.HEURISTIC_FACE_SUBSETS,
        heuristic_face_offsets=extraction_config.HEURISTIC_FACE_OFFSETS,
        symmetry=symmetry,
    )


def count_shards(formatter: Formatter, timeout_minutes: float) -> Tuple[int, int]:
    """Count shards with and without symmetry reduction, under a timeout.

    :param formatter: The CMF formatter (e.g. ``pFq(log(2), 2, 1, -1)``).
    :param timeout_minutes: Per-run wall-clock budget, in minutes.
    :return: ``(s1, s2)`` -- shard count without symmetry, and with symmetry.
    """
    cmf_data = formatter.to_cmf()
    hyperplanes = extract_cmf_hyperplanes(cmf_data)
    print(f'Extracted {len(hyperplanes)} hyperplanes from the CMF.')

    # The v2 extractors work in the shifted lattice, so shift the hyperplanes
    # once and reuse them for both runs (matches ShardExtractor._discover_via_v2).
    shifted_hps = [hp.apply_shift(cmf_data.shift) for hp in hyperplanes]
    max_seconds = timeout_minutes * 60.0

    if not shifted_hps:
        # No hyperplanes -> the whole space is a single shard; symmetry can't
        # reduce below one.
        return 1, 1

    # 1) No symmetry reduction.
    s1 = len(_make_manager(max_seconds, symmetry=None).extract(shifted_hps))

    # 2) With symmetry reduction (None for CMF families with no registered
    #    symmetry, in which case s2 == s1).
    symmetry = symmetry_for_cmf(cmf_data.cmf, list(cmf_data.shift.values()))
    s2 = len(_make_manager(max_seconds, symmetry=symmetry).extract(shifted_hps))

    return s1, s2


def main() -> None:
    """CLI entry point: run the worked example (or a chosen timeout)."""
    parser = argparse.ArgumentParser(
        description="Count CMF shards with/without symmetry reduction."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-run timeout in minutes (default: 1).",
    )
    args = parser.parse_args()

    from dreamer.loading import pFq
    from dreamer import log

    formatter = pFq(log(2), 5, 4, -1)
    s1, s2 = count_shards(formatter, args.timeout)

    x = args.timeout
    print(f"#shards found after {x} minutes (timeout) = {s1}")
    print(
        f"#shards after symmetry consideration found after {x} minutes "
        f"(timeout) = {s2}"
    )


if __name__ == "__main__":
    main()
