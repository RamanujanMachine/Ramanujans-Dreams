"""
Shared flatland lattice-realization helpers.

The gradient-based optimizers operate on a *real-valued* direction; δ is
continuous in the direction's angle.  To evaluate δ we must realize a real
direction as an integer-coordinate trajectory.  :func:`snap_to_trajectory`
returns the integer flatland direction whose angle best matches a real
direction, subject to a length cap (so trajectories stay short / cheap) and
shard-cone membership.

This lives in the ``flatland`` package (rather than alongside the gradient-ascent
optimizers) because it is a generic lattice primitive: besides the continuous
optimizers, the shared discrete micro-hill-climb finalization
(:func:`dreamer.search.methods.flatland.discrete_local_max.discrete_micro_climb`)
uses it to re-snap angular-resolution-doubling probes, so every search method's
finalization can reach it without importing a method-specific package.
"""

from typing import Optional

import numpy as np

from dreamer.search.methods.flatland.geometry import FlatlandGeometry


def snap_to_trajectory(
    d: np.ndarray,
    geom: FlatlandGeometry,
    max_norm: float,
    norm: str = "l2",
) -> Optional[np.ndarray]:
    """
    Realize a real direction *d* as the angle-best, length-capped, in-cone integer direction.

    Scans candidate integer flatland directions obtained by scaling the unit
    direction to a range of integer lengths and rounding; keeps those whose
    **real shard-space length** (``geom.traj_norm``) is ``<= max_norm`` and that
    lie inside the shard cone, and returns the one maximizing the cosine
    similarity to ``d`` (i.e. minimizing the angle).

    The length cap is measured in real *shard* space (the basis the trajectory is
    actually walked in), not in the flatland lattice basis — flatland L2 length
    is in a different (LLL-reduced) basis and does not match the cost / geometry
    of the realized trajectory.

    :param d: Real direction vector (length ``geom.d_flat``).
    :param geom: Flatland geometry providing the cone-membership + length tests.
    :param max_norm: Maximum shard-space length of the returned integer direction.
    :param norm: Which shard-space norm bounds the length (``'linf'`` | ``'l1'``
        | ``'l2'``); see :meth:`FlatlandGeometry.traj_norm`.
    :return: The best integer flatland direction, or ``None`` if no in-cone,
        non-zero candidate exists within the length cap.
    """
    d = np.asarray(d, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm == 0.0:
        return None

    unit = d / d_norm

    # The flatland length needed to reach a given shard-space length depends on
    # the basis: a flatland step of length 1 maps to a real vector of length
    # ``||Z_reduced @ unit||``.  Generate flatland lengths up to the ceiling that
    # reaches ``max_norm`` in shard space (plus a margin), then filter exactly by
    # the shard-space norm below.
    unit_real = geom.Z_reduced.astype(np.float64) @ unit
    unit_real_len = float(np.linalg.norm(unit_real)) or 1.0
    max_flat_len = int(np.ceil(max_norm / unit_real_len)) + 1
    lengths = np.arange(1, max_flat_len + 1, dtype=np.int64)
    candidates = np.rint(np.outer(lengths, unit)).astype(np.int64)  # (L, d)

    flat_norms = np.linalg.norm(candidates, axis=1)
    candidates = candidates[flat_norms > 0.0]
    if candidates.shape[0] == 0:
        return None

    # Cap by real shard-space length (primitive ray length actually walked).
    real_norms = geom.traj_norm_many(candidates, norm)
    keep = real_norms <= max_norm
    if not np.any(keep):
        return None
    candidates = candidates[keep]

    # Dedup identical rounded directions (small L collapse to the same vector).
    candidates = np.unique(candidates, axis=0)

    # Keep only in-cone candidates (single batched NumPy cone test).
    inside = geom.is_inside_many(candidates)
    if not np.any(inside):
        return None
    candidates = candidates[inside]

    # Pick the angle-best candidate: maximal cosine similarity to ``d``.
    cos = (candidates @ unit) / np.linalg.norm(candidates, axis=1)
    return candidates[int(np.argmax(cos))].astype(np.int64)
