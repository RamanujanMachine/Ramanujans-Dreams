"""
Lattice realization helpers for Gradient Ascent.

The optimizer operates on a *real-valued* direction; delta is continuous in the
direction's angle.  To evaluate delta we must realize a real direction as an
integer-coordinate trajectory.  :func:`snap_to_trajectory` (now defined in the
shared :mod:`dreamer.search.methods.flatland.lattice` module and re-exported here
for backward compatibility) returns the integer flatland direction whose angle
best matches a real direction, subject to a length cap (so trajectories stay
short / cheap) and shard-cone membership.

:func:`rotate_toward` produces the small angular perturbations used to estimate
the gradient by forward differences in angle space.
"""

import numpy as np

# Re-export so existing call sites (``gradient_ascent.lattice.snap_to_trajectory``)
# keep working after the function moved to the shared flatland package.
from dreamer.search.methods.flatland.lattice import snap_to_trajectory  # noqa: F401


def rotate_toward(d: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """
    Rotate direction *d* by *angle* radians toward basis axis *axis*, preserving length.

    The rotation happens in the plane spanned by ``d`` and the unit basis vector
    ``e_axis``.  If that axis is (nearly) parallel to ``d`` the rotation is a
    no-op (the plane is degenerate) and ``d`` is returned unchanged.

    :param d: Real direction vector (length ``dim``).
    :param axis: Index of the coordinate basis vector to rotate toward.
    :param angle: Rotation angle in radians.
    :return: The rotated direction (same L2 norm as ``d``).
    """
    d = np.asarray(d, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return d.copy()

    u = d / norm
    e = np.zeros_like(u)
    e[axis] = 1.0

    perp = e - np.dot(e, u) * u
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-12:
        return d.copy()  # axis parallel to d — no well-defined rotation plane.
    perp_unit = perp / perp_norm

    rotated_unit = np.cos(angle) * u + np.sin(angle) * perp_unit
    return rotated_unit * norm
