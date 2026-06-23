from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from ramanujantools import Position

from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger


def trajectory_to_seed(
    geom: FlatlandGeometry, trajectory: Position
) -> Optional[np.ndarray]:
    """Map a user-supplied real-space trajectory to a validated flatland seed.

    Used by every search method to convert an injected ``initial_trajectory``
    (a real-space direction the user already believes has a good δ) into the
    integer flatland coordinate ``z`` that seeds the optimiser.

    :param geom: The shard's flatland geometry.
    :param trajectory: Real-space trajectory direction (``Position`` over the
        shard symbols).
    :return: The integer flatland direction ``z`` (``geom.to_flatland`` rounded)
        when it is a non-zero recession direction of the shard cone, otherwise
        ``None`` — i.e. the trajectory is *geometrically invalid* for this shard
        (maps to the zero vector or falls outside the cone), and the caller
        should fall back to its default reservoir seed.

    .. note::
        Trajectory *length* is intentionally **not** checked here: even a long
        trajectory is a legitimate starting point for the optimiser to explore
        from.  Only cone membership (geometry) gates usability as a seed.
    """
    z = geom.to_flatland(trajectory)
    if not geom.is_inside(z):
        return None
    return z


def resolve_injected_seed(
    geom: FlatlandGeometry,
    initial_trajectory: Optional[Position],
    shard_id: str,
    constant: Constant,
    identify_fn: Optional[Callable[[np.ndarray], bool]] = None,
) -> Optional[np.ndarray]:
    """Resolve a user-supplied initial trajectory into a search seed, with messaging.

    Shared by all search methods so the seed-handling policy is implemented once:

    * **Geometrically invalid** (maps to ``0`` / outside the cone) → emit a
      user-facing WARNING and return ``None`` so the caller falls back to its
      default reservoir seed.
    * **Valid but does not identify** the constant → emit a user-facing WARNING and
      **use it anyway** (search maximises δ regardless of identification).

    :param geom: The shard's flatland geometry.
    :param initial_trajectory: The user-supplied direction, or ``None`` (returns
        ``None`` immediately → default seeding).
    :param shard_id: Structural shard id (for the log message).
    :param constant: The constant being searched (for the log message).
    :param identify_fn: Optional predicate ``z -> bool`` telling whether the seed
        identifies the constant; when given, a non-identifying seed is logged.
    :return: The flatland seed ``z`` to use, or ``None`` to fall back to the
        method's default seed selection.
    """
    if initial_trajectory is None:
        return None

    z = trajectory_to_seed(geom, initial_trajectory)
    if z is None:
        Logger(
            f"Supplied initial trajectory {initial_trajectory} is not a valid "
            f"recession direction for shard {shard_id}; falling back to the "
            f"default reservoir seed.",
            Logger.Levels.warning,
        ).log()
        return None

    if identify_fn is not None and not identify_fn(z):
        Logger(
            f"Supplied initial trajectory for shard {shard_id} does not identify "
            f"{constant.name}; using it as the search seed anyway.",
            Logger.Levels.warning,
        ).log()
    return z
