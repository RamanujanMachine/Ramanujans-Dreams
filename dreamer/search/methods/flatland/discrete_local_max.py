"""Discrete orthogonal-neighbour hill-climb — the lattice local-maximum certificate.

Shared by the gradient-based flatland search methods (Gradient Ascent, Hybrid
SPSA).  A continuous / stochastic ascent realises each direction through
:func:`snap_to_trajectory`, so its objective ``δ(snap(d))`` is piecewise-constant
on the integer lattice: when the continuous loop stops, the returned trajectory
is only a *true* local maximum **at the lattice resolution** if no minimal integer
move — one coordinate ``±1``, in-cone, within the length cap — yields a strictly
larger δ.

:func:`discrete_hill_climb` provides that certificate: it greedily climbs the
``2·d_flat`` orthogonal neighbours until none strictly improves δ, then declares
the discrete local maximum.  Neighbours are cone-filtered (``A·v ≤ 0``) so invalid
shards are never walked, and the batch is optionally evaluated across a per-shard
process pool.  Running it as the final step of an ascent guarantees the method
returns a genuine ±1 local max — the honest "the resolution is exhausted" stop.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.parallel_eval import evaluate_batch


def orthogonal_neighbours(
    z: np.ndarray,
    geom: FlatlandGeometry,
    max_norm: float,
    traj_norm: str,
) -> List[np.ndarray]:
    """Return the in-cone, length-capped ``±1`` orthogonal neighbours of ``z``.

    Builds the ``2·d_flat`` candidates (one coordinate ``±1``, a *raw* minimal
    integer step — not GCD-reduced — so the move is the smallest faithful lattice
    step), then keeps only those inside the shard cone (``A·v ≤ 0`` via
    ``is_inside_many``) and within the real-space norm cap, so invalid shards are
    never walked.

    :param z: Current integer flatland trajectory.
    :param geom: Flatland geometry (cone filter + norm).
    :param max_norm: Trajectory norm cap.
    :param traj_norm: Norm used for the length cap (``SEARCH_TRAJ_NORM``).
    :return: List of admissible neighbour vectors (possibly empty).
    """
    z = np.asarray(z, dtype=np.int64)
    d_flat = geom.d_flat
    cands = np.repeat(z[None, :], 2 * d_flat, axis=0)
    for i in range(d_flat):
        cands[2 * i, i] += 1
        cands[2 * i + 1, i] -= 1

    inside = geom.is_inside_many(cands)
    within = geom.traj_norm_many(cands, traj_norm) <= max_norm
    keep = inside & within
    return [cands[j] for j in np.nonzero(keep)[0]]


def evaluate_neighbours(
    neighbours: List[np.ndarray],
    eval_ctx: dict,
    pool=None,
) -> List[Tuple[float, bool]]:
    """Evaluate a neighbour batch, optionally across a per-shard process pool.

    :param neighbours: Admissible neighbour vectors.
    :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
    :param pool: Optional persistent per-shard process pool.
    :return: ``(delta, identified)`` per neighbour, in input order.
    """
    if pool is not None and len(neighbours) > 1:
        return evaluate_batch(neighbours, eval_ctx=eval_ctx, pool=pool)
    return [evaluate_in_flatland(z, **eval_ctx) for z in neighbours]


def discrete_hill_climb(
    cur_z: np.ndarray,
    cur_delta: float,
    *,
    geom: FlatlandGeometry,
    eval_ctx: dict,
    max_norm: float,
    traj_norm: str,
    improve_threshold: float,
    pool=None,
    on_local_max: Optional[Callable[[np.ndarray, float], None]] = None,
) -> Tuple[np.ndarray, float]:
    """Greedily climb the ``2·d_flat`` minimal integer neighbours until a local max.

    Repeatedly evaluates the in-cone, length-capped orthogonal neighbours of
    ``cur_z`` and moves to the strictly-best one (δ greater than the current by
    more than ``improve_threshold``).  When no neighbour strictly improves, ``cur_z``
    is the true discrete local maximum at the lattice resolution and the climb stops.

    Calling this once at the end of an ascent is the local-maximum *certificate*:
    if the continuous/stochastic phase already sat on a ±1 local max it returns
    immediately (one neighbour sweep); otherwise it climbs the improving moves the
    continuous phase left on the table.

    :param cur_z: Current integer flatland trajectory (must be identified / valid).
    :param cur_delta: δ at ``cur_z``.
    :param geom: Flatland geometry (cone filter + norm).
    :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
    :param max_norm: Trajectory norm cap (the lattice resolution).
    :param traj_norm: Norm used for the length cap (``SEARCH_TRAJ_NORM``).
    :param improve_threshold: Minimum δ gain for a neighbour to count as strictly
        better (a neighbour must beat ``cur_delta`` by more than this to be taken).
    :param pool: Optional per-shard process pool for the neighbour batch.
    :param on_local_max: Optional callback ``(z, delta)`` invoked once when the
        discrete local maximum is reached (for caller-specific logging).
    :return: ``(z, delta)`` at the discrete local maximum.
    """
    while True:
        neighbours = orthogonal_neighbours(cur_z, geom, max_norm, traj_norm)
        if not neighbours:
            break  # boxed in by the cone / norm cap — current point is maximal.

        results = evaluate_neighbours(neighbours, eval_ctx, pool)

        best_z, best_delta = cur_z, cur_delta
        for z_n, (delta_n, identified_n) in zip(neighbours, results):
            if identified_n and delta_n > best_delta + improve_threshold:
                best_z, best_delta = z_n, delta_n

        if best_delta <= cur_delta + improve_threshold:
            # No strictly-better orthogonal neighbour -> discrete local maximum.
            if on_local_max is not None:
                on_local_max(cur_z, cur_delta)
            break

        cur_z, cur_delta = best_z, best_delta

    return cur_z, cur_delta
