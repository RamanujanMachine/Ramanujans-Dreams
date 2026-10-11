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

import math
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from dreamer.extraction.utils.fast_gcd import reduce_to_primitive
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.flatland.lattice import snap_to_trajectory
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


def primitive_ray_key(z: np.ndarray, geom: FlatlandGeometry) -> Tuple[int, ...]:
    """Canonical identity of the trajectory ``z`` realises: its primitive real ray.

    δ depends only on a trajectory's *ray angle*, so scaled copies (``z`` and
    ``2z``) and any two flatland vectors mapping to the same GCD-reduced real
    direction are the *same* trajectory.  This returns that canonical key — the
    primitive (GCD == 1) real-space direction as an int tuple — so a caller can
    deduplicate work across micro-climbs without re-walking the same ray.

    :param z: Integer flatland coordinate vector.
    :param geom: Flatland geometry (provides the ``Z_reduced`` real mapping).
    :return: Tuple of ints — the primitive real-space ray.
    """
    v = geom.Z_reduced @ np.asarray(z, dtype=np.int64)
    v = reduce_to_primitive(v)
    return tuple(int(x) for x in v)


def doublings_to_resolution(center_len: float, max_norm: float) -> int:
    """Number of angular-resolution-doubling levels to reach the max-length cap.

    The probe ``2^j · center ± e_i`` deviates from ``center`` by an angle
    ``≈ 1/(2^j · |center|)``; the lattice cannot resolve a finer angle than
    ``≈ 1/max_norm`` (the longest representable ray), so doublings beyond
    ``2^j · |center| ≈ max_norm`` buy no new resolution.  This returns that
    ``K = ⌈log2(max_norm / |center|)⌉`` (at least 1, so the max-length resolution
    is always probed once).  The max-length radius *is* the bound — there is no
    separate round-count knob.

    :param center_len: Real shard-space length of the (primitive) center ray.
    :param max_norm: Trajectory length cap (the lattice resolution radius).
    :return: Number of doubling levels ``K ≥ 1``.
    """
    if center_len <= 0.0 or max_norm <= center_len:
        return 1  # already at/over the cap: just the final max-length round.
    return max(1, int(math.ceil(math.log2(max_norm / center_len))))


def resolution_probe_rays(
    z: np.ndarray,
    geom: FlatlandGeometry,
    max_norm: float,
    traj_norm: str,
    visited: Optional[Set[Tuple[int, ...]]] = None,
) -> List[np.ndarray]:
    """Phase-B angular-resolution probes for the center ``z`` — the full fan.

    Sweeps the doubling levels ``j = 1 … K`` (``K`` from
    :func:`doublings_to_resolution`, i.e. up to the **max-length resolution**) and,
    for each, treats every ``2^j · z ± e_i`` as a *continuous* directional probe —
    its raw norm may break the length cap, so it is **never** evaluated directly —
    re-snapping it with :func:`snap_to_trajectory` into the primitive, length-capped,
    in-cone integer ray along that finer interstitial angle.  Larger ``j`` probes a
    finer angular offset from ``z``; the last level's probe length reaches /
    exceeds ``max_norm`` and is therefore snapped down to the max-length ray (the
    finest the resolution allows).  ``snap_to_trajectory`` guarantees cone
    membership, the length cap, GCD == 1 primitivity, and rejects the zero vector,
    so every returned ray is a safe, walkable trajectory.

    Returning the whole coarse→finest fan in one list lets the caller evaluate it
    as a **single parallel batch**.  Rays equal to ``z``, duplicates, and — when
    ``visited`` is supplied — rays already explored are dropped, so nothing is
    checked twice.

    :param z: Current integer flatland center (a ±1 local maximum).
    :param geom: Flatland geometry (cone filter + norm + real mapping).
    :param max_norm: Trajectory norm cap (passed straight to ``snap_to_trajectory``).
    :param traj_norm: Norm used for the length cap (``SEARCH_TRAJ_NORM``).
    :param visited: Optional set of primitive-ray keys already evaluated; matching
        rays are skipped (cross-climb deduplication).
    :return: List of distinct, novel, in-cone integer rays spanning the resolutions.
    """
    # Probe from the *primitive* flatland center so the doubling sequence starts at
    # the smallest faithful representation of the ray.
    center = reduce_to_primitive(np.asarray(z, dtype=np.int64))
    center_len = geom.traj_norm(center, traj_norm)
    z_key = primitive_ray_key(center, geom)
    d_flat = geom.d_flat
    K = doublings_to_resolution(center_len, max_norm)

    rays: List[np.ndarray] = []
    seen_keys: Set[Tuple[int, ...]] = set()
    for j in range(1, K + 1):
        base = (float(2 ** j) * center.astype(np.float64))
        for i in range(d_flat):
            for sign in (1, -1):
                probe = base.copy()
                probe[i] += sign
                ray = snap_to_trajectory(probe, geom, max_norm, traj_norm)
                if ray is None:
                    continue
                key = primitive_ray_key(ray, geom)
                if key == z_key:
                    continue  # snapped back onto the current center — no new angle.
                if key in seen_keys:
                    continue  # duplicate probe (a coarser/finer level collided).
                if visited is not None and key in visited:
                    continue  # already explored by an earlier climb — don't redo it.
                seen_keys.add(key)
                rays.append(ray)
    return rays


def discrete_micro_climb(
    cur_z: np.ndarray,
    cur_delta: float,
    *,
    geom: FlatlandGeometry,
    eval_ctx: dict,
    max_norm: float,
    traj_norm: str,
    improve_threshold: float,
    pool=None,
    visited: Optional[Set[Tuple[int, ...]]] = None,
    on_local_max: Optional[Callable[[np.ndarray, float], None]] = None,
) -> Tuple[np.ndarray, float]:
    """Discrete micro-hill-climb with angular resolution doubling (Phase A + B).

    The lattice endgame run as the assurance finalization of *any* search method:

    * **Phase A** — :func:`discrete_hill_climb`: greedily climb the ``2·d_flat``
      orthogonal ±1 neighbours to a true ±1 lattice local maximum.
    * **Phase B** — when no ±1 neighbour improves, the peak may sit at a finer
      fractional angle *inside* the current cell, reachable only by a longer (finer)
      lattice ray.  Probe the **whole coarse→finest fan** ``2^j · z ± e_i`` for
      ``j = 1 … K`` — doubling the angular resolution each level **up to the
      max-length resolution** (``K`` from :func:`doublings_to_resolution`; the last
      level's probe is snapped down to the ``SEARCH_MAX_TRAJ_LEN`` ray, the finest
      the cap allows) — and evaluate the whole fan in **one parallel batch**
      (:func:`resolution_probe_rays` + :func:`evaluate_neighbours` over ``pool``).
      If a superior ray is found, recenter on it, re-certify with Phase A, and
      probe again; once the fan up to the max-length resolution yields no
      improvement the resolution is exhausted and the climb stops.

    Doubling continues regardless of whether an intermediate level improved — a
    plateau at a coarse resolution is exactly *why* a finer one is probed — so the
    only stop conditions are (a) the max-length-resolution fan found nothing
    better, or (b) no novel in-cone ray exists.  The angular search is therefore
    bounded by the ``max_norm`` radius, not an arbitrary round count.

    All evaluation flows through :func:`evaluate_in_flatland` / :func:`evaluate_batch`,
    so walks are cached by primitive ray (``eval_ctx['seen_trajectories']``) and
    the same trajectory is never re-walked.  ``visited`` additionally prunes whole
    re-checks across multiple climbs (e.g. several tied best trajectories).

    :param cur_z: Starting integer flatland trajectory (identified / in-cone).
    :param cur_delta: δ at ``cur_z``.
    :param geom: Flatland geometry (cone filter + norm + real mapping).
    :param eval_ctx: Evaluation context for :func:`evaluate_in_flatland`.
    :param max_norm: Trajectory norm cap (the lattice resolution radius).
    :param traj_norm: Norm used for the length cap (``SEARCH_TRAJ_NORM``).
    :param improve_threshold: Minimum δ gain for a move/ray to count as strictly better.
    :param pool: Optional per-shard process pool for the neighbour / fan batches.
    :param visited: Optional set of primitive-ray keys, updated in place with every
        ray this climb evaluates; pass the same set across climbs to avoid
        re-checking shared trajectories.
    :param on_local_max: Optional callback ``(z, delta)`` invoked once when the
        final (resolution-exhausted) local maximum is reached.
    :return: ``(z, delta)`` at the refined discrete local maximum.
    """
    if visited is None:
        visited = set()

    # Phase A: reach a genuine ±1 lattice local maximum first.
    cur_z, cur_delta = discrete_hill_climb(
        cur_z, cur_delta,
        geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
        traj_norm=traj_norm, improve_threshold=improve_threshold, pool=pool,
    )
    visited.add(primitive_ray_key(cur_z, geom))

    # Phase B: probe the coarse→finest fan (up to the max-length resolution) around
    # the current center in one parallel batch; recenter on any improver and repeat
    # until the finest resolution yields nothing better.
    while True:
        rays = resolution_probe_rays(cur_z, geom, max_norm, traj_norm, visited)
        if not rays:
            break  # no novel in-cone ray up to the resolution — exhausted.

        results = evaluate_neighbours(rays, eval_ctx, pool)  # single parallel batch

        best_z, best_delta = cur_z, cur_delta
        for ray, (delta_n, identified_n) in zip(rays, results):
            visited.add(primitive_ray_key(ray, geom))
            if identified_n and delta_n > best_delta + improve_threshold:
                best_z, best_delta = ray, delta_n

        if best_delta <= cur_delta + improve_threshold:
            break  # max-length-resolution fan found nothing better -> stop.

        # Lock onto the superior ray and re-certify it with a fresh ±1 climb.
        cur_z, cur_delta = discrete_hill_climb(
            best_z, best_delta,
            geom=geom, eval_ctx=eval_ctx, max_norm=max_norm,
            traj_norm=traj_norm, improve_threshold=improve_threshold, pool=pool,
        )
        visited.add(primitive_ray_key(cur_z, geom))

    if on_local_max is not None:
        on_local_max(cur_z, cur_delta)
    return cur_z, cur_delta


# ---------------------------------------------------------------------------
# Concurrent (breadth-first) micro-climb of several anchors
# ---------------------------------------------------------------------------

class _Climb:
    """Mutable per-anchor state for :func:`parallel_micro_climb`.

    Advanced one *round* at a time so many climbs can share a single evaluation
    batch.  Mirrors :func:`discrete_micro_climb`: phase ``"A"`` is the greedy ±1
    ascent, phase ``"B"`` the resolution-doubling fan; ``visited`` is per-climb
    (matching the fresh ``visited`` the sequential version gets per anchor) and
    prunes only that climb's Phase-B re-probes.
    """

    __slots__ = ("center", "delta", "phase", "visited", "done")

    def __init__(self, center: np.ndarray, delta: float):
        self.center = center
        self.delta = delta
        self.phase = "A"
        self.visited: Set[Tuple[int, ...]] = set()
        self.done = False


def _advance_climb(
    c: _Climb,
    cands: List[np.ndarray],
    results: List[Tuple[float, bool]],
    geom: FlatlandGeometry,
    improve_threshold: float,
) -> None:
    """Advance one climb by a single round from its candidates' ``(δ, identified)``.

    Exactly the per-step logic of :func:`discrete_hill_climb` (phase A) and the
    Phase-B body of :func:`discrete_micro_climb` (phase B), applied to one round's
    worth of already-evaluated candidates.
    """
    if c.phase == "A":
        best_z, best_delta = c.center, c.delta
        for z_n, (delta_n, identified_n) in zip(cands, results):
            if identified_n and delta_n > best_delta + improve_threshold:
                best_z, best_delta = z_n, delta_n
        if best_delta > c.delta + improve_threshold:
            c.center, c.delta = best_z, best_delta          # moved; stay in phase A
        else:
            c.visited.add(primitive_ray_key(c.center, geom))  # ±1 local max → phase B
            c.phase = "B"
    else:  # phase "B"
        best_z, best_delta = c.center, c.delta
        for ray, (delta_n, identified_n) in zip(cands, results):
            c.visited.add(primitive_ray_key(ray, geom))
            if identified_n and delta_n > best_delta + improve_threshold:
                best_z, best_delta = ray, delta_n
        if not cands or best_delta <= c.delta + improve_threshold:
            c.done = True                                    # resolution exhausted
        else:
            c.center, c.delta = best_z, best_delta
            c.phase = "A"                                    # re-certify the improver


def parallel_micro_climb(
    anchors,
    *,
    geom: FlatlandGeometry,
    eval_ctx: dict,
    max_norm: float,
    traj_norm: str,
    improve_threshold: float,
    pool=None,
) -> List[Tuple[np.ndarray, float]]:
    """Micro-climb many anchors concurrently — one saturated batch per round.

    Semantically identical to calling :func:`discrete_micro_climb` on each anchor
    with its own fresh ``visited`` set: every climb's move depends only on its own
    candidates' δ (read from the shared walk cache ``eval_ctx['seen_trajectories']``),
    so interleaving the climbs changes only *scheduling*, not the result.

    The win is core utilisation.  Each round unions the candidate rays of **all**
    still-active climbs — their Phase-A ±1 neighbours or their Phase-B fans — into a
    single :func:`evaluate_neighbours` call, so a shard with *N* tied bests dispatches
    a few ~N×-wider batches instead of *N* separate thin ones (``2·d_flat`` at a
    time), keeping every core busy.  :func:`evaluate_batch` de-duplicates shared rays
    by trajectory id, so a ray on several climbs' frontiers is still walked once.

    :param anchors: iterable of ``(z, delta)`` start points (identified, in-cone).
    :param geom: flatland geometry (cone filter + norm + real mapping).
    :param eval_ctx: evaluation context for :func:`evaluate_neighbours`.
    :param max_norm: trajectory norm cap (lattice resolution radius).
    :param traj_norm: norm used for the length cap (``SEARCH_TRAJ_NORM``).
    :param improve_threshold: minimum δ gain for a move/ray to count as better.
    :param pool: optional per-shard process pool for the batched walks.
    :return: refined ``(z, delta)`` per anchor, aligned with *anchors*.
    """
    climbs = [_Climb(z, delta) for z, delta in anchors]
    while True:
        batch: List[np.ndarray] = []
        plan: List[Tuple[_Climb, int, List[np.ndarray]]] = []
        for c in climbs:
            if c.done:
                continue
            if c.phase == "A":
                cands = orthogonal_neighbours(c.center, geom, max_norm, traj_norm)
            else:
                cands = resolution_probe_rays(c.center, geom, max_norm, traj_norm, c.visited)
            plan.append((c, len(batch), cands))
            batch.extend(cands)
        if not plan:                       # every climb has converged
            break
        results = evaluate_neighbours(batch, eval_ctx, pool) if batch else []
        for c, start, cands in plan:
            _advance_climb(c, cands, results[start:start + len(cands)], geom, improve_threshold)
    return [(c.center, c.delta) for c in climbs]
