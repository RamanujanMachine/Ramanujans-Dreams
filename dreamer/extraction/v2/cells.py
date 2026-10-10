r"""
Sign-pattern (cell) enumeration for a hyperplane arrangement.

Given ``N`` hyperplanes ``A x + c = 0`` in :math:`\mathbb{R}^D`, the
arrangement partitions space into open cells; each cell is uniquely
labelled by a sign vector ``s in {-1, +1}^N``.  Iterating over all
``2^N`` sign vectors is infeasible past ``N ~ 25``, but the number of
*non-empty* cells is bounded by :math:`O(N^D)`.

Enumeration: Avis–Fukuda reverse search
=======================================

We enumerate cells by **memoryless reverse search** (Avis & Fukuda,
"Reverse search for enumeration").  Pick a generic *base* cell (sign
vector ``base``); every other non-empty cell ``c`` is assigned a unique
*parent*

    parent(c) = flip ``c`` at the minimum index ``i`` such that
                ``c[i] != base[i]``  (separating)  and
                ``flip(c, i)`` is itself a non-empty cell.

Because ``parent`` strictly reduces the Hamming distance to ``base``,
iterating it from any cell reaches ``base`` — so the parent pointers
form a spanning tree rooted at the base cell.  We enumerate by walking
that tree *forwards*: a neighbour ``c'`` of ``c`` (one sign flipped) is
a child of ``c`` iff ``parent(c') == c``.

Why reverse search over BFS?  No ``seen`` set is needed (each cell is
reached through its unique parent exactly once), so

* memory is ``O(tree depth)`` instead of ``O(#cells)``, and
* disjoint subtrees can be dispatched to separate processes with zero
  shared state — see ``num_workers``.

Feasibility backend (hot-started stateful solver)
=================================================

The per-cell non-emptiness test dominates runtime.  We build the LP
**once** and navigate by swapping variable *bounds* only:

* Coordinate variables ``x in [-bound, bound]^D``.
* Auxiliary variables ``y_i`` with static constraints
  ``A_i . x + c_i - y_i = 0`` (never modified — the simplex basis is
  reused across checks).
* A sign vector maps to ``y`` bounds: ``s_i = +1`` => ``y_i in [eps, inf)``;
  ``s_i = -1`` => ``y_i in (-inf, -eps]``.  Feasibility of those bounds
  proves the open cell has interior of slack ``>= eps``.

CBC (via ``python-mip``) hot-starts from the previous basis because
only bounds change.  When ``python-mip`` is unavailable we fall back to
the original per-call ``scipy`` LP so the module never hard-fails.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .symmetry import SymmetryStrategy

try:  # Optional fast path — see module docstring.
    import mip

    _HAS_MIP = True
except ImportError:  # pragma: no cover - depends on the environment
    mip = None  # type: ignore[assignment]
    _HAS_MIP = False


SignTuple = Tuple[int, ...]

# Feasibility predicate: maps a candidate sign vector to "does this open
# cell have non-empty interior?".
FeasibilityChecker = Callable[[np.ndarray], bool]


class ExtractionTimeout(RuntimeError):
    """Raised when cell enumeration passes its wall-clock ``deadline``.

    Carries an optional ``partial`` payload so a caller that was
    streaming results (e.g. the exact extractor interleaving
    classification) can attach whatever it had completed before the
    deadline, letting the manager salvage it instead of discarding.
    """

    def __init__(self, *args, partial=None):
        super().__init__(*args)
        self.partial = partial


# ---------------------------------------------------------------------------
# Hot-started python-mip feasibility solver (bound swapping)
# ---------------------------------------------------------------------------

class _StatefulFeasibilitySolver:
    """
    Hot-started LP feasibility oracle for the cells of an arrangement.

    Builds the LP once; ``feasible(sign_vector)`` only rewrites the
    ``y`` bounds and re-optimises, reusing the simplex basis in memory.
    """

    def __init__(
        self,
        A: np.ndarray,
        c: np.ndarray,
        *,
        bound: float = 1e6,
        epsilon: float = 1e-6,
    ):
        if not _HAS_MIP:  # pragma: no cover - guarded by caller
            raise RuntimeError("python-mip is not available")
        self.n, self.d = A.shape
        # CBC's primal feasibility tolerance is ~1e-7: an epsilon below
        # it would let CBC treat an empty cell's contradictory bounds
        # (y_i >= eps AND y_i <= -eps) as satisfiable at y_i ~ 0, wrongly
        # reporting the cell feasible.  Floor the slack comfortably above
        # that tolerance.  Safe for integer arrangements — a genuinely
        # full-dimensional cell has O(1) slack, so this never rejects a
        # real cell.
        self.epsilon = max(float(epsilon), 1e-6)

        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0  # suppress solver chatter for speed

        self._x = [model.add_var(lb=-bound, ub=bound) for _ in range(self.d)]
        self._y = [model.add_var(lb=-mip.INF, ub=mip.INF) for _ in range(self.n)]

        for i in range(self.n):
            model += (
                mip.xsum(float(A[i, j]) * self._x[j] for j in range(self.d))
                + float(c[i])
                - self._y[i]
                == 0
            )
        model.objective = 0  # feasibility only — any feasible point is optimal

        self._model = model

    def feasible(self, sign_vector: np.ndarray) -> bool:
        """
        Return ``True`` iff the open cell labelled by ``sign_vector``
        has interior of slack ``>= epsilon``.
        """
        eps = self.epsilon
        for i in range(self.n):
            if sign_vector[i] > 0:
                self._y[i].lb = eps
                self._y[i].ub = mip.INF
            else:
                self._y[i].lb = -mip.INF
                self._y[i].ub = -eps
        status = self._model.optimize()
        return status in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        )


# ---------------------------------------------------------------------------
# Fallback: per-call scipy LP (used when python-mip is unavailable)
# ---------------------------------------------------------------------------

def _solve_interior(
    A: np.ndarray,
    c: np.ndarray,
    sign_vector: np.ndarray,
    *,
    bound: float = 1e6,
):
    """
    Maximise the interior slack of a cell.

    Returns ``(slack, x)`` where ``slack`` is the maximum Chebyshev-style
    slack (``-inf`` if the LP fails) and ``x`` is the slack-maximising
    interior point (``None`` on failure).  ``x`` is strictly inside the
    open cell whenever ``slack > 0``.
    """
    n, d = A.shape
    obj = np.zeros(d + 1, dtype=np.float64)
    obj[-1] = -1.0
    A_ub = np.zeros((n, d + 1), dtype=np.float64)
    A_ub[:, :d] = -(sign_vector[:, None] * A)
    A_ub[:, -1] = 1.0
    b_ub = (sign_vector * c).astype(np.float64)

    bounds = [(-bound, bound)] * d + [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return float("-inf"), None
    return float(res.x[-1]), np.asarray(res.x[:d], dtype=np.float64)


def _interior_slack(
    A: np.ndarray,
    c: np.ndarray,
    sign_vector: np.ndarray,
    *,
    bound: float = 1e6,
) -> float:
    """
    Return the maximum interior slack of a cell, or ``-inf`` if the LP
    fails.  Kept as the fallback feasibility backend for environments
    without ``python-mip``.
    """
    return _solve_interior(A, c, sign_vector, bound=bound)[0]


def _make_feasibility_checker(
    A: np.ndarray,
    c: np.ndarray,
    *,
    epsilon: float,
    bound: float = 1e6,
) -> FeasibilityChecker:
    """
    Build the per-cell feasibility predicate.

    Prefers the hot-started ``python-mip`` solver; transparently falls
    back to the per-call scipy LP when mip is unavailable.
    """
    if _HAS_MIP:
        solver = _StatefulFeasibilitySolver(A, c, bound=bound, epsilon=epsilon)
        return solver.feasible
    return lambda sign_vector: _interior_slack(A, c, sign_vector, bound=bound) > epsilon


# Predicate returning a cell's interior point (or ``None`` if empty).
# Used by the symmetry-aware canonical enumerator, which needs an actual
# interior point to teleport into the fundamental domain.
InteriorPointFinder = Callable[[np.ndarray], Optional[np.ndarray]]


def _make_interior_point_finder(
    A: np.ndarray,
    c: np.ndarray,
    *,
    epsilon: float,
    bound: float = 1e6,
) -> InteriorPointFinder:
    """
    Build a predicate ``sign_vector -> interior point or None``.

    Returns the **slack-maximising** (deep-interior) point of the open
    cell, or ``None`` if the cell is empty (slack ``<= epsilon``).

    The deep-interior point is essential for the symmetry fingerprint.  A
    plain feasibility LP (e.g. CBC with a zero objective) returns a
    *vertex*, which lies on the arrangement's hyperplanes and frequently
    on a symmetry wall ``x_i = x_j``; teleporting such a degenerate point
    into the fundamental domain yields an inconsistent canonical signature
    (it under- or over-counts orbits).  The slack-maximising point is
    strictly inside every facet, so its teleported signature is a sound
    orbit invariant — a cell that genuinely straddles a wall ``x_i = x_j``
    is itself invariant under that swap, so its deep-interior points all
    teleport to the same canonical cell.

    We always use the scipy slack-maximising LP here (the hot-started CBC
    feasibility solver maximises no slack).  Symmetry reduction targets
    moderate arrangements (the exact method is not run on huge CMFs), so
    the per-call cost is acceptable.
    """
    def finder(sign_vector: np.ndarray) -> Optional[np.ndarray]:
        slack, x = _solve_interior(A, c, sign_vector, bound=bound)
        if x is None or slack <= epsilon:
            return None
        return x

    return finder


# ---------------------------------------------------------------------------
# Unboundedness via the recession cone (replaces a per-cell lrs subprocess)
# ---------------------------------------------------------------------------
#
# A non-empty (full-dimensional) cell C = { x : s_i (A_i . x + c_i) > 0 }
# is *unbounded* iff its recession cone  K = { d : s_i (A_i . d) >= 0 }
# contains a nonzero direction (a ray along which C escapes to infinity).
# The constants c_i drop out — only the linear parts and the signs matter.
#
# Two cases:
#
#   1. Lineality.  If rank(A) < D there is a nonzero d with A.d = 0 (hence
#      s_i A_i d = 0 >= 0 for every cell), so *every* cell is unbounded.
#      rank(A) is the same for all cells (sign flips don't change rank),
#      so we test it once.
#
#   2. Full rank (rank(A) = D, no lineality).  Then A.d = 0 only at d = 0,
#      so any nonzero d in K has some s_i (A_i d) > 0.  We maximise
#      f(d) = sum_i s_i (A_i . d)  over  K ∩ [-1,1]^D.  f >= 0 always
#      (every term is >= 0 on K); f = 0 exactly when K = {0} (bounded),
#      and f > 0 when a nonzero recession direction exists (unbounded).
#
# The LP uses the same hot-started bound-swap trick as the feasibility
# solver: y_i = A_i . d are auxiliary variables whose sign bounds encode
# the cone (s_i = +1 -> y_i >= 0, s_i = -1 -> y_i <= 0); only those bounds
# and the objective coefficients (the signs) change between cells.

UnboundedChecker = Callable[[np.ndarray], bool]

# f(d) is exactly 0 for bounded cells and bounded-below by an O(1) value
# for unbounded ones on integer arrangements; any small positive tol
# cleanly separates the two (well above CBC's ~1e-7 tolerance).
_UNBOUNDED_TOL = 1e-6


class _StatefulUnboundedSolver:
    """
    Hot-started LP oracle: ``unbounded(sign_vector)`` for the cells of an
    arrangement.  Build once, swap ``y`` sign bounds + objective signs
    per cell.
    """

    def __init__(self, A: np.ndarray, *, tol: float = _UNBOUNDED_TOL):
        if not _HAS_MIP:  # pragma: no cover - guarded by caller
            raise RuntimeError("python-mip is not available")
        self.n, self.d = A.shape
        self.tol = tol

        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0

        # Direction d (unit box) and y_i = A_i . d.
        self._d = [model.add_var(lb=-1.0, ub=1.0) for _ in range(self.d)]
        self._y = [model.add_var(lb=-mip.INF, ub=mip.INF) for _ in range(self.n)]

        for i in range(self.n):
            # Skip zero coefficients: a 0.0 * var term can confuse CBC's
            # presolve (see the feasibility-solver notes).
            terms = [
                float(A[i, j]) * self._d[j]
                for j in range(self.d)
                if A[i, j] != 0
            ]
            model += mip.xsum(terms) - self._y[i] == 0
        self._model = model

    def unbounded(self, sign_vector: np.ndarray) -> bool:
        # Recession cone: s_i = +1 -> y_i >= 0 ; s_i = -1 -> y_i <= 0.
        for i in range(self.n):
            if sign_vector[i] > 0:
                self._y[i].lb, self._y[i].ub = 0.0, mip.INF
            else:
                self._y[i].lb, self._y[i].ub = -mip.INF, 0.0
        # Maximise sum_i s_i y_i  ==  minimise sum_i (-s_i) y_i.
        self._model.objective = mip.minimize(
            mip.xsum(-float(sign_vector[i]) * self._y[i] for i in range(self.n))
        )
        status = self._model.optimize()
        if status not in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        ):
            # d = 0 is always feasible, so this should not happen; treat
            # an unexpected solver state as "bounded" (conservative).
            return False
        # objective_value = -max(sum s_i y_i); > tol ⇒ nonzero recession.
        return (-self._model.objective_value) > self.tol


def _recession_unbounded_scipy(
    A: np.ndarray, sign_vector: np.ndarray, *, tol: float = _UNBOUNDED_TOL
) -> bool:
    """scipy fallback for :class:`_StatefulUnboundedSolver` (full-rank case)."""
    n, d = A.shape
    s = sign_vector.astype(np.float64)
    sA = s[:, None] * A  # rows s_i A_i
    # maximise sum_i (s_i A_i) . d  ==  minimise -(sum_i s_i A_i) . d
    obj = -sA.sum(axis=0)
    # cone: s_i A_i . d >= 0  =>  -(s_i A_i) . d <= 0
    A_ub = -sA
    b_ub = np.zeros(n)
    bounds = [(-1.0, 1.0)] * d
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return False
    return (-float(res.fun)) > tol


def make_unbounded_checker(
    A: np.ndarray, *, tol: float = _UNBOUNDED_TOL
) -> UnboundedChecker:
    """
    Build the per-cell unboundedness predicate (recession-cone LP).

    Handles the lineality case once via ``rank(A)``: if ``rank(A) < D``
    every cell is unbounded.  Otherwise prefers the hot-started
    ``python-mip`` solver, falling back to the per-call scipy LP when mip
    is unavailable.  Needs only ``A`` (the constants ``c`` do not affect
    unboundedness).
    """
    A = np.asarray(A, dtype=np.int64)
    d = A.shape[1]
    if np.linalg.matrix_rank(A) < d:
        # A nonzero direction d with A.d = 0 is a recession direction of
        # every cell -> all cells unbounded.
        return lambda sign_vector: True
    if _HAS_MIP:
        return _StatefulUnboundedSolver(A, tol=tol).unbounded
    return lambda sign_vector: _recession_unbounded_scipy(
        A, np.asarray(sign_vector, dtype=np.int64), tol=tol
    )


# ---------------------------------------------------------------------------
# Starting / base cell
# ---------------------------------------------------------------------------

def _sign_at(A: np.ndarray, c: np.ndarray, x: np.ndarray) -> Optional[np.ndarray]:
    """Return the sign vector at ``x`` or :data:`None` if ``x`` lies on a hyperplane."""
    vals = A @ x + c
    if np.any(vals == 0):
        return None
    return np.where(vals > 0, 1, -1).astype(np.int64)


def _find_start_cell(
    A: np.ndarray,
    c: np.ndarray,
    *,
    rng: np.random.Generator,
    max_attempts: int = 64,
    radius: int = 1000,
) -> np.ndarray:
    """
    Find any non-empty cell (the reverse-search base/root) by random
    integer sampling, **biased toward the origin**.

    Reverse search reaches every cell regardless of the base — the base
    only sets the traversal *order*.  Near the origin the arrangement's
    cells are "fat" and integer-rich, so a base there front-loads the
    shard-yielding cells; under a wall-clock deadline that roughly doubles
    the salvaged shard count vs a far-out base (measured ~1.6x on the
    7-D 4F3 arrangement).  We therefore sample a tight near-origin box
    first and only widen it (geometric growth up to ``radius``) if no
    off-hyperplane point turns up — tight boxes land exactly on a
    hyperplane more often, so the wider fallback guarantees a base.

    :param max_attempts: Samples drawn at *each* radius level before
        widening.  The tight first level gets a generous floor (it is the
        one we most want to succeed at) regardless of this value.
    :raises RuntimeError: If no off-hyperplane point is found at any level.
    """
    d = A.shape[1]
    rad = min(3, radius)
    # Try hard at the tightest box (we want a near-origin base); fall back
    # to the caller's per-level budget once we start widening.
    attempts_here = max(max_attempts, 300)
    total = 0
    while True:
        for _ in range(attempts_here):
            total += 1
            x = rng.integers(-rad, rad + 1, size=d, dtype=np.int64)
            sig = _sign_at(A, c, x)
            if sig is not None:
                return sig
        if rad >= radius:
            break
        rad = min(rad * 4, radius)
        attempts_here = max_attempts
    raise RuntimeError(
        f"Could not find a starting cell after {total} random samples"
    )


# ---------------------------------------------------------------------------
# Reverse-search primitives
# ---------------------------------------------------------------------------

def _parent(
    base: np.ndarray, sign: np.ndarray, is_feasible: FeasibilityChecker, n: int
) -> Optional[np.ndarray]:
    """
    Return the parent of ``sign`` in the reverse-search tree, or
    :data:`None` if ``sign`` is the base cell.

    The parent flips the minimum-index *separating* hyperplane (one
    whose sign differs from ``base``) whose flip yields a non-empty
    cell.  This reduces the Hamming distance to ``base`` by one, so the
    parent chain always terminates at the root.
    """
    for i in range(n):
        if sign[i] != base[i]:  # separating hyperplane
            cand = sign.copy()
            cand[i] = -cand[i]
            if is_feasible(cand):
                return cand
    return None


def _reverse_search_iter(
    base: np.ndarray,
    root: np.ndarray,
    is_feasible: FeasibilityChecker,
    n: int,
    *,
    max_cells: Optional[int],
    deadline: Optional[float],
):
    """
    Yield every cell in the subtree rooted at ``root`` (with parent
    pointers defined relative to ``base``).

    A DFS over the spanning tree: from each cell we emit it, then push
    every neighbour whose parent is the current cell.  No ``seen`` set —
    the unique-parent property guarantees each cell is visited once.

    Yielding (rather than returning a list) lets a consumer interleave
    work per cell and salvage partial results on a deadline hit.

    :raises ExtractionTimeout: If ``deadline`` (wall-clock ``time.time``)
        is passed.
    :raises RuntimeError: If ``max_cells`` is not ``None`` and more than
        that many cells are emitted.
    """
    stack: List[np.ndarray] = [root.copy()]
    count = 0

    while stack:
        # Each cell costs ~2N LP solves, so a per-iteration time.time()
        # check (microseconds) is free relative to the work and gives
        # tight deadline adherence -- the caller can fall back promptly.
        if deadline is not None and time.time() > deadline:
            raise ExtractionTimeout(
                f"Cell enumeration passed its deadline after {count} cells"
            )
        sig = stack.pop()
        yield tuple(sig.tolist())
        count += 1
        # ``max_cells`` is an optional safety ceiling; the deadline is the
        # primary stop.  Reverse search is memoryless (O(depth) stack, no
        # ``seen`` set), so a large/None cap does not bloat RAM during
        # enumeration -- only the caller's kept output grows.
        if max_cells is not None and count > max_cells:
            raise RuntimeError(f"Cell enumeration exceeded max_cells={max_cells}")

        for i in range(n):
            child = sig.copy()
            child[i] = -child[i]
            if not is_feasible(child):
                continue
            par = _parent(base, child, is_feasible, n)
            if par is not None and np.array_equal(par, sig):
                stack.append(child)


def iter_cells(
    A: np.ndarray,
    c: np.ndarray,
    *,
    max_cells: Optional[int] = None,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
    deadline: Optional[float] = None,
):
    """
    Stream the non-empty cells of the arrangement (serial reverse
    search), yielding each sign-encoding as it is discovered.

    Unlike :func:`enumerate_cells` this does not collect or sort — it is
    meant for a consumer that processes cells incrementally (e.g. the
    exact extractor, which classifies + locates each cell on the fly so
    it can salvage partial results if a ``deadline`` is hit).

    Parameters mirror :func:`enumerate_cells` (minus ``num_workers`` —
    streaming is inherently serial).
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")

    rng = np.random.default_rng(seed)
    base = _find_start_cell(A, c, rng=rng)
    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)
    yield from _reverse_search_iter(
        base, base, is_feasible, A.shape[0], max_cells=max_cells, deadline=deadline
    )


def reverse_search_seeds(
    A: np.ndarray,
    c: np.ndarray,
    *,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Return ``(base, root_children)`` for parallel reverse search.

    ``base`` is a generic starting cell; ``root_children`` are its
    feasible neighbours — each the root of a **disjoint** subtree (every
    other cell has a unique parent chain leading back through exactly one
    of them).  A caller can dispatch each subtree to its own process and
    must additionally handle ``base`` itself (it belongs to no subtree).
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    base = _find_start_cell(A, c, rng=rng)
    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)
    children: List[np.ndarray] = []
    for i in range(n):
        child = base.copy()
        child[i] = -child[i]
        if is_feasible(child):
            children.append(child)
    return base, children


def iter_subtree(
    A: np.ndarray,
    c: np.ndarray,
    base: np.ndarray,
    root: np.ndarray,
    *,
    max_cells: Optional[int] = None,
    epsilon: float = 1e-6,
    deadline: Optional[float] = None,
):
    """
    Stream the cells of the subtree rooted at ``root`` (parent pointers
    relative to ``base``).  Used by parallel workers, each of which
    builds its own feasibility solver.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    base = np.asarray(base, dtype=np.int64)
    root = np.asarray(root, dtype=np.int64)
    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)
    yield from _reverse_search_iter(
        base, root, is_feasible, A.shape[0], max_cells=max_cells, deadline=deadline
    )


# ---------------------------------------------------------------------------
# Parallel subtree worker (top-level so it is picklable)
# ---------------------------------------------------------------------------

def _subtree_worker(args) -> List[SignTuple]:
    """
    Enumerate one subtree in a child process.

    Each worker builds its **own** feasibility solver (CBC models are
    not picklable, and a per-process solver is exactly what we want for
    parallel scaling).
    """
    A, c, base, root, epsilon, max_cells, deadline = args
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    base = np.asarray(base, dtype=np.int64)
    root = np.asarray(root, dtype=np.int64)
    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)
    return list(
        _reverse_search_iter(
            base, root, is_feasible, A.shape[0], max_cells=max_cells, deadline=deadline
        )
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enumerate_cells(
    A: np.ndarray,
    c: np.ndarray,
    *,
    max_cells: Optional[int] = None,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
    deadline: Optional[float] = None,
    num_workers: int = 1,
) -> List[SignTuple]:
    """
    Enumerate every non-empty cell of the arrangement via reverse search.

    :param A: Hyperplane coefficient matrix, shape ``(N, D)``.
    :param c: Hyperplane constants, shape ``(N,)``.
    :param max_cells: Optional safety ceiling (``None`` = unbounded;
        the ``deadline`` is the intended stop).  Raises if exceeded.
        Reverse search is memoryless, so a large/``None`` cap does not
        bloat RAM during enumeration.
    :param seed: RNG seed used to find the base/root cell.
    :param epsilon: Minimum interior slack for a cell to count as
        non-empty.  Floored at 1e-6 (above CBC's feasibility tolerance).
    :param deadline: Optional wall-clock (``time.time()``) instant after
        which enumeration aborts with :class:`ExtractionTimeout`.  This
        is what lets a caller bound the exact method and fall back to a
        heuristic instead of running unbounded.
    :param num_workers: ``> 1`` dispatches disjoint root subtrees across
        that many processes (reverse search needs no shared state).
        ``1`` (default) runs serially.    :return: Sorted list of sign-encoding tuples (``+1`` / ``-1``).
    :raises ExtractionTimeout: If ``deadline`` is passed.
    :raises RuntimeError: If ``max_cells`` is exceeded or no starting
        cell can be found.
    :raises ValueError: For inconsistent input shapes.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    base = _find_start_cell(A, c, rng=rng)

    if num_workers is None or num_workers <= 1:
        is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)
        result = list(
            _reverse_search_iter(
                base, base, is_feasible, n, max_cells=max_cells, deadline=deadline
            )
        )
        return sorted(result)

    return _enumerate_parallel(
        A, c, base, n,
        max_cells=max_cells,
        epsilon=epsilon,
        deadline=deadline,
        num_workers=num_workers,
    )


def _enumerate_parallel(
    A: np.ndarray,
    c: np.ndarray,
    base: np.ndarray,
    n: int,
    *,
    max_cells: Optional[int],
    epsilon: float,
    deadline: Optional[float],
    num_workers: int,
) -> List[SignTuple]:
    """
    Parallel reverse search: enumerate the base cell's children serially
    (cheap), then dispatch each child's disjoint subtree to a worker.
    """
    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)

    # The base cell's children are exactly its feasible neighbours (each
    # such neighbour's parent is the base, by the min-index rule, since
    # they differ from base in a single bit).
    root_children: List[np.ndarray] = []
    for i in range(n):
        child = base.copy()
        child[i] = -child[i]
        if is_feasible(child):
            root_children.append(child)

    if len(root_children) < 2:
        # Nothing to parallelise — fall back to a single serial sweep.
        result = list(
            _reverse_search_iter(
                base, base, is_feasible, n, max_cells=max_cells, deadline=deadline
            )
        )
        return sorted(result)

    tasks = [
        (A, c, base, child, epsilon, max_cells, deadline)
        for child in root_children
    ]
    # 'spawn' would re-import; 'fork' (default on Linux/WSL) is cheap and
    # lets workers inherit the imported modules.
    from dreamer.utils.resilient_pool import ResilientPool
    collected = {tuple(base.tolist())}
    with ResilientPool(min(num_workers, len(tasks)), name="cell enumeration") as pool:
        try:
            for subtree in pool.imap_unordered(_subtree_worker, tasks):
                collected.update(subtree)
                if max_cells is not None and len(collected) > max_cells:
                    raise RuntimeError(
                        f"Cell enumeration exceeded max_cells={max_cells}"
                    )
        except BaseException:
            pool.terminate()
            raise
    return sorted(collected)


# ---------------------------------------------------------------------------
# Symmetry-aware enumeration (canonical-teleportation BFS)
# ---------------------------------------------------------------------------
#
# When a SymmetryStrategy is active we want exactly *one* representative
# cell per symmetry orbit.  Avis-Fukuda reverse search cannot be pruned
# on the fly — dropping a symmetric branch breaks its spanning tree and
# silently loses non-symmetric cells reachable only through it.  We
# therefore switch to a BFS over the (full, unconstrained) cell graph that
# deduplicates by *canonical signature*:
#
#   * For each feasible cell, take a strictly-interior point, teleport it
#     into the fundamental domain via ``symmetry.apply``, and fingerprint
#     it as ``sign(A @ canonical_point + c)``.
#   * If the fingerprint was seen, discard the cell and DO NOT expand it.
#   * Otherwise record the fingerprint, emit the cell, and queue its
#     neighbours.
#
# Completeness (one representative per orbit, none missed) holds *because*
# the arrangement is invariant under the symmetry group: a group element
# mapping cell ``r'`` to the chosen representative ``r`` also maps any
# neighbour of ``r'`` to a neighbour of ``r``, so every orbit adjacent to
# ``r``'s orbit is reachable from ``r`` alone.  Verified empirically
# against brute force on invariant arrangements.  Memory is O(#orbits)
# (the ``seen`` set), which is the whole point — #orbits << #cells.


def _canonical_signature(
    A: np.ndarray,
    c: np.ndarray,
    point: np.ndarray,
    symmetry: SymmetryStrategy,
) -> SignTuple:
    """Fingerprint a cell by the sign vector of its teleported interior point."""
    canon = symmetry.canonical_point(point)
    vals = A @ canon + c
    return tuple(np.sign(vals).astype(np.int64).tolist())


def iter_cells_canonical(
    A: np.ndarray,
    c: np.ndarray,
    symmetry: SymmetryStrategy,
    *,
    max_cells: Optional[int] = None,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
    deadline: Optional[float] = None,
):
    """
    Stream one representative cell per symmetry orbit (canonical-teleport
    BFS).  Yields each representative's sign-encoding as it is discovered.

    :param symmetry: The CMF family's :class:`SymmetryStrategy`; its
        ``apply`` maps interior points into a fundamental domain so
        orbit-equivalent cells share a canonical signature.
    :raises ExtractionTimeout: If ``deadline`` is passed.
    :raises RuntimeError: If ``max_cells`` representatives are exceeded.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    base = _find_start_cell(A, c, rng=rng)
    point_of = _make_interior_point_finder(A, c, epsilon=epsilon)

    seen = set()  # canonical signatures already represented
    queue: deque = deque()
    count = 0

    base_pt = point_of(base)
    if base_pt is None:  # pragma: no cover - base came from a real point
        return
    seen.add(_canonical_signature(A, c, base_pt, symmetry))
    queue.append(base)
    yield tuple(base.tolist())
    count += 1

    while queue:
        if deadline is not None and time.time() > deadline:
            raise ExtractionTimeout(
                f"Canonical cell enumeration passed its deadline after "
                f"{count} representatives"
            )
        sig = queue.popleft()
        for i in range(n):
            child = sig.copy()
            child[i] = -child[i]
            pt = point_of(child)
            if pt is None:
                continue  # empty cell — not a neighbour in the arrangement graph
            csig = _canonical_signature(A, c, pt, symmetry)
            if csig in seen:
                continue  # symmetric duplicate — prune the whole orbit branch
            seen.add(csig)
            queue.append(child)
            yield tuple(child.tolist())
            count += 1
            if max_cells is not None and count > max_cells:
                raise RuntimeError(
                    f"Canonical enumeration exceeded max_cells={max_cells}"
                )


def enumerate_cells_canonical(
    A: np.ndarray,
    c: np.ndarray,
    symmetry: SymmetryStrategy,
    *,
    max_cells: Optional[int] = None,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
    deadline: Optional[float] = None,
) -> List[SignTuple]:
    """
    Enumerate one representative per symmetry orbit; sorted list of
    sign-encodings.  See :func:`iter_cells_canonical`.
    """
    return sorted(
        iter_cells_canonical(
            A, c, symmetry,
            max_cells=max_cells, seed=seed, epsilon=epsilon, deadline=deadline,
        )
    )
