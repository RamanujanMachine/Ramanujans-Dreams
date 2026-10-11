from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache, cached_property
from typing import List, Optional, TYPE_CHECKING, Tuple

import sympy as sp
from sympy.abc import n

from LIReC.db.access import db
from ramanujantools import LinearRecurrence, Matrix, Limit, linear_recurrence
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searchable import Searchable
from dreamer.configs import config

search_config = config.search

if TYPE_CHECKING:
    from dreamer.utils.storage.dtos import TrajectoryDTO


# ---------------------------------------------------------------------------
# Module-level helpers — stable IDs and position conversion
# ---------------------------------------------------------------------------

def _stable_id(*parts: str, length: int = 16) -> str:
    """SHA-256 of pipe-joined parts, truncated to ``length`` hex chars.

    Deterministic across runs and processes (unlike Python's built-in ``hash``).
    """
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:length]


def _mpc_to_sympy(value) -> sp.Expr:
    """Convert an ``mpmath`` real/complex number to a sympy expression.

    Real values (including ``mpf``) become a :class:`sympy.Float`; complex values
    become ``re + im*I``.  ``mpmath.inf`` maps to :data:`sympy.oo`.  Used to keep
    the coboundary eigenvalue path's return type identical to the symbolic
    ``Matrix.sorted_eigenvals`` it replaces.
    """
    import mpmath as mp

    if value == mp.inf:
        return sp.oo
    if value == -mp.inf:
        return -sp.oo
    im = mp.im(value)
    if im == 0:
        return sp.Float(str(mp.nstr(mp.re(value), 30)))
    return sp.Float(str(mp.nstr(mp.re(value), 30))) + sp.Float(str(mp.nstr(im, 30))) * sp.I


def _position_to_tuple(pos) -> tuple:
    """Convert a ramanujantools.Position (dict-like) to a plain tuple of
    JSON-serializable values.

    Integer coordinates become ``int``; non-integer coordinates (e.g. a
    ``sympy.Rational`` like ``7/2`` arising from a rational shift) become a
    ``str`` that round-trips cleanly through ``sympify``.  Using
    :func:`_pq_to_jsonsafe` here is important: ``int(sympy.Rational(7, 2))``
    returns ``3`` *without raising*, so a naive ``int()`` would silently
    truncate fractional start points to whole numbers.
    """
    return tuple(_pq_to_jsonsafe(v) for v in pos.values())


def _pq_to_jsonsafe(v) -> object:
    """Return ``v`` as ``int`` when it's an integer (sympy or Python),
    otherwise as ``str``.  p/q coefficients from LIReC are usually ints
    but can be ``sympy.Rational`` like ``1/2``; converting fractions to
    ``int`` would either truncate or raise — strings round-trip cleanly
    through ``sympify`` if needed downstream.
    """
    try:
        if getattr(v, "is_Integer", False) or isinstance(v, int):
            return int(v)
    except Exception:
        pass
    return str(v)


def _trajectory_norm(trajectory) -> float:
    """Euclidean norm of a ``Position`` (or any dict-like) used as a
    trajectory direction — mirrors ``np.linalg.norm`` over its values."""
    return math.sqrt(sum(float(v) ** 2 for v in trajectory.values()))


def _serialize_encoding(shard) -> str:
    """Canonical string form of the shard's ±1 sign vector.

    The extractor produces hyperplanes in a canonical sorted order, so
    ``shard.encoding[i]`` unambiguously refers to ``cmf.hyperplanes[i]``.
    Joining the ±1 values with commas gives a deterministic, compact
    label suitable for hashing into ``shard_id`` / ``trajectory_id``.
    Whole-space shards (no encoding) produce a fixed placeholder.
    """
    encoding = getattr(shard, "encoding", None)
    if not encoding:
        return "whole_space"
    return ",".join(str(int(s)) for s in encoding)


def derive_cmf_and_shard_ids(shard) -> tuple[str, str, str]:
    """Return ``(cmf_id, shard_id, shard_encoding_str)`` for *shard*.

    * ``cmf_id`` — the CMF name (unique per CMF in the current system).
    * ``shard_id`` — structural id ``"{cmf_id}__{encoding_hash}"`` where
      ``encoding_hash`` is a stable SHA-256 truncation of
      ``(cmf_id, shard_encoding_str)``.  Embedding the cmf_id literally
      makes shard ids self-describing — any record's shard id discloses
      its parent CMF without a separate lookup, and the filenames written
      by the pipeline can simply be ``{shard_id}.jsonl``.
    * ``shard_encoding_str`` — canonical ±1 sign vector string (see
      :func:`_serialize_encoding`).  Also used as part of trajectory ids
      so the two levels stay consistent.
    """
    cmf_id = shard.cmf_name
    shard_encoding_str = _serialize_encoding(shard)
    encoding_hash = _stable_id(cmf_id, shard_encoding_str)
    shard_id = f"{cmf_id}__{encoding_hash}"
    return cmf_id, shard_id, shard_encoding_str


def walk_depth_for(cmf, direction) -> int:
    """Walk depth a trajectory in *direction* through *cmf* will use.

    Mirrors the default depth resolution inside
    :meth:`TrajectoryAttributesHandler.from_cmf` —
    ``search.DEPTH_FROM_TRAJECTORY_LEN(||direction||, cmf.dim())`` — so callers
    can predict the depth of a (not-yet-built) trajectory cheaply (no walk).
    Used by :func:`tier1_config_fingerprint` to detect when a re-run requests a
    different (e.g. deeper) walk than a cached record was computed with.
    """
    return int(search_config.DEPTH_FROM_TRAJECTORY_LEN(_trajectory_norm(direction), cmf.dim()))


def tier1_config_fingerprint(walk_depth: int) -> str:
    """Stable fingerprint of the config knobs that influence Tier-1 values.

    Two trajectory computations with the same ``trajectory_id`` are
    interchangeable **only** when every configuration input that feeds the
    Tier-1 attributes (``delta``, ``identified``, ``limit``, ``p``/``q``) is
    unchanged.  When any of them differs, a cached record is stale and must be
    recomputed — this is what lets a later run with, e.g., a deeper walk
    (``DEPTH_FROM_TRAJECTORY_LEN``) or a different walk style
    (``DEFAULT_USES_INV_T``) override previously stored values instead of
    silently reusing them.

    The inputs, and the attributes they affect:

    * ``walk_depth`` — the per-trajectory walk depth (passed in; derived from
      ``DEPTH_FROM_TRAJECTORY_LEN`` and the trajectory length).  Affects every
      walk-derived value: ``limit``, ``delta``, ``p``/``q``.
    * ``DEFAULT_USES_INV_T`` (walk type 1 vs 2) — changes the walked matrix, so
      affects all of the above.
    * ``DEPTH_CONVERGENCE_THRESHOLD``, ``LIMIT_DIFF_ERROR_BOUND`` — the
      convergence sanity check inside ``delta``.
    * ``MIN_ESTIMATE_DENOMINATOR`` — the denominator floor in ``delta_sequence``.
    * ``CACHE_ACCEPTANCE_THRESHOLD``, ``IDENTIFY_CHECK_THRESHOLD`` — the LIReC
      identification / cache-acceptance tolerances (``identified``, ``p``/``q``).
    * ``CONSTANT_NO_DIGITS_HIGH_RES`` / ``CONSTANT_NO_DIGITS_LOW_RES`` — the
      precision the target constant is evaluated at for identification and δ.

    Deliberately **excluded**: ``IDENTIFY_DEPTH``.  It only changes *where* the
    (depth-independent) p/q relation is identified, with a fallback to the full
    depth on failure, so it never changes the stored Tier-1 values — including it
    would spuriously invalidate every cached record.

    :param walk_depth: The walk depth used (or to be used) for this trajectory.
    :return: A 16-hex-char stable fingerprint string.
    """
    walk_type = 1 if search_config.DEFAULT_USES_INV_T else 2
    # The result depends only on these config values + walk_depth, so memoise
    # the JSON-serialise + hash step keyed on them.  Reading the config fresh on
    # every call (cheap attribute lookups) and keying on the values means the
    # cache stays correct under *any* config change — both ``config.configure``
    # and a direct ``setattr`` (e.g. test monkeypatching) — with no manual
    # invalidation, while still skipping the repeated dumps/hash on cache hits.
    key = (
        int(walk_depth),
        walk_type,
        tuple(float(x) for x in search_config.DEPTH_CONVERGENCE_THRESHOLD),
        float(search_config.LIMIT_DIFF_ERROR_BOUND),
        int(search_config.MIN_ESTIMATE_DENOMINATOR),
        float(search_config.CACHE_ACCEPTANCE_THRESHOLD),
        float(search_config.IDENTIFY_CHECK_THRESHOLD),
        int(search_config.CONSTANT_NO_DIGITS_HIGH_RES),
        int(search_config.CONSTANT_NO_DIGITS_LOW_RES),
    )
    return _tier1_fingerprint_for_key(key)


@lru_cache(maxsize=8192)
def _tier1_fingerprint_for_key(key: tuple) -> str:
    """Memoised core of :func:`tier1_config_fingerprint`.

    Reconstructs the exact same payload dict and serialisation the function
    used before caching was added, so previously-stored fingerprints still
    match (a cache that changed the bytes would force a spurious full recompute
    of every cached trajectory).

    :param key: Tuple of the Tier-1 config values, in a fixed order.
    :return: A 16-hex-char stable fingerprint string.
    """
    (walk_depth, walk_type, depth_conv, limit_diff, min_denom,
     cache_acc, identify_thr, digits_hi, digits_lo) = key
    payload = {
        "walk_depth": walk_depth,
        "walk_type": walk_type,
        "depth_convergence_threshold": list(depth_conv),
        "limit_diff_error_bound": limit_diff,
        "min_estimate_denominator": min_denom,
        "cache_acceptance_threshold": cache_acc,
        "identify_check_threshold": identify_thr,
        "constant_digits_high_res": digits_hi,
        "constant_digits_low_res": digits_lo,
    }
    return _stable_id(json.dumps(payload, sort_keys=True))


def derive_trajectory_id(
    shard_id: str,
    cmf_name: str,
    shard_encoding_str: str,
    start_tuple,
    direction_tuple,
) -> str:
    """Return a structural trajectory id ``"{shard_id}__{traj_hash}"``.

    The trailing ``traj_hash`` is a stable SHA-256 truncation of
    ``(cmf_name, shard_encoding_str, start, direction)`` — the same data
    that previously formed the entire id, just hashed onto the back of
    the shard id so the result is self-describing (you can recover the
    cmf and shard by rsplitting on ``"__"``).
    """
    traj_hash = _stable_id(
        cmf_name, shard_encoding_str, str(start_tuple), str(direction_tuple),
    )
    return f"{shard_id}__{traj_hash}"


# ---------------------------------------------------------------------------
# DTO factory
# ---------------------------------------------------------------------------

def build_trajectory_dtos(
    handler: "TrajectoryAttributesHandler",
    *,
    cmf_id: str,
    shard_id: str,
    cmf_name: str,
    shard_encoding_str: str,
    start,
    direction,
    constants=None,
    compute_recurrence: bool = False,
    extra_metrics: tuple = (),
) -> "List[TrajectoryDTO]":
    """Build **one** flat :class:`TrajectoryDTO` **per constant** from a handler.

    The unit of a result is a ``(trajectory, constant)`` pair (almost every
    attribute is constant-dependent through identification), so this returns a
    *list* — one row per constant in *constants*.  The trajectory-matrix walk is
    computed **once** and shared across the constants via
    :meth:`TrajectoryAttributesHandler.compute_for_constant` (constant swap, no
    re-walk); the constant-independent columns (start/direction/walk metadata,
    recurrence) are duplicated onto each row.

    Each row's ``delta`` / ``identified`` / ``p_vector`` / ``q_vector`` are the
    Tier-1 scalars for that constant.  The **active optimisation objective**
    (``system.OPTIMIZATION_OBJECTIVE``), when it is not a Tier-1 field like δ, is
    computed synchronously for that constant and stored under its own name in the
    flat ``extra`` (so analysis ranking and the search optimisers have it without
    waiting for the async Tier-2 workers, and it lands under the same key Tier-2
    would use).

    :param constants: Iterable of :class:`Constant` objects (preferred) or sympy
        expressions.  ``None`` falls back to ``handler.constant()``.
    :param compute_recurrence: Also populate ``recurrence_relation`` / order
        (constant-independent; computed once, duplicated on each row).
    :param extra_metrics: Additional registry attribute names to compute
        synchronously into each row's ``extra`` (per constant).
    :return: One :class:`TrajectoryDTO` per constant, in *constants* order.
    """
    from dreamer.utils.storage.dtos import TrajectoryDTO  # lazy import avoids circular dep
    from dreamer.utils.constants.constant import Constant as _Constant  # local import avoids circular
    from dreamer.utils.storage.optimization_objectives import objective_metric_attribute

    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)
    trajectory_id = derive_trajectory_id(
        shard_id, cmf_name, shard_encoding_str, start_t, dir_t,
    )

    if constants is None:
        c_expr = handler.constant()
        constants_list = [c_expr] if c_expr is not None else []
    else:
        constants_list = list(constants)

    # Which registry attributes to compute synchronously into each per-constant
    # row: the active objective (when it is stored as an ``extra`` metric rather
    # than a core field like δ) plus any explicitly-requested extras.
    sync_attrs: list = list(extra_metrics)
    obj_attr = objective_metric_attribute(config.system.OPTIMIZATION_OBJECTIVE)
    if obj_attr is not None and obj_attr not in sync_attrs:
        sync_attrs.append(obj_attr)

    # Recurrence is constant-independent — compute it once, duplicate on each row.
    recurrence_relation = handler.formula_str() if compute_recurrence else None
    recurrence_order = handler.order() if compute_recurrence else None

    walk_type = int(handler.walk_type())
    walk_depth = int(handler.walk_depth())
    fingerprint = tier1_config_fingerprint(handler.walk_depth())

    dtos: List[TrajectoryDTO] = []
    for c in constants_list:
        if isinstance(c, _Constant):
            c_name = c.name
            c_sympy = c.value_sympy
        else:
            c_name = str(c)   # backward-compat: raw sympy expression
            c_sympy = c

        delta, p, q, ided, extra = handler.compute_for_constant(
            c_sympy, sync_attributes=tuple(sync_attrs),
        )

        dtos.append(TrajectoryDTO(
            trajectory_id=trajectory_id,
            cmf_id=cmf_id,
            shard_id=shard_id,
            constant=c_name,
            start_point=start_t,
            direction=dir_t,
            identified=bool(ided),
            delta=float(delta),
            p_vector=tuple(_pq_to_jsonsafe(x) for x in p) if p else None,
            q_vector=tuple(_pq_to_jsonsafe(x) for x in q) if q else None,
            walk_type=walk_type,
            walk_depth=walk_depth,
            config_fingerprint=fingerprint,
            projection_column=handler.projection_column(),
            recurrence_relation=recurrence_relation,
            recurrence_order=recurrence_order,
            extra=extra,
        ))
    return dtos


#: Cache-key prefixes whose values depend **only** on the trajectory matrix
#: (constant-independent) and may therefore be preserved across the constant
#: swaps inside :meth:`TrajectoryAttributesHandler.compute_for_constant`.
#: Every other cached attribute is treated as constant-dependent and cleared on
#: a constant swap.  Being conservative here is a correctness requirement: an
#: attribute wrongly listed as independent would leak one constant's value into
#: the next.  The listed spectral / structural attributes derive purely from
#: ``trajectory_matrix_typed`` (no p/q, no target constant), so they are safe to
#: keep — which spares the (expensive) eigenvalue recompute per constant.
_CONST_INDEPENDENT_CACHE_KEYS: tuple = (
    "sorted_eigenvalues", "eigenvalue_errors", "spectral_gap", "coboundary_rank",
    "linear_recurrence", "companion", "order", "relation", "coeff_degrees",
    "recurrence_coeffs", "formula_str", "asymptotics",
)


class TrajectoryAttributesHandler:
    """
    Lazy-computed container for a recurrence relation extracted from a
    CMF trajectory matrix.

    Nothing is computed at __init__. Each method computes on first call
    and caches the result.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        traj_matrix: Matrix,
        constant: Optional[sp.Expr] = None,
        walk_depth: int = 1500,
        walk_type: Optional[int] = None,
        searchable: Optional[Searchable] = None,
        cmf=None,
        trajectory=None,
        start_point=None,
    ):
        """
        Parameters
        ----------
        traj_matrix : ramanujantools.Matrix
            Symbolic d×d trajectory matrix M(n); free parameters (e.g. ``z``)
            must already be substituted for numeric computation.
        constant : sympy.Expr, optional
            The target constant this trajectory approximates (e.g. ``sp.pi``).
            Required for Tier-1 attributes (``delta``, ``limit``, p/q vectors);
            may be ``None`` in worker contexts that only need Tier-2/3 attrs.
        walk_depth : int
            Default number of recurrence steps for walks.
        walk_type : int, optional
            ``1`` → walk uses ``M.inv().T`` (the dual recurrence);
            ``2`` → walk uses ``M`` directly.
            When omitted, resolved from ``search.DEFAULT_USES_INV_T``
            (``True`` → 1, ``False`` → 2).
        searchable : Searchable, optional
            The ``Searchable`` (typically the ``Shard``) this trajectory was
            sampled from.  When provided, its ``cache`` is consulted/updated
            for p/q vectors so repeated identification calls are avoided.
        cmf : optional
            The originating CMF.  Stored only to enable the fast coboundary
            eigenvalue path for ``pFq`` CMFs (see :meth:`sorted_eigenvalues`).
            ``None`` when the handler is built directly from a matrix.
        trajectory, start_point : ramanujantools.Position, optional
            The trajectory direction and start point the matrix was walked
            from.  Required (together with ``cmf``) for the coboundary path.
        """
        self._traj = traj_matrix
        self._cmf = cmf
        self._trajectory = trajectory
        self._start_point = start_point
        self._constant = constant
        self._depth = walk_depth
        self._cache: dict = {}
        if walk_type is None:
            walk_type = 1 if search_config.DEFAULT_USES_INV_T else 2
        self._walk_type = walk_type
        self._searchable = searchable
        self._utility_cache: dict = {}  # separate cache for non-core attributes like p/q vectors
        # Per-instance constant resolution state — avoids the class-level shared-state
        # bug where one trajectory's precision escalation would pollute all subsequent ones.
        self._constant_resolution: int = search_config.CONSTANT_NO_DIGITS_HIGH_RES
        self._high_res_constant: Optional[sp.Expr] = None
        # The walk-matrix column index chosen for p/q projection.  Selected once
        # (during identification) from the first normalisable column and then reused
        # by *every* projection — both the manual delta path and the ``Limit``-based
        # path (``final_projection``) — so they always agree on which column they read.
        # ``None`` until a column has been selected; geometry-only, so it persists
        # across constants (``compute_for_constant`` does not reset it).
        self._projection_column: Optional[int] = None

    @classmethod
    def from_cmf(
        cls,
        cmf,
        trajectory,
        start_point,
        constant: Optional[sp.Expr] = None,
        walk_depth: Optional[int] = None,
        walk_type: Optional[int] = None,
        searchable: Optional[Searchable] = None,
    ) -> "TrajectoryAttributesHandler":
        """Build a handler by computing ``cmf.trajectory_matrix(trajectory, start_point)``.

        ``walk_depth`` defaults to ``search.DEPTH_FROM_TRAJECTORY_LEN(||traj||, cmf.dim())``
        — the same per-trajectory depth ``Searchable.calc_delta`` uses.
        See ``__init__`` for ``constant``, ``walk_type``, ``searchable``.
        """
        tmat = cmf.trajectory_matrix(trajectory, start_point)
        # sp.cancel() intentionally NOT called here: the walk (numerical mpmath) and
        # LIReC identification (numerical p/q) work correctly on the unsimplified form.
        # Simplification is done lazily inside linear_recurrence() — only when symbolic
        # Tier-2 attributes (eigenvalues, kamidelta, …) are actually requested.
        if walk_depth is None:
            walk_depth = search_config.DEPTH_FROM_TRAJECTORY_LEN(
                _trajectory_norm(trajectory), cmf.dim(),
            )
        return cls(
            tmat, constant, walk_depth, walk_type, searchable,
            cmf=cmf, trajectory=trajectory, start_point=start_point,
        )

    # ------------------------------------------------------------------
    #  Cache helpers
    # ------------------------------------------------------------------

    def __get(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def __get_utility(self, key: str, fn):
        if key not in self._utility_cache:
            self._utility_cache[key] = fn()
        return self._utility_cache[key]

    def clear_cache(self):
        """Drop all cached core (Tier-1/2) attribute results."""
        self._cache.clear()

    def clear_utility_cache(self):
        """Drop all cached utility results (walks, p/q vectors)."""
        self._utility_cache.clear()

    def _trajectory_repr(self) -> str:
        """Human-readable ``start`` / ``direction`` for diagnostic logs.

        Every failure/skip log in this handler appends this so a problematic
        trajectory can be reproduced from the logs.  Falls back to the matrix
        shape when the handler was built directly from a matrix (no CMF /
        trajectory retained).
        """
        if self._start_point is not None and self._trajectory is not None:
            return f"[start={self._start_point}, direction={self._trajectory}]"
        return f"[matrix-only handler, shape={self._traj.shape}]"

    def computed_attributes(self) -> list:
        """:return: Names of the core attributes computed and cached so far."""
        return list(self._cache.keys())

    # ==================================================================
    #  TRAJECTORY MATRIX
    # ==================================================================

    @cached_property
    def trajectory_matrix(self) -> Matrix:
        """Raw d×d symbolic trajectory matrix M(n)."""
        return self._traj

    @cached_property
    def trajectory_matrix_typed(self) -> Matrix:
        """The trajectory matrix used by recurrence-level attributes.

        For ``walk_type == 1`` this is ``M.inv().T`` (the dual recurrence);
        for ``walk_type == 2`` it is ``M`` itself.  Recurrence-level attributes
        (linear_recurrence, companion, eigenvalues, gcd_slope) operate on this.

        IMPORTANT: this must **not** mutate ``self._traj``.  ``trajectory_matrix``
        (raw ``M``) and the walked/limit paths read ``self._traj`` directly and
        rely on it staying raw; an earlier version reassigned
        ``self._traj = self._traj.inv().T`` here, which — depending on which
        attribute was computed first — could double-apply the transform or feed
        an already-inverted matrix into the walk.  Computing the transform into a
        separate cached value avoids that order-dependent corruption.
        """
        if self.walk_type() == 1:
            return self._traj.inv().T
        return self._traj

    def traj_size(self) -> int:
        """Dimension d of the trajectory matrix."""
        return self._traj.shape[0]

    def _walked_matrix(self, depth: int) -> Optional[Matrix]:
        """Walk M to ``depth``, then apply ``inv().T`` when ``walk_type==1``.

        Returns ``None`` when the walk fails (e.g. ZeroDivisionError on a
        degenerate trajectory, or singular product when inv() is needed) —
        matches the broad ``try/except`` in :meth:`Searchable.calc_delta`.
        Downstream methods propagate the ``None`` to skip the trajectory.
        Cached per depth.
        """
        def compute():
            try:
                walked = self.trajectory_matrix.walk({n: 1}, depth, {n: 0})
                return walked if self.walk_type() == 2 else walked.inv().T
            except Exception as e:
                Logger(
                    f'Walk failed at depth {depth}: {e} {self._trajectory_repr()}',
                    Logger.Levels.warning,
                ).log()
                return None
        return self.__get_utility(f"walked_{depth}", compute)

    def _initial_values(self) -> Optional[Matrix]:
        """The ``[[p], [q]]`` initial-values matrix for the ``Limit`` machinery.

        Returns a ``2×d`` matrix whose rows are the identified p/q projection
        vectors, or ``None`` when the trajectory is **not identified** (no p/q).
        Returning ``None`` is the signal that nothing should be computed for this
        trajectory — every ``Limit``-based caller (:meth:`_limits`,
        :meth:`gcd_slope`) treats ``None`` as "skip".  This replaces an earlier
        version that built ``Matrix([None, None])`` (a degenerate ``2×1`` matrix),
        which produced the ``Matrix size mismatch: (1, 1) * (2, 2)`` errors when
        fed to ``Matrix.limit`` as ``initial_values``.
        """
        vectors, _ = self._pq_vector()
        if vectors is None:
            return None
        p, q = vectors
        return Matrix([list(p), list(q)])

    # ==================================================================
    #  LINEAR RECURRENCE  (the core object from ramanujantools)
    # ==================================================================

    def linear_recurrence(self) -> LinearRecurrence:
        """
        The LinearRecurrence object built from the trajectory matrix.

        This is the central object. It wraps the trajectory matrix by:
        1. Calling as_companion() to get the companion matrix
        2. Reading the last column (reversed) to get the relation coefficients
        3. Storing as relation = [a_0(n), a_1(n), ..., a_d(n)]
           where Σ a_i(n) · f(n-i) = 0

        All downstream attributes (recurrence_matrix, kamidelta, asymptotics)
        are methods on this object.
        """
        def _build():
            # sp.cancel() commented out: LinearRecurrence extracts recurrence
            # coefficients correctly from the unsimplified symbolic matrix.
            # Re-enable here if symbolic simplification is needed for Tier-2 attributes.
            # if hasattr(tmat, 'applyfunc'):
            #     tmat = tmat.applyfunc(sp.cancel)
            # elif hasattr(tmat, 'matrix'):
            #     tmat.matrix = tmat.matrix.applyfunc(sp.cancel)
            return LinearRecurrence(self.trajectory_matrix_typed)
        return self.__get("linear_recurrence", _build)

    # ==================================================================
    #  COMPANION MATRIX
    # ==================================================================

    def companion(self) -> Matrix:
        """
        The companion (recurrence) matrix.

        Structure for d=2:
            [[0, c_2(n)],    ← right column: coeff of f(n-2)
             [1, c_1(n)]]    ← right column: coeff of f(n-1)

        All recurrence coefficients live in the LAST column.
        The 0s and 1s in the left columns are structural.
        """
        return self.__get("companion", lambda: self.trajectory_matrix_typed.as_companion())

    # ==================================================================
    #  RECURRENCE FORMULA ATTRIBUTES
    # ==================================================================

    def order(self) -> int:
        """
        Order d of the recurrence.
        d=2 → f(n) depends on f(n-1) and f(n-2).
        """
        return self.__get("order", lambda: self.linear_recurrence().order())

    def relation(self) -> list:
        """
        Raw recurrence relation [a_0(n), a_1(n), ..., a_d(n)].

        These define:  Σ a_i(n) · f(n-i) = 0
        i.e.:  a_0(n)·f(n) + a_1(n)·f(n-1) + ... + a_d(n)·f(n-d) = 0

        Rearranging: f(n) = -[a_1(n)·f(n-1) + ... + a_d(n)·f(n-d)] / a_0(n)
        """
        return self.__get("relation", lambda: self.linear_recurrence().relation)

    def recurrence_coeffs(self) -> list:
        """
        The 'friendly' coefficients [c_1(n), ..., c_d(n)] such that:
            f(n) = c_1(n)·f(n-1) + c_2(n)·f(n-2) + ... + c_d(n)·f(n-d)

        Derived from the last column of the companion matrix (bottom to top).
        """
        def compute():
            C = self.companion()
            d = self.order()
            return [sp.simplify(C[d - i, -1]) for i in range(1, d + 1)]
        return self.__get("recurrence_coeffs", compute)

    def coeff_degrees(self) -> list:
        """
        Polynomial degrees of the relation coefficients.
        Uses LinearRecurrence.degrees() which returns degrees of [a_0, ..., a_d].
        """
        return self.__get("coeff_degrees", lambda: self.linear_recurrence().degrees())

    def formula_str(self) -> str:
        """
        Human-readable recurrence formula.
        Uses LinearRecurrence.__str__() which gives: Σ a_i(n)·p(n-i) = 0
        """
        return self.__get("formula_str", lambda: str(self.linear_recurrence()))

    # ==================================================================
    #  EIGENVALUES
    # ==================================================================

    def sorted_eigenvalues(self) -> list:
        """
        Poincaré eigenvalues of the companion matrix, sorted by |λ| descending.
        Uses Matrix.sorted_eigenvals() which computes eigenvalues of the
        Poincaré characteristic polynomial (the asymptotic limit of the
        charpoly as n→∞).

        For a constant-coefficient recurrence, these are the actual
        characteristic roots. For polynomial coefficients, these are
        the leading-term roots that govern asymptotic growth.

        Fast path: when this handler originates from a ``pFq`` CMF (built via
        :meth:`from_cmf`), the eigenvalues are obtained through a diagonal
        coboundary transform (see :meth:`_pfq_coboundary_eigenvalues`), which
        is ~3x faster than the generic symbolic ``sorted_eigenvals`` at higher
        dimensions.  For every other CMF — or when the coboundary path fails —
        the generic implementation is used.
        """
        def compute() -> list:
            fast = self._pfq_coboundary_eigenvalues()
            if fast is not None:
                return fast
            return self.trajectory_matrix_typed.sorted_eigenvals()
        return self.__get("sorted_eigenvalues", compute)

    def _pfq_coboundary_eigenvalues(self) -> Optional[list]:
        r"""Eigenvalues of a ``pFq`` trajectory matrix via a coboundary transform.

        For a ``pFq(p, q, z)`` CMF the trajectory matrix :math:`M(n)` admits a
        *diagonal* coboundary :math:`C(n) = \operatorname{diag}(1, r, r^2, \dots)`
        with ratio

        .. math:: r(n) = \frac{\prod_k y_k(n)}{\prod_k x_k(n)},

        evaluated along the trajectory (``x_k(n) = start_k + n·traj_k``).  The
        transformed matrix :math:`\tilde M(n) = C(n)^{-1} M(n) C(n+1)` has a
        finite element-wise limit as :math:`n\to\infty`, whose eigenvalues are
        the Poincaré eigenvalues of :math:`M`.

        Because :math:`C` is diagonal the transform is pure element-wise
        scaling — no symbolic inverse or matmul — so
        :math:`\tilde M_{ij} = \lim_{n\to\infty} M_{ij}\, r(n+1)^j / r(n)^i`.
        The eigenvalues of the resulting numeric matrix are extracted by rooting
        its exact characteristic polynomial with ``mpmath`` at high precision
        (robust to the large condition numbers that defeat float eigensolvers on
        the small, convergence-governing eigenvalue).

        Returns the eigenvalues as sympy ``Float``/``Add`` expressions sorted by
        :math:`|\lambda|` descending (matching :meth:`Matrix.sorted_eigenvals`),
        or ``None`` when the fast path is not applicable (non-``pFq`` CMF, missing
        trajectory/start, or any failure) so the caller can fall back.

        For ``walk_type == 1`` the handler walks ``M.inv().T`` whose eigenvalues
        are the reciprocals of :math:`M`'s; they are reciprocated here so the
        result matches the generic path for both walk types.
        """
        cmf = self._cmf
        if cmf is None or self._trajectory is None or self._start_point is None:
            return None
        # Duck-typed pFq detection (avoids a hard import dependency at module load).
        if type(cmf).__name__ != "pFq" or not (hasattr(cmf, "p") and hasattr(cmf, "q")):
            return None
        try:
            import mpmath as mp

            x_axes = sp.symbols(f"x:{cmf.p}")
            y_axes = sp.symbols(f"y:{cmf.q}")
            traj = self._trajectory
            start = self._start_point

            def at(sym):
                return start[sym] + n * traj[sym]

            r = sp.prod([at(y) for y in y_axes]) / sp.prod([at(x) for x in x_axes])
            r_next = r.subs({n: n + 1})

            # Raw pFq trajectory matrix (before the handler's inv().T transform).
            tmat = cmf.trajectory_matrix(traj, start)
            d = tmat.shape[0]
            lim = sp.zeros(d, d)
            for i in range(d):
                for j in range(d):
                    lim[i, j] = sp.limit(tmat[i, j] * r_next ** j / r ** i, n, sp.oo)

            charpoly = Matrix(lim).charpoly()
            with mp.workdps(50):
                coeffs = [mp.mpf(str(sp.N(c, 50))) if sp.im(c) == 0
                          else mp.mpc(complex(c)) for c in charpoly.all_coeffs()]
                roots = mp.polyroots(coeffs, maxsteps=200, extraprec=200)
            if self.walk_type() == 1:
                # Eigenvalues of M.inv().T are the reciprocals of M's.  Drop
                # (near-)zero roots instead of mapping them to infinity: an
                # infinite eigenvalue is not a usable Poincaré eigenvalue and
                # would poison downstream ratios (log of infinity → the "delta
                # from infinity" errors).
                roots = [1 / rt for rt in roots if abs(rt) > 1e-30]
            eigs = [_mpc_to_sympy(rt) for rt in roots]
            return sorted(eigs, key=lambda e: self._eigenvalue_norm(e), reverse=True)
        except Exception as exc:  # noqa: BLE001 — any failure → generic fallback
            Logger(
                f"pFq coboundary eigenvalue path failed ({exc!r}); "
                f"falling back to symbolic sorted_eigenvals. {self._trajectory_repr()}",
                Logger.Levels.debug,
            ).log()
            return None

    @staticmethod
    def _eigenvalue_norm(e, dps: int = 100):
        r"""High-precision norm :math:`|\lambda|` of an eigenvalue, as a sympy ``Float``.

        Computes ``sp.Abs(e)`` **symbolically** first, then evaluates at *dps*
        digits.  This is essential for the small subdominant eigenvalue: a
        Poincaré eigenvalue of the form ``A - B·√k`` (huge rationals ``A``, ``B``
        with ``A ≈ B√k``) catastrophically cancels at the default 15-digit
        precision, and ``evalf(chop=True)`` would zero a legitimately tiny
        magnitude (e.g. ``9.6e-30``).  Working from the exact symbolic form with
        adaptive high precision and **no chop** preserves it; ``sp.Abs`` also
        collapses a real ``x + 0i`` to ``|x|`` without chopping.

        Returned as a high-precision sympy ``Float`` (not a Python ``float``) so
        downstream digit/error computations stay symbolic for maximum accuracy;
        callers convert to ``float`` only where a Python float is needed (sorting,
        dedup) or at the storage boundary.  Returns ``sp.oo`` for a non-finite
        eigenvalue so callers can apply a *true*-zero / finiteness filter.
        """
        try:
            return sp.Abs(e).evalf(dps)
        except (TypeError, ValueError):
            return sp.oo

    @staticmethod
    def _log10(x, dps: int = 50) -> float:
        r"""``log₁₀(x)`` of a positive (high-precision) value, returned as a float.

        Evaluated symbolically at *dps* digits then cast to ``float`` only at the
        end — there is no cancellation in a log of a single positive magnitude, so
        the float result is exact to double precision.
        """
        return float(sp.log(x, 10).evalf(dps))

    def _unique_eigenvalue_pairs(self) -> list:
        r"""All ordered tuples ``(λᵢ, λⱼ, |λᵢ|, |λⱼ|)`` with ``|λᵢ| > |λⱼ|``.

        Operates on the **actual (symbolic) eigenvalues** and takes their norms via
        :meth:`_eigenvalue_norm` (high precision, no chop), so a legitimately tiny
        subdominant eigenvalue is kept rather than rounded/chopped to zero.  Uses a
        **true-zero filter**: only exactly-zero (or non-finite) norms are dropped —
        a ``9.6e-30`` eigenvalue is a valid ``λⱼ``.  Deduplicates by *relative*
        magnitude so equal-|λ| multiplicities don't generate redundant pairs and
        equal-magnitude pairs (convergence rate 0) are excluded.

        The eigenvalues themselves are returned unevaluated (symbolic where the
        generic path produced radicals) so downstream consumers can re-evaluate at
        whatever precision they need; the norms are returned alongside so callers
        need not recompute them.
        """
        unique: list = []   # (eigenvalue, norm[high-precision sympy Float])
        seen: list = []     # float norms, for dedup / ordering only
        for e in self.sorted_eigenvalues():
            norm = self._eigenvalue_norm(e)
            nf = float(norm)
            if not math.isfinite(nf) or nf == 0.0:   # true-zero / finiteness filter
                continue
            if not any(abs(nf - s) <= 1e-12 * max(1.0, nf) for s in seen):
                unique.append((e, norm))
                seen.append(nf)
        pairs = []
        for i, (ei, ni) in enumerate(unique):
            for j, (ej, nj) in enumerate(unique):
                if i == j:
                    continue
                if float(ni) > float(nj) * (1 + 1e-12):   # strictly larger magnitude
                    pairs.append((ei, ej, ni, nj))
        return pairs

    def eigenvalue_errors(self) -> list:
        """
        log|λ₁/λᵢ| for i = 2, ..., d.
        These are the 'error terms' — the log-ratios between the dominant
        eigenvalue and each subdominant one. Used internally by delta_prediction().

        errors[0] = log|λ₁/λ₂|  (primary convergence rate)
        errors[1] = log|λ₁/λ₃|  (if d ≥ 3), etc.

        Norms are taken from the actual (symbolic) eigenvalues at high precision
        (see :meth:`_eigenvalue_norm`), so a tiny subdominant eigenvalue yields a
        large finite error rather than ``zoo``.  A genuinely zero subdominant
        eigenvalue (true-zero) is skipped.
        """
        def compute():
            norms = [self._eigenvalue_norm(e) for e in self.sorted_eigenvalues()]
            if not norms:
                return []
            n0 = norms[0]
            deltas = []
            for ni in norms[1:]:
                nif = float(ni)
                if not math.isfinite(nif) or nif == 0.0:
                    continue
                # high-precision natural log of the (well-separated) norm ratio
                deltas.append(float(sp.log(n0 / ni)))
            return deltas
        return self.__get("eigenvalue_errors", compute)

    def spectral_gap(self) -> Optional[float]:
        """
        |λ₁| − |λ₂| from the Poincaré eigenvalues.
        Large gap → fast convergence, small gap → slow/noisy.
        """
        def compute():
            eigs = self.sorted_eigenvalues()
            if len(eigs) >= 2:
                return float(self._eigenvalue_norm(eigs[0]) - self._eigenvalue_norm(eigs[1]))
            return None
        return self.__get("spectral_gap", compute)

    # ==================================================================
    #  WALKING — LIMIT
    # ==================================================================

    def constant(self) -> sp.Expr:
        """The target constant that this trajectory is approximating (e.g., π)."""
        return self._constant

    def walk_type(self) -> int:
        """Return the walk type (1 or 2) for this handler."""
        return self._walk_type

    def walk_depth(self) -> int:
        """Return the walk depth this handler walks the trajectory matrix to."""
        return self._depth

    def projection_column(self) -> Optional[int]:
        """The walk-matrix column index used to project the p/q vectors.

        Tier-1 attribute, persisted on the DTO so a stored result can be
        reconstructed exactly: the same column must be used as the
        ``Limit.final_projection`` to recover the identical p_n / q_n sequence.

        Selected on first walk (see :meth:`_select_projection_column`); this
        triggers that selection if it has not run yet.  It is geometry, chosen
        independently of identification, so it can be set even for an
        unidentified trajectory.  Returns ``None`` only when the walk failed /
        no column was normalisable.
        """
        if self._projection_column is None:
            self._effective_walk_values(self._depth)
        return self._projection_column

    def _select_projection_column(self, walked: sp.Matrix) -> Optional[int]:
        """Pick the walk-matrix column index to project p/q onto.

        Skips columns with a zero top entry (cannot normalise without dividing by
        zero), prefers a column with *no* zero entries (LIReC-friendly), and
        otherwise falls back to the first normalisable column.  Returns the column
        index, or ``None`` when no column is normalisable.
        """
        first_normalizable = None
        for col_ind in range(sp.shape(walked)[1]):
            if walked[0, col_ind].is_zero:
                continue
            col = (walked / walked[0, col_ind]).col(col_ind)
            if first_normalizable is None:
                first_normalizable = col_ind
            if all(not v.is_zero for v in col):
                return col_ind  # no-zero column preferred
        return first_normalizable

    def _effective_walk_values(
        self, depth: Optional[int] = None, walk_matrix: Optional[sp.Matrix] = None,
    ) -> Tuple[Optional[list], Optional[int]]:
        """Return ``(column_values, column_index)`` used for p/q projection.

        The column index is selected **once** (the first time this runs, during
        identification) via :meth:`_select_projection_column` and stored in
        ``self._projection_column``; every later call reuses it so the manual
        delta path and the ``Limit``-based path (which gets ``final_projection =
        e_{column}``) read the *same* column.  This fixes the bug where the
        carefully-chosen normalisable column was known only here while the
        ``Limit`` path silently defaulted to the last column (``e_{-1}``),
        yielding ``zoo`` / ``mpf`` errors when that column was degenerate.

        Pass ``depth`` to walk internally (cached, ``inv().T`` applied after the
        walk for ``walk_type==1``).  ``walk_matrix`` may be an already-transformed
        ``Limit.current`` snapshot — used as-is.

        Returns ``(None, None)`` when the walk fails, no column is normalisable,
        or the chosen column is degenerate at this depth.
        """
        if depth is None and walk_matrix is None:
            Logger(
                'No depth or walk matrix provided for effective walk values. '
                f'This was not supposed to happen. Skipping trajectory... {self._trajectory_repr()}',
                Logger.Levels.exception,
            ).log()
            return None, None

        depth = depth or self._depth

        def compute():
            walked = walk_matrix if walk_matrix is not None else self._walked_matrix(depth)
            if walked is None:
                return None, None

            col_index = self._projection_column
            if col_index is None:
                col_index = self._select_projection_column(walked)
                if col_index is None:
                    Logger(
                        'Could not normalize any walk matrix column. '
                        f'Skipping trajectory... {self._trajectory_repr()}',
                        Logger.Levels.warning,
                    ).log()
                    return None, None
                self._projection_column = col_index

            top = walked[0, col_index]
            if top.is_zero:
                # The established projection column is degenerate at this depth —
                # a legitimate, non-fatal outcome (e.g. a non-converging tail).
                Logger(
                    f'Projection column {col_index} is degenerate at depth {depth}; '
                    f'skipping this depth. {self._trajectory_repr()}',
                    Logger.Levels.debug,
                ).log()
                return None, None

            col = (walked / top).col(col_index)
            return [item for item in col], col_index

        if walk_matrix is not None:
            return compute()
        return self.__get_utility(f"effective_walk_{depth}", compute)

    def _final_projection(self) -> Optional[Matrix]:
        """``final_projection`` (``e_column`` for both p and q) for the ``Limit``.

        Uses the same projection column the manual path selected
        (:meth:`_effective_walk_values`) so ``Limit.as_rational`` reads the
        identical column.  Returns ``None`` (⇒ ramanujantools default, the last
        column) only when no column has been selected yet.
        """
        _, col_index = self._effective_walk_values(self._depth)
        if col_index is None:
            return None
        N = self.traj_size()
        return Matrix.hstack(Matrix.e(N, col_index), Matrix.e(N, col_index))

    def _limits(self, depths: list) -> list[Limit]:
        """
        Internal: get Limit objects at specified depths.

        Returns ``[]`` when the trajectory is **not identified** (no p/q vectors)
        or when the walk fails (singular matrix, ZeroDivisionError on degenerate
        trajectories, etc.) — callers must handle this.  Per project policy we do
        not compute ``Limit`` values for an unidentified trajectory: feeding the
        absent p/q as ``initial_values`` is what produced the earlier
        ``Matrix size mismatch`` / ``mpf from zoo`` errors.
        """
        initial_values = self._initial_values()
        if initial_values is None:
            return []  # not identified — nothing to compute
        final_projection = self._final_projection()
        try:
            limits = self.trajectory_matrix.limit(
                {n: 1}, depths, {n: 0}, initial_values, final_projection,
            )
            if isinstance(limits, Limit):
                limits = [limits]

            if self.walk_type() == 1:
                for i, l in enumerate(limits):
                    limits[i].current = l.current.inv().T
            return limits
        except Exception as e:
            Logger(
                f'_limits walk failed at depths={depths}: {e} {self._trajectory_repr()}',
                Logger.Levels.warning,
            ).log()
            return []

    def limit(self, depth: Optional[int] = None) -> float:
        """
        Numerical estimate of L = lim(n→∞) p_n/q_n.

        Returns ``float('nan')`` when the walk fails — keeps the DTO
        constructible (``float(NaN)`` is a valid float) instead of leaking
        the walk exception to callers.
        """
        depth = depth or self._depth
        def compute() -> float:
            try:
                limits = self._limits([depth])
                if not limits:
                    return float('nan')
                return float(limits[0].as_float())
            except Exception as e:
                Logger(
                    f'limit failed at depth {depth}: {e} {self._trajectory_repr()}',
                    Logger.Levels.warning,
                ).log()
                return float('nan')
        return self.__get(f"limit_{depth}", compute)

    # ==================================================================
    #  DELTA — IRRATIONALITY MEASURE
    # ==================================================================

    def delta(self, depth: Optional[int] = None) -> float:
        """
        Irrationality measure δ at the given depth.
            |p/q − L| = 1 / q^(1+δ)

        For any irrational L: δ ≥ 1 (Dirichlet's theorem).
        Returns ``float('-inf')`` when the walk fails, identification
        fails, or the convergence sanity check fails — the documented
        non-convergence sentinel.
        """
        depth = depth or self._depth
        def compute() -> float:
            try:
                converges, _ = self._convergence_sanity_check(depth)
                if not converges:
                    return float('-inf')
                delta_res = self.delta_sequence([depth])
                if len(delta_res) == 0:
                    return float('-inf')
                return delta_res[0]
            except Exception as e:
                Logger(
                    f'delta failed at depth {depth}: {e} {self._trajectory_repr()}',
                    Logger.Levels.warning,
                ).log()
                return float('-inf')
        return self.__get(f"delta_{depth}", compute)

    def __compute_delta(
            self, p: sp.Matrix, q: sp.Matrix, effective_walk_values: list, constant: sp.Expr
    ) -> Tuple[bool, float | sp.Expr]:
        """
        Computes the delta value for evaluating an approximation using a rational estimator.

        Uses per-instance ``_constant_resolution`` / ``_high_res_constant`` so that
        precision escalation for one trajectory does not affect any other handler.

        :param p: Row vector providing weights for numerator computation.
        :param q: Row vector providing weights for denominator computation.
        :param effective_walk_values: Walk column values used to compute the weighted sum.
        :param constant: The symbolic constant whose approximation is to be validated.
        :returns: ``(success, delta)`` — ``success`` is False and ``delta`` is ``-inf``
            when the approximation fails the denominator guard or precision ceiling.
        """
        walk_col = sp.Matrix(effective_walk_values)
        numerator = p.dot(walk_col)
        denom = q.dot(walk_col)
        # Guard a zero denominator before building the Rational: q·walk == 0 makes
        # sp.Rational(numerator, 0) == zoo, which later raises "cannot create mpf
        # from zoo".  A zero denominator is a legitimate, non-fatal outcome on a
        # degenerate/non-converging step — treat it as a failed estimate.
        if denom == 0:
            Logger(
                f'Zero denominator in delta estimate (q·walk == 0); skipping. '
                f'{self._trajectory_repr()}',
                Logger.Levels.debug,
            ).log()
            return False, float('-inf')
        estimated = sp.Abs(sp.Rational(numerator, denom))
        denom_int = sp.Abs(sp.denom(estimated))

        if sp.Abs(denom_int) <= search_config.MIN_ESTIMATE_DENOMINATOR:
            Logger(f'Guardrail reached - not good enough approximation. Ignoring trajectory\n'
                   f'Reason: Denominator <= {search_config.MIN_ESTIMATE_DENOMINATOR}',
                   Logger.Levels.debug).log()
            return False, float('-inf')

        if self._high_res_constant is None:
            self._high_res_constant = constant.evalf(self._constant_resolution)

        while self._constant_resolution <= search_config.MAX_CONSTANT_RESOLUTION:
            err = sp.Abs(estimated - self._high_res_constant)
            # Match Searchable.calc_delta: use the integer denominator
            # of the rational estimate for the delta formula, not the
            # raw symbolic q·walk (which can be a fractional Rational).
            delta = -1 - sp.log(err) / sp.log(denom_int)
            if delta == sp.oo or delta == sp.zoo:
                if self._constant_resolution == search_config.MAX_CONSTANT_RESOLUTION:
                    Logger(
                        f'Guardrail reached - could not approximate with constant at resolution: '
                        f'{search_config.MAX_CONSTANT_RESOLUTION}'
                        f'\nYou might want to rerun the search with a higher constant resolution to get a better approximation.',
                        Logger.Levels.warning,
                    ).log()
                    break
                self._constant_resolution = min(
                    self._constant_resolution * 2, search_config.MAX_CONSTANT_RESOLUTION
                )
                self._high_res_constant = constant.evalf(self._constant_resolution)
                continue
            return True, delta
        return False, float('-inf')

    def delta_sequence(self, depth: Optional[int | list] = None) -> list:
        """
        δ values at every step from 1 to depth.
        Shows how the irrationality measure evolves with walk depth.

        Uses Limit.delta(L) at each step.
        """
        depth = depth or self._depth
        if isinstance(depth, int):
            depth = list(range(1, depth + 1))

        def compute():
            # Identification is depth-independent and cached (see _pq_vector); the
            # returned walk column is at the identification depth, so it is NOT used
            # for the δ values here — each requested depth walks its own column.
            vectors, _ = self._pq_vector()
            if vectors is None:
                return []
            p, q = vectors
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            deltas = []

            if len(depth) > 1:
                limits = self._limits(depth)
                for l in limits:
                    walk_values, _ = self._effective_walk_values(None, l.current)
                    if walk_values is None:
                        continue
                    success, delta = self.__compute_delta(p, q, walk_values, self.constant())
                    deltas.append(float(delta.evalf(10)) if success else delta)
            else:
                # Single depth: walk the requested depth's column explicitly (the
                # identification walk may have been at a shallower IDENTIFY_DEPTH).
                walk_values, _ = self._effective_walk_values(depth[0])
                if walk_values is None:
                    return []
                success, delta = self.__compute_delta(p, q, walk_values, self.constant())
                deltas.append(float(delta.evalf(10)) if success else delta)
            return deltas
        
        return self.__get(f"delta_seq_{depth}", compute)

    def delta_prediction(self, depth: int = 20) -> Optional[dict]:
        """Predict δ by selecting the eigenvalue pair that best fits the actual delta.

        Algorithm:
          1. Enumerate all pairs (λᵢ, λⱼ) with |λᵢ| > |λⱼ| from the unique
             Poincaré eigenvalues (see :meth:`_unique_eigenvalue_pairs`).
          2. For each pair, compute the kamidelta formula:
             ``predicted = -1 + log(|λᵢ| / |λⱼ|) / gcd_slope(depth)``.
          3. Compare each ``predicted`` to the actual ``self.delta()`` and pick
             the pair with minimum ``|predicted − actual_delta|``.

        Returns a dict ``{"predicted_delta": float, "lambda_1": expr, "lambda_2": expr}``
        where ``lambda_1`` / ``lambda_2`` are the matched dominant / subdominant
        eigenvalues (sympy expressions, evalf'd).
        Returns ``None`` when the actual delta is not finite, no valid pairs exist,
        or ``gcd_slope`` is zero.
        """
        def compute() -> Optional[dict]:
            actual_delta = self.delta()
            if not math.isfinite(actual_delta):
                return None

            pairs = self._unique_eigenvalue_pairs()
            if not pairs:
                return None

            slope = self.gcd_slope(depth)
            try:
                slope_f = float(slope)
            except Exception:
                return None
            if abs(slope_f) < 1e-30:
                return None

            best: Optional[dict] = None
            best_diff = float('inf')
            for lam1, lam2, norm1, norm2 in pairs:
                try:
                    log_ratio = float(sp.log(norm1 / norm2))
                    predicted = float(-1 + log_ratio / slope_f)
                    diff = abs(predicted - actual_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best = {
                            "predicted_delta": predicted,
                            "lambda_1": lam1,
                            "lambda_2": lam2,
                            # high-precision norms (see _eigenvalue_norm), reused by
                            # error_formula_ratio / digits_approximation
                            "norm_1": norm1,
                            "norm_2": norm2,
                        }
                except Exception:
                    continue
            return best

        return self.__get(f"delta_prediction_{depth}", compute)

    def gcd_slope(self, depth: int = 20):
        r"""
        Linear fit slope of ``log(q̃_n) = log(q_n / gcd(p_n, q_n))``.
        This measures how fast the **reduced denominator** of the convergents
        grows, and it is the denominator of the kamidelta formula
        (``δ ≈ -1 + log|λ₁/λ₂| / gcd_slope``).

        IMPORTANT — must use the **same walk as δ**.  ``ramanujantools.Matrix.gcd_slope``
        walks the reduced-denominator sequence starting at ``{n: 1}``, whereas this
        handler identifies p/q and computes δ on the walk starting at ``{n: 0}``
        (see :meth:`_limits` / :meth:`_walked_matrix`) and, for ``walk_type == 1``,
        applies ``inv().T`` to each walked matrix.  Those are *different* integer
        sequences: the identified p/q vectors belong to the ``{n: 0}`` walk, so
        projecting them onto the ``{n: 1}`` walk yields a mismatched fraction with
        far less gcd cancellation and a denominator that grows ~30 % faster.  Using
        the ramanujantools slope therefore drove kamidelta far below the true δ
        (e.g. δ≈0.26 but kamidelta≈-0.03).  We instead fit ``log(q̃_n)`` over the
        **identical** ``Limit`` objects δ uses (:meth:`_limits`), so the eigenvalue
        ratio and the denominator slope refer to the same convergents.

        Used by :meth:`delta_prediction`.  Returns ``None`` for an unidentified
        trajectory (no p/q vectors) or when the walk/fit fails.
        """
        def compute():
            import numpy as np

            if self._initial_values() is None:
                return None  # not identified — nothing to fit
            depths = list(range(1, depth))
            if len(depths) < 2:
                return None
            try:
                # Same walk δ uses (start {n: 0}, walk_type-aware) so the reduced
                # denominator matches the convergents δ / the p/q vectors live on.
                limits = self._limits(depths)
                if not limits:
                    return None
                xs, ys = [], []
                for d, lim in zip(depths, limits):
                    try:
                        q = lim.as_rational().q
                        ys.append(float(sp.log(q).evalf(30)))
                        xs.append(d)
                    except Exception:
                        continue  # degenerate step — skip this depth
                if len(xs) < 2:
                    return None
                import mpmath as mp
                slope = np.polyfit(np.asarray(xs, dtype=float),
                                   np.asarray(ys, dtype=float), 1)[0]
                return mp.mpf(float(slope))
            except Exception as e:
                Logger(
                    f'gcd_slope failed at depth {depth}: {e} {self._trajectory_repr()}',
                    Logger.Levels.warning,
                ).log()
                return None
        return self.__get(f"gcd_slope_{depth}", compute)

    # ==================================================================
    #  CONVERGENCE RATE
    # ==================================================================

    def _convergence_sanity_check(self, depth: Optional[int] = None) -> Tuple[bool, List[Limit]]:
        """Check that the estimated limit stabilises across the depths configured
        in ``search.DEPTH_CONVERGENCE_THRESHOLD``.

        Returns ``(converges, limits)`` — ``converges`` is True iff successive
        estimates differ by less than ``search.LIMIT_DIFF_ERROR_BOUND``.
        """
        depth = depth or self._depth
        # Check identification *before* walking limits — an unidentified trajectory
        # has no p/q, so there is nothing to sanity-check and ``_limits`` would have
        # nothing to project anyway.
        vectors, _ = self._pq_vector(depth)
        if vectors is None:
            return False, []
        p, q = vectors
        p = sp.Matrix(p).T
        q = sp.Matrix(q).T

        limits = self._limits([round(coef * depth) for coef in search_config.DEPTH_CONVERGENCE_THRESHOLD])
        if not limits:
            return False, []

        # extract estimated limit at each depth (each ``limit.current`` is a
        # different walked matrix — they must be used, not the same one).
        floats = []
        for limit in limits:
            walk_col, _ = self._effective_walk_values(depth, limit.current)
            if walk_col is None:
                continue
            values_vec = sp.Matrix(walk_col)
            numerator = p.dot(values_vec)
            denom = q.dot(values_vec)
            if denom == 0:
                continue
            estimated = sp.Abs(sp.Rational(numerator, denom))
            floats.append(estimated)

        if len(floats) < 2:
            # Not enough usable estimates to judge convergence.
            return False, limits

        # check that the estimated limits are consistent (within error bound)
        diffs = [abs(floats[i] - floats[i-1]) for i in range(1, len(floats))]
        return all(diff < search_config.LIMIT_DIFF_ERROR_BOUND for diff in diffs), limits

    def precision_at(self, depth: Optional[int] = None) -> int:
        # TODO: just compute the error (log())
        """
        Number of correct decimal digits at the given depth.
        Uses Limit.precision() which compares the last two walk steps.
        """
        depth = depth or self._depth
        def compute():
            limits = self._limits([depth])
            if not limits:
                return None
            return limits[0].precision()
        return self.__get(f"precision_{depth}", compute)

    def digits_per_step(self, max_depth: Optional[int] = None) -> list:
        """
        Δd(k) = precision(k) − precision(k-1) for each step k.
        Shows how many new digits each recurrence step contributes.

        Interpretation:
            roughly constant → exponential convergence  (~8 for Gosper N=29)
            growing with k   → factorial convergence    (super-exponential)
            shrinking with k → polynomial convergence   (slow)
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            depths = list(range(1, max_depth + 1))
            limits = self._limits(depths)
            precisions = [lim.precision() for lim in limits]
            return [
                (k + 1, precisions[k] - precisions[k - 1])
                for k in range(1, len(precisions))
                if precisions[k - 1] > 0
            ]
        return self.__get(f"dps_{max_depth}", compute)

    def approximated_digits_per_step(self, prediction_depth: int = 20) -> Optional[float]:
        """Approximated correct digits gained **per step** (eigenvalue-based).

        Formula:  ``-log₁₀(|λ₂ / λ₁|) = log₁₀(|λ₁| / |λ₂|)``

        The asymptotic per-step digit gain implied by the dominant/subdominant
        eigenvalue ratio of the pair matched by :meth:`delta_prediction`.  Computed
        at high precision from the symbolic eigenvalue norms (see
        :meth:`_eigenvalue_norm`).  Returns ``None`` when delta_prediction is
        unavailable (non-identified, no valid eigenvalue pair, or zero gcd_slope).
        """
        def compute() -> Optional[float]:
            pred = self.delta_prediction(prediction_depth)
            if pred is None:
                return None
            n1, n2 = pred["norm_1"], pred["norm_2"]
            if float(n1) == 0.0 or float(n2) == 0.0 or not (float(n2) < float(n1)):
                return None
            return self._log10(n1 / n2)   # = -log10(n2/n1)
        return self.__get(f"approx_dps_{prediction_depth}", compute)

    def convergence_rate(self, prediction_depth: int = 20) -> Optional[float]:
        r"""Length-normalised spectral convergence rate.

        Formula:  ``approximated_digits_per_step / ||trajectory||_2``
                  ``= log10(|λ1| / |λ2|) / ||v||_2``

        The eigenvalue-based digits-per-step gain (see
        :meth:`approximated_digits_per_step`) divided by the Euclidean norm of the
        trajectory direction ``v`` — i.e. correct digits gained per step **per unit
        trajectory length**.  Normalising by ``||v||`` makes the rate comparable
        across trajectories whose direction vectors have different magnitudes (a
        longer direction advances further per recurrence step, so the raw per-step
        gain is not directly comparable).

        Because ``approximated_digits_per_step`` selects its eigenvalue pair via the
        δ-matched :meth:`delta_prediction`, this quantity is identification- and
        constant-dependent (larger ⇒ faster convergence).

        Returns ``None`` when ``approximated_digits_per_step`` is unavailable
        (non-identified, no valid eigenvalue pair, or zero gcd_slope) or when the
        trajectory direction is missing / has zero norm (e.g. a matrix-only
        handler).

        :param prediction_depth: Depth forwarded to
            :meth:`approximated_digits_per_step` (``delta_prediction``'s ``gcd_slope``).
        """
        def compute() -> Optional[float]:
            per_step = self.approximated_digits_per_step(prediction_depth)
            if per_step is None:
                return None
            if self._trajectory is None:
                return None
            norm = _trajectory_norm(self._trajectory)
            if norm == 0:
                return None
            return per_step / norm
        return self.__get(f"convergence_rate_{prediction_depth}", compute)

    def digits_approximation(self, depth: Optional[int] = None, prediction_depth: int = 20) -> Optional[float]:
        """Approximated number of correct digits **at** *depth* (eigenvalue-based).

        Formula:  ``#digits(n) = -n · log₁₀(|λ₂ / λ₁|) = n · approximated_digits_per_step``

        The convergence rate ``approximated_digits_per_step`` extrapolated to the
        walk depth.  Returns ``None`` when delta_prediction is unavailable.

        :param depth: Depth to predict at (defaults to this handler's walk depth).
        :param prediction_depth: Depth passed to :meth:`delta_prediction` for ``gcd_slope``.
        """
        depth = depth or self._depth
        def compute() -> Optional[float]:
            per_step = self.approximated_digits_per_step(prediction_depth)
            if per_step is None:
                return None
            return depth * per_step
        return self.__get(f"digits_approx_{depth}_{prediction_depth}", compute)

    def error_formula_ratio(self, prediction_depth: int = 20) -> Optional[float]:
        """Eigenvalue ratio ``|λ₂ / λ₁|`` of the matched pair (per-step error decay).

        ``err(n) ≈ ratio^n``.  Returns ``None`` when delta_prediction is unavailable.
        """
        def compute() -> Optional[float]:
            pred = self.delta_prediction(prediction_depth)
            if pred is None:
                return None
            n1, n2 = pred["norm_1"], pred["norm_2"]
            if float(n1) == 0.0:
                return None
            return float(n2 / n1)
        return self.__get(f"error_ratio_{prediction_depth}", compute)

    def digits_computed(self, depth: Optional[int] = None) -> Optional[float]:
        r"""Number of **correct** digits of the p/q approximation vs. the constant.

        Formula:  ``#digits = -log₁₀ |p_n/q_n − C|``

        Walk-based and constant-dependent: the *actual* accuracy of the identified
        convergent against the target constant ``C`` (whereas
        :meth:`digits_approximation` *predicts* it from the eigenvalue ratio).  Uses
        the same high-resolution constant the δ computation escalated to, so the
        result is bounded by ``MAX_CONSTANT_RESOLUTION``.  Returns ``None`` when the
        trajectory is not identified / δ is not finite, or when the convergent
        matches the constant exactly within the available precision.
        """
        depth = depth or self._depth
        def compute() -> Optional[float]:
            if not math.isfinite(self.delta(depth)):
                return None
            vectors, _ = self._pq_vector(depth)
            if vectors is None:
                return None
            p, q = vectors
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            walk_col, _ = self._effective_walk_values(depth)
            if walk_col is None:
                return None
            wc = sp.Matrix(walk_col)
            denom = q.dot(wc)
            if denom == 0:
                return None
            estimated = sp.Abs(sp.Rational(p.dot(wc), denom))
            # delta() has populated _high_res_constant at the escalated precision.
            constant = self._high_res_constant
            if constant is None:
                constant = self.constant().evalf(self._constant_resolution)
            err = sp.Abs(estimated - constant)
            if err == 0:
                return None   # exact within available precision
            return -self._log10(err)
        return self.__get(f"digits_computed_{depth}", compute)

    def avg_computed_digits_per_step(self, depth: Optional[int] = None) -> Optional[float]:
        """Average correct digits per step:  ``digits_computed / depth``."""
        depth = depth or self._depth
        def compute() -> Optional[float]:
            dc = self.digits_computed(depth)
            if dc is None:
                return None
            return dc / depth
        return self.__get(f"avg_computed_dps_{depth}", compute)

    # ==================================================================
    #  PENDING IMPLEMENTATIONS  (stubs — user will fill in later)
    # ==================================================================

    def _pq_vector(self, depth: Optional[int] = None) -> Tuple[Tuple[list, list], list] | Tuple[None, None]:
        """Numerator and denominator projection vectors (p, q) such that constant = p·walk / q·walk.

        The (p, q) integer relation is **depth-independent**, so it is identified
        once (via LIReC) at the cheap fixed depth ``search.IDENTIFY_DEPTH`` and then
        cached and reused for the deeper δ / spectral computations.  LIReC — and the
        walk that feeds it (whose convergents are huge rationals that must be
        ``evalf``'d) — get much more expensive at large depth, so this avoids a
        redundant deep identification walk.  If identification fails at that depth
        (a slow-converging trajectory whose convergents are not yet accurate enough
        there), it falls back to the full walk depth, so the result is **identical**
        to identifying at the full depth — only faster in the common case.

        The ``depth`` argument is retained for API compatibility but no longer
        selects the identification depth (that is ``min(IDENTIFY_DEPTH, walk_depth)``);
        callers that need a walk column at a specific depth use
        :meth:`_effective_walk_values` directly.
        """
        def compute():
            if self.constant() is None:
                # No target constant ⇒ identification cannot run (it needs the
                # constant for LIReC).  Matches ``identified()``'s own None guard
                # and keeps Limit-based utilities (``limit``) from dereferencing
                # ``None`` when called on a constant-less handler (e.g. the
                # ``constant=None`` handler the analyzer builds before injecting
                # constants via ``compute_for_constant``).
                return None, None
            id_depth = min(int(search_config.IDENTIFY_DEPTH), int(self._depth))
            vectors, walk_values = self._identify_at(id_depth)
            if vectors is None and self._depth > id_depth:
                # Cheap identification failed — retry at the full walk depth so a
                # slow-converging trajectory still identifies (result-preserving).
                vectors, walk_values = self._identify_at(self._depth)
            return vectors, walk_values

        return self.__get_utility("pq_vector", compute)

    def _identify_at(self, depth: int):
        """Identify the (p, q) integer relation from the walk column at ``depth``.

        :return: ``((p, q), walk_values)`` on success; ``(None, walk_values)`` when
            the walk succeeded but no verified relation was found (LIReC error /
            empty / failed the ``IDENTIFY_CHECK_THRESHOLD`` reconstruction check);
            ``(None, None)`` when the walk itself failed.  ``walk_values`` is the
            effective walk column at ``depth``.
        """
        walk_values, _ = self._effective_walk_values(depth)
        if walk_values is None:
            # Walk failed (singular matrix, ZeroDivisionError, …) — no
            # identification possible.  ``identified()`` reads this as False;
            # ``p_vector``/``q_vector`` propagate ``None``.
            return None, None
        low_res_constant = self.constant().evalf(search_config.CONSTANT_NO_DIGITS_LOW_RES)
        walk_col = sp.Matrix(walk_values)

        # If searchable is provided, try to find a cached p/q pair that matches the effective walk values.
        if self._searchable:
            def matcher(v):
                v1, v2 = v
                v1 = sp.Matrix(v1).T
                v2 = sp.Matrix(v2).T
                numerator = v1.dot(walk_col)
                denom = v2.dot(walk_col)
                err = sp.Abs(sp.Abs(sp.Rational(numerator, denom)) - low_res_constant)
                return sp.N(err, 25) < search_config.CACHE_ACCEPTANCE_THRESHOLD

            if matched := self._searchable.cache.find(matcher):
                # Cache hit: matcher verified this (p, q) reconstructs the
                # constant.  Returning a non-None result is itself the
                # signal that identification succeeded (see ``identified``).
                return matched, walk_values

        # Compute p, q using LIReC (values evaluated at CONSTANT_NO_DIGITS_LOW_RES).
        try:
            res = db.identify(
                [low_res_constant] + [v.evalf(search_config.CONSTANT_NO_DIGITS_LOW_RES) for v in walk_values[1:]]
            )
        except Exception as e:
            Logger(f'Error while identifying constant. LIReC failed with: "{e}"', Logger.Levels.warning).log()
            return None, walk_values

        # LIReC may also return an empty list when it cannot identify the constant
        if len(res) == 0:
            return None, walk_values

        # extract p, q from LIReC result
        res = res[0]
        res.include_isolated = 0
        estimated_expr = sp.nsimplify(str(res).rsplit(' ', 1)[0], rational=True)
        numerator, denom = sp.fraction(estimated_expr)
        p_dict = numerator.as_coefficients_dict()
        q_dict = denom.as_coefficients_dict()
        syms = sp.symbols(f'c:{self.traj_size()}')[1:]
        ext_syms = [1] + list(syms)
        # Keep coefficients as sympy Numbers — they can be Rational
        # (e.g. ``1/2``) and ``int(Rational)`` raises.  JSON-safety is
        # handled at the DTO boundary by ``_pq_to_jsonsafe``.
        p = [p_dict[sym] for sym in ext_syms]
        q = [q_dict[sym] for sym in ext_syms]

        estimated = estimated_expr.subs({sym: v for sym, v in zip(ext_syms, list(walk_values))})
        err = sp.Abs(estimated - self.constant())
        if sp.N(err, 15) > search_config.IDENTIFY_CHECK_THRESHOLD:
            # LIReC provided a combination that does not reconstruct the constant
            # to tolerance (spurious relation) — treat as not identified.
            return None, walk_values

        if self._searchable:
            self._searchable.cache.append((tuple(p), tuple(q)))
        return (p, q), walk_values

    def p_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector p such that p·walk gives the numerator sequence."""
        pq, _ = self._pq_vector(depth or self._depth)
        return pq[0] if pq is not None else None

    def q_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector q such that q·walk gives the denominator sequence."""
        pq, _ = self._pq_vector(depth or self._depth)
        return pq[1] if pq is not None else None

    # ==================================================================
    #  ASYMPTOTICS  (Birkhoff-Trjitzinsky)
    # ==================================================================

    def asymptotics(self, precision=None) -> list:
        """
        Formal asymptotic basis for the recurrence solutions.

        Uses LinearRecurrence.asymptotics() which runs the Birkhoff-Trjitzinsky
        reduction algorithm to find the canonical fundamental matrix.

        Returns a list of sympy expressions — one per solution of the recurrence.
        The last column of the CFM (transposed) gives the asymptotic behavior
        of p_n and q_n.

        These encode the growth rates η (factorial), γ (exponential), β (polynomial)
        from the NeurIPS 2024 paper in symbolic form.
        """
        def compute():
            import random
            rand = random.randint(1, 1_000_000)
            Logger(f'computing asymptotics [id={rand}] ... ').log()
            Logger(f'the linear recurrence [id={rand}] is: {self.linear_recurrence()}').log()
            precision = 5
            result = self.linear_recurrence().asymptotics(precision)
            Logger(f'asymptotics computed for prec = 5 [id={rand}]').log()
            result = self.linear_recurrence().asymptotics(None)
            Logger(f'computation successful [id={rand}]!').log()
            return result

        return self.__get(f"asymptotics_{precision}", compute)

    def identified(self) -> bool:
        """Whether the trajectory both identifies and converges to the target.

        A trajectory is identified iff *all* of:
          1. ``_pq_vector()`` produced numerator/denominator coefficients
             (LIReC succeeded, or a cache hit matched the constant).
          2. The path converges to the target constant (the convergence
             sanity check inside ``delta`` passes).
          3. The resulting ``delta`` is a well-defined finite float.

        All three conditions collapse to a single check:
        ``math.isfinite(self.delta())``.  ``delta`` returns ``float('-inf')``
        whenever any of them fails (walk error, ``_pq_vector`` is ``None``,
        non-converging path, LIReC silent failure).  ``delta`` is cached, so
        asking ``identified`` after ``delta`` is O(1); asking it first
        triggers the same computation that ``delta`` would have anyway.

        Worker handlers without a constant return ``False`` because the
        identification pipeline can't run.
        """
        if self._constant is None:
            return False
        return math.isfinite(self.delta())

    def _clear_constant_dependent_cache(self) -> None:
        """Drop every cached value that depends on the target constant.

        Called around a constant swap in :meth:`compute_for_constant`.  Keeps the
        constant-independent spectral / structural attributes (see
        :data:`_CONST_INDEPENDENT_CACHE_KEYS`) so the eigenvalue work is not
        repeated per constant, and always drops the constant-dependent p/q vector.

        This is deliberately broader than clearing only ``"delta"``-keyed entries:
        ``delta_prediction`` / ``gcd_slope`` / ``approximated_digits_per_step`` /
        ``convergence_rate`` / ``limit`` are all constant-dependent (they read the
        identified p/q or match the actual δ) but their cache keys do **not**
        contain ``"delta"``, so a substring clear would leak one constant's value
        into the next.
        """
        self._utility_cache.pop("pq_vector", None)
        for key in list(self._cache.keys()):
            if not key.startswith(_CONST_INDEPENDENT_CACHE_KEYS):
                del self._cache[key]

    def compute_for_constant(self, constant, sync_attributes: tuple = ()) -> tuple:
        """Evaluate delta / p / q / identified (+ optional extra metrics) for *constant*.

        Reuses the cached walk matrices from this handler (the walk is
        constant-independent).  Only the LIReC identification and constant-dependent
        derived values are recomputed.

        When *sync_attributes* is given, those registry attributes are computed
        **within the same constant-set scope** (so eigenvalue-based metrics share
        δ's walk + spectral work in a single pass) and returned as a flat
        ``{attr_name: value}`` dict — this is what makes the per-``(trajectory,
        constant)`` row carry the correct, constant-specific spectral metrics.

        Returns ``(delta, p_vector, q_vector, identified, extra)`` where ``delta``
        is a ``float``, ``extra`` is the (possibly empty) metrics dict, and the
        rest match :meth:`delta` / :meth:`p_vector` / :meth:`q_vector` /
        :meth:`identified`.
        """
        # Ensure the walk is cached before swapping the constant (walk is
        # constant-independent, so we prime it here if not already done).
        _ = self._effective_walk_values(self._depth)

        old_constant = self._constant
        self._constant = constant
        self._clear_constant_dependent_cache()

        try:
            delta = self.delta()
            p = self.p_vector()
            q = self.q_vector()
            ided = self.identified()
            extra: dict = {}
            if sync_attributes:
                # Local import avoids a module-load cycle with attribute_registry.
                from dreamer.utils.storage.attribute_registry import compute_attributes
                extra = compute_attributes(self, sync_attributes, on_error="store")
        finally:
            # Restore the previous constant regardless of exceptions, and clear
            # again so subsequent calls see the right constant.
            self._constant = old_constant
            self._clear_constant_dependent_cache()
        return delta, p, q, ided, extra

    def companion_coboundary_rank(self) -> int:
        """
        Rank of the coboundary matrix of the companion.
        """
        return self.__get("coboundary_rank", lambda:
            self.trajectory_matrix_typed.companion_coboundary_matrix().rank()
        )