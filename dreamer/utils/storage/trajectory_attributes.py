from __future__ import annotations

import hashlib
import json
import math
import warnings
from functools import lru_cache
from typing import List, Optional, TYPE_CHECKING, Tuple

import sympy as sp
from sqlalchemy.engine import result
from sympy.abc import n

from LIReC.db.access import db
from ramanujantools import LinearRecurrence, Matrix, Limit
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

def build_trajectory_dto(
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
) -> "TrajectoryDTO":
    """Build a ``TrajectoryDTO`` carrying Tier-1 attributes from a handler.

    The trajectory matrix walk is computed once; delta / p / q / identified
    are evaluated for every constant in *constants* by calling
    :meth:`TrajectoryAttributesHandler.compute_for_constant` which swaps the
    constant and clears only the per-constant caches, reusing the expensive
    walk.  Results are stored as ``{const_name: value}`` dicts.

    Parameters
    ----------
    handler:
        A ``TrajectoryAttributesHandler`` constructed from the trajectory
        (``constant`` may be ``None``; each constant is injected via
        ``compute_for_constant``).
    cmf_id, shard_id:
        Identifiers for the parent CMF and shard.
    cmf_name, shard_encoding_str:
        Used together to build the deterministic trajectory id.
    start, direction:
        The ``ramanujantools.Position`` objects for this trajectory.
    constants:
        Iterable of ``sympy.Expr`` objects to evaluate.  When ``None``,
        falls back to ``handler.constant()`` (backward-compatible path for
        single-constant callers).
    compute_recurrence:
        When ``True``, populate ``recurrence_relation`` (``formula_str``) and
        ``recurrence_order`` (``order``).  Default ``False`` because building
        the symbolic ``LinearRecurrence`` dominates the per-trajectory cost
        (~80% in profiling) and is rarely needed on the hot search/analysis
        path.  Request it through the Tier-2 ``"formula"`` / ``"order"``
        attributes instead when only some trajectories need it.
    """
    from dreamer.utils.storage.dtos import TrajectoryDTO  # lazy import avoids circular dep

    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)
    trajectory_id = derive_trajectory_id(
        shard_id, cmf_name, shard_encoding_str, start_t, dir_t,
    )

    # Resolve which constants to evaluate.
    # ``constants`` should be a list of ``Constant`` objects (preferred) or
    # sympy expressions (legacy path).  Using ``Constant`` objects lets us
    # store the human-readable name as the dict key while using the sympy
    # expression for computation — keeping keys consistent across
    # ``build_trajectory_dto`` and the ``c.name`` checks in the analyzer.
    from dreamer.utils.constants.constant import Constant as _Constant  # local import avoids circular

    if constants is None:
        c_expr = handler.constant()
        constants_list = [c_expr] if c_expr is not None else []
    else:
        constants_list = list(constants)

    delta_dict: dict = {}
    p_dict: dict = {}
    q_dict: dict = {}
    identified_dict: dict = {}

    for c in constants_list:
        if isinstance(c, _Constant):
            c_name = c.name
            c_sympy = c.value_sympy
        else:
            # Backward-compat: raw sympy expression.
            c_name = str(c)
            c_sympy = c

        delta, p, q, ided = handler.compute_for_constant(c_sympy)
        delta_dict[c_name] = float(delta)
        p_dict[c_name] = tuple(_pq_to_jsonsafe(x) for x in p) if p else None
        q_dict[c_name] = tuple(_pq_to_jsonsafe(x) for x in q) if q else None
        identified_dict[c_name] = bool(ided)

    # Recurrence (formula + order) builds the symbolic LinearRecurrence — the
    # dominant per-trajectory cost — so compute it only when explicitly asked.
    recurrence_relation = handler.formula_str() if compute_recurrence else None
    recurrence_order = handler.order() if compute_recurrence else None

    return TrajectoryDTO(
        trajectory_id=trajectory_id,
        cmf_id=cmf_id,
        shard_id=shard_id,
        start_point=start_t,
        direction=dir_t,
        recurrence_relation=recurrence_relation,
        recurrence_order=recurrence_order,
        limit_value=float(handler.limit()),
        delta_estimate=delta_dict,
        p_vector=p_dict if p_dict else None,
        q_vector=q_dict if q_dict else None,
        identified=identified_dict,
        walk_type=int(handler.walk_type()),
        walk_depth=int(handler.walk_depth()),
        config_fingerprint=tier1_config_fingerprint(handler.walk_depth()),
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
        """
        self._traj = traj_matrix
        self._constant = constant
        self._depth = walk_depth
        self._cache: dict = {}
        if walk_type is None:
            walk_type = 1 if search_config.DEFAULT_USES_INV_T else 2
        self._walk_type = walk_type
        self._searchable = searchable
        self._utility_cache: dict = {}  # separate cache for non-core attributes like p/q vectors

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
        return cls(tmat, constant, walk_depth, walk_type, searchable)

    # ------------------------------------------------------------------
    #  Cache helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _get_utility(self, key: str, fn):
        if key not in self._utility_cache:
            self._utility_cache[key] = fn()
        return self._utility_cache[key]

    def clear_cache(self):
        """Drop all cached core (Tier-1/2) attribute results."""
        self._cache.clear()

    def clear_utility_cache(self):
        """Drop all cached utility results (walks, p/q vectors)."""
        self._utility_cache.clear()

    def computed_attributes(self) -> list:
        """:return: Names of the core attributes computed and cached so far."""
        return list(self._cache.keys())

    # ==================================================================
    #  TRAJECTORY MATRIX
    # ==================================================================

    def trajectory_matrix(self) -> Matrix:
        """Raw d×d symbolic trajectory matrix M(n).

        The ``inv().T`` transform for ``walk_type=1`` is applied *after*
        walking (see :meth:`_walked_matrix`), not to ``M`` itself —
        ``walk(M.inv().T) ≠ walk(M).inv().T`` in general.  Recurrence-level
        attributes (linear_recurrence, companion, eigenvalues, kamidelta,
        gcd_slope) all operate on raw ``M``.
        """
        if self.walk_type() == 1:
            self._traj = self._traj.inv().T
        return self._traj

    def traj_size(self) -> int:
        """Dimension d of the trajectory matrix."""
        return self.trajectory_matrix().shape[0]

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
                return self.trajectory_matrix().walk({n: 1}, depth, {n: 0})
            except Exception as e:
                Logger(
                    f'Walk failed at depth {depth}: {e}',
                    Logger.Levels.warning,
                ).log()
                return None
        return self._get_utility(f"walked_{depth}", compute)

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
            tmat = self.trajectory_matrix()
            # sp.cancel() commented out: LinearRecurrence extracts recurrence
            # coefficients correctly from the unsimplified symbolic matrix.
            # Re-enable here if symbolic simplification is needed for Tier-2 attributes.
            # if hasattr(tmat, 'applyfunc'):
            #     tmat = tmat.applyfunc(sp.cancel)
            # elif hasattr(tmat, 'matrix'):
            #     tmat.matrix = tmat.matrix.applyfunc(sp.cancel)
            return LinearRecurrence(tmat)
        return self._get("linear_recurrence", _build)

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
        return self._get("companion", lambda: self.trajectory_matrix().as_companion())

    # ==================================================================
    #  RECURRENCE FORMULA ATTRIBUTES
    # ==================================================================

    def order(self) -> int:
        """
        Order d of the recurrence.
        d=2 → f(n) depends on f(n-1) and f(n-2).
        """
        return self._get("order", lambda:
            self.linear_recurrence().order()
        )

    def relation(self) -> list:
        """
        Raw recurrence relation [a_0(n), a_1(n), ..., a_d(n)].

        These define:  Σ a_i(n) · f(n-i) = 0
        i.e.:  a_0(n)·f(n) + a_1(n)·f(n-1) + ... + a_d(n)·f(n-d) = 0

        Rearranging: f(n) = -[a_1(n)·f(n-1) + ... + a_d(n)·f(n-d)] / a_0(n)
        """
        return self._get("relation", lambda:
            self.linear_recurrence().relation
        )

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
        return self._get("recurrence_coeffs", compute)

    def coeff_degrees(self) -> list:
        """
        Polynomial degrees of the relation coefficients.
        Uses LinearRecurrence.degrees() which returns degrees of [a_0, ..., a_d].
        """
        return self._get("coeff_degrees", lambda:
            self.linear_recurrence().degrees()
        )

    def formula_str(self) -> str:
        """
        Human-readable recurrence formula.
        Uses LinearRecurrence.__str__() which gives: Σ a_i(n)·p(n-i) = 0
        """
        return self._get("formula_str", lambda:
            str(self.linear_recurrence())
        )

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
        """
        return self._get("sorted_eigenvalues", lambda: self.companion().sorted_eigenvals())

    def eigenvalue_errors(self) -> list:
        """
        log|λ₁/λᵢ| for i = 2, ..., d.
        These are the 'error terms' — the log-ratios between the dominant
        eigenvalue and each subdominant one. Used internally by kamidelta().

        From Matrix.errors():
            errors[0] = log|λ₁/λ₂|  (primary convergence rate)
            errors[1] = log|λ₁/λ₃|  (if d ≥ 3)
            etc.
        """
        def compute():
            # Using Ramanujan-tools implementation of errors() for caching
            lambdas = [e.evalf(chop=True) for e in self.sorted_eigenvalues()]
            deltas = []
            for i in range(1, len(lambdas)):
                deltas.append(sp.log(abs(lambdas[0]) / abs(lambdas[i])))
            return deltas
        return self._get("eigenvalue_errors", compute)

    def spectral_gap(self) -> Optional[float]:
        """
        |λ₁| − |λ₂| from the Poincaré eigenvalues.
        Large gap → fast convergence, small gap → slow/noisy.
        """
        def compute():
            eigs = self.sorted_eigenvalues()
            if len(eigs) >= 2:
                return float(abs(eigs[0]).evalf() - abs(eigs[1]).evalf())
            return None
        return self._get("spectral_gap", compute)

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

    def _effective_walk_values(self, depth: Optional[int] = None, walk_matrix: Optional[sp.Matrix] = None) -> Optional[list]:
        """Return the column of the (walked-and-transformed) matrix used for p/q projection.

        Picks the first column with a non-zero top entry, prefers one with no
        zero entries (LIReC-friendly), and normalises by the top entry.
        Pass ``depth`` to compute internally (cached via :meth:`_walked_matrix`,
        so the ``inv().T`` transform for ``walk_type==1`` is applied AFTER the
        walk).  ``walk_matrix`` may be a raw ``Limit.current`` snapshot —
        ``inv().T`` is applied here when ``walk_type==1``.
        """
        if depth is None and walk_matrix is None:
            Logger(
                'No depth or walk matrix provided for effective walk values. '
                'This was not supposed to happen. Skipping trajectory...',
                Logger.Levels.exception,
            ).log()
            return None

        depth = depth or self._depth

        def compute():
            if walk_matrix is None:
                walked = self._walked_matrix(depth)
            else:
                try:
                    walked = walk_matrix
                    if self.walk_type() == 1:
                        walked = walked.inv().T
                except Exception as e:
                    Logger(
                        f'inv().T transform failed: {e}',
                        Logger.Levels.warning,
                    ).log()
                    walked = None
            if walked is None:
                return None

            lirec_valid_col = None
            normalized_col = None
            for col_ind in range(sp.shape(walked)[1]):
                if walked[0, col_ind].is_zero:
                    continue

                col = (walked / walked[0, col_ind]).col(col_ind)
                if normalized_col is None:
                    normalized_col = col

                if all([not v.is_zero for v in col]):
                    lirec_valid_col = col
                    break

            if normalized_col is None:
                Logger(
                    'Could not normalize any walk matrix column. '
                    'Skipping trajectory...',
                    Logger.Levels.warning,
                ).log()
                return None

            if lirec_valid_col is not None:
                normalized_col = lirec_valid_col

            return [item for item in normalized_col]

        if walk_matrix is not None:
            return compute()
        return self._get_utility(f"effective_walk_{depth}", compute)

    def _limits(self, depths: list) -> list:
        """
        Internal: get Limit objects at specified depths.

        Returns ``[]`` when the walk fails (singular matrix, ZeroDivisionError
        on degenerate trajectories, etc.) — callers must handle this.
        """
        try:
            return self.trajectory_matrix().limit({n: 1}, depths, {n: 0})
        except Exception as e:
            Logger(
                f'_limits walk failed at depths={depths}: {e}',
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
                    f'limit failed at depth {depth}: {e}',
                    Logger.Levels.warning,
                ).log()
                return float('nan')
        return self._get(f"limit_{depth}", compute)

    def limit_rational(self, depth: Optional[int] = None):
        """
        Limit as an exact sympy Rational p/q.
        """
        depth = depth or self._depth
        return self._get(f"limit_rational_{depth}", lambda:
            self._limits([depth])[0].as_rational()
        )

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
                    f'delta failed at depth {depth}: {e}',
                    Logger.Levels.warning,
                ).log()
                return float('-inf')
        return self._get(f"delta_{depth}", compute)

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
            limits = self._limits(depth)
            vectors = self._pq_vector(depth[-1])
            if vectors is None:
                return []
            
            p, q = vectors
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            deltas = []
            high_res_constant = self.constant().evalf(search_config.CONSTANT_NO_DIGITS_HIGH_RES)

            for limit in limits:    
                walk_col = sp.Matrix(self._effective_walk_values(None, limit.current))
                numerator = p.dot(walk_col)
                denom = q.dot(walk_col)
                estimated = sp.Abs(sp.Rational(numerator, denom))
                err = sp.Abs(estimated - high_res_constant)
                # Match Searchable.calc_delta: use the integer denominator
                # of the rational estimate for the delta formula, not the
                # raw symbolic q·walk (which can be a fractional Rational).
                denom_int = sp.denom(estimated)
                
                if sp.Abs(denom_int) <= search_config.MIN_ESTIMATE_DENOMINATOR:
                    # probably didn't converge for some reason
                    deltas.append(float('-inf'))
                    continue

                delta = -1 - sp.log(err) / sp.log(denom_int)

                # This part is not supposed to be reached at all, these are the final guardrails
                if delta == sp.oo or delta == sp.zoo:
                    deltas.append(float('-inf'))
                    # Logger(f"Warning: Infinite delta estimate at step. Marking delta as -inf.", Logger.Levels.warning).log()
                    continue

                deltas.append(float(delta.evalf(10)))
            return deltas
        
        return self._get(f"delta_seq_{depth}", compute)

    def kamidelta(self, depth: int = 20) -> list:
        """
        BLIND irrationality measure prediction — NO knowledge of L needed.

        Uses the Kamidelta algorithm from ramanujantools:
            1. errors() computes log|λ₁/λᵢ| from Poincaré eigenvalues
            2. gcd_slope(depth) fits log(q̃_n) linearly
            3. kamidelta = -1 + error / slope

        Returns a list of predicted δ values (one per eigenvalue pair).
        For order-2 recurrences, this is a single-element list.
        """
        def compute():
            # Copied implementation from Ramanujan-Tools - utilizing cache here.
            errors = self.eigenvalue_errors()
            slope = self.gcd_slope(depth)
            return [-1 + error / slope for error in errors]
        return self._get(f'kamidelta_{depth}', compute)

    def gcd_slope(self, depth: int = 20):
        """
        Linear fit slope of log(q̃_n) = log(q_n / gcd(p_n, q_n)).
        This measures how fast the reduced denominator grows.

        Used internally by kamidelta, but also useful on its own.
        """
        # return self._get(f"gcd_slope_{depth}", lambda: self.trajectory_matrix().gcd_slope(depth))
        return self._get(f"gcd_slope_{depth}", lambda: self.companion().gcd_slope(depth))

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
        limits = self._limits([round(coef * depth) for coef in search_config.DEPTH_CONVERGENCE_THRESHOLD])
        floats = []
        vectors = self._pq_vector(depth)
        if vectors is None:
            return False, limits
        p, q = vectors
        p = sp.Matrix(p).T
        q = sp.Matrix(q).T

        # extract estimated limit
        for limit in limits:
            walk_col = self._effective_walk_values(depth, limit.current)
            values_vec = sp.Matrix(walk_col)
            numerator = p.dot(values_vec)
            denom = q.dot(values_vec)
            estimated = sp.Abs(sp.Rational(numerator, denom))
            floats.append(estimated)

        # check that the estimated limits are consistent (within error bound)
        diffs = [abs(floats[i] - floats[i-1]) for i in range(1, len(floats))]
        return all(diff < search_config.LIMIT_DIFF_ERROR_BOUND for diff in diffs), limits

    def precision_at(self, depth: Optional[int] = None) -> int:
        """
        Number of correct decimal digits at the given depth.
        Uses Limit.precision() which compares the last two walk steps.
        """
        depth = depth or self._depth
        return self._get(f"precision_{depth}", lambda:
            self._limits([depth])[0].precision()
        )

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
        return self._get(f"dps_{max_depth}", compute)

    def asymptotic_digits_per_step(self, max_depth: Optional[int] = None) -> Optional[float]:
        """
        Mean Δd in the tail (last 25%) of the digits-per-step trajectory.
        This is the stable long-run convergence rate.

        Expected:  Gosper N=29 → ≈8,  Chudnovsky N=41 → ≈14
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            dps = self.digits_per_step(max_depth)
            if not dps:
                return None
            tail = dps[max(0, int(0.75 * len(dps))):]
            if not tail:
                return None
            return sum(d for _, d in tail) / len(tail)
        return self._get(f"asymp_dps_{max_depth}", compute)

    def convergence_class(self, max_depth: Optional[int] = None) -> str:
        """
        Classify convergence by comparing early vs. late Δd values:
            'factorial'    — Δd grows (super-exponential)
            'exponential'  — Δd roughly constant
            'polynomial'   — Δd shrinks
            'unknown'      — not enough data
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            dps = self.digits_per_step(max_depth)
            if len(dps) < 8:
                return "unknown"
            vals = [d for _, d in dps]
            mid = len(vals) // 2
            early = sum(vals[:mid]) / mid
            late = sum(vals[mid:]) / len(vals[mid:])
            if early == 0:
                return "unknown"
            r = late / early
            if   r > 1.25: return "factorial"
            elif r < 0.80: return "polynomial"
            else:          return "exponential"
        return self._get(f"conv_class_{max_depth}", compute)

    # ==================================================================
    #  PENDING IMPLEMENTATIONS  (stubs — user will fill in later)
    # ==================================================================

    def _pq_vector(self, depth: Optional[int] = None) -> Optional[tuple[list, list]]:
        """Numerator and denominator projection vectors (p, q) such that constant = p·walk / q·walk."""
        depth = depth or self._depth

        def compute():
            walk_values = self._effective_walk_values(depth)
            if walk_values is None:
                # Walk failed (singular matrix, ZeroDivisionError, …) — no
                # identification possible.  ``identified()`` reads this as
                # False; ``p_vector``/``q_vector`` propagate ``None``.
                return None
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
                    return matched
            
            # Compute p, q using LIReC
            try:
                res = db.identify([low_res_constant] + walk_values[1:])
            except Exception as e:
                Logger(f'Error while identifing constnat. LIReC failed with: "{e}"', Logger.Levels.warning).log()
                return None

            # LIReC may also return an empty list when it cannot identify the constant
            if len(res) == 0:
                return None

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
            err = sp.Abs(estimated - self.constant().evalf(search_config.CONSTANT_NO_DIGITS_HIGH_RES))
            if sp.N(err, 15) > search_config.IDENTIFY_CHECK_THRESHOLD:
                d = len(walk_values)
                return None

            if self._searchable:
                self._searchable.cache.append((tuple(p), tuple(q)))
            return p, q

        return self._get_utility("pq_vector", compute)

    def p_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector p such that p·walk gives the numerator sequence."""
        pq = self._pq_vector(depth or self._depth)
        return pq[0] if pq is not None else None

    def q_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector q such that q·walk gives the denominator sequence."""
        pq = self._pq_vector(depth or self._depth)
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

        return self._get(f"asymptotics_{precision}", compute)

        # return self._get(f"asymptotics_{precision}", lambda:
        #     self.linear_recurrence().asymptotics(precision)
        # )

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

    def compute_for_constant(self, constant) -> tuple:
        """Evaluate delta / p_vector / q_vector / identified for *constant*.

        Reuses the cached walk matrices from this handler (the walk is
        constant-independent).  Only the LIReC identification and derived
        values are recomputed.

        Returns ``(delta, p_vector, q_vector, identified)`` where
        ``delta`` is a ``float`` and the others match the normal return
        types of :meth:`delta`, :meth:`p_vector`, :meth:`q_vector`,
        :meth:`identified`.
        """
        # Ensure the walk is cached before swapping the constant (walk is
        # constant-independent, so we prime it here if not already done).
        _ = self._effective_walk_values(self._depth)

        old_constant = self._constant
        self._constant = constant

        # Clear only the constant-dependent entries from both caches.
        self._utility_cache.pop("pq_vector", None)
        for key in list(self._cache.keys()):
            if "delta" in key:
                del self._cache[key]

        try:
            delta = self.delta()
            p = self.p_vector()
            q = self.q_vector()
            ided = self.identified()
        finally:
            # Restore the previous constant regardless of exceptions.
            self._constant = old_constant
            # Clear again so subsequent calls see the right constant.
            self._utility_cache.pop("pq_vector", None)
            for key in list(self._cache.keys()):
                if "delta" in key:
                    del self._cache[key]

        return delta, p, q, ided

    def companion_coboundary_rank(self) -> int:
        """
        Rank of the coboundary matrix of the companion.
        """
        return self._get("coboundary_rank", lambda:
            self.trajectory_matrix().companion_coboundary_matrix().rank()
        )