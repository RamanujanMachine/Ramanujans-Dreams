"""
Central registry of trajectory attributes that can be computed by the pipeline.

A dictionary maps an attribute name (the public, config-facing string) to a
function that, given a fully constructed ``TrajectoryAttributesHandler``,
returns a JSON-serialisable value.

The pipeline stages — analysis, search-stage producer, and search-stage
background workers — read their own ordered list of attribute *specs* from
config and call ``compute_attributes(handler, specs)``.  Each spec is either
a bare attribute name (always compute) or a ``(name, predicate)`` tuple where
``predicate`` is either a callable on the handler or a string key into
``PREDICATES`` (e.g. ``"if_identified"``).  The predicate gates the
computation: if it returns ``False``, the attribute is silently skipped.

Adding a new attribute takes one line in :data:`ATTRIBUTE_REGISTRY` plus, if
needed, a new method on the handler.

The serialiser layer is intentionally tight:
    handler -> raw value -> JSON-safe value (str / float / int / list / None / ...)
This keeps SymPy / mpmath / numpy objects out of the JSONL files.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler


# ---------------------------------------------------------------------------
# Small helpers that normalise raw handler outputs to JSON-safe forms
# ---------------------------------------------------------------------------

def _opt_float(value) -> float | None:
    """Return ``float(value)`` or ``None`` when the value is None."""
    return None if value is None else float(value)


def _opt_int(value) -> int | None:
    """Return ``int(value)`` or ``None`` when the value is None."""
    return None if value is None else int(value)


_FLOAT_STR_DIGITS = 30  # significant-digit cap for mpmath/sympy floats in JSONL


def _numeric_str(value, n: int = _FLOAT_STR_DIGITS) -> str:
    """Stringify *value* with at most *n* significant digits for float types.

    mpmath ``mpf``/``mpc`` objects use ``mp.nstr``; sympy ``Float`` uses
    ``str(round(v, n))``.  All other types fall back to plain ``str()``.
    """
    try:
        import mpmath as mp
        if isinstance(value, mp.mpf):
            return mp.nstr(value, n)
        if isinstance(value, mp.mpc):
            return mp.nstr(value, n)
    except ImportError:
        pass
    if getattr(value, "is_Float", False):
        try:
            from sympy import Float
            return str(Float(value, n))
        except Exception:
            pass
    return str(value)


def _list_of_str(values) -> list[str]:
    """Stringify each element with limited precision for numeric types."""
    return [_numeric_str(v) for v in values]


_EIGENVALUE_STR_DIGITS = 100  # eigenvalues are stored at high precision for reconstruction


def _eigenvalue_str(value, n: int = _EIGENVALUE_STR_DIGITS) -> str:
    """High-precision numeric string of an eigenvalue for storage.

    Evaluates the (possibly symbolic) eigenvalue to *n* digits — resolving any
    radical cancellation via SymPy's adaptive ``evalf`` — and applies a *true*
    zero filter: a value that is really ``x + 0i`` is collapsed to ``x`` (only an
    exactly-zero imaginary part is dropped; magnitudes are never chopped).
    """
    try:
        v = sp.sympify(value).evalf(n)
        if sp.im(v) == 0:
            v = sp.re(v)
        return str(v)
    except Exception:
        return str(value)


def _list_of_eigenvalue_str(values) -> list[str]:
    """High-precision eigenvalue strings (see :func:`_eigenvalue_str`)."""
    return [_eigenvalue_str(v) for v in values]


def _list_of_int(values) -> list[int]:
    """Coerce each element to ``int`` — used for integer-valued sequences."""
    return [int(v) for v in values]


def _list_of_float(values) -> list[float]:
    """Coerce each element to ``float`` — used for numeric sequences."""
    return [float(v) for v in values]


def _serialize_delta_prediction(pred) -> Optional[dict]:
    """Convert a delta_prediction dict to a JSON-safe form.

    The handler returns ``{"predicted_delta": float, "lambda_1": sympy_expr,
    "lambda_2": sympy_expr}`` or ``None``.  Eigenvalue expressions are
    stringified with limited precision via :func:`_numeric_str`.
    """
    if pred is None:
        return None
    return {
        "predicted_delta": float(pred["predicted_delta"]),
        "lambda_1": _eigenvalue_str(pred["lambda_1"]),
        "lambda_2": _eigenvalue_str(pred["lambda_2"]),
    }


def _pq_jsonsafe_list(pq) -> list | None:
    """Convert a p/q coefficient list to JSON-safe form, or ``None`` when missing.

    Mirrors the per-element logic in
    :func:`dreamer.utils.storage.trajectory_attributes._pq_to_jsonsafe`:
    integers stay as ``int``, fractions/symbolics become ``str``.
    """
    if pq is None:
        return None
    out = []
    for v in pq:
        try:
            if getattr(v, "is_Integer", False) or isinstance(v, int):
                out.append(int(v))
                continue
        except Exception:
            pass
        out.append(str(v))
    return out


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

AttributeComputer = Callable[["TrajectoryAttributesHandler"], Any]

ATTRIBUTE_REGISTRY: Dict[str, AttributeComputer] = {
    # ----- Tier-1 — core scalars (cheap: walk-based), computed on the main thread. -----
    "delta":                       lambda h: float(h.delta()),
    "identified":                  lambda h: bool(h.identified()),
    "p_vector":                    lambda h: _pq_jsonsafe_list(h.p_vector()),
    "q_vector":                    lambda h: _pq_jsonsafe_list(h.q_vector()),
    "traj_size":                   lambda h: int(h.traj_size()),
    "projection_column":           lambda h: _opt_int(h.projection_column()),

    # ----- Tier-2 — heavier numerical / spectral / recurrence attributes. -----
    # ``order`` / ``formula`` / ``relation`` / ``coeff_degrees`` / ``recurrence_coeffs``
    # all build the symbolic ``LinearRecurrence`` (the dominant per-trajectory cost),
    # so they are Tier-2: computed only when requested, not on the hot path.
    "order":                       lambda h: int(h.order()),
    "formula":                     lambda h: h.formula_str(),
    "coeff_degrees":               lambda h: _list_of_int(h.coeff_degrees()),
    "relation":                    lambda h: _list_of_str(h.relation()),
    "recurrence_coeffs":           lambda h: _list_of_str(h.recurrence_coeffs()),
    "eigenvalues":                 lambda h: _list_of_eigenvalue_str(h.sorted_eigenvalues()),
    "eigenvalue_errors":           lambda h: _list_of_float(h.eigenvalue_errors()),
    "spectral_gap":                lambda h: _opt_float(h.spectral_gap()),
    "companion_coboundary_rank":   lambda h: int(h.companion_coboundary_rank()),
    # "asymptotics":                 lambda h: _list_of_str(h.asymptotics()),
    "gcd_slope":                   lambda h: _opt_float(h.gcd_slope()),
    "delta_prediction":            lambda h: _serialize_delta_prediction(h.delta_prediction()),
    "error_formula_ratio":         lambda h: _opt_float(h.error_formula_ratio()),
    # Eigenvalue-based digit predictions (per-step and at the walk depth).
    "approximated_digits_per_step": lambda h: _opt_float(h.approximated_digits_per_step()),
    "digits_approximation":        lambda h: _opt_float(h.digits_approximation()),
    # Length-normalised spectral convergence rate: approximated_digits_per_step
    # divided by the L2 norm of the trajectory direction (see
    # TrajectoryAttributesHandler.convergence_rate).  The single definition of
    # "convergence_rate" in the system — record_metrics reads this stored value.
    "convergence_rate":            lambda h: _opt_float(h.convergence_rate()),

    # ----- Tier-3 — symbolic / expensive attributes (post-process). -----
    "precision_at":                lambda h: _opt_int(h.precision_at()),
    # Walk-based correct digits of p/q vs. the constant, and its per-step average.
    "digits_computed":             lambda h: _opt_float(h.digits_computed()),
    "avg_computed_digits_per_step": lambda h: _opt_float(h.avg_computed_digits_per_step()),
    # ``delta_sequence`` defaults to the handler's full walk depth (often
    # 1500) and compares each step against the constant at 50 000 digits —
    # that is hours per trajectory in practice.  Cap the registry-driven
    # call to ``min(_depth, 100)`` so it mirrors ``digits_per_step`` and
    # stays inside a budget the post-process stage can actually complete.
    # Callers that want the full sequence can invoke ``h.delta_sequence()``
    # directly with an explicit depth.
    "delta_sequence":              lambda h: _list_of_float(
        h.delta_sequence(min(h._depth, 100))
    ),
    "digits_per_step":             lambda h: [[int(k), int(d)] for k, d in h.digits_per_step()],
}


# ---------------------------------------------------------------------------
# Named preconditions for conditional attribute computation
# ---------------------------------------------------------------------------

#: Predicate signature.  Two arities are accepted:
#:
#: * ``f(handler) -> bool`` — handler-only; everything the predicate needs is
#:   computable from the trajectory in isolation.  Examples: ``if_identified``,
#:   ``if_has_degree_2`` (``2 in h.coeff_degrees()``).
#: * ``f(handler, context) -> bool`` — handler plus a caller-supplied dict.
#:   Use for shard-level questions such as "is this trajectory in the top-N
#:   by delta", where the answer depends on neighbours.  The caller of
#:   ``compute_attributes`` builds the ``context`` once per shard.
#:
#: ``compute_attributes`` dispatches on arity via :func:`inspect.signature`,
#: so existing single-arg predicates keep working.
Predicate = Callable[..., bool]

PREDICATES: Dict[str, Predicate] = {
    # The trajectory both identifies the constant and yields a finite delta.
    # Typical use: gate expensive symbolic attrs that are only meaningful for
    # converging trajectories (asymptotics, eigenvalues, kamidelta, ...).
    "if_identified": lambda h: h.identified(),

    # Recurrence has at least one polynomial coefficient of degree 2 — a
    # handler-only example of a "complex" predicate that still answers from
    # the trajectory alone.  Users can register their own analogues for
    # other degrees / shapes via :func:`register_predicate`.
    "if_has_degree_2": lambda h: 2 in h.coeff_degrees(),

    # Trajectory is in the top-N by delta within its shard — a shard-level
    # predicate, so it reads ``context``.  The caller (typically the
    # post-process producer) precomputes the set of qualifying trajectory
    # ids and supplies them alongside the trajectory's own id::
    #
    #     ctx = {
    #         "trajectory_id": record["trajectory_id"],
    #         "top_n_ids": {r["trajectory_id"] for r in top_n_records},
    #     }
    #     compute_attributes(handler, specs, context=ctx)
    "if_top_n_delta": lambda _h, ctx: (
        ctx is not None
        and ctx.get("trajectory_id") in ctx.get("top_n_ids", set())
    ),
}


def register_predicate(name: str, fn: Predicate) -> None:
    """Register a named precondition for use in config-driven attribute lists.

    Named predicates let configs stay plain data (strings) instead of needing
    lambdas; the lookup happens inside :func:`compute_attributes`.
    """
    PREDICATES[name] = fn


# ---------------------------------------------------------------------------
# Attribute specs — string name OR (name, predicate) tuple
# ---------------------------------------------------------------------------

AttributeSpec = Union[str, Tuple[str, Union[str, Predicate]]]


def attribute_name(spec: AttributeSpec) -> str:
    """Return the attribute name regardless of whether ``spec`` is a bare
    string or a ``(name, predicate)`` tuple.  Used by callers that need to
    test ``name in extended_metrics`` on a mixed config list.
    """
    return spec if isinstance(spec, str) else spec[0]


def _resolve_predicate(pred: Union[str, Predicate]) -> Predicate:
    """Resolve a predicate reference to a callable.

    Delegates to :func:`dreamer.utils.storage.predicate_specs.parse_predicate_spec`,
    which understands the declarative grammar (``"max_degree below N"``,
    ``"top N highest <metric> in shard|cmf"``) and falls back to the named
    :data:`PREDICATES` registry for plain keys (``"if_identified"``, ...).
    Callables are returned as-is.  Raises ``KeyError`` on unknown names/metrics
    so misspelled configs fail loudly.
    """
    from dreamer.utils.storage.predicate_specs import parse_predicate_spec
    return parse_predicate_spec(pred)


def _predicate_arity(pred: Predicate) -> int:
    """Return the number of positional parameters a predicate accepts.

    Used to decide whether to pass ``context`` alongside the handler.
    Falls back to ``1`` for callables whose signature can't be introspected
    (rare — e.g. some C-implemented builtins).
    """
    try:
        params = inspect.signature(pred).parameters.values()
        return sum(
            1 for p in params
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (TypeError, ValueError):
        return 1


def _call_predicate(pred: Predicate, handler, context: Optional[dict]) -> bool:
    """Invoke ``pred`` with ``handler`` (and ``context`` when the predicate
    declares a second positional parameter).  Keeps single-arg predicates
    working unchanged while letting shard-level ones read the dict."""
    if _predicate_arity(pred) >= 2:
        return bool(pred(handler, context))
    return bool(pred(handler))


def compute_attribute(handler: "TrajectoryAttributesHandler", name: str) -> Any:
    """Compute a single registered attribute on *handler*.

    Raises ``KeyError`` if *name* is not registered — the misspelled-config
    case should fail loudly rather than silently skip an attribute.
    """
    if name not in ATTRIBUTE_REGISTRY:
        raise KeyError(
            f"Unknown attribute '{name}'. Registered: {sorted(ATTRIBUTE_REGISTRY)}"
        )
    return ATTRIBUTE_REGISTRY[name](handler)


def compute_attributes(
    handler: "TrajectoryAttributesHandler",
    specs,
    on_error: str = "store",
    context: Optional[dict] = None,
    watchdog_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute many attributes and collect them into a dict keyed by name.

    Parameters
    ----------
    handler:
        The handler to compute against.
    specs:
        Iterable of attribute specs.  Each spec is either:

        * a string — the attribute name (always compute), or
        * a ``(name, predicate)`` tuple — compute only when the predicate
          is truthy.  ``predicate`` may be a callable (``f(h)`` or
          ``f(h, ctx)``) or a string key into :data:`PREDICATES`
          (e.g. ``"if_identified"``, ``"if_top_n_delta"``).

        Mixed lists are supported, so a config can be plain data:
        ``("delta", "spectral_gap", ("eigenvalues", "if_identified"))``.
    on_error:
        ``'store'`` — record the exception message under ``<name>_error`` and
        continue (default; lets a single misbehaving attribute not poison the
        whole pipeline).  ``'raise'`` — re-raise the first exception.
    context:
        Optional dict passed to two-argument predicates.  Use this to thread
        shard-level state into predicates that need it (e.g. the set of
        top-N trajectory ids for an ``if_top_n_delta`` gate).  Single-arg
        predicates ignore it.
    watchdog_detail:
        Optional descriptor (e.g. ``"traj_id=… start=… direction=…"``) appended
        to the per-attribute watchdog warning so a heavy/stuck attribute (e.g.
        ``asymptotics``) can be traced back to its trajectory.  ``None`` keeps
        the warning generic.

    Returns
    -------
    dict mapping each *computed* attribute name to its value (or an
    ``<name>_error`` key when ``on_error='store'`` and a failure occurred).
    Attributes whose predicate returned False are silently absent from the
    output — that's the signal "we decided not to compute this".
    """
    out: Dict[str, Any] = {}
    for spec in specs:
        if isinstance(spec, str):
            name, predicate = spec, None
        else:
            name, pred_ref = spec
            predicate = _resolve_predicate(pred_ref)

        if predicate is not None and not _call_predicate(predicate, handler, context):
            continue

        try:
            from dreamer.utils.logger import Logger
            from dreamer.configs.logging import logging_config
            with Logger.watchdog(
                f"attribute:{name}",
                logging_config.WATCHDOG_ATTRIBUTE_SECONDS,
                detail=watchdog_detail,
            ):
                out[name] = compute_attribute(handler, name)
        except Exception as exc:  # pragma: no cover — behaviour driven by tests
            if on_error == "raise":
                raise
            out[f"{name}_error"] = str(exc)
    return out


def register_attribute(name: str, fn: AttributeComputer) -> None:
    """Register a custom attribute at runtime (e.g. for experiments).

    Overwrites any existing entry under the same name.
    """
    ATTRIBUTE_REGISTRY[name] = fn
