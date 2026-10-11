r"""
Declarative grammar for Tier-3 attribute *predicates* (selectors).

A predicate gates whether an attribute is computed for a trajectory.  Config
entries stay plain data (strings); this module parses the user-facing grammar
into callables that :func:`dreamer.utils.storage.attribute_registry.compute_attributes`
can dispatch (it inspects each callable's arity to decide whether to pass the
shard/CMF ``context``).

Grammar
-------
Two parameterised forms plus full backward-compatibility:

* ``"max_degree below N"`` / ``"max_degree above N"`` — handler-only.
  ``below N`` ⇔ ``max(coeff_degrees) < N`` (the recurrence's polynomial degree is
  under N); ``above N`` ⇔ ``max(coeff_degrees) > N``.

* ``"top N highest <metric> in <scope>"`` / ``"... lowest ..."`` — a
  :class:`TopNSelector`, a *shard/CMF-level* gate.  ``<metric>`` must be a key in
  :data:`dreamer.utils.storage.record_metrics.METRIC_EXTRACTORS`; ``<scope>`` is
  ``shard`` or ``cmf``.  The trajectory passes iff it is among the top-N by that
  stored metric within its scope.  The producer precomputes the qualifying id-set
  and threads it through ``context["top_n_sets"][selector.key]``.

* Anything else — looked up in
  :data:`dreamer.utils.storage.attribute_registry.PREDICATES` (so ``if_identified``,
  ``if_has_degree_2``, ``if_top_n_delta`` keep working unchanged).

All parse failures (unknown metric, unknown scope, bad number, unknown predicate
name) raise loudly so a misspelled config fails fast rather than silently
skipping an attribute.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from dreamer.utils.storage.record_metrics import METRIC_EXTRACTORS

Predicate = Callable[..., bool]

_TOP_N_RE = re.compile(
    r"^top\s+(?P<n>\d+)\s+(?P<dir>highest|lowest)\s+(?P<metric>\w+)\s+in\s+(?P<scope>shard|cmf)$",
    re.IGNORECASE,
)
_MAX_DEGREE_RE = re.compile(
    r"^max_degree\s+(?P<op>below|above)\s+(?P<n>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TopNSelector:
    """A shard/CMF-scoped "top-N by metric" gate.

    Instances are *callable* with the two-argument predicate signature
    ``(handler, context) -> bool`` so they slot straight into
    ``compute_attributes`` arity dispatch.  The actual ranking is done once per
    scope by the producer, which fills ``context["top_n_sets"][self.key]`` with
    the set of qualifying ``trajectory_id``s; here we merely test membership.

    :param metric: key into :data:`record_metrics.METRIC_EXTRACTORS`.
    :param n: how many trajectories to keep.
    :param highest: ``True`` → keep the N largest metric values, ``False`` → the N smallest.
    :param scope: ``"shard"`` or ``"cmf"`` — the population the ranking is over.
    """

    metric: str
    n: int
    highest: bool
    scope: str

    @property
    def key(self) -> str:
        """Stable canonical id, shared by the producer (when building the id-set)
        and the predicate (when reading it back from ``context``)."""
        direction = "highest" if self.highest else "lowest"
        return f"top_{self.n}_{direction}_{self.metric}_in_{self.scope}"

    def __call__(self, _handler, context: Optional[dict] = None) -> bool:
        if not context:
            return False
        ids = (context.get("top_n_sets") or {}).get(self.key)
        if ids is None:
            return False
        return context.get("trajectory_id") in ids


def _make_max_degree_predicate(op: str, n: int) -> Predicate:
    """Build a handler-only predicate on the recurrence's polynomial degree."""
    def _pred(handler) -> bool:
        degrees = handler.coeff_degrees()
        if not degrees:
            return False
        top = max(int(d) for d in degrees)
        return top < n if op == "below" else top > n
    return _pred


def parse_predicate_spec(spec) -> Predicate:
    """Resolve a predicate reference to a callable.

    * Callables are returned unchanged.
    * Strings are matched against the grammar (``max_degree``/``top N``); on no
      match they fall back to the named-predicate registry.

    :raises KeyError: unknown metric, scope, or predicate name.
    :raises ValueError: malformed numeric arguments.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"Predicate spec must be a string or callable, got {type(spec)!r}")

    text = spec.strip()

    m = _MAX_DEGREE_RE.match(text)
    if m:
        return _make_max_degree_predicate(m.group("op").lower(), int(m.group("n")))

    m = _TOP_N_RE.match(text)
    if m:
        metric = m.group("metric").lower()
        if metric not in METRIC_EXTRACTORS:
            raise KeyError(
                f"Unknown ranking metric '{metric}' in predicate '{spec}'. "
                f"Registered: {sorted(METRIC_EXTRACTORS)}"
            )
        n = int(m.group("n"))
        if n <= 0:
            raise ValueError(f"top-N count must be positive, got {n} in '{spec}'")
        return TopNSelector(
            metric=metric,
            n=n,
            highest=m.group("dir").lower() == "highest",
            scope=m.group("scope").lower(),
        )

    # Fall back to the named-predicate registry (if_identified, ...).
    from dreamer.utils.storage.attribute_registry import PREDICATES
    if text in PREDICATES:
        return PREDICATES[text]
    raise KeyError(
        f"Unknown predicate '{spec}'. Expected the grammar "
        f"('max_degree below/above N', 'top N highest/lowest <metric> in shard/cmf') "
        f"or a registered name: {sorted(PREDICATES)}"
    )


def iter_top_n_selectors(specs) -> list["TopNSelector"]:
    """Return the :class:`TopNSelector` instances referenced by an attribute-spec list.

    Used by the producer to discover which rankings it must precompute.  Bare
    strings and ``(name, predicate)`` tuples are both handled; non-top-N
    predicates are ignored.
    """
    selectors: list[TopNSelector] = []
    for spec in specs:
        if isinstance(spec, str):
            continue
        _name, pred_ref = spec
        try:
            resolved = parse_predicate_spec(pred_ref)
        except (KeyError, ValueError, TypeError):
            # Resolution errors surface later in compute_attributes; the producer
            # only needs the well-formed top-N selectors here.
            continue
        if isinstance(resolved, TopNSelector):
            selectors.append(resolved)
    return selectors
