"""Tests for per-attribute fingerprints and the reuse policy.

The central guarantee is the **dependency test**: the declared dependencies in
``attribute_fingerprints.ATTRIBUTE_META`` must cover everything the code
actually reads.  It works statically:

1. Parse ``TrajectoryAttributesHandler``: per method, collect the config knobs
   read (``search_config.X`` / ``config.<section>.X``), the instance state read
   (``self._depth``, ``self._walk_type``, ...) and the methods called
   (``self.m(...)``).
2. Parse ``ATTRIBUTE_REGISTRY``: per attribute, the handler members its lambda
   touches (``h.method(...)``, ``h._depth``).
3. Close transitively and map state to knobs (walk depth → ``@depth``, walk
   type → ``DEFAULT_USES_INV_T``, constant resolution →
   ``CONSTANT_NO_DIGITS_HIGH_RES``).
4. Fail if a knob the code reads (minus the justified ``EXEMPT_KNOBS``) is not
   declared by the attribute or one of its (transitive) dependencies, or if an
   attribute calls another registered attribute's method without declaring it
   as a dependency.

A missing dependency would silently reuse a stale value — the one failure mode
fine-grained fingerprints must never have.  The remaining tests pin the
fingerprint semantics (each declared knob invalidates exactly its dependents)
and the three reuse policies.
"""

import ast
import inspect

import pytest

from dreamer.configs import config
from dreamer.utils.storage import attribute_fingerprints as af
from dreamer.utils.storage import attribute_registry as ar
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.attribute_fingerprints import (
    ATTRIBUTE_META,
    EXEMPT_KNOBS,
    TIER1,
    TIER1_ATTRIBUTES,
    attribute_fingerprint,
    stale_attributes,
    tier1_fingerprint,
    tier1_is_fresh,
)

#: Instance state that stands for a config knob / per-trajectory input.
STATE_KNOBS = {
    "_depth": "@depth",
    "_walk_type": af.WALK_TYPE_KNOB,
    "_constant_resolution": "search.CONSTANT_NO_DIGITS_HIGH_RES",
    "_high_res_constant": "search.CONSTANT_NO_DIGITS_HIGH_RES",
}


# ---------------------------------------------------------------------------
# Static scan
# ---------------------------------------------------------------------------

def _scan_handler():
    """``{method: (knobs, state_knobs, called_members)}`` for the handler class."""
    cls = ast.parse(inspect.getsource(ta.TrajectoryAttributesHandler)).body[0]
    out = {}
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        knobs, state, calls = set(), set(), set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Attribute):
                continue
            v = node.value
            if isinstance(v, ast.Name) and v.id.endswith("_config"):
                knobs.add(f"{v.id[: -len('_config')]}.{node.attr}")
            elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "config":
                knobs.add(f"{v.attr}.{node.attr}")
            elif isinstance(v, ast.Name) and v.id == "self":
                calls.add(node.attr)
                if node.attr in STATE_KNOBS:
                    state.add(STATE_KNOBS[node.attr])
        out[fn.name] = (knobs, state, calls)
    return out


def _scan_registry():
    """``{attribute: set of handler members its registry lambda touches}``."""
    tree = ast.parse(inspect.getsource(ar))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ATTRIBUTE_REGISTRY":
            mapping = node.value
            break
    else:
        raise AssertionError("ATTRIBUTE_REGISTRY not found")
    out = {}
    for key, value in zip(mapping.keys, mapping.values):
        members = {
            n.attr for n in ast.walk(value)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "h"
        }
        out[key.value] = members
    return out


HANDLER = _scan_handler()
REGISTRY = _scan_registry()
# Mangled private helpers (``self.__get``) appear as ``__get`` in source.
METHODS = set(HANDLER)


def _closure(members):
    """Transitive (knobs, members) reached from *members*."""
    knobs, seen, stack = set(), set(), list(members)
    while stack:
        m = stack.pop()
        if m in STATE_KNOBS:
            knobs.add(STATE_KNOBS[m])
        if m in seen or m not in METHODS:
            continue
        seen.add(m)
        k, s, calls = HANDLER[m]
        knobs |= k | s
        stack.extend(calls)
    # Built in __init__ / from_cmf: walk type and depth come from config there.
    return knobs, seen


def _declared(name, _seen=None):
    """Knobs + attributes covered by *name*'s declaration, transitively."""
    _seen = _seen or set()
    meta = ATTRIBUTE_META[name]
    knobs = set(meta.config) | ({"@depth"} if meta.uses_depth else set())
    attrs = {name}
    for d in meta.deps:
        if d in _seen:
            continue
        _seen.add(d)
        k, a = _declared(d, _seen)
        knobs |= k
        attrs |= a
    if name == TIER1:
        attrs |= set(TIER1_ATTRIBUTES)
    return knobs, attrs


#: Public handler method that "is" each registry attribute (for dependency checks).
PRIMARY_METHOD = {
    name: sorted(m for m in members if m in METHODS and not m.startswith("_"))[0]
    for name, members in REGISTRY.items()
    if any(m in METHODS and not m.startswith("_") for m in members)
}


# ---------------------------------------------------------------------------
# 1. The dependency test
# ---------------------------------------------------------------------------

class TestDependencyDeclarations:
    def test_every_registry_attribute_is_declared(self):
        missing = sorted(set(ar.ATTRIBUTE_REGISTRY) - set(ATTRIBUTE_META))
        assert not missing, f"attributes without a dependency declaration: {missing}"

    def test_declared_dependencies_exist(self):
        for name, meta in ATTRIBUTE_META.items():
            for d in meta.deps:
                assert d in ATTRIBUTE_META, f"{name} depends on unknown {d!r}"

    @pytest.mark.parametrize("name", sorted(REGISTRY))
    def test_code_reads_are_declared(self, name):
        """Every knob / depth / walk-type read reachable from the attribute is
        declared by it or one of its dependencies (minus justified exemptions)."""
        code_knobs, _ = _closure(REGISTRY[name])
        declared, _ = _declared(name)
        undeclared = sorted(k for k in code_knobs - declared if k not in EXEMPT_KNOBS)
        assert not undeclared, (
            f"{name!r} reads {undeclared} but does not declare them (directly or via deps) "
            f"in attribute_fingerprints.ATTRIBUTE_META — a stale value would be reused"
        )

    @pytest.mark.parametrize("name", sorted(REGISTRY))
    def test_attribute_calls_are_declared_dependencies(self, name):
        """If the attribute's code calls another registered attribute's method,
        that attribute must be a declared (transitive) dependency — unless the
        method is shared machinery that a declared dependency's own code already
        runs (e.g. ``delta()`` is implemented with the ``delta_sequence()``
        method, so Tier-1-based attributes reach it through ``tier1``)."""
        _, reached = _closure(REGISTRY[name])
        _, declared_attrs = _declared(name)
        covered = set()
        for dep in declared_attrs - {name}:
            covered |= _closure(REGISTRY.get(dep, set()))[1]
        own = PRIMARY_METHOD.get(name)
        for other, method in PRIMARY_METHOD.items():
            if other == name or method == own or method not in reached or method in covered:
                continue
            assert other in declared_attrs, (
                f"{name!r} uses {other!r} (via {method}()) but does not depend on it"
            )

    def test_exemptions_are_justified(self):
        for knob, why in EXEMPT_KNOBS.items():
            assert len(why) > 40, f"exemption for {knob} needs a real justification"

    def test_scan_sees_known_reads(self):
        """Guard the scanner itself: known reads must be detected (else every
        other test here passes vacuously)."""
        knobs, reached = _closure(REGISTRY["delta"])
        assert "search.LIMIT_DIFF_ERROR_BOUND" in knobs and "@depth" in knobs
        assert "search.MAX_CONSTANT_RESOLUTION" in knobs
        _, reached = _closure(REGISTRY["delta_prediction"])
        assert {"gcd_slope", "sorted_eigenvalues"} <= reached


# ---------------------------------------------------------------------------
# 2. Fingerprint semantics
# ---------------------------------------------------------------------------

def _bump(monkeypatch, knob):
    section, name = knob.split(".", 1)
    sec = getattr(config, section)
    cur = getattr(sec, name)
    if isinstance(cur, bool):
        new = not cur
    elif isinstance(cur, (int, float)):
        new = cur * 2 + 1
    elif isinstance(cur, tuple):
        new = tuple(x * 2 + 1 for x in cur)
    else:
        new = f"{cur}-changed"
    monkeypatch.setattr(sec, name, new)


class TestFingerprints:
    @pytest.mark.parametrize("knob", ATTRIBUTE_META[TIER1].config)
    def test_tier1_knob_invalidates_tier1_and_dependents_only(self, monkeypatch, knob):
        before = {n: attribute_fingerprint(n, 1000) for n in ATTRIBUTE_META}
        _bump(monkeypatch, knob)
        after = {n: attribute_fingerprint(n, 1000) for n in ATTRIBUTE_META}
        changed = {n for n in before if before[n] != after[n]}
        assert TIER1 in changed and "delta_prediction" in changed
        if knob != af.WALK_TYPE_KNOB:
            # Spectral / recurrence attributes do not depend on identification knobs.
            assert "eigenvalues" not in changed and "order" not in changed

    def test_dependency_change_propagates_through_a_chain(self, monkeypatch):
        """If (b)'s inputs change while (a)'s own inputs do not, (a) is still
        recomputed: fp(a) contains fp(b).  Checked along the full chain
        tier1 → gcd_slope → delta_prediction → approximated_digits_per_step →
        convergence_rate, and via stale_attributes on a stored row."""
        monkeypatch.setattr(config.system, "REUSE_POLICY", "strict")
        chain = ["gcd_slope", "delta_prediction", "approximated_digits_per_step", "convergence_rate"]
        row = {n: 1.0 for n in chain}
        row["fingerprints"] = {n: attribute_fingerprint(n, 1000) for n in chain}
        assert stale_attributes(row, chain, 1000) == []
        # convergence_rate declares no knob of its own; only tier1 (at the root) changes.
        assert ATTRIBUTE_META["convergence_rate"].config == ()
        _bump(monkeypatch, "search.DEPTH_CONVERGENCE_THRESHOLD")
        assert stale_attributes(row, chain, 1000) == chain

    def test_walk_type_invalidates_spectral(self, monkeypatch):
        before = attribute_fingerprint("eigenvalues", 1000)
        _bump(monkeypatch, af.WALK_TYPE_KNOB)
        assert attribute_fingerprint("eigenvalues", 1000) != before

    def test_depth_only_matters_where_used(self):
        assert tier1_fingerprint(1000) != tier1_fingerprint(800)
        assert attribute_fingerprint("eigenvalues", 1000) == attribute_fingerprint("eigenvalues", 800)
        assert attribute_fingerprint("delta_prediction", 1000) != attribute_fingerprint("delta_prediction", 800)

    def test_exempt_knob_changes_nothing(self, monkeypatch):
        before = {n: attribute_fingerprint(n, 1000) for n in ATTRIBUTE_META}
        _bump(monkeypatch, "search.IDENTIFY_DEPTH")
        assert before == {n: attribute_fingerprint(n, 1000) for n in ATTRIBUTE_META}

    def test_version_bump_invalidates_dependents(self, monkeypatch):
        before_ep = attribute_fingerprint("eigenvalue_errors", 1000)
        before_gcd = attribute_fingerprint("gcd_slope", 1000)
        meta = ATTRIBUTE_META["eigenvalues"]
        monkeypatch.setitem(ATTRIBUTE_META, "eigenvalues", af.AttributeMeta(
            config=meta.config, deps=meta.deps, uses_depth=meta.uses_depth, version=meta.version + 1,
        ))
        assert attribute_fingerprint("eigenvalue_errors", 1000) != before_ep
        assert attribute_fingerprint("gcd_slope", 1000) == before_gcd

    def test_undeclared_attribute_depends_on_everything(self, monkeypatch):
        before = attribute_fingerprint("my_runtime_attr", 1000)
        monkeypatch.setattr(config.search, "SA_PATIENCE", config.search.SA_PATIENCE + 1)
        assert attribute_fingerprint("my_runtime_attr", 1000) != before


# ---------------------------------------------------------------------------
# 3. Reuse policy
# ---------------------------------------------------------------------------

def _row(depth=1000, **extra):
    fps = {n: attribute_fingerprint(n, depth) for n in ("eigenvalues", "gcd_slope")}
    return {"delta": 1.0, "config_fingerprint": tier1_fingerprint(depth),
            "eigenvalues": ["1"], "gcd_slope": 0.5, "fingerprints": fps, **extra}


class TestReusePolicy:
    def test_strict(self, monkeypatch):
        monkeypatch.setattr(config.system, "REUSE_POLICY", "strict")
        row = _row()
        assert tier1_is_fresh(row, 1000) and not tier1_is_fresh(row, 900)
        assert stale_attributes(row, ["eigenvalues", "gcd_slope", "spectral_gap"], 1000) == ["spectral_gap"]
        _bump(monkeypatch, "search.LIMIT_DIFF_ERROR_BOUND")
        assert not tier1_is_fresh(row, 1000)
        # gcd_slope depends on tier1 → stale; eigenvalues does not → reused.
        assert stale_attributes(row, ["eigenvalues", "gcd_slope"], 1000) == ["gcd_slope"]

    def test_keep(self, monkeypatch):
        monkeypatch.setattr(config.system, "REUSE_POLICY", "keep")
        row = _row()
        _bump(monkeypatch, "search.LIMIT_DIFF_ERROR_BOUND")
        assert tier1_is_fresh(row, 1000)
        assert stale_attributes(row, ["eigenvalues", "gcd_slope", "spectral_gap"], 1000) == ["spectral_gap"]

    def test_recompute_ignores_earlier_runs_only(self, monkeypatch):
        """'recompute' folds the run id into every fingerprint: rows written in an
        earlier run are stale, rows written in this run (same id) are reused."""
        monkeypatch.setattr(config.system, "REUSE_POLICY", "strict")
        old_row = _row()                                  # written by an earlier run
        monkeypatch.setattr(config.system, "REUSE_POLICY", "recompute")
        monkeypatch.setattr(config.system, "RUN_ID", "run-B")
        assert not tier1_is_fresh(old_row, 1000)
        assert stale_attributes(old_row, ["eigenvalues"], 1000) == ["eigenvalues"]
        this_run = _row()                                 # written during run-B
        assert tier1_is_fresh(this_run, 1000)
        assert stale_attributes(this_run, ["eigenvalues"], 1000) == []
        monkeypatch.setattr(config.system, "RUN_ID", "run-C")
        assert not tier1_is_fresh(this_run, 1000)         # next run: stale again

    def test_missing_record_and_error_entries(self, monkeypatch):
        monkeypatch.setattr(config.system, "REUSE_POLICY", "strict")
        assert not tier1_is_fresh(None, 1000)
        assert stale_attributes(None, ["eigenvalues"], 1000) == ["eigenvalues"]
        row = _row()
        row.pop("gcd_slope")
        row["gcd_slope_error"] = "boom"
        assert stale_attributes(row, ["gcd_slope"], 1000) == []  # same fp → error is final

    def test_invalid_policy(self, monkeypatch):
        monkeypatch.setattr(config.system, "REUSE_POLICY", "sometimes")
        with pytest.raises(ValueError):
            tier1_is_fresh({}, 1000)
