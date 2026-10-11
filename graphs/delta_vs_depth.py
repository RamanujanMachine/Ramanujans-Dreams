"""
δ-versus-depth curves with the eigenvalue-based **kamidelta** prediction overlaid.

Given a CMF, a start point and a list of trajectory directions, this script walks
each trajectory and plots how the irrationality measure δ evolves with the walk
depth.  Over each δ curve it also plots the *kamidelta* prediction — δ predicted
from the trajectory matrix eigenvalues,
``δ̂(n) = -1 + log(|λ₁|/|λ₂|) / gcd_slope(n)`` (see
:meth:`TrajectoryAttributesHandler.delta_prediction`) — evaluated at each depth.

This reuses the production attribute handler, so the δ and kamidelta values match
exactly what the search / post-process stages would compute.

This is a testing tool: everything is computed from scratch via the production
attribute handler, so it uses LIReC to identify the recurrence (finding the p/q
``initial_values`` and the projection column / ``final_projection``) exactly as
the pipeline does — no precomputed data is required.

Inputs
------
The CMF is built directly from a constructor expression, e.g. ``--cmf "pFq(4, 3, 1)"``
(or, for an existing run, an export cmf id like ``pFq_2_1_-1__0_0_0`` with
``--export-cmfs <dir>``).  The constant is a sympy expression or registered name,
e.g. ``--const "zeta(2)"``.  The start point and trajectories are given as:

    --start "(a, b, c, ...)"
    --trajectories "(x1, y1, z1, ...)" "(x2, y2, z2, ...)" ...

``--use_inv_t`` selects the inverse-transpose walk (walk type 1) instead of the
direct matrix walk (walk type 2, the default).

Final-value error bounds
------------------------
Both the δ curve and the kamidelta prediction converge but keep jittering even
after convergence (arithmetic / gcd fluctuations in the denominator), so each
final value is reported as ``mean ± σ`` over the last ``--error-window``
(default 100) **consecutive integer depths**: the tail mean, its sample standard
deviation σ (the drawn band on each curve), with the standard error of the mean
``SEM = σ/√n`` as the tighter (i.i.d.-assuming) bound.  A tail *trend* check
flags the case where a sequence is still drifting rather than merely scattering
(``drift > σ``) — a signal that the walk is too shallow for the bound to be
meaningful.  The curve legend and these descriptions are drawn in a footer strip
at the bottom of the figure so they never overlap the plot.

Constant resolution
-------------------
δ is computed at the production constant precision — the target constant is
``evalf``'d at ``CONSTANT_NO_DIGITS_HIGH_RES`` (10_000 digits), auto-doubling up
to ``MAX_CONSTANT_RESOLUTION`` (pinned here to 200_000) whenever a deep
approximation is good enough to underflow the current precision.  So higher
depths automatically get more digits; this script sets no per-call ``evalf`` of
its own (the ``.evalf(10)`` in the δ path is only output rounding for the plot).

Example
-------
    python graphs/delta_vs_depth.py --const "zeta(2)" --cmf "pFq(4, 3, 1)" \
        --start "(1, 1, 1, 1, 1, 1, 1)" --trajectories "(1, 1, 1, 1, 1, 1, 1)" \
        --max-depth 400 --num-points 20 --error-window 100 --use_inv_t \
        --save delta_vs_depth.png

Run with the WSL conda env ``rama``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ramanujantools import Position

from dreamer.configs import config
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
from dreamer.utils.types import CMFData


_HERE = Path(__file__).resolve().parent
_DEFAULT_EXPORT_CMFS = str(_HERE / ".." / "examples" / "CMFs")


# ===========================================================================
# Input parsing
# ===========================================================================

def parse_point(text: str) -> Tuple:
    """Parse ``"(a, b, c)"`` (or ``"a,b,c"``) into a tuple of rational numbers.

    Each coordinate may be an integer or a fraction, so both start points and
    trajectory directions can have rational coordinates, e.g.
    ``"(0, 1, 3, -1/2, 5, 7, 7/2)"`` →
    ``(0, 1, 3, -sympy.Rational(1, 2), 5, 7, sympy.Rational(7, 2))``.  Whole
    numbers come back as ``sympy.Integer``.

    :param text: the coordinate string, with or without surrounding brackets.
    :return: the coordinates as a tuple of sympy rationals/integers.
    :raises ValueError: if a coordinate is not a valid rational number.
    """
    import sympy as sp

    cleaned = text.strip().strip("()[]")
    parts = [p for p in cleaned.replace(" ", "").split(",") if p != ""]
    out: List = []
    for p in parts:
        try:
            out.append(sp.Rational(p))
        except (TypeError, ValueError, sp.SympifyError):
            raise ValueError(
                f"Coordinate {p!r} in {text!r} is not a rational number "
                f"(use an integer or a fraction like -1/2)."
            )
    if not out:
        raise ValueError(f"No coordinates parsed from {text!r}.")
    return tuple(out)


def parse_trajectories(items: Sequence[str]) -> List[Tuple]:
    """Parse a list of ``"(x, y, z, ...)"`` strings into rational tuples.

    :param items: trajectory strings as given on the command line.
    :return: list of trajectory-direction tuples (sympy rationals/integers).
    """
    return [parse_point(it) for it in items]


def depth_grid(max_depth: int, num_points: int, min_depth: int = 2) -> List[int]:
    """Build an increasing list of integer depths to sample δ / kamidelta at.

    :param max_depth: the largest walk depth.
    :param num_points: how many depths to sample (roughly evenly spaced).
    :param min_depth: the smallest depth (a couple of steps are needed before δ
        is meaningful).
    :return: a sorted list of unique integer depths in ``[min_depth, max_depth]``.
    """
    if max_depth < min_depth:
        raise ValueError(f"max_depth ({max_depth}) must be >= {min_depth}.")
    raw = np.linspace(min_depth, max_depth, max(2, num_points))
    return sorted({int(round(d)) for d in raw})


# ===========================================================================
# CMF / constant loading
# ===========================================================================

def load_cmf_data(export_cmfs: str, constant: Constant, cmf_id: str) -> CMFData:
    """Load the ``CMFData`` for *cmf_id* from the formatter JSON export.

    Mirrors the formatter-loading half of
    :func:`dreamer.utils.storage.atlas_writer.load_shards_from_export` but
    returns just the CMF (no shards), since this script supplies its own start
    point and trajectories.

    :param export_cmfs: ``EXPORT_CMFS`` directory.
    :param constant: the constant (selects the per-constant subdirectory).
    :param cmf_id: the CMF id (formatter JSON stem, e.g. ``pFq_2_1_-1__0_0_0``).
    :raises FileNotFoundError: if the formatter JSON does not exist.
    :return: the reconstructed ``CMFData``.
    """
    from dreamer.loading.funcs.formatter import Formatter

    safe = "".join(c for c in constant.name if c.isalnum() or c in ("-", "_"))
    path = Path(export_cmfs) / safe / f"{cmf_id}.json"
    if not path.is_file():
        avail = sorted(
            f.stem for f in (Path(export_cmfs) / safe).glob("*.json")
        ) if (Path(export_cmfs) / safe).is_dir() else []
        raise FileNotFoundError(
            f"No formatter JSON for cmf_id={cmf_id!r} under constant "
            f"{constant.name!r} at {path}. Available: {avail}"
        )
    with open(path, "r") as fh:
        raw = json.load(fh)
    return Formatter.from_json_obj(raw).to_cmf()


def build_cmf_from_expr(spec: str, use_inv_t: bool) -> CMFData:
    """Construct a fresh ``CMFData`` from a constructor expression like ``pFq(4, 3, 1)``.

    Currently the ``pFq`` family is supported: ``pFq(p, q, z)`` builds
    ``ramanujantools.cmf.pFq(p, q, z)`` with a zero coordinate shift.  ``z`` is
    parsed with sympy, so ``pFq(2, 1, -1)`` and ``pFq(3, 2, 1/2)`` both work.

    :param spec: the constructor string, e.g. ``"pFq(4, 3, 1)"``.
    :param use_inv_t: whether the walk should use the inverse-transpose recurrence.
    :raises ValueError: on a malformed expression or unsupported family.
    :return: a freshly built ``CMFData`` (no stored shards/data).
    """
    import re
    import sympy as sp
    from ramanujantools.cmf import pFq as rt_pFq

    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", spec)
    if not m:
        raise ValueError(f"Could not parse CMF expression {spec!r} (expected e.g. 'pFq(4, 3, 1)').")
    name, argstr = m.group(1), m.group(2)
    args = [a.strip() for a in argstr.split(",") if a.strip() != ""]

    if name.lower() == "pfq":
        if len(args) != 3:
            raise ValueError(f"pFq expects 3 arguments (p, q, z), got {args}.")
        p, q = int(args[0]), int(args[1])
        z = sp.sympify(args[2])
        cmf = rt_pFq(p, q, z)
        cmf_name = f"pFq({p}, {q}, {z})"
    else:
        raise ValueError(
            f"Unsupported CMF family {name!r}.  Supported: pFq(p, q, z). "
            f"(Pass an export cmf id instead, e.g. --cmf pFq_2_1_-1__0_0_0.)"
        )

    shift = Position({k: 0 for k in cmf.matrices.keys()})
    return CMFData(cmf, shift, use_inv_t=use_inv_t, cmf_name=cmf_name)


def build_cmf(spec: str, constant: Constant, export_cmfs: str, use_inv_t: bool) -> CMFData:
    """Build a ``CMFData`` either from a code expression or the JSONL export.

    A *spec* containing ``"("`` (e.g. ``"pFq(4, 3, 1)"``) is constructed from
    scratch via :func:`build_cmf_from_expr`; otherwise it is treated as an export
    CMF id and loaded via :func:`load_cmf_data`.

    :param spec: a constructor expression or an export cmf id.
    :param constant: the constant (only used for the export-loading path).
    :param export_cmfs: ``EXPORT_CMFS`` directory (export path only).
    :param use_inv_t: walk inverse-transpose flag (applied to the result).
    :return: the ``CMFData``.
    """
    if "(" in spec:
        return build_cmf_from_expr(spec, use_inv_t)
    cmf_data = load_cmf_data(export_cmfs, constant, spec)
    cmf_data.use_inv_t = use_inv_t  # honour the explicit --use_inv_t for testing
    return cmf_data


def _resolve_constant(name: str, value_str: Optional[str]) -> Constant:
    """Resolve a constant from a registered name or a sympy expression.

    Resolution order: registered name (``e``, ``pi``, …) → explicit
    ``--constant-value`` → the ``log-2`` shortcut → ``sympify(name)`` for an
    expression such as ``"zeta(2)"`` or ``"sqrt(2)"`` (accepted only when it has
    no free symbols, i.e. is a concrete number).

    :param name: the constant name or a sympy expression string.
    :param value_str: optional explicit sympy expression for the value.
    :raises SystemExit: if *name* cannot be resolved to a concrete constant.
    :return: the resolved :class:`Constant` (its ``name`` is *name* for labels).
    """
    import sympy as sp
    try:
        import dreamer.utils.constants.ready_made_consts  # noqa: F401  (registers e/pi/…)
    except Exception:
        pass
    if name in Constant.registry:
        return Constant.registry[name]
    if value_str:
        return Constant(name, sp.sympify(value_str))
    if name == "log-2":
        return Constant(name, sp.log(2))
    try:
        expr = sp.sympify(name)
    except (sp.SympifyError, TypeError, ValueError):
        expr = None
    if expr is not None and not expr.free_symbols and not expr.is_Symbol:
        return Constant(name, expr)
    raise SystemExit(
        f"Unknown constant {name!r}.  Use a registered name (e, pi, …), a sympy "
        f"expression with no free symbols (e.g. 'zeta(2)', 'sqrt(2)'), or pass "
        f"--constant-value '<sympy expr>'."
    )


# ===========================================================================
# δ / kamidelta computation
# ===========================================================================

def compute_curves(
    cmf_data: CMFData,
    constant: Constant,
    start: Tuple[int, ...],
    trajectory: Tuple[int, ...],
    depths: List[int],
    walk_type: Optional[int] = None,
    error_window: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict], Optional[dict]]:
    """Compute the δ and kamidelta curves for one trajectory over *depths*.

    p/q are identified once (at the deepest depth); δ is then evaluated at every
    depth.  Kamidelta is the eigenvalue prediction (:meth:`delta_prediction`)
    evaluated at each depth.

    :param cmf_data: the CMF (object + shift) the trajectory lives in.
    :param constant: the target constant.
    :param start: start-point coordinates (CMF-symbol order); ints or sympy
        rationals.
    :param trajectory: trajectory-direction coordinates (ints or rationals).
    :param depths: depths to evaluate at (increasing).
    :param walk_type: ``1`` (inv-transpose) / ``2`` (direct); ``None`` → derive
        from the CMF's ``use_inv_t``.
    :param error_window: how many consecutive integer depths at the deep end to
        use for the tail error bounds (see :func:`_tail_stats`); ``0`` / ``1``
        disables them.
    :return: ``(delta, kami, delta_error_bound, kami_error_bound)`` — the
        ``delta``/``kami`` float arrays aligned with *depths* (``NaN`` where the
        value is unavailable / non-finite), and the tail error-bound dicts for
        each curve (``None`` if disabled / unavailable).
    """
    import sympy as sp

    symbols = list(cmf_data.cmf.matrices.keys())
    if len(start) != len(symbols) or len(trajectory) != len(symbols):
        raise ValueError(
            f"start ({len(start)}) and trajectory ({len(trajectory)}) must each "
            f"have {len(symbols)} coordinates for CMF {cmf_data.cmf_name!r}."
        )
    if walk_type is None:
        walk_type = 1 if cmf_data.use_inv_t else 2

    # Keep coordinates as sympy numbers so rational start points / trajectories
    # (e.g. -1/2) flow through to cmf.trajectory_matrix unchanged.
    start_pos = Position({s: sp.Rational(v) for s, v in zip(symbols, start)})
    traj_pos = Position({s: sp.Rational(v) for s, v in zip(symbols, trajectory)})

    handler = TrajectoryAttributesHandler.from_cmf(
        cmf_data.cmf, traj_pos, start_pos, constant.value_sympy,
        walk_depth=max(depths), walk_type=walk_type,
    )

    # Tail block of consecutive integer depths for the error bounds, merged with
    # the (sparse) display depths so δ and kamidelta are each walked only once;
    # the tail values are sliced back out for the statistics afterwards.
    tail_depths = _tail_depths(max(depths), error_window)
    all_depths = sorted(set(depths) | set(tail_depths or ()))

    # δ over the merged depths — fast path reuses one p/q identification; fall
    # back to the per-depth path if delta_sequence drops a depth (rare walk
    # failure).  Then split into the display array and the tail statistics.
    seq = handler.delta_sequence(all_depths)
    raw = seq if len(seq) == len(all_depths) else [handler.delta(d) for d in all_depths]
    delta_map = {d: _finite_or_nan(_as_float(v)) for d, v in zip(all_depths, raw)}
    delta = np.array([delta_map[d] for d in depths])

    # Kamidelta (eigenvalue prediction) over the merged depths.  Fast path: one
    # walk for all depths; fall back to the per-depth handler call otherwise.
    finite_delta = delta[np.isfinite(delta)]
    actual_delta = float(finite_delta[-1]) if finite_delta.size else float("nan")
    kami_all = _fast_kamidelta(handler, all_depths, actual_delta)
    if kami_all is None:
        kami_all = np.array([
            (float(p["predicted_delta"]) if (p := handler.delta_prediction(d)) else np.nan)
            for d in all_depths
        ])
    kami_map = {d: v for d, v in zip(all_depths, kami_all)}
    kami = np.array([kami_map[d] for d in depths])

    # Tail error bounds — the *same* statistic applied to both curves, each over
    # the last error_window integer depths (reuses the merged walk above).
    delta_eb = kami_eb = None
    if tail_depths:
        delta_eb = _tail_stats(tail_depths, [delta_map[d] for d in tail_depths])
        kami_eb = _tail_stats(tail_depths, [kami_map[d] for d in tail_depths])
    return delta, kami, delta_eb, kami_eb


def _tail_depths(max_depth: int, window: Optional[int]) -> Optional[List[int]]:
    """The consecutive integer depths ``[max_depth-window+1 … max_depth]`` used
    for a tail error bound, or ``None`` when disabled (``window < 2``) or too
    short.  The low end is clamped to 2 (δ is not meaningful below that)."""
    if window is None or window < 2:
        return None
    lo = max(2, max_depth - window + 1)
    tail = list(range(lo, max_depth + 1))
    return tail if len(tail) >= 2 else None


def _tail_stats(tail_depths: List[int], vals) -> Optional[dict]:
    """Error-bound statistics over a tail block of δ (or kamidelta) values.

    A converging-but-noisy curve is summarised by its tail: the value is the mean
    of the last stretch, and the honest uncertainty is the spread there — not a
    single depth's value.  Over the *finite* entries of *vals* (aligned with
    *tail_depths*; non-finite entries are dropped):

    * ``mean`` — the tail mean (the reported final value);
    * ``std`` — sample standard deviation (``ddof=1``): the tail **scatter**, the
      headline ± bound;
    * ``sem`` — ``std / √n``, the standard error of the **mean** (tighter, but
      i.i.d.-assuming, so optimistic on an autocorrelated tail);
    * ``slope`` / ``drift`` — a linear fit of value vs depth and the total change
      it predicts across the window (``|slope| · span``); ``converged = drift ≤
      std`` flags a tail that is still *trending* rather than merely scattering.

    :return: the stats dict, or ``None`` when fewer than 2 finite values remain.
    """
    v = np.asarray(vals, dtype=float)
    fin = np.isfinite(v)
    finite = v[fin]
    fdepths = np.asarray(tail_depths, dtype=float)[fin]
    if finite.size < 2:
        return None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    sem = std / np.sqrt(finite.size)
    slope = float(np.polyfit(fdepths, finite, 1)[0])
    span = float(fdepths[-1] - fdepths[0])
    drift = abs(slope) * span
    return {
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "slope": slope,
        "drift": drift,
        "last": float(finite[-1]),
        "n": int(finite.size),
        "depth_lo": int(fdepths[0]),
        "depth_hi": int(fdepths[-1]),
        "converged": bool(drift <= std),
    }


def compute_error_bound(handler, max_depth: int,
                        window: int) -> Optional[dict]:
    """Tail error bound on the final δ from a handler (see :func:`_tail_stats`).

    Thin convenience wrapper for direct/programmatic use and tests; the plotting
    path computes the δ and kamidelta bounds together inside
    :func:`compute_curves` (sharing a single walk).
    """
    tail = _tail_depths(max_depth, window)
    if tail is None:
        return None
    seq = handler.delta_sequence(tail)
    raw = seq if len(seq) == len(tail) else [handler.delta(d) for d in tail]
    return _tail_stats(tail, [_finite_or_nan(_as_float(v)) for v in raw])


def _fast_kamidelta(handler, depths: List[int],
                    actual_delta: float) -> Optional[np.ndarray]:
    """Kamidelta at every depth from a **single** walk (else ``None`` to fall back).

    The production :meth:`delta_prediction` recomputes ``gcd_slope(d)`` per depth,
    and each call re-walks from ``n=1`` to ``d`` — so plotting K depths costs
    O(Σ depths) of redundant walking.  Here the reduced-denominator sequence
    ``log(q̃_n)`` is walked **once** to the deepest depth via ``handler._limits``
    — the *same* ``Limit`` objects δ is computed on (start ``{n: 0}``, walk_type
    applied) — so the denominator slope refers to the same convergents as δ.  Each
    depth's ``gcd_slope`` is then a cheap linear fit of a prefix, and the
    eigenvalue-pair selection matches :meth:`delta_prediction` exactly.

    (This deliberately does **not** replicate ``ramanujantools.Matrix.gcd_slope``,
    which walks the denominator from ``{n: 1}`` — a *different* integer sequence
    from the ``{n: 0}`` walk δ / the identified p/q live on.  Using it made the
    denominator grow ~30 % too fast and pushed kamidelta far below δ; see the note
    in :meth:`TrajectoryAttributesHandler.gcd_slope`.)

    :param handler: the trajectory attribute handler (already identified).
    :param depths: increasing depths to predict at.
    :param actual_delta: the actual δ at the deepest depth (for pair selection);
        ``NaN`` when not identified.
    :return: kamidelta array aligned with *depths*, or ``None`` if the fast path
        cannot run (caller should use the slow per-depth path).
    """
    import math
    import warnings
    import sympy as sp
    from sympy.abc import n

    if not math.isfinite(actual_delta):
        return np.full(len(depths), np.nan)
    try:
        if handler._initial_values() is None:
            return np.full(len(depths), np.nan)  # not identified
        pairs = handler._unique_eigenvalue_pairs()
        if not pairs:
            return np.full(len(depths), np.nan)

        max_n = max(depths)
        walk_ns = list(range(1, max_n))
        limits = handler._limits(walk_ns)  # same walk as δ (start {n: 0}, walk_type)
        if not limits or len(limits) != len(walk_ns):
            return None  # walk failed / dropped a depth — fall back to slow path
        y = np.array([float(sp.log(lim.as_rational().q).evalf(30)) for lim in limits])
        x = np.arange(1, max_n)  # n values aligned with y

        # Precompute log(|λ_i|/|λ_j|) per candidate pair once, from the
        # high-precision norms (pairs are (λ_i, λ_j, |λ_i|, |λ_j|)), matching
        # delta_prediction's ``float(sp.log(norm1 / norm2))``.
        ratios = []
        for _lam1, _lam2, norm1, norm2 in pairs:
            try:
                ratios.append(float(sp.log(norm1 / norm2)))
            except Exception:
                ratios.append(None)

        out = np.full(len(depths), np.nan)
        for i, d in enumerate(depths):
            k = d - 1  # gcd_slope(d) fits n = 1 .. d-1 (matches ramanujantools)
            if k < 1:
                continue
            with warnings.catch_warnings():  # a 1-point fit is rank-deficient
                warnings.simplefilter("ignore")
                slope = float(np.polyfit(x[:k], y[:k], 1)[0])
            if abs(slope) < 1e-30:
                continue
            best, best_diff = np.nan, float("inf")
            for r in ratios:
                if r is None:
                    continue
                predicted = -1.0 + r / slope
                diff = abs(predicted - actual_delta)
                if diff < best_diff:
                    best_diff, best = diff, predicted
            out[i] = best
        return out
    except Exception:
        return None  # fall back to the slow, well-tested per-depth path


def _as_float(x) -> float:
    """Coerce a δ value (float / sympy / ``-inf``) to a plain float, NaN on failure."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_nan(f: float) -> float:
    """Return *f* if finite, else NaN (so ``-inf`` non-convergence breaks the line)."""
    return f if np.isfinite(f) else float("nan")


# ===========================================================================
# Plotting
# ===========================================================================

def _value_norm(results, show_kamidelta: bool):
    """A ``Normalize`` over all finite δ (and kamidelta) values across trajectories.

    Drives the value→colour spectrum: ``vmin`` (lowest δ) maps to deep blue,
    ``vmax`` (highest δ) to bright red.  Returns ``None`` when there is no finite
    data (nothing to colour).
    """
    vals: List[float] = []
    for res in results:
        for idx in (1, 2) if show_kamidelta else (1,):
            arr = np.asarray(res[idx], dtype=float)
            vals.extend(arr[np.isfinite(arr)].tolist())
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5  # constant field — widen so colours vary
    return plt.Normalize(vmin=lo, vmax=hi)


def _add_value_line(ax, x: np.ndarray, y: np.ndarray, norm, cmap, *,
                    linestyle: str = "-", linewidth: float = 1.8, alpha: float = 1.0,
                    marker: Optional[str] = None, markersize: float = 4.0) -> None:
    """Draw ``y`` vs ``x`` coloured by **value** along a blue→red spectrum.

    Each segment is coloured by ``cmap(norm(midpoint))`` so the lowest δ is deep
    blue and the highest bright red — the magnitude trend is visible at a glance.
    Segments touching a ``NaN`` (non-identified depth) are skipped.

    :param ax: target axis.
    :param x: x values (depths).
    :param y: y values (δ or kamidelta), possibly containing ``NaN``.
    :param norm: shared value ``Normalize`` (see :func:`_value_norm`).
    :param cmap: the diverging colormap (blue→red).
    :param linestyle: ``"-"`` for δ, ``"--"`` for kamidelta.
    :param linewidth: stroke width.
    :param alpha: opacity.
    :param marker: optional point marker drawn at finite samples (value-coloured).
    :param markersize: marker size.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fin = np.isfinite(y)

    segs, vals = [], []
    for i in range(len(x) - 1):
        if fin[i] and fin[i + 1]:
            segs.append([(x[i], y[i]), (x[i + 1], y[i + 1])])
            vals.append(0.5 * (y[i] + y[i + 1]))
    if segs:
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=linewidth,
                            linestyles=linestyle, alpha=alpha)
        lc.set_array(np.asarray(vals))
        ax.add_collection(lc)
    if marker is not None and fin.any():
        ax.scatter(x[fin], y[fin], c=y[fin], cmap=cmap, norm=norm, s=markersize ** 2,
                   marker=marker, alpha=alpha, edgecolor="none", zorder=5)


# Per-kind styling:
# (value index, linestyle, marker, linewidth, alpha, label, error-bound index).
_KIND_DELTA = (1, "-", None, 1.8, 1.0, "δ", 3)
_KIND_KAMI = (2, "--", "o", 1.4, 0.85, "kamiδ", 4)


def _error_bound_of(res, eb_idx: int) -> Optional[dict]:
    """The tail error-bound dict at *eb_idx* in a result tuple, or ``None``."""
    return res[eb_idx] if len(res) > eb_idx else None


def _draw_error_bounds(ax, results, kind, *, value_color: bool, norm, spec_cmap, tab) -> None:
    """Shade the ±σ tail band and draw the mean line for one curve *kind*.

    The band spans the error-window depth range at ``[mean − σ, mean + σ]`` (the
    tail *scatter*, the headline bound); a dotted line marks the mean.  Colour
    matches the curve (value spectrum in ``value_color`` mode, per-trajectory
    otherwise).  Applies to whichever axis draws *kind* (δ or kamidelta).
    """
    eb_idx = kind[6]
    for i, res in enumerate(results):
        eb = _error_bound_of(res, eb_idx)
        if not eb:
            continue
        lo, hi = eb["depth_lo"], eb["depth_hi"]
        mean, std = eb["mean"], eb["std"]
        color = spec_cmap(norm(mean)) if (value_color and norm is not None) else tab(i % 10)
        ax.fill_between([lo, hi], [mean - std, mean - std], [mean + std, mean + std],
                        color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.hlines(mean, lo, hi, color=color, linestyle=":", linewidth=1.3,
                  alpha=0.9, zorder=6)


def _error_summary_lines(results, kinds) -> List[str]:
    """``label = mean ± σ`` summary lines, one per (trajectory, curve *kind*).

    Covers every curve drawn (δ and, when shown, kamidelta), so both error bounds
    are reported.  Labels are padded so the ``=`` columns align in the monospace
    footer box.
    """
    width = max((len(k[5]) for k in kinds), default=1)
    lines: List[str] = []
    for res in results:
        for kind in kinds:
            eb = _error_bound_of(res, kind[6])
            if not eb:
                continue
            flag = "" if eb["converged"] else "  ⚠ drift>σ (not converged)"
            lines.append(
                f"traj {res[0]}  {kind[5]:<{width}} = {eb['mean']:.4g} ± {eb['std']:.2g}"
                f"  (SEM {eb['sem']:.2g}, n={eb['n']}, depths {eb['depth_lo']}–{eb['depth_hi']}){flag}"
            )
    return lines


def _draw_series(ax, x, results, kind, *, value_color: bool, norm, spec_cmap, tab) -> List[float]:
    """Draw one kind of curve (δ or kamidelta) for every trajectory on *ax*.

    :param ax: target axis.
    :param x: depths.
    :param results: ``[(trajectory, delta_curve, kami_curve, ...), ...]``.
    :param kind: one of ``_KIND_DELTA`` / ``_KIND_KAMI`` (index + style).
    :param value_color: colour by δ value (spectrum) vs one colour per trajectory.
    :param norm: shared value ``Normalize`` (spectrum mode).
    :param spec_cmap: the blue→red spectrum colormap.
    :param tab: the per-trajectory colormap (non-spectrum mode).
    :return: the finite y values drawn (for axis limit fitting).
    """
    idx, linestyle, marker, lw, alpha, kind_label, _eb = kind
    finite: List[float] = []
    for i, res in enumerate(results):
        traj, y = res[0], res[idx]
        finite.extend(np.asarray(y)[np.isfinite(y)].tolist())
        if value_color and norm is not None:
            _add_value_line(ax, x, y, norm, spec_cmap, linestyle=linestyle,
                            linewidth=lw, alpha=alpha, marker=marker, markersize=3)
        else:
            ax.plot(x, y, linestyle=linestyle, color=tab(i % 10), linewidth=lw,
                    alpha=alpha, marker=marker, markersize=3,
                    label=f"traj {traj}: {kind_label}")
    return finite


def _finish_axis(ax, x, results, ylabel, finite_vals, kinds, *,
                 value_color: bool, log_scale: bool, linthresh: float) -> None:
    """Apply scale, limits, grid and the empty-data note to one axis.

    The legend is *not* drawn here — it lives in a shared figure footer (see
    :func:`_footer_legend_handles`) so it never overlaps the curves.
    """
    if log_scale:
        ylabel += "  (symlog)"
        ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylabel(ylabel, fontsize=13)
    if len(x) > 1:
        ax.set_xlim(float(x.min()), float(x.max()))
    ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)

    # LineCollection segments don't drive autoscale reliably — set y from data.
    if finite_vals:
        lo, hi = min(finite_vals), max(finite_vals)
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(-1.0, 1.0)
        ax.text(0.5, 0.5,
                "No δ identified at these depths.\n"
                "Try a deeper --max-depth, or the right --const for this CMF.",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="firebrick",
                bbox=dict(boxstyle="round", fc="white", ec="firebrick", alpha=0.9))


def _footer_legend_handles(results, value_color: bool, kinds):
    """Handles + labels for the shared figure-footer legend.

    In spectrum mode colour encodes the δ value (shown by the colorbar), so the
    legend only distinguishes line styles (δ vs kamiδ) and lists the
    trajectories.  In per-trajectory mode each (trajectory, kind) gets its own
    coloured line entry.
    :return: ``(handles, labels, ncol)``.
    """
    handles: List[Line2D] = []
    if value_color:
        for _idx, linestyle, marker, _lw, _alpha, kind_label, _eb in kinds:
            handles.append(Line2D([], [], color="grey", linestyle=linestyle,
                                  marker=marker or "", markersize=4, label=kind_label))
        for res in results:
            handles.append(Line2D([], [], color="none", label=f"traj {res[0]}"))
    else:
        tab = plt.get_cmap("tab10")
        for i, res in enumerate(results):
            for idx, linestyle, marker, _lw, _alpha, kind_label, _eb in kinds:
                handles.append(Line2D([], [], color=tab(i % 10), linestyle=linestyle,
                                      marker=marker or "", markersize=4,
                                      label=f"traj {res[0]}: {kind_label}"))
    labels = [h.get_label() for h in handles]
    ncol = max(1, min(len(handles), 4))
    return handles, labels, ncol


def plot_curves(
    depths: List[int],
    results: List[Tuple[Tuple[int, ...], np.ndarray, np.ndarray]],
    *,
    constant_name: str = "",
    title: Optional[str] = None,
    show_kamidelta: bool = True,
    value_color: bool = True,
    log_scale: bool = False,
    linthresh: float = 0.05,
    separate: bool = False,
    cmap_name: str = "coolwarm",
    show_error_bound: bool = True,
):
    """Plot δ-vs-depth, with kamidelta either overlaid or on a second axis.

    :param depths: the x-axis depths.
    :param results: ``[(trajectory, delta_curve, kami_curve, delta_eb, kami_eb), ...]``.
    :param show_error_bound: shade the ±σ tail band + mean line on **both** the δ
        and the kamidelta curves, and report ``mean ± σ`` for each in the footer.
    :param constant_name: constant name, for axis text.
    :param title: optional figure title.
    :param show_kamidelta: include the kamidelta prediction.
    :param value_color: colour the curves by δ **value** along a blue→red spectrum
        (default, with a colorbar); ``False`` uses one colour per trajectory.
    :param log_scale: symmetric-log (``symlog``) y axis.
    :param linthresh: half-width of the linear band around 0 for ``symlog``.
    :param separate: draw δ and kamidelta on **two stacked axes** instead of
        overlaying them (ignored when ``show_kamidelta`` is False).
    :param cmap_name: the spectrum colormap (lowest δ → deep blue, highest →
        bright red).
    :return: the matplotlib ``Figure``.
    """
    tab = plt.get_cmap("tab10")
    spec_cmap = plt.get_cmap(cmap_name)
    x = np.asarray(depths)
    base = r"irrationality measure $\delta$"
    if constant_name:
        base += f"  ({constant_name})"

    norm = _value_norm(results, show_kamidelta) if value_color else None
    draw_kw = dict(value_color=value_color, norm=norm, spec_cmap=spec_cmap, tab=tab)
    fin_kw = dict(value_color=value_color, log_scale=log_scale, linthresh=linthresh)

    if separate and show_kamidelta:
        fig, (ax_d, ax_k) = plt.subplots(2, 1, sharex=True, figsize=(9, 8), dpi=200)
        fv_d = _draw_series(ax_d, x, results, _KIND_DELTA, **draw_kw)
        fv_k = _draw_series(ax_k, x, results, _KIND_KAMI, **draw_kw)
        _finish_axis(ax_d, x, results, base, fv_d, [_KIND_DELTA], **fin_kw)
        _finish_axis(ax_k, x, results, "kamidelta prediction", fv_k, [_KIND_KAMI], **fin_kw)
        ax_k.set_xlabel("walk depth $n$", fontsize=13)
        cbar_axes = [ax_d, ax_k]
        kinds = [_KIND_DELTA, _KIND_KAMI]
        band_axes = [(ax_d, _KIND_DELTA), (ax_k, _KIND_KAMI)]  # each band on its own axis
        if title:
            fig.suptitle(title, fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
        kinds = [_KIND_DELTA] + ([_KIND_KAMI] if show_kamidelta else [])
        fv = []
        for kind in kinds:
            fv += _draw_series(ax, x, results, kind, **draw_kw)
        _finish_axis(ax, x, results, base, fv, kinds, **fin_kw)
        ax.set_xlabel("walk depth $n$", fontsize=13)
        cbar_axes = [ax]
        band_axes = [(ax, k) for k in kinds]  # both bands on the shared axis
        if title:
            ax.set_title(title, fontsize=14)

    if show_error_bound:
        for band_ax, kind in band_axes:
            _draw_error_bounds(band_ax, results, kind, value_color=value_color,
                               norm=norm, spec_cmap=spec_cmap, tab=tab)

    # Reserve the bottom footer strip BEFORE the colorbar so the colorbar is
    # placed relative to the shrunk axes; then fill the strip with the legend and
    # the error-bound description (both kept off the plot area).
    _layout_with_footer(fig, results, kinds, value_color=value_color,
                        show_error_bound=show_error_bound, has_colorbar=norm is not None,
                        separate=separate and show_kamidelta, has_title=bool(title))

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=spec_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=cbar_axes, fraction=0.046, pad=0.02)
        cbar.set_label(base, fontsize=12)

    _fill_footer(fig, results, kinds, value_color=value_color,
                 show_error_bound=show_error_bound)
    return fig


# Vertical geometry of the bottom footer, in figure-fraction terms.
_FOOT_PAD = 0.03      # margin below the error-text box
_FOOT_GAP = 0.02      # gap between the error box and the legend
_FOOT_BOX = 0.02      # padding inside each text/legend block


def _footer_line_frac(fig, pt: float = 10.0) -> float:
    """Figure-fraction height of one text line at *pt* points."""
    return pt / 72.0 / fig.get_size_inches()[1]


def _footer_extent(fig, results, kinds, *, value_color, show_error_bound):
    """Footer sizing: ``(bottom_margin, err_h, legend_top_y)`` in fig fractions.

    ``bottom_margin`` is how much of the figure to reserve below the axes;
    ``err_h`` the height of the error-text block; ``legend_top_y`` the y at which
    the legend's lower edge is anchored (just above the error text)."""
    handles, _labels, ncol = _footer_legend_handles(results, value_color, kinds)
    err_lines = _error_summary_lines(results, kinds) if show_error_bound else []
    line = _footer_line_frac(fig)
    legend_rows = int(np.ceil(len(handles) / ncol)) if handles else 0
    err_h = (len(err_lines) * line + _FOOT_BOX) if err_lines else 0.0
    legend_h = (legend_rows * (line + 0.004) + _FOOT_BOX) if handles else 0.0
    legend_y = _FOOT_PAD + err_h + _FOOT_GAP
    # Room between the legend top and the axes bottom for the x tick labels and
    # the "walk depth n" x-label (which live just below the axes spine).
    xlabel_allow = 0.6 / fig.get_size_inches()[1]
    bottom = legend_y + legend_h + xlabel_allow
    return bottom, err_h, legend_y


def _layout_with_footer(fig, results, kinds, *, value_color, show_error_bound,
                        has_colorbar, separate, has_title) -> None:
    """Reserve figure margins so the footer (legend + error text) never overlaps
    the axes.  Uses ``subplots_adjust`` (deterministic, and — unlike
    ``tight_layout`` — compatible with the colorbar added afterwards)."""
    bottom, _err_h, _legend_y = _footer_extent(
        fig, results, kinds, value_color=value_color, show_error_bound=show_error_bound)
    top = 0.93 if has_title else 0.965
    right = 0.9 if has_colorbar else 0.95
    fig.subplots_adjust(bottom=bottom, top=top, left=0.13, right=right,
                        hspace=0.12 if separate else 0.2)


def _fill_footer(fig, results, kinds, *, value_color, show_error_bound) -> None:
    """Draw the shared legend and the error-bound description in the reserved
    bottom strip (legend just under the axes, error text below it)."""
    _bottom, err_h, legend_y = _footer_extent(
        fig, results, kinds, value_color=value_color, show_error_bound=show_error_bound)
    handles, labels, ncol = _footer_legend_handles(results, value_color, kinds)
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
                   ncol=ncol, fontsize=9, framealpha=0.9,
                   handlelength=2.2, columnspacing=1.4)
    err_lines = _error_summary_lines(results, kinds) if show_error_bound else []
    if err_lines:
        fig.text(0.5, _FOOT_PAD, "\n".join(err_lines), ha="center", va="bottom",
                 family="monospace", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))


# ===========================================================================
# CLI
# ===========================================================================

def _build_cli():
    """Construct the argument parser for the δ-vs-depth CLI."""
    import argparse

    p = argparse.ArgumentParser(
        prog="delta_vs_depth.py",
        description="Plot δ vs walk depth (with the eigenvalue kamidelta "
                    "prediction) for given trajectories of a CMF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--const", "--constant", dest="constant", default="log-2",
                   help="Constant: a registered name (e, pi, …), a sympy expression "
                        "with no free symbols (e.g. 'zeta(2)', 'sqrt(2)'), or 'log-2'.")
    p.add_argument("--constant-value", default=None,
                   help="Explicit sympy expression for the constant's value.")
    p.add_argument("--cmf", required=True,
                   help="A constructor expression, e.g. 'pFq(4, 3, 1)', built from "
                        "scratch; or an export cmf id, e.g. pFq_2_1_-1__0_0_0.")
    p.add_argument("--export-cmfs", default=_DEFAULT_EXPORT_CMFS,
                   help="EXPORT_CMFS directory (used only for the export-id form of --cmf).")
    p.add_argument("--start", required=True,
                   help='Start point, e.g. "(-3, 1, -1)".')
    p.add_argument("--trajectories", required=True, nargs="+",
                   help='Trajectory directions, e.g. "(-1, 3, 0)" "(-1, 2, -1)".')
    p.add_argument("--max-depth", type=int, default=200,
                   help="Largest walk depth (must be deep enough for LIReC to "
                        "identify p/q, else δ stays -inf).")
    p.add_argument("--min-depth", type=int, default=2,
                   help="Smallest walk depth shown (skip the early steps, e.g. "
                        "--min-depth 100 to plot only depths 100..max).")
    p.add_argument("--num-points", type=int, default=20,
                   help="Number of depths sampled between --min-depth and --max-depth.")
    p.add_argument("--error-window", type=int, default=100,
                   help="Number of consecutive integer depths at the deep end used "
                        "for the final-δ error bound (mean ± sample std, with SEM). "
                        "0 disables the error bound.")
    p.add_argument("--use_inv_t", action="store_true",
                   help="Walk with the inverse-transpose recurrence (walk type 1) "
                        "instead of the direct matrix (walk type 2, the default).")
    p.add_argument("--no-kamidelta", action="store_true",
                   help="Plot only δ, without the kamidelta prediction overlay.")
    p.add_argument("--per-trajectory-color", "--no-sign-color", dest="per_trajectory",
                   action="store_true",
                   help="Use one colour per trajectory instead of the default "
                        "value spectrum (lowest δ deep blue → highest bright red).")
    p.add_argument("--log", action="store_true",
                   help="Symmetric-log (symlog) y axis — log magnitude on both the "
                        "positive and negative δ sides, linear near 0.")
    p.add_argument("--linthresh", type=float, default=0.05,
                   help="Half-width of the linear band around 0 for --log (symlog).")
    p.add_argument("--separate", action="store_true",
                   help="Draw δ and kamidelta on two stacked graphs instead of "
                        "overlaying them on one.")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--save", default=None, help="Save to this path instead of showing.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show().")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: parse flags, compute curves, render the plot."""
    args = _build_cli().parse_args(argv)

    # δ is computed at the production constant resolution (CONSTANT_NO_DIGITS_HIGH_RES,
    # 10_000 digits), auto-escalating up to MAX_CONSTANT_RESOLUTION whenever a deep
    # approximation is good enough to underflow the current resolution.  Pin that
    # ceiling explicitly so deep walks here always get the full precision headroom.
    config.configure(search={"MAX_CONSTANT_RESOLUTION": 200_000})

    const = _resolve_constant(args.constant, args.constant_value)
    cmf_data = build_cmf(args.cmf, const, args.export_cmfs, args.use_inv_t)
    walk_type = 1 if args.use_inv_t else 2

    start = parse_point(args.start)
    trajectories = parse_trajectories(args.trajectories)
    depths = depth_grid(args.max_depth, args.num_points, args.min_depth)
    symbols = list(cmf_data.cmf.matrices.keys())
    print(f"CMF {cmf_data.cmf_name!r} (dim {len(symbols)}, symbols "
          f"{[str(s) for s in symbols]}); constant {const.name!r} = "
          f"{const.value_sympy}; walk_type {walk_type} (use_inv_t={args.use_inv_t}).")
    identify_depth = min(int(config.search.IDENTIFY_DEPTH), max(depths))
    print(f"{len(trajectories)} trajectory(ies), {len(depths)} depths in "
          f"[{min(depths)}, {max(depths)}] (LIReC identification at depth {identify_depth}, "
          f"p/q reused for the deeper δ / kamidelta).")

    results = []
    for traj in trajectories:
        print(f"  computing δ / kamidelta for trajectory {traj} ...")
        delta, kami, delta_eb, kami_eb = compute_curves(
            cmf_data, const, start, traj, depths, walk_type=walk_type,
            error_window=args.error_window)
        n_delta = int(np.isfinite(delta).sum())
        n_kami = int(np.isfinite(kami).sum())
        print(f"    δ finite at {n_delta}/{len(depths)} depths; "
              f"kamidelta at {n_kami}/{len(depths)}.")
        for label, eb in (("δ", delta_eb), ("kamiδ", kami_eb)):
            if eb is None:
                continue
            print(f"    final {label} = {eb['mean']:.6g} ± {eb['std']:.3g} "
                  f"(sample std) over depths {eb['depth_lo']}–{eb['depth_hi']} "
                  f"(n={eb['n']}); SEM = {eb['sem']:.3g}, last {label} = {eb['last']:.6g}.")
            if not eb["converged"]:
                print(f"    WARNING: {label} tail drift ({eb['drift']:.3g}) exceeds scatter "
                      f"({eb['std']:.3g}) — still trending, so the ±std band is NOT a "
                      f"converged error bound.  Increase --max-depth.")
        results.append((traj, delta, kami, delta_eb, kami_eb))

    total_finite = sum(int(np.isfinite(d).sum()) for _, d, _, _, _ in results)
    if total_finite == 0:
        print(
            "WARNING: no trajectory produced a finite δ — LIReC did not identify "
            f"{const.name!r} for any of them at depth {max(depths)}.  Likely causes: "
            "the walk is too shallow (raise --max-depth), or this CMF does not "
            "approximate the chosen constant along these trajectories (check --const)."
        )

    title = args.title or f"{cmf_data.cmf_name}  start={start}"
    fig = plot_curves(depths, results, constant_name=const.name, title=title,
                      show_kamidelta=not args.no_kamidelta,
                      value_color=not args.per_trajectory,
                      log_scale=args.log, linthresh=args.linthresh,
                      separate=args.separate,
                      show_error_bound=args.error_window >= 2)

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    if not args.no_show and not args.save:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
