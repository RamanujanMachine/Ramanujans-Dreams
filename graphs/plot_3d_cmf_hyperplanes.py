"""
Interactive 3-D visualisation of a CMF's hyperplane arrangement.

Given a CMF described by a :class:`~dreamer.loading.funcs.formatter.Formatter`
object whose CMF lives in **three** symbols (e.g. ``pFq(log(2), 2, 1, -1)`` has
symbols ``x0, x1, y0``), this script draws every hyperplane of the arrangement as
a translucent plane inside a cube, together with the three coordinate axes.

Per request, **no grid lines** are drawn: the figure shows only the hyperplanes
and the axes.  The result is an interactive plotly figure (rotate / zoom / pan),
saved to an HTML file and opened in the browser.

Run directly for the worked example::

    python examples/plot_3d_cmf_hyperplanes.py

or import :func:`plot_cmf_hyperplanes` and pass your own formatter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go

from dreamer.extraction.extractor import extract_cmf_hyperplanes
from dreamer.loading.funcs.formatter import Formatter


def _plane_surface(
    coeffs: np.ndarray,
    free: float,
    box: float,
    resolution: int,
    color: str,
    name: str,
) -> go.Surface:
    """Build a clipped :class:`plotly.graph_objects.Surface` for one hyperplane.

    The hyperplane is ``a*x + b*y + c*z + free = 0`` with ``coeffs = (a, b, c)``.
    The plane is rendered by solving for the coordinate with the largest absolute
    coefficient over a meshgrid of the other two coordinates, then clipping the
    solved coordinate to the drawing cube ``[-box, box]^3`` (points outside become
    NaN and are not drawn).

    :param coeffs: Length-3 array of the linear coefficients ``(a, b, c)``.
    :param free: The free (constant) term of the hyperplane equation.
    :param box: Half-side of the cube the plane is clipped to.
    :param resolution: Number of samples per axis in the meshgrid.
    :param color: Solid fill colour for the plane.
    :param name: Legend / hover label for the plane.
    :return: A plotly ``Surface`` trace.
    """
    a, b, c = (float(v) for v in coeffs)
    free = float(free)

    grid = np.linspace(-box, box, resolution)
    u, v = np.meshgrid(grid, grid)

    # Solve for the dominant axis to avoid dividing by a near-zero coefficient.
    dominant = int(np.argmax(np.abs([a, b, c])))
    if dominant == 2:        # solve for z: z = -(a*x + b*y + free) / c
        x, y = u, v
        z = -(a * x + b * y + free) / c
    elif dominant == 1:      # solve for y: y = -(a*x + c*z + free) / b
        x, z = u, v
        y = -(a * x + c * z + free) / b
    else:                    # solve for x: x = -(b*y + c*z + free) / a
        y, z = u, v
        x = -(b * y + c * z + free) / a

    # Clip to the cube: NaN-out the solved coordinate where it leaves the box.
    solved = (x, y, z)[dominant]
    outside = (solved < -box) | (solved > box)
    solved = np.where(outside, np.nan, solved)
    coords = [x, y, z]
    coords[dominant] = solved
    x, y, z = coords

    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np.zeros_like(u),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=0.5,
        name=name,
        showlegend=True,
        hoverinfo="name",
        contours={
            "x": {"show": False},
            "y": {"show": False},
            "z": {"show": False},
        },
    )


def _axis_lines(box: float, symbols) -> list:
    """Build the three coordinate-axis line traces through the origin.

    :param box: Half-length of each axis line.
    :param symbols: The CMF's three symbols (used as axis labels).
    :return: A list of plotly traces (one line + one label per axis).
    """
    colors = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z
    traces = []
    for i, (sym, color) in enumerate(zip(symbols, colors)):
        ends = np.zeros((2, 3))
        ends[0, i], ends[1, i] = -box, box
        traces.append(
            go.Scatter3d(
                x=ends[:, 0],
                y=ends[:, 1],
                z=ends[:, 2],
                mode="lines",
                line={"color": color, "width": 4},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        label = np.zeros(3)
        label[i] = box * 1.05
        traces.append(
            go.Scatter3d(
                x=[label[0]],
                y=[label[1]],
                z=[label[2]],
                mode="text",
                text=[str(sym)],
                textfont={"color": color, "size": 16},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def plot_cmf_hyperplanes(
    formatter: Formatter,
    box: float = 10.0,
    resolution: int = 60,
    output_html: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """Draw the hyperplanes of a 3-D CMF on an interactive 3-D graph.

    :param formatter: A formatter (e.g. ``pFq(log(2), 2, 1, -1)``) whose CMF lives
        in exactly three symbols.
    :param box: Half-side of the cube the planes/axes are drawn within.
    :param resolution: Meshgrid samples per axis for each plane (higher = smoother
        plane edges, slower).
    :param output_html: Path to write the interactive figure to.  Defaults to
        ``<cmf_name>_hyperplanes.html`` next to the current working directory.
    :param show: When True, open the figure in the default browser.
    :raises ValueError: If the CMF is not three-dimensional.
    :return: The constructed plotly :class:`~plotly.graph_objects.Figure`.
    """
    cmf_data = formatter.to_cmf()
    symbols = list(cmf_data.cmf.matrices.keys())
    if len(symbols) != 3:
        raise ValueError(
            f"plot_cmf_hyperplanes requires a 3-D CMF (3 symbols); "
            f"'{cmf_data.cmf_name}' has {len(symbols)} symbols: {symbols}"
        )

    hyperplanes = extract_cmf_hyperplanes(cmf_data)
    if not hyperplanes:
        raise ValueError(f"CMF '{cmf_data.cmf_name}' has no hyperplanes to draw.")

    palette = pcolors.qualitative.Plotly
    traces = []
    for idx, hp in enumerate(hyperplanes):
        # hp.symbols order matches the CMF symbol order, so coeffs line up with
        # the (x, y, z) drawing axes.
        coeffs, free = hp.vectors
        traces.append(
            _plane_surface(
                np.asarray(coeffs, dtype=float),
                free,
                box=box,
                resolution=resolution,
                color=palette[idx % len(palette)],
                name=str(hp.expr),
            )
        )

    import sympy as sp
    x1, x2, x3 = sp.symbols('x1, x2, x3')
    traces.extend(_axis_lines(box, [x1, x2, x3]))

    # Hide all default axis decoration so only the hyperplanes and our own
    # axis lines remain (no grid lines, no panes, no ticks).
    blank_axis = {
        "visible": False,
        "showgrid": False,
        "zeroline": False,
        "showline": False,
        "showbackground": False,
        "showticklabels": False,
        "title": "",
    }
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hyperplanes of {cmf_data.cmf_name}",
        scene={
            "xaxis": blank_axis,
            "yaxis": blank_axis,
            "zaxis": blank_axis,
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    if output_html is None:
        output_html = f"{cmf_data.cmf_name}_hyperplanes.html"
    fig.write_html(output_html, auto_open=show)
    print(f"Wrote interactive figure to {output_html}")
    return fig


if __name__ == "__main__":
    from dreamer.loading import pFq
    from dreamer import log

    plot_cmf_hyperplanes(pFq(log(2), 2, 1, -1))
