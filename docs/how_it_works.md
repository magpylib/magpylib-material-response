---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# How it works

Magpylib computes the field of magnets whose polarization is fixed. Real parts
answer back: an iron plate near a magnet becomes a magnet itself, and a
permanent magnet loses a little polarization in the field of its neighbours.
This package computes that answer. It cuts each part into small cells, finds the
polarization of every cell, and hands back ordinary magpylib magnets that you
use like any other.

This page shows what happens on the way, with 3D views you can turn and charts
you can drag. The [method page](method.md) has the equations.

```{code-cell} ipython3
:tags: [hide-input]

import html
import itertools
import time
import warnings

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from comm import DummyComm
from IPython.display import HTML
from loguru import logger
from magpylib_studio.widget import SceneWidget
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from magpylib_material_response import configure_logging, disable_logging
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.demag_fft import analyze_collection
from magpylib_material_response.meshing import (
    mesh_Cuboid,
    mesh_TriangularMesh,
    voxelize,
)
from magpylib_material_response.newell import demag_block_general, point_block

MM = 1e-3  # lengths are in metres; this reads them in millimetres
ORANGE, BLUE, GREY = "#eb6834", "#2a78d6", "#8a8984"
logger.remove()  # loguru's own default sink would print the package log twice


def views(*titled, height=430):
    """magpylib-studio views side by side, saved into the page so that they
    turn and zoom without a running Python. In a notebook, show the view
    itself, or draw any magpylib objects with
    ``magpy.show(*objects, backend="widget")``."""
    cells = []
    for title, view in titled:
        page = html.escape(view.to_html(title))
        cells.append(
            '<figure style="flex: 1 1 340px; margin: 0">'
            f'<iframe class="magpy-studio-frame" title="{html.escape(title)}" '
            f'srcdoc="{page}" style="width: 100%; height: {height}px; border: 0">'
            f'</iframe><figcaption style="text-align: center">{title}</figcaption>'
            "</figure>"
        )
    return HTML(
        '<div style="display: flex; flex-wrap: wrap; gap: 12px">'
        + "".join(cells)
        + "</div>"
    )


def view_of(*objects, **kwargs):
    """A studio view made only to be saved into the page: on a channel to
    nowhere, so the notebook does not also keep it as a live widget."""
    return SceneWidget(*objects, comm=DummyComm(), **kwargs)


def leaves(node):
    """The ids of the objects under a node of a view's object tree."""
    if not node["children"]:
        return [node["id"]]
    return [i for child in node["children"] for i in leaves(child)]


def cut_open(parts, coloured, j_max):
    """A view of solved parts cut open at y = 0, with the field lines of B in
    that plane. ``parts`` maps a name to the part's cells; the cells of the
    parts named in ``coloured`` are coloured by the strength of their
    polarization, up to ``j_max``, and the others show their own north and
    south. The front halves are hidden, not removed: the field lines are still
    those of the whole parts, and the legend shows the halves again."""
    halves, front = [], []
    for name, cells in parts.items():
        cells = [c.copy() for c in cells]
        if name in coloured:
            strength = [np.linalg.norm(c.polarization) / j_max for c in cells]
            for cell, colour in zip(
                cells, sample_colorscale("Viridis", strength), strict=True
            ):
                cell.style.color = colour
                cell.style.magnetization.show = False
        else:
            for cell in cells:
                cell.style.color, cell.style.magnetization.show = "#d9534f", False
        back = magpy.Collection(*(c for c in cells if c.position[1] > 0))
        back.style.label = name
        halves.append(back)
        front_half = magpy.Collection(*(c for c in cells if c.position[1] < 0))
        front_half.style.label = f"{name}, front"
        halves.append(front_half)
        front.append(front_half.style.label)
    x, z = np.linspace(-25, 25, 26) * MM, np.linspace(-12, 12, 13) * MM
    X, Z = np.meshgrid(x, z, indexing="ij")
    plane = magpy.Sensor(pixel=np.stack([X, 0 * X, Z], axis=-1))
    plane.style.label = "field lines of B"
    plane.style.pixel.size = 0
    plane.style.pixel.field.source = "B"
    for axis in "xyz":
        getattr(plane.style.arrows, axis).show = False
    with warnings.catch_warnings():
        # the halves were built in a loop, which matters only for editing
        warnings.filterwarnings("ignore", message="magpylib-studio", category=UserWarning)
        view = view_of(*halves, plane, field_lines=True)
    view.hidden = [
        i for node in view.tree if node["label"] in front for i in leaves(node)
    ]
    # Look at, and turn about, the middle of the cut rather than the middle of
    # the halves left showing: the cut faces and the lines are what to see.
    # (Studio takes the camera in metres; it has no argument for it yet.)
    target = np.array([0.0, 0.0, -1.75 * MM])
    look = np.array([0.55, -0.68, 0.48]) / np.linalg.norm([0.55, -0.68, 0.48])
    view._camera = {
        "projection": "perspective",
        "position": (target + 0.045 * look).tolist(),
        "target": target.tolist(),
        "up": [0.0, 0.0, 1.0],
        "zoom": 1.0,
        "height": 0.06,
        "near": 1e-5,
        "far": 1.0,
    }
    return view


def field_on_grid(get, xs, zs, y, sources):
    """``get`` (``getB`` or ``getJ`` of a collection of ``sources`` cells) on
    the grid xs x zs of the plane at ``y``, shape (len(zs), len(xs), 3): a slice
    of the points at a time, so that a call holds at most 200,000 cell-point
    pairs."""
    X, Z = np.meshgrid(xs, zs)
    points = np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()])
    step = max(1, 200_000 // sources)
    parts = [np.reshape(get(points[i : i + step]), (-1, 3)) for i in range(0, len(points), step)]
    return np.nan_to_num(np.concatenate(parts)).reshape(*X.shape, 3)


def even_field_lines(xs, zs, U, V, spacing):
    """Evenly spaced lines of the field (U, V) on the grid xs x zs, as
    matplotlib's streamplot spaces them: a line stops where it enters a square
    of side ``spacing`` another line already crosses, and lines start from
    squares no line crosses yet."""
    nx, nz = int(np.ceil((xs[-1] - xs[0]) / spacing)), int(np.ceil((zs[-1] - zs[0]) / spacing))
    taken = np.zeros((nz, nx), dtype=bool)
    size = np.hypot(U, V)
    u = np.divide(U, size, out=np.zeros_like(U), where=size > 0)
    v = np.divide(V, size, out=np.zeros_like(V), where=size > 0)
    dx, dz, h = xs[1] - xs[0], zs[1] - zs[0], spacing / 4

    def direction(p):
        fx, fz = (p[0] - xs[0]) / dx, (p[1] - zs[0]) / dz
        i, j = int(fz), int(fx)
        if not (0 <= i < len(zs) - 1 and 0 <= j < len(xs) - 1):
            return None
        a, b = fz - i, fx - j
        w = ((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b)
        d = np.array([
            w[0] * f[i, j] + w[1] * f[i, j + 1] + w[2] * f[i + 1, j] + w[3] * f[i + 1, j + 1]
            for f in (u, v)
        ])
        n = np.hypot(*d)
        return None if n < 1e-9 else d / n

    def square(p):
        return int((p[1] - zs[0]) / spacing), int((p[0] - xs[0]) / spacing)

    def half(seed, sign, mine):
        p, path, at = np.array(seed), [], square(seed)
        for _ in range(4000):
            d = direction(p)
            d = None if d is None else direction(p + 0.5 * h * sign * d)
            if d is None:
                break
            p = p + h * sign * d
            c = square(p)
            if not (0 <= c[0] < nz and 0 <= c[1] < nx):
                break
            if c != at:
                if taken[c] or c in mine:
                    break
                mine.add(c)
                at = c
            path.append(p)
        return path

    lines = []
    edge_first = sorted(np.ndindex(nz, nx), key=lambda c: min(c[0], nz - 1 - c[0], c[1], nx - 1 - c[1]))
    for i, j in edge_first:
        if taken[i, j]:
            continue
        seed = (xs[0] + (j + 0.5) * spacing, zs[0] + (i + 0.5) * spacing)
        mine = {(i, j)}
        back, ahead = half(seed, -1.0, mine), half(seed, 1.0, mine)
        if len(mine) >= 3:
            for c in mine:
                taken[c] = True
            lines.append(np.array([*back[::-1], seed, *ahead]))
    return lines
```

+++ {"user_expressions": []}

## A part becomes a set of small magnets

Before solving, each part is cut into cells: small cuboids, or tetrahedra for
any shape. Each cell gets one uniform polarization, so magpylib computes its
field exactly. The solver finds that one polarization per cell. More cells
follow the real, smoothly varying polarization more closely.

Here a magnet sits 1.5 mm below a 3 mm iron plate. Both are meshed, and
`apply_demag` solves them together:

```{code-cell} ipython3
def magnet_and_plate(chi):
    """A 10 x 10 x 5 mm magnet below a 30 x 30 x 3 mm plate of susceptibility chi."""
    magnet = magpy.magnet.Cuboid(
        dimension=(10 * MM, 10 * MM, 5 * MM),
        position=(0, 0, -4 * MM),
        polarization=(0, 0, 1.2),
    )
    magnet.susceptibility = 0.05  # a sintered magnet answers a little too
    plate = magpy.magnet.Cuboid(
        dimension=(30 * MM, 30 * MM, 3 * MM),
        position=(0, 0, 1.5 * MM),
        polarization=(0, 0, 0),
    )
    plate.susceptibility = chi
    model = magpy.Collection(
        mesh_Cuboid(magnet, (4, 4, 2)), mesh_Cuboid(plate, (16, 16, 4))
    )
    return apply_demag(model)


iron = magnet_and_plate(1000)
```

+++ {"user_expressions": []}

The view shows both parts cut open in the middle, with the field lines of B in
the cut. The magnet is red. The plate's cells are coloured by the strength of
their polarization, from dark blue for none to yellow for 1 T. Turn the view
with the mouse; the legend brings back the front halves.

```{code-cell} ipython3
:tags: [hide-input]

magnet_cells, plate_cells = (part.sources_all for part in iron.children)
j_max = max(np.linalg.norm(c.polarization) for c in plate_cells)
views(("a magnet below an iron plate, cut open", cut_open({"magnet": magnet_cells, "plate": plate_cells}, {"plate"}, j_max)))
```

+++ {"user_expressions": []}

The cells just above the magnet's edges carry the most polarization: that is
where the flux turns into the plate. What the plate does with it depends on its
susceptibility. The cut below shows the strength of B in colour, on a log scale,
and its direction with evenly spaced field lines, through the iron too. Drag the
slider from χ = 0, where the plate is not there for the field, to soft iron.

```{code-cell} ipython3
:tags: [hide-input]

from scipy.ndimage import median_filter

xs, zs = np.linspace(-25, 25, 196), np.linspace(-12, 12, 96)  # mm, off the cell faces
LOW, HIGH = 0.0, 3.1  # log10 of |B| in mT: 1 mT to 1.26 T


def cut_traces(solved, chi):
    """|B| in colour and evenly spaced field lines, 0.05 mm behind the cut. In
    a plate cell, the solver's own B, J (1 + 1/chi) at its centre: the field
    inside a uniformly polarized cell swings from point to point, which is the
    mesh's, not the iron's."""
    cells = len(solved.sources_all)
    y = 0.05 * MM
    B = field_on_grid(solved.getB, xs * MM, zs * MM, y, cells)
    plate = solved.children[1]
    J = field_on_grid(plate.getJ, xs * MM, zs * MM, y, cells)
    iron = np.linalg.norm(J, axis=-1) > 0
    if chi:
        B = np.where(iron[..., None], J * (1 + 1 / chi), B)
    level = (np.log10(np.maximum(np.linalg.norm(B, axis=-1) * 1e3, 1e-3)) - LOW) / (HIGH - LOW)
    # a cell edge just outside the iron leaves a dark speck one sample wide
    level = np.where(iron, level, median_filter(level, size=3))
    xl, zl = [], []
    for line in even_field_lines(xs, zs, B[..., 0], B[..., 2], spacing=1.0):
        line = np.round(line[::2], 2)
        xl += [*line[:, 0], None]
        zl += [*line[:, 1], None]
    return [
        go.Heatmap(
            x=xs, y=zs, z=np.round(255 * np.clip(level, 0, 1)).astype(np.uint8), zmin=0, zmax=255,
            colorscale="Viridis", hoverinfo="skip",
            colorbar={"title": {"text": "|B|"}, "tickvals": [255 * (k - LOW) / (HIGH - LOW) for k in range(4)],
                      "ticktext": ["1 mT", "10 mT", "100 mT", "1 T"]},
        ),
        go.Scatter(x=xl, y=zl, mode="lines", line={"color": "rgba(255, 255, 255, 0.8)", "width": 1}, hoverinfo="skip", showlegend=False),
    ]


chis = (0, 10, 100, 1000)
cuts = {chi: cut_traces(iron if chi == 1000 else magnet_and_plate(chi), chi) for chi in chis}
fig = go.Figure(data=cuts[1000])
fig.add_shape(type="rect", x0=-5, x1=5, y0=-6.5, y1=-1.5, line={"color": ORANGE, "width": 2})
fig.add_shape(type="rect", x0=-15, x1=15, y0=0, y1=3, line={"color": "white", "width": 1})
fig.frames = [go.Frame(name=str(chi), data=cuts[chi]) for chi in chis]
fig.update_layout(
    xaxis={"title": {"text": "x (mm)"}, "range": [-25, 25], "constrain": "domain"},
    yaxis={"title": {"text": "z (mm)"}, "range": [-12, 12], "scaleanchor": "x", "constrain": "domain"},
    height=400,
    margin={"l": 60, "r": 20, "t": 20, "b": 40},
    sliders=[{
        "active": len(chis) - 1, "pad": {"t": 50},
        "currentvalue": {"prefix": "plate susceptibility χ = "},
        "steps": [
            {"label": str(chi), "method": "animate",
             "args": [[str(chi)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
            for chi in chis
        ],
    }],
)
fig
```

+++ {"user_expressions": []}

At χ = 10 the lines still cross the plate and go on above it. Soft iron draws
the flux in, carries it sideways inside the plate and lets it out at the ends,
and the space above the plate goes dark. On the axis, 3 mm above the plate, the
field falls from 65 mT without the plate to 31 mT at χ = 10 and to 10 mT for
soft iron.

## Every cell feels every other cell

How strongly a cell is magnetized depends on the field where it sits. That field
is the applied one plus the field of every other cell, and those cells are
magnetized in turn. For a linear material this is one equation per cell:

$$
\mathbf{J}_i = \mathbf{J}_{r,i} + \chi_i \Bigl(\mathbf{B}_\text{applied} +
\sum_{j} \mathsf{T}_{ij}\,\mathbf{J}_j\Bigr)
$$

where $\mathbf{J}_r$ is the remanence of a permanent magnet (zero for iron) and
$\mathsf{T}_{ij}$ is the field of cell $j$ at cell $i$ per unit of its
polarization. With three components per cell, N cells give 3N equations in 3N
unknowns, solved all at once so that every cell agrees with every other.

The interactions $\mathsf{T}_{ij}$ are the heart of it: N cells have N² of them.
There are two ways to take the field of one cell at another:

- **at its centre** (`interaction="point"`, the default), for any cell type;
- **averaged over it** (`interaction="newell"`), for cuboids all turned the same
  way.

Drag the slider to move a neighbour away from a cell, for cubes and for flat
cells.

```{code-cell} ipython3
:tags: [hide-input]


def pair_figure():
    """The field of one cell on a neighbour, at the neighbour's centre and
    averaged over it, for cubes and for flat cells."""
    shapes = (
        ("cubes", np.array([1.0, 1.0, 1.0]), "solid"),
        ("flat cells", np.array([1.0, 1.0, 0.25]), "dot"),
    )
    steps = [1.0, 2.0, 3.0, 4.0]
    xs, zs = np.linspace(-1.2, 5.2, 128), np.linspace(-0.9, 0.9, 56)
    X, Z = np.meshgrid(xs, zs)
    grid = np.stack([X, 0 * X, Z], axis=-1)
    dist = np.linspace(1.0, 5.0, 81)
    fig = make_subplots(
        rows=3, cols=1, row_heights=[0.3, 0.3, 0.4], vertical_spacing=0.08,
        subplot_titles=(
            "cubes: the z-field of the left cell",
            "flat cells, 4 : 1: the z-field of the left cell",
            "the field each puts on its neighbour",
        ),
    )
    values = {}
    for row, (name, dim, dash) in enumerate(shapes, start=1):
        lim = 0.12
        field = np.clip(-point_block(grid, dim)[..., 2, 2], -lim, lim).astype(np.float32)
        fig.add_trace(go.Heatmap(x=xs, y=zs, z=field, zmin=-lim, zmax=lim, colorscale="RdBu_r", showscale=False, hoverinfo="skip"), row, 1)
        centre = [-point_block(np.array([d, 0, 0]), dim)[2, 2] for d in dist]
        average = [-demag_block_general(np.array([d, 0, 0]), dim, dim)[2, 2] for d in dist]
        fig.add_trace(go.Scatter(x=dist, y=centre, name=f"{name}, at its centre (point)", line={"color": ORANGE, "width": 2, "dash": dash}), 3, 1)
        fig.add_trace(go.Scatter(x=dist, y=average, name=f"{name}, averaged over it (Newell)", line={"color": BLUE, "width": 2, "dash": dash}), 3, 1)
        for d in steps:
            values[(row, d)] = (
                dim[2] / 2,
                -point_block(np.array([d, 0, 0]), dim)[2, 2],
                -demag_block_general(np.array([d, 0, 0]), dim, dim)[2, 2],
            )
        fig.update_xaxes(range=[-1.2, 5.2], constrain="domain", row=row, col=1)
        fig.update_yaxes(scaleanchor="x" if row == 1 else "x2", range=[-0.9, 0.9], constrain="domain", showticklabels=False, row=row, col=1)
    fig.update_xaxes(title_text="distance between centres (cell widths)", row=3, col=1)
    fig.update_yaxes(title_text="Hz per unit Jz", row=3, col=1)
    titles = [a.to_plotly_json() for a in fig.layout.annotations]

    def layout_for(d):
        shapes_, notes = [], list(titles)
        for row in (1, 2):
            h, p, n = values[(row, d)]
            xa, ya = ("x", "y") if row == 1 else ("x2", "y2")
            shapes_ += [
                {"type": "rect", "x0": -0.5, "x1": 0.5, "y0": -h, "y1": h, "xref": xa, "yref": ya, "line": {"color": ORANGE, "width": 2}},
                {"type": "rect", "x0": d - 0.5, "x1": d + 0.5, "y0": -h, "y1": h, "xref": xa, "yref": ya, "line": {"color": BLUE, "width": 2}},
            ]
            notes.append({
                "text": f"centre {p:+.4f}, average {n:+.4f}: {abs(p - n) / abs(n):.1%} apart",
                "x": 5.1, "y": -0.72, "xref": xa, "yref": ya, "xanchor": "right", "showarrow": False, "font": {"size": 12},
            })
        shapes_.append({"type": "line", "x0": d, "x1": d, "y0": 0, "y1": 1, "xref": "x3", "yref": "y3 domain", "line": {"color": GREY, "dash": "dot"}})
        return {"shapes": shapes_, "annotations": notes}

    fig.update_layout(
        **layout_for(steps[0]),
        height=780,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"x": 0.99, "y": 0.02, "xanchor": "right", "yanchor": "bottom", "bgcolor": "rgba(0,0,0,0)"},
        sliders=[{
            "active": 0, "pad": {"t": 50},
            "currentvalue": {"prefix": "distance between centres: ", "suffix": " cell widths"},
            "steps": [{"label": f"{d:g}", "method": "relayout", "args": [layout_for(d)]} for d in steps],
        }],
    )
    return fig


pair_figure()
```

+++ {"user_expressions": []}

For cubes the two ways agree within about 1 % at every distance on a grid. For
flat cells side by side the field changes so much across the neighbour that its
centre value is only half the average, and the gap closes only a few widths
away. Both rules converge to the same answer as the cells shrink: on the cube
below they differ by 0.1 to 0.3 %, against a mesh error of 1 to 4 %. Keep the
default; choose `"newell"` for meshes of flat cuboids, or for a mesh study,
where its moment rises steadily toward the converged value.

## How many cells?

The answer converges as the cells shrink, slowly: halving the cell size, which
takes eight times the cells, cuts the error to about a third. Soft iron is the
hard case, because its polarization gathers at edges and corners, where small
cells are needed. Here is a 10 mm cube of soft iron (χ = 1000) in a 10 mT field.
Its magnetic moment is known to 0.1 %:

```{code-cell} ipython3
def soft_cube():
    cube = magpy.magnet.Cuboid(dimension=(10 * MM,) * 3, polarization=(0, 0, 0))
    cube.susceptibility, cube.H_ext = 1000, (0, 0, 0.01)
    return cube


def soft_cube_surface():
    corners = np.array(list(itertools.product((-5 * MM, 5 * MM), repeat=3)))
    body = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
    body.susceptibility, body.H_ext = 1000, (0, 0, 0.01)
    return body


def moment(solved):
    """The z-moment of the solved cells, in A m²."""
    return sum(c.volume * c.orientation.apply(c.polarization)[2] for c in solved.sources_all) / magpy.mu_0


EXACT = 0.02889  # A m², the converged moment of this cube
runs = []
for rule in ("point", "newell"):
    for n in (27, 64, 216, 512, 1728, 4096):
        cells = mesh_Cuboid(soft_cube(), n)
        t0 = time.perf_counter()
        solved = apply_demag(cells, interaction=rule)
        runs.append((f"cuboids, {rule}", len(cells.sources_all), moment(solved), time.perf_counter() - t0))
for n in (30, 100, 300, 1000):
    cells = mesh_TriangularMesh(soft_cube_surface(), n)
    t0 = time.perf_counter()
    solved = apply_demag(cells)
    runs.append(("tetrahedra", len(cells.sources_all), moment(solved), time.perf_counter() - t0))
```

```{code-cell} ipython3
:tags: [hide-input]

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06)
for name, colour, dash in (("cuboids, point", ORANGE, "solid"), ("cuboids, newell", BLUE, "dash"), ("tetrahedra", "#1baf7a", "solid")):
    rows = [r for r in runs if r[0] == name]
    err = [100 * abs(m - EXACT) / EXACT for _, _, m, _ in rows]
    hover = [f"{name}<br>{n:,} cells<br>moment {100 * (m - EXACT) / EXACT:+.2f} %<br>{t:.2f} s" for _, n, m, t in rows]
    style = {"color": colour, "dash": dash, "width": 2}
    fig.add_trace(go.Scatter(x=[r[1] for r in rows], y=err, name=name, mode="lines+markers", line=style, hovertext=hover, hoverinfo="text"), 1, 1)
    fig.add_trace(go.Scatter(x=[r[3] for r in rows], y=err, name=name, mode="lines+markers", line=style, hovertext=hover, hoverinfo="text", showlegend=False), 1, 2)
fig.update_xaxes(type="log", title_text="cells", row=1, col=1)
fig.update_xaxes(type="log", title_text="solve time (s)", row=1, col=2)
fig.update_yaxes(type="log", title_text="error of the moment (%)", row=1, col=1)
fig.update_layout(height=400, margin={"l": 60, "r": 20, "t": 20, "b": 40}, legend={"orientation": "h", "y": -0.25})
fig
```

+++ {"user_expressions": []}

Tetrahedra reach the same error with fewer cells. A cuboid grid, though, solves
far faster per cell, with FFT products the next section explains, so on this
cube the cuboids win per second of solving. On other parts the balance shifts:
on the core of the [U-core example](examples/ucore_electromagnet.md), a few
hundred tetrahedra come closer than thousands of cuboids. The two interaction
rules sit almost on top of each other. In practice: start coarse, multiply the
cells by four to eight, and stop when the quantity you need, a force or a field
at a sensor, changes by less than you can accept.

## Solving: direct or iterative

N cells give 3N unknowns and a matrix of 9N² numbers: 1 GB for 3,700 cells. The
**direct** solver stores that matrix and solves it exactly, which is quick up to
a few thousand cells. The **iterative** solver never stores it whole. It only
needs the field of all cells on all cells, again and again, until the
polarizations stop changing. On a grid of identical cuboids that field is a
convolution, so FFTs compute it in N log N time and little memory.

The default, `solver="auto"`, looks at how the cells are arranged and picks.
Here three parts are meshed three ways, and coloured by how the solver sees
them: a cuboid mesh is one grid (blue); a voxelized sphere is many small full
boxes of grid (blue) and an irregular rim (orange); tetrahedra lie on no grid
(grey).

```{code-cell} ipython3
block = soft_cube()
block.position = (-22 * MM, 0, 0)
ball = magpy.magnet.Sphere(diameter=12 * MM, polarization=(0, 0, 0))
ball.susceptibility, ball.H_ext = 1000, (0, 0, 0.01)
cone = magpy.magnet.TriangularMesh.from_ConvexHull(
    polarization=(0, 0, 0),
    points=np.vstack(
        [[(16 + 6 * np.cos(a)) * MM, 6 * np.sin(a) * MM, -5 * MM] for a in np.linspace(0, 2 * np.pi, 13)[:-1]]
        + [[16 * MM, 0, 6 * MM]]
    ),
)
cone.susceptibility, cone.H_ext = 1000, (0, 0, 0.01)
model = magpy.Collection(
    mesh_Cuboid(block, 1000), voxelize(ball, 1500), mesh_TriangularMesh(cone, 250)
)

configure_logging(sink=lambda message: print(message.record["message"]), min_log_time=60)
solved = apply_demag(model)
disable_logging()
```

```{code-cell} ipython3
:tags: [hide-input]

kinds = {"grid": ("grids (FFT)", BLUE), "loose": ("other cuboids", ORANGE), "generic": ("tetrahedra", GREY)}
cells = [c.copy() for c in solved.sources_all]
groups = {kind: [] for kind in kinds}
for cluster in analyze_collection(solved.sources_all)[1]:
    for i in cluster["indices"]:
        cells[i].style.color = kinds[cluster["kind"]][1]
        cells[i].style.magnetization.show = False
        groups[cluster["kind"]].append(cells[i])
views(("how the solver sees the cells", view_of(*(magpy.Collection(*group, style_label=kinds[k][0]) for k, group in groups.items() if group))), height=380)
```

+++ {"user_expressions": []}

The first log line counts the clusters, the second says what `"auto"` made of
them. It takes the iterative solver once FFT products cover a tenth of the
interaction matrix, the cells of each grid among themselves and large grids of
one spacing between each other. Below that, many small bodies or cells off any
grid make the iterative solver slower, and it takes the direct one. Both give
the same answer, to about `solver_tol`; the {ref}`method page <auto-solver>` has
the measurements behind the rule.

## Where to go next

- The [quickstart](quickstart.md) sets susceptibilities and applied fields.
- The [examples](examples/index.md) compare with FEM, drive a core with a coil
  and mesh any shape with tetrahedra.
- The [method page](method.md) has the equations, the two interaction rules and
  the fast paths.
