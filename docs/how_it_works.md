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

import itertools
import time
import warnings

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
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
# studio says that objects made in a loop are copies: it matters only for editing
warnings.filterwarnings("ignore", message="magpylib-studio", category=UserWarning)
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

In the view, the plate's cells are coloured by the strength of their
polarization, from dark blue for none to yellow for 1 T, and studio draws the
field lines of B in the plane y = 0. Turn the view to look at the plate from
below: the cells above the magnet's edges carry the most, where the flux turns
into the plate.

```{code-cell} ipython3
:tags: [hide-input]

magnet, plate = iron.children
iron.style.label, magnet.style.label, plate.style.label = "solved", "magnet", "iron plate"
strength = np.array([np.linalg.norm(c.polarization) for c in plate.sources_all])
for cell, colour in zip(plate.sources_all, sample_colorscale("Viridis", list(strength / strength.max())), strict=True):
    cell.style.color, cell.style.magnetization.show = colour, False
for cell in magnet.sources_all:
    cell.style.color, cell.style.magnetization.show = "#d9534f", False
x, z = np.meshgrid(np.linspace(-25, 25, 26) * MM, np.linspace(-12, 12, 13) * MM, indexing="ij")
lines = magpy.Sensor(
    pixel=np.stack([x, 0 * x, z], axis=-1),
    style_label="field lines of B",
    style_pixel_size=0,  # the lines, without the pixels
    style_pixel_field_source="B",
    style_arrows_x_show=False,
    style_arrows_y_show=False,
    style_arrows_z_show=False,
)
SceneWidget(iron, lines, field_lines=True)
```

+++ {"user_expressions": []}

What the plate does with the flux depends on its susceptibility. On the axis
above the plate:

```{code-cell} ipython3
:tags: [hide-input]

above = np.linspace(4, 20, 81) * MM
fig = go.Figure()
for chi, colour in ((0, GREY), (10, "#86b6ef"), (100, BLUE), (1000, "#0b3c8c")):
    solved = iron if chi == 1000 else magnet_and_plate(chi)
    Bz = solved.getB(np.column_stack([0 * above, 0 * above, above]))[:, 2]
    name = "χ = 0, as if there were no plate" if chi == 0 else f"χ = {chi}"
    fig.add_trace(go.Scatter(x=above / MM, y=Bz * 1e3, name=name, line={"color": colour, "width": 2}))
fig.update_layout(
    xaxis={"title": {"text": "z on the axis (mm); the plate's top is at 3 mm"}},
    yaxis={"title": {"text": "Bz (mT)"}, "type": "log"},
    height=360,
    margin={"l": 60, "r": 20, "t": 20, "b": 40},
    legend={"x": 0.98, "y": 0.98, "xanchor": "right"},
)
fig
```

+++ {"user_expressions": []}

3 mm above the plate, at z = 6 mm, the field falls from 65 mT without the plate
to 31 mT at χ = 10 and to 10 mT for soft iron. The iron draws the flux in,
carries it sideways inside the plate and lets it out at its ends, and the space
above it is shielded.

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
Its magnetic moment, 0.02889 A m², is known to 0.1 %: finite-element solutions
of rising order and mesh density converge to it
([#76](https://github.com/magpylib/magpylib-material-response/issues/76)). The
polarizability of a cube of infinite susceptibility, 3.6443 times its volume
(Helsing and Perfekt 2013), gives 0.028895 A m², 0.02 % away, through $m = VH /
(1/3.6443 + 1/\chi)$, which is exact for ellipsoids:

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


EXACT = 0.02889  # A m², the converged moment of this cube (#76)
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
magpy.show(*(magpy.Collection(*group, style_label=kinds[k][0]) for k, group in groups.items() if group), backend="widget")
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

## References

1. Helsing, J., & Perfekt, K.-M. (2013). _On the polarizability and capacitance
   of the cube._ Applied and Computational Harmonic Analysis, 34(3), 445–468.
   [doi:10.1016/j.acha.2012.07.006](https://doi.org/10.1016/j.acha.2012.07.006).
   The limit polarizability of a cube, 3.6443 times its volume, is in Table 1.
