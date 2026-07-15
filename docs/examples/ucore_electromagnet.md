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

# U-Core Electromagnet

+++

All other examples drive the material response with hard magnets or a uniform
external field. This one uses **current sources**: a soft-magnetic U-core with a
multi-turn coil wound around its middle bar. Current objects are first-class
citizens of the workflow — `mesh_all` passes them through unmeshed, and
`apply_demag` adds their field to the applied field seen by every magnet cell.

The coil itself is built with the `polyline` helpers: `create_polyline_fillet`
rounds the corners of a square loop, and `move_grid_along_polyline` sweeps the
winding-pack cross-section (turns × layers) along it, producing one closed
polyline per winding.

At the end, the same core is re-meshed with tetrahedral cells and both
discretizations are compared on the same field map.

## Build the U-core

Three soft-iron cuboids — a horizontal yoke and two legs — with µr = 1000. The
susceptibility is set on the parent collection and inherited by all children
(and later by their mesh cells). All quantities are SI (m, T, A).

```{code-cell} ipython3
import time

import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_all, mesh_TriangularMesh
from magpylib_material_response.polyline import (
    create_polyline_fillet,
    move_grid_along_polyline,
)

magpy.defaults.display.backend = "plotly"

a = 0.02  # core cross-section side
yoke = magpy.magnet.Cuboid(dimension=(0.10, a, a), style_label="yoke")
leg_left = magpy.magnet.Cuboid(
    dimension=(a, a, 0.06), position=(-0.04, 0, 0.04), style_label="left leg"
)
leg_right = magpy.magnet.Cuboid(
    dimension=(a, a, 0.06), position=(0.04, 0, 0.04), style_label="right leg"
)

core = magpy.Collection(yoke, leg_left, leg_right, style_label="U-core")
core.susceptibility = 999  # µr = 1000
```

+++ {"user_expressions": []}

## Wind the coil with the polyline helpers

One turn is a square loop around the yoke cross-section with filleted corners.
The loop starts and ends on an edge midpoint so that the seam of the closed
polyline lies on a straight segment, away from the fillets.

```{code-cell} ipython3
h = 0.014  # loop half-width: 4 mm clearance around the 20 mm core
square = np.array(
    [
        [0, h, 0],  # seam on the edge midpoint
        [0, h, h],
        [0, -h, h],
        [0, -h, -h],
        [0, h, -h],
        [0, h, 0],
    ]
)
turn = create_polyline_fillet(square, max_radius=0.006, N=6)
```

+++ {"user_expressions": []}

The winding pack is a grid of wire positions in the cross-sectional plane of the
coil: 8 turns along the yoke axis (x) times 2 radial layers, anchored at the
seam point of the turn. Sweeping this grid along the rounded turn returns one
closed polyline per winding — outer layers automatically get slightly larger
fillet radii, exactly like a real winding.

```{code-cell} ipython3
pitch = 0.0022  # wire pitch
x_turns = (np.arange(8) - 3.5) * pitch  # 8 turns centered on the yoke middle
y_layers = h + np.arange(2) * pitch  # 2 radial layers
pack = np.array([(xt, yl, 0) for xt in x_turns for yl in y_layers])

windings = move_grid_along_polyline(turn, pack)  # shape (16, n_vertices, 3)

coil = magpy.Collection(
    *(magpy.current.Polyline(current=25, vertices=v) for v in windings),
    style_label="coil",
)
print(f"{len(coil)} windings × 25 A = {len(coil) * 25:.0f} ampere-turns")
```

+++ {"user_expressions": []}

## Assemble the system and the field plane

The field is evaluated on a plane 10 mm above the pole faces, normal to the legs
(the z-direction). A decimated copy of the sensor grid is added to the scene to
visualize where the field will be evaluated.

```{code-cell} ipython3
system = magpy.Collection(core, coil, style_label="U-core electromagnet")

x = np.linspace(-0.08, 0.08, 81)
y = np.linspace(-0.04, 0.04, 41)
X, Y = np.meshgrid(x, y, indexing="ij")
pixel = np.stack([X, Y, np.full_like(X, 0.08)], axis=-1)  # 10 mm above poles

plane = magpy.Sensor(pixel=pixel, style_label="field plane")
plane_shown = magpy.Sensor(
    pixel=pixel[::10, ::5], style_label="field plane", style_pixel_size=0.5
)

magpy.show(system, plane_shown)
```

+++ {"user_expressions": []}

## Mesh and apply the material response

`mesh_all` walks the collection and meshes the three cuboids into cells, passing
the current polylines through untouched. `apply_demag` then solves for the
self-consistent polarization of every cell: the right-hand side of the linear
system contains the coil field evaluated at each cell, on top of the cell-cell
interactions.

```{code-cell} ipython3
system_meshed = mesh_all(system, target_elems=250)
n_cells = sum(isinstance(s, magpy.magnet.Cuboid) for s in system_meshed.sources_all)
print(f"{n_cells} magnet cells")
magpy.show(system_meshed, plane_shown)
```

```{code-cell} ipython3
def timed_demag(collection):
    """apply_demag with wall-clock timing, quoted in the plot legends below."""
    t0 = time.perf_counter()
    out = apply_demag(collection)
    return out, time.perf_counter() - t0


system_demag, t_cub = timed_demag(system_meshed)
print(f"{n_cells} cuboid cells solved in {t_cub:.1f} s")
```

+++ {"user_expressions": []}

## Field on the plane above the poles

The returned collection contains the magnetized cells _and_ the coil, so it is a
drop-in source for the total field. The normal component $B_z$ maps the two pole
footprints — positive above one leg, negative above the other:

```{code-cell} ipython3
B_full = plane.getB(system_demag)  # shape (81, 41, 3)

fig = px.imshow(
    B_full[..., 2].T * 1e3,  # transpose to (y, x) image orientation
    x=x * 1e3,
    y=y * 1e3,
    origin="lower",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    labels={"x": "x (mm)", "y": "y (mm)", "color": "Bz (mT)"},
    title="Bz 10 mm above the pole faces",
)
fig
```

+++ {"user_expressions": []}

How much of this is the core? Compare with the bare coil along the line y = 0.
Without the core the same 400 ampere-turns produce a field two orders of
magnitude weaker on this plane — the soft core collects the coil flux and
delivers it to the poles:

```{code-cell} ipython3
B_coil = plane.getB(coil)  # bare coil, no core

iy = np.argmin(np.abs(y))  # index of the y=0 line
cut = pd.DataFrame(
    {
        "coil only": B_coil[:, iy, 2] * 1e3,
        f"coil + core response ({t_cub:.1f} s solve)": B_full[:, iy, 2] * 1e3,
    },
    index=pd.Index(x * 1e3, name="x (mm)"),
)
fig = px.line(
    cut,
    labels={"value": "Bz (mT)", "variable": ""},
    title="Bz along y=0, 10 mm above the pole faces",
)

amp = np.abs(B_full[..., 2]).max() / np.abs(B_coil[..., 2]).max()
print(f"peak Bz amplification by the core: ×{amp:.0f}")
fig
```

+++ {"user_expressions": []}

## Cuboid cells vs tetrahedral cells

The three-cuboid core is convenient, but nothing forces cuboid cells. The same
solid can be built as a **single closed `TriangularMesh`** — the U profile in
the x–z plane, extruded along y — and handed to `mesh_TriangularMesh`, which
drives [TetGen](https://tetgen.org) (optional dependency,
`pip install magpylib-material-response[tetgen]`, see the
[tetrahedral example](tetrahedral_meshes.md)):

```{code-cell} ipython3
u_profile = np.array(
    [  # U outline in the x–z plane, counterclockwise
        (-0.05, -0.01),
        (0.05, -0.01),
        (0.05, 0.07),
        (0.03, 0.07),
        (0.03, 0.01),
        (-0.03, 0.01),
        (-0.03, 0.07),
        (-0.05, 0.07),
    ]
)
cap = [(0, 1, 4), (1, 3, 4), (1, 2, 3), (0, 4, 5), (0, 5, 7), (5, 6, 7)]

# extrude the profile along y into a closed prism: two caps + side walls
n = len(u_profile)
vertices = np.array([(ux, uy, uz) for uy in (-0.01, 0.01) for ux, uz in u_profile])
faces = list(cap) + [tuple(i + n for i in tri) for tri in cap]
for i in range(n):
    j = (i + 1) % n
    faces += [(i, j, j + n), (i, j + n, i + n)]

u_iron = magpy.magnet.TriangularMesh(
    vertices=vertices, faces=np.array(faces), reorient_faces=True
)

core_tet = mesh_TriangularMesh(u_iron, target_elems=250)
core_tet.susceptibility = 999
n_tets = len(core_tet.sources_all)
print(f"{n_tets} tetrahedra")

magpy.show(
    {"objects": [u_iron], "col": 1},
    {"objects": [core_tet], "col": 2},
)
```

+++ {"user_expressions": []}

TetGen grades the cells: they are small along the edges and corners of the U,
where the magnetization gradients live, and coarse in the bulk. Solve both
discretizations at the same three refinement levels (`target_elems` of 250, 500
and 1000). Every solve is timed, and the wall times end up in the plot legend:

```{code-cell} ipython3
targets = [250, 500, 1000]
runs = {}

for target in targets:  # cuboid meshes
    meshed = mesh_all(system, target_elems=target)
    nc = sum(isinstance(s, magpy.magnet.Cuboid) for s in meshed.sources_all)
    demag, t = timed_demag(meshed)
    runs[f"cuboids: {nc} cells, {t:.1f} s"] = plane.getB(demag)

for target in targets:  # tetrahedral meshes
    core_t = mesh_TriangularMesh(u_iron, target_elems=target)
    core_t.susceptibility = 999
    demag, t = timed_demag(magpy.Collection(core_t, coil.copy()))
    runs[f"tetrahedra: {len(core_t.sources_all)} cells, {t:.1f} s"] = plane.getB(demag)
```

+++ {"user_expressions": []}

In the comparison the hue distinguishes the cell type and darker means finer:

```{code-cell} ipython3
for name, B in runs.items():
    print(f"{name:>30}: peak |Bz| = {np.abs(B[..., 2]).max() * 1e3:.2f} mT")

cut = pd.DataFrame(
    {name: B[:, iy, 2] * 1e3 for name, B in runs.items()},
    index=pd.Index(x * 1e3, name="x (mm)"),
)
shades = px.colors.sequential.Blues[4::2] + px.colors.sequential.Oranges[4::2]
px.line(
    cut,
    color_discrete_sequence=shades,
    labels={"value": "Bz (mT)", "variable": ""},
    title="Cuboid vs tetrahedral mesh — Bz along y=0",
)
```

+++ {"user_expressions": []}

Both discretizations converge to the same field, approaching it from opposite
sides: refining the tetrahedra by 4× moves the peak by only a few percent — the
graded mesh is nearly converged already at ~400 cells — while the uniform cuboid
grid climbs from below and needs several thousand cells for the same accuracy.
Sharp corners and µr = 1000 put the burden on resolving the edge regions, which
is exactly where TetGen concentrates its cells. The solve times in the legend
show the other side of the trade: tetrahedra take the point-matched _generic_
interaction path with dense N² cost, so refining them quickly gets expensive,
whereas axis-aligned cuboid cells use the analytical volume-averaged path, stay
much cheaper per cell, and remain FFT-accelerable to far larger cell counts (see
[solvers and performance](solver_performance.md)).

+++ {"user_expressions": []}

Notes:

- The demo meshes are deliberately coarse to keep the page fast; refine
  `target_elems` to converge the field values (see the
  [cuboid example](cuboids_demagnetization.md) for a convergence study).
- With µr = 1000 the result is the linear material response; saturation of real
  core materials is not modeled.
