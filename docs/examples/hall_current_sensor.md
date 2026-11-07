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

# A Hall current sensor

An open-loop Hall current sensor, as in drives, chargers and battery packs: a
slotted ferrite ring around a busbar, and a Hall element in the slot. The ring
gathers the busbar's field into the slot, and the Hall element reads it. Four
questions from its datasheet: how many mT per ampere does the Hall element see,
how does the slot's width set that, how much does the reading move when the
busbar sits off-centre, and at what current does the ferrite saturate?

The ring comes from CAD, as a triangle mesh. Face charge solves it on its
triangles, and cells on tetrahedra give the field inside it.

## The model

A MnZn ferrite ring (χ = 2000), 16 mm inside, 26 mm outside, 6 mm tall, with a 2
mm slot. [`hall_core.py`](hall_core.py) writes the triangles a CAD export of it
gives; with a real part, read its STL file with a mesh library instead. The
busbar is a line current along the ring's axis, which is what a round conductor
gives outside itself. It closes 0.5 m away, where its return adds 0.07 % to the
field in the slot.

```{code-cell} ipython3
import time

import magpylib as magpy
import numpy as np
from hall_core import core_mesh

from magpylib_material_response import solve_face_charge

MM = 1e-3
LOOP = 0.5  # m: where the busbar's current returns


def busbar_path(x=0.0, y=0.0):
    """Along the ring's axis, shifted by (x, y) mm, and back 0.5 m away."""
    x, y = x * MM, y * MM
    return [(x, y, -LOOP), (x, y, LOOP), (LOOP, y, LOOP), (LOOP, y, -LOOP), (x, y, -LOOP)]


def current_sensor(gap=2.0, current=100.0):
    vertices, faces = core_mesh(gap)
    core = magpy.magnet.TriangularMesh(vertices=vertices * MM, faces=faces, polarization=(0, 0, 0))
    core.susceptibility = 2000  # MnZn ferrite
    busbar = magpy.current.Polyline(current=current, vertices=busbar_path())
    return magpy.Collection(core, busbar)


hall = np.array([10.5, 0, 0]) * MM  # the Hall element, in the middle of the slot
```

```{code-cell} ipython3
:tags: [hide-input]

solved = solve_face_charge(current_sensor(), level=2)
# the core, a short piece of the busbar through it, and the Hall element
piece = magpy.current.Polyline(current=100, vertices=[(0, 0, -0.015), (0, 0, 0.015)])
magpy.show(solved.children[0], piece, magpy.Sensor(position=hall), backend="widget")
```

## How many mT per ampere?

```{code-cell} ipython3
t0 = time.perf_counter()
solved = solve_face_charge(current_sensor(current=100))
sensitivity = magpy.getB(solved, hall)[1] / 100  # T per A, across the slot
print(f"{sensitivity * 1e3:.4f} mT/A in {time.perf_counter() - t0:.1f} s")
print(f"textbook mu0 I / g: {magpy.mu_0 / (2 * MM) * 1e3:.4f} mT/A")
```

Finite elements give 0.5975 mT/A (order 3, from a run not repeated on this
page), 0.015 % from face charge. The textbook estimate μ0 I / g is 5 % high: it
takes the ring for a perfect conductor of flux and the slot's field for uniform.

## How does the slot's width set it?

```{code-cell} ipython3
rows = ["slot    sensitivity   mu0 / g      off it"]
for gap in (1.0, 1.5, 2.0, 3.0, 4.0):
    k = magpy.getB(solve_face_charge(current_sensor(gap), level=2), hall)[1] / 100
    ideal = magpy.mu_0 / (gap * MM)
    rows.append(f"{gap:.1f} mm   {k * 1e3:.3f} mT/A    {ideal * 1e3:.3f} mT/A   {k / ideal - 1:+.1%}")
print("\n".join(rows))
```

A narrower slot gives more signal, close to as 1 / g, and the estimate stays 4.5
to 7 % high throughout, for two reasons. The ring's own path, 66 mm long at a
permeability of 2001, acts as 0.03 mm more slot: 3 % of a 1 mm slot, under 1 %
of a 4 mm one. The rest is flux that leaves the ring before the slot, or spreads
out of it.

## What if the busbar sits off-centre?

A busbar rarely sits at the centre of the window. The sweep below moves it 2 and
4 mm in four directions. Only the current moves, so every solve keeps the
factorization of the first ([Many solves](../guide/loops.md)).

```{code-cell} ipython3
model = current_sensor()
busbar = model.children[1]
solved, readings, t0 = True, {}, time.perf_counter()
for r in (0, 2, 4):
    for angle in (0,) if r == 0 else (0, 90, 180, 270):
        busbar.vertices = busbar_path(r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle)))
        solved = solve_face_charge(model, level=2, reuse=solved)
        readings[r, angle] = magpy.getB(solved, hall)[1]
print(f"9 solves in {time.perf_counter() - t0:.1f} s, factorization kept: {solved.face_charge_info['reused']}")
for (r, angle), b in readings.items():
    toward = {0: "toward the slot", 90: "sideways", 180: "away from the slot", 270: "sideways"}[angle]
    print(f"{r} mm {toward if r else 'centred':<19} {b / readings[0, 0] - 1:+.2%}")
```

The reading moves by at most 0.4 %, toward the slot, at 4 mm off-centre: the
ring gathers the field from wherever the busbar runs through it. Without the
ring, the Hall element 10.5 mm from the axis would read 62 % more with the
busbar 4 mm closer.

## At what current does the ferrite saturate?

The slot's field, 0.6 mT/A, says little about the ferrite: part of the ring's
flux closes through the air before it reaches the slot, so the ring carries more
far from it. Face charge gives the field outside the ring only. Cells give it
inside: TetGen cuts the ring's triangles into tetrahedra (the `tetgen` extra,
`pip install "magpylib-material-response[tetgen]"`), and each cell's solved
polarization is, at χ = 2000, its flux density to 0.05 %.

```{code-cell} ipython3
import warnings

from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_TriangularMesh

core, busbar = current_sensor(current=1).children
t0 = time.perf_counter()
with warnings.catch_warnings():
    # TetGen keeps every triangle of the surface and asks for more cells than
    # the target, which it warns about
    warnings.simplefilter("ignore")
    cells = mesh_TriangularMesh(core, target_elems=800)
in_cells = solve_cells(magpy.Collection(cells, busbar.copy()))
tetrahedra = [c for c in in_cells.sources_all if isinstance(c, magpy.magnet.Tetrahedron)]
J = np.array([np.linalg.norm(c.polarization) for c in tetrahedra])  # T per A
centres = np.array([c.centroid for c in tetrahedra]) / MM
# the ring in 24 sectors of 15 degrees, 0 at the slot
angle = np.abs(np.degrees(np.arctan2(centres[:, 1], centres[:, 0])))
sector = np.minimum((angle // 15).astype(int), 11)
mean = np.array([J[sector == k].mean() for k in range(12)])
print(f"{len(tetrahedra):,} tetrahedra, {time.perf_counter() - t0:.1f} s")
print(f"in the slot: {magpy.getB(in_cells, hall)[1] * 1e3:.3f} mT/A")
print("ferrite, mean over each 15 degrees, from the slot round to opposite it, mT/A:")
print("  " + "  ".join(f"{m * 1e3:.2f}" for m in mean))
```

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go

fig = go.Figure(go.Scatter(x=centres[:, 0], y=centres[:, 1], mode="markers",
                           marker={"color": J * 1e3, "colorscale": "Viridis", "size": 5, "cmin": 0, "cmax": 3.5,
                                   "colorbar": {"title": "mT/A"}},
                           hovertemplate="%{marker.color:.2f} mT/A<extra></extra>"))
fig.add_trace(go.Scatter(x=[hall[0] / MM], y=[0], mode="markers", marker={"symbol": "x", "size": 10, "color": "#eb6834"},
                         hovertemplate="the Hall element<extra></extra>"))
fig.update_xaxes(title_text="x (mm)", scaleanchor="y")
fig.update_yaxes(title_text="y (mm)")
fig.update_layout(height=420, showlegend=False, margin={"l": 60, "r": 20, "t": 20, "b": 50})
fig
```

Opposite the slot the ferrite carries about four times the slot's flux density,
most at its inner edge, where finite elements put it at 2.9 mT/A. A few cells at
the slot's corners read more: a sliver there saturates first. MnZn ferrite
saturates near 0.5 T cold and 0.4 T hot, so this ring stays linear to about 140
A, where the slot reads only 83 mT. Past that the sensitivity above no longer
holds and the reading falls behind the current: give the ring a `BHCurve` and
solve it with `solve_cells` ([Saturation](saturation.md)). Face charge refuses a
part on a curve.

## The answers

- **Sensitivity:** 0.598 mT/A in a 2 mm slot, 5 % under the textbook μ0 / g.
- **Slot width:** close to 1 / g, 4.5 to 7 % under the estimate from 1 to 4 mm.
- **Busbar off-centre:** at most 0.4 % at 4 mm, the most toward the slot.
- **Saturation:** from about 140 A, opposite the slot, where the ferrite carries
  four to five times the slot's flux density. Past it, solve on a `BHCurve` with
  cells.
