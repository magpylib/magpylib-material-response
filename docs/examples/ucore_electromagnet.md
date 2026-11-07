---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A U-core electromagnet

A coil of 16 windings at 25 A, 400 ampere-turns, sits on the yoke of an iron
U-core. The core gathers the coil's flux and brings it out at its two poles. How
strong is the field 30 mm above the poles, and how much of it is the core's? The
flux density inside the iron needs cells: how fine a mesh do they need, and
should they be cuboids or tetrahedra? And how hard is the iron driven at this
current: does a linear susceptibility hold?

## The model

The core is iron with χ = 999 (μr = 1000), as three cuboids of 20 mm square
section: a 100 mm yoke and two 60 mm legs. The coil comes from the `polyline`
helpers: `create_polyline_fillet` rounds the corners of one square turn, and
`move_grid_along_polyline` sweeps a grid of wire positions, 8 along the yoke in
2 layers, along that turn, one closed polyline per winding. Both functions are
in [`ucore_model.py`](ucore_model.py), which
[the same core at high current](saturation.md) shares:

```{literalinclude} ucore_model.py
:pyobject: coil
```

```{literalinclude} ucore_model.py
:pyobject: core
```

```{code-cell} ipython3
import time

import magpylib as magpy
import numpy as np
from ucore_model import coil, core, core_mesh, line_above_poles

from magpylib_material_response import solve_cells, solve_face_charge, study_cells
from magpylib_material_response.meshing import mesh_TriangularMesh

system = magpy.Collection(core(susceptibility=999), coil(current=25))
magpy.show(system, backend="widget")
```

## How strong is the field above the poles?

The core is linear and each of its parts uniform, so face charge solves it on
its surfaces, with no volume mesh. The field is read on a line 30 mm above the
pole faces, along the U at y = 0:

```{code-cell} ipython3
line = line_above_poles()
solved = solve_face_charge(system)
B = magpy.getB(solved, line)
B_coil = magpy.getB(system.children[1], line)  # the coil alone

print(f"{solved.face_charge_info['unknowns']:,} unknowns, {solved.face_charge_info['seconds']:.2f} s")
print(f"peak Bz: {np.abs(B[:, 2]).max() * 1e3:.3f} mT, coil alone {np.abs(B_coil[:, 2]).max() * 1e3:.3f} mT")
```

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

import plotly.graph_objects as go

COLOURS = {"face charge": "#eb6834", "cuboids": "#2a78d6", "tetrahedra": "#1baf7a"}
reference = json.loads(Path("../../validation/references/ucore.json").read_text())
Bz_ref = np.array(reference["groups"]["30 mm above the poles"]["B"])[:, 0]


def error(B):
    """Relative error of Bz on the line against the finite-element reference."""
    return np.linalg.norm(B[:, 2] - Bz_ref) / np.linalg.norm(Bz_ref)


x = line[:, 0] * 1e3
fig = go.Figure()
fig.add_scatter(x=x, y=B_coil[:, 2] * 1e3, name="coil alone", line={"color": "#999999"})
fig.add_scatter(x=x, y=B[:, 2] * 1e3, name="face charge", line={"color": COLOURS["face charge"], "width": 3})
fig.add_scatter(x=x, y=Bz_ref * 1e3, name="finite-element reference", line={"color": "#555555", "dash": "dot"})
fig.update_xaxes(title_text="x (mm)")
fig.update_yaxes(title_text="Bz 30 mm above the poles (mT)")
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 20, "b": 50}, legend={"orientation": "h", "y": -0.25})
print(f"face charge against the reference: {error(B):.2%}")
print(f"the core multiplies the peak by {np.abs(B[:, 2]).max() / np.abs(B_coil[:, 2]).max():.0f}")
fig
```

The peak is 2.15 mT, 71 times the coil's own 0.030 mT: nearly all of the field
is the core's. The reference is the validation suite's finite-element solve of
this model, good to 0.25 % ([How accurate is a result?](../guide/accuracy.md));
face charge lies within 0.1 % of it, in under a second.

## How fine a mesh do cells need?

Inside a part, face charge gives μ0H without the part's own polarization. The
flux density in the iron needs the core cut into cells and solved with
`solve_cells` ([Cells](../guide/cells.md)). `study_cells` meshes the three
cuboids at 2, 4 and 8 cells across their 20 mm section, eight times the cells
each time, solves each mesh and estimates the error of the finest:

```{code-cell} ipython3
cuboids = study_cells(system, magpy.Sensor(pixel=line), levels=(2, 4, 8))
for n, seconds in zip(cuboids.cells, cuboids.seconds, strict=True):
    print(f"{n:5,} cuboid cells: {seconds:.2f} s")
print(f"estimated error of the finest: {cuboids.error:.1%}")
```

The same U as one closed triangle mesh, `core_mesh` in
[`ucore_model.py`](ucore_model.py), goes to `mesh_TriangularMesh`, which cuts it
into tetrahedra with TetGen {cite}`si2015` (the `tetgen` extra, see
[Getting started](../getting_started.md)):

```{code-cell} ipython3
tetrahedra = []  # (cells, seconds, solved model)
for target in (25, 100, 700):
    model = magpy.Collection(mesh_TriangularMesh(core_mesh(999), target), coil(25))
    t0 = time.perf_counter()
    solved_tets = solve_cells(model)
    tetrahedra.append((len(model.children[0].sources_all), time.perf_counter() - t0, solved_tets))
```

```{code-cell} ipython3
:tags: [hide-input]

runs = {
    "cuboids": [(n, t, error(v)) for n, t, v in zip(cuboids.cells, cuboids.seconds, cuboids.values, strict=True)],
    "tetrahedra": [(n, t, error(magpy.getB(s, line))) for n, t, s in tetrahedra],
}
fig = go.Figure()
for family, rows in runs.items():
    for n, t, e in rows:
        print(f"{n:5,} {family:<10} {t:6.2f} s   {e:6.2%} off the reference")
    fig.add_scatter(
        x=[t for _, t, _ in rows], y=[100 * e for _, _, e in rows], name=family, mode="lines+markers",
        line={"color": COLOURS[family], "width": 2}, text=[f"{n:,} cells" for n, _, _ in rows],
        hovertemplate="%{text}<br>%{x:.2f} s: %{y:.2f} %<extra></extra>",
    )
fig.add_scatter(
    x=[solved.face_charge_info["seconds"]], y=[100 * error(B)], name="face charge", mode="markers",
    marker={"color": COLOURS["face charge"], "size": 10}, hovertemplate="%{x:.2f} s: %{y:.2f} %<extra></extra>",
)
fig.update_xaxes(title_text="solve time (s)", type="log")
fig.update_yaxes(title_text="error of Bz above the poles (%)", type="log")
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 20, "b": 50}, legend={"orientation": "h", "y": -0.25})
fig
```

Cuboid cells converge slowly here: 37, 15 and 4.9 % off at 88, 704 and 5,632
cells, the error falling 2.4 and then 3.1 times with each eightfold refinement.
The study's estimate, 12 %, covers the true 4.9 %: without a reference, it is
the error bar to quote. Tetrahedra are 2.8, 1.6 and 0.8 % off at 53, 229 and
1,491 cells: 229 of them beat 5,632 cuboids, in under a fortieth of the time. On
these three tetrahedral meshes `study_cells` gives no estimate, as their results
don't yet change steadily. Face charge, at 0.08 %, beats both.

## How hard is the iron driven?

The finest tetrahedral solve gives each cell's polarization. Grouped by where
the cells sit:

```{code-cell} ipython3
cells = solved_tets.children[0].sources_all
J = np.linalg.norm([c.polarization for c in cells], axis=1)
x_c, z_c = np.array([c.centroid for c in cells]).T[[0, 2]]
yoke = z_c < 0.01  # the yoke spans z from -10 to 10 mm
for name, where in (
    ("yoke, middle 20 mm", yoke & (np.abs(x_c) < 0.01)),
    ("yoke, ends", yoke & (np.abs(x_c) > 0.03)),
    ("legs", ~yoke),
):
    print(f"{name:<19} mean polarization {J[where].mean():.2f} T")
magpy.show(solved_tets, backend="widget")
```

The middle of the yoke, inside the coil, carries 0.28 T; the legs carry about a
third of that, as flux leaves the iron across the gap between them. Iron
saturates near 1.6 to 2 T, so at 400 ampere-turns the yoke is at under a fifth
of that, and the linear susceptibility holds. The polarization grows with the
current: [the same core at high current](saturation.md) runs it to 16 times the
current, where the linear answer fails.

## The answers

- **Field:** 2.15 mT peak 30 mm above the poles, 71 times what the coil gives
  alone. Face charge finds it within 0.1 % of the finite-element reference, in
  under a second.
- **Mesh:** for cells, tetrahedra. 229 of them put the field above the poles
  within 1.6 %; 5,632 cuboid cells are still 4.9 % off. With cuboids, refine
  eightfold per step, and quote `study_cells`'s estimate.
- **Iron:** 0.28 T in the middle of the yoke, under a fifth of saturation, so
  the linear model holds at this current.

This model on the <a href="../inside/map.html#ucore">solver map</a>: the path it
takes through the solver, box by box.
