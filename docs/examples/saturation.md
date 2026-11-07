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

# A U-core electromagnet at high current

At 25 A per winding, the [U-core electromagnet](ucore_electromagnet.md) is
linear: the middle of its yoke carries 0.28 T, and the field 30 mm above the
poles is 2.15 mT. A constant susceptibility lets iron carry any flux, so at 16
times the current it would put the yoke at 16 times the polarization, 4.4 T.
Real iron saturates near 1.6 to 2 T. A B–H curve in place of the constant
captures that, if the solver follows it: does it? Then, at what current does
this core saturate, and how much field above the poles does that cost? And what
does a curve cost in solve time?

## The curve

The iron here is a stand-in: J = χ0 μ0 H / (1 + χ0 μ0 H / Js), with the core's
χ0 = 999 at low field and Js = 1.6 T. `BHCurve` takes it as B against H, as a
datasheet gives it ([Saturation](../guide/saturation.md) in the guide shows how
to enter one):

```{literalinclude} saturation_study.py
:pyobject: iron_curve
```

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from saturation_study import CHI0, iron_curve
from scipy.optimize import brentq
from ucore_model import coil, core_mesh, line_above_poles

from magpylib_material_response import UniformField, solve_cells
from magpylib_material_response.meshing import mesh_TriangularMesh

COLOURS = {"linear": "#999999", "cells": "#1baf7a", "finite elements": "#555555"}
curve = iron_curve()
mu0H = np.linspace(0, 0.02, 200)  # T
fig = go.Figure()
fig.add_scatter(x=mu0H * 1e3, y=curve.polarization(mu0H / magpy.mu_0), name="the curve")
fig.add_scatter(x=mu0H * 1e3, y=CHI0 * mu0H, name="χ = 999", line={"color": COLOURS["linear"], "dash": "dash"})
fig.update_xaxes(title_text="μ0H inside the iron (mT)")
fig.update_yaxes(title_text="polarization J (T)", range=[0, 2])
fig.update_layout(height=320, margin={"l": 60, "r": 20, "t": 20, "b": 50}, legend={"orientation": "h", "y": -0.3})
fig
```

## Does the solver follow the curve?

A sphere in a uniform applied field stays uniformly polarized while it
saturates, so its answer is exact: μ0H inside is $B_\text{ext}$ − J/3, and J
lies on the curve at that field. A 10 mm sphere, a triangle mesh cut into
tetrahedra, in four applied fields:

```{code-cell} ipython3
:tags: [hide-input]

def icosphere(diameter, splits=2):
    """A sphere as a closed triangle mesh: an icosahedron, its faces split in
    four `splits` times, the new vertices pushed out onto the sphere."""
    t = (1 + 5**0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = list(V / np.linalg.norm(V, axis=1)[:, None])
    for _ in range(splits):
        mid, new = {}, []
        for a, b, c in F:
            m = []
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                if key not in mid:
                    p = V[i] + V[j]
                    V.append(p / np.linalg.norm(p))
                    mid[key] = len(V) - 1
                m.append(mid[key])
            new += [(a, m[0], m[2]), (b, m[1], m[0]), (c, m[2], m[1]), (m[0], m[1], m[2])]
        F = new
    return magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=np.array(V) * diameter / 2, faces=np.array(F)
    )
```

```{code-cell} ipython3
sphere = icosphere(0.01)
sphere.susceptibility = curve
mesh = mesh_TriangularMesh(sphere, 150)
volumes = np.array([c.volume for c in mesh.sources_all])

for B_ext in (0.05, 0.5, 1.0, 2.0):  # T
    solved = solve_cells(magpy.Collection(mesh.copy(), UniformField(B=(0, 0, B_ext))))
    J = np.array([c.polarization[2] for c in solved.children[0].sources_all]) @ volumes / volumes.sum()
    # J on the curve at the field inside, mu0 H = B_ext - J / 3
    exact = brentq(lambda j, b=B_ext: curve.polarization(max(b - j / 3, 0) / magpy.mu_0) - j, 0, 3 * B_ext)
    print(f"{B_ext:4.2f} T applied: J = {J:.4f} T, exact {exact:.4f} T, {J / exact - 1:+.3%}")
```

On 517 tetrahedra the sphere's polarization is within 0.02 % of the exact one,
from the linear range at 0.05 T to deep saturation at 2 T.

## When does the core saturate?

The core goes on the curve as one closed triangle mesh, `core_mesh` in
[`ucore_model.py`](ucore_model.py), cut into tetrahedra: the U-core example
found them converging far faster than cuboid cells. A coarse mesh at 400 A per
winding:

```{code-cell} ipython3
core = mesh_TriangularMesh(core_mesh(susceptibility=curve), target_elems=100)
solved = solve_cells(magpy.Collection(core, coil(current=400)))

B = magpy.getB(solved, line_above_poles())
J = np.linalg.norm([c.polarization for c in solved.children[0].sources_all], axis=1)
print(f"{len(J)} tetrahedra: peak Bz {np.abs(B[:, 2]).max() * 1e3:.1f} mT, J up to {J.max():.2f} T")
```

The study script [`saturation_study.py`](saturation_study.py) runs that from 25
to 400 A, on 229 and 1,491 tetrahedra, and checks it with `solve_fem` at orders
2 and 3, which take up to a minute each. It stores the results in
`saturation.json`:

```{code-cell} ipython3
:tags: [hide-input]

stored = json.loads(Path("saturation.json").read_text())
currents = stored["currents"]
cells = stored["cells"]["1491"]["runs"]
fem = stored["fem"]["3"]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=("peak Bz 30 mm above the poles", "polarization in the yoke's middle"))
fig.add_scatter(x=currents, y=stored["linear"], name="linear, χ = 999", line={"color": COLOURS["linear"], "dash": "dash"}, row=1, col=1)
fig.add_scatter(x=currents, y=[r["peak"] for r in cells], name="cells, 1,491 tetrahedra", mode="lines+markers",
                line={"color": COLOURS["cells"], "width": 3}, row=1, col=1)
fig.add_scatter(x=[r["current"] for r in fem], y=[r["peak"] for r in fem], name="finite elements, order 3",
                mode="markers", marker={"color": COLOURS["finite elements"], "size": 10, "symbol": "diamond"}, row=1, col=1)
fig.add_scatter(x=currents, y=[r["J"]["yoke middle"] for r in cells], mode="lines+markers", showlegend=False,
                line={"color": COLOURS["cells"], "width": 3}, row=1, col=2)
fig.add_hline(y=1.6, line={"color": COLOURS["linear"], "dash": "dot"}, annotation_text="Js = 1.6 T", row=1, col=2)
fig.update_xaxes(title_text="current per winding (A)")
fig.update_yaxes(title_text="mT", rangemode="tozero", row=1, col=1)
fig.update_yaxes(title_text="T", range=[0, 1.8], row=1, col=2)
fig.update_layout(height=400, margin={"l": 60, "r": 20, "t": 40, "b": 50}, legend={"orientation": "h", "y": -0.25})
fig
```

```{code-cell} ipython3
:tags: [hide-input]

linear = dict(zip(currents, stored["linear"], strict=True))
print("current   yoke J   peak Bz: linear  finite elements   cells     ratio to linear")
for run, current in zip(cells, currents, strict=True):
    f = next((r["peak"] for r in fem if r["current"] == current), None)
    fe = f"{f:8.2f} mT" if f else "        –  "
    ratio = f"{f / linear[current]:.2f}" if f else "–"
    print(f"{current:5} A   {run['J']['yoke middle']:5.2f} T   {linear[current]:7.2f} mT"
          f"   {fe}   {run['peak']:6.2f} mT   {ratio}")
```

At 25 A the curve changes nothing: finite elements on the curve agree with the
linear answer within 0.3 %. At 100 A the yoke's middle carries 1.05 T, past the
knee, and the field is 5 % below the linear one. By 200 A the yoke is at 1.54 T,
close to the curve's 1.6 T. At 400 A the field is 14.3 mT, 0.41 times the linear
34.5 mT: the last fourfold rise of the current, from 100 to 400 A, gives only
1.75 times the field.

The cells differ from finite elements by 0.9 to 1.3 % at every current. The two
finite-element orders differ by 1.6 to 1.9 %, so order 2 alone would not be a
check at that level.

## What does a curve cost?

```{code-cell} ipython3
:tags: [hide-input]

for n, study in stored["cells"].items():
    runs = study["runs"]
    print(f"cells, {int(n):,} tetrahedra: linear {study['linear_seconds']:.2f} s; on the curve "
          f"{runs[0]['solves']} to {runs[-1]['solves']} solves, {runs[0]['seconds']:.2f} to {runs[-1]['seconds']:.2f} s")
for order, runs in stored["fem"].items():
    print(f"finite elements, order {order}: {runs[0]['seconds']:.0f} to {runs[-1]['seconds']:.0f} s")
```

The further into saturation, the more solves: 6 at 25 A, 29 at 400 A.
`solve_cells` builds its interaction matrix once, so 29 solves on 1,491
tetrahedra take only about 2.4 times the linear solve. Finite elements assemble
again at every solve: order 3 takes 15 s at 25 A and about a minute at 400 A.
Face charge refuses a part on a curve ([Saturation](../guide/saturation.md) in
the guide).

## The answers

- **Following the curve:** the sphere is within 0.02 % of its exact answer from
  0.05 to 2 T; on the U-core, cells and finite elements differ by 0.9 to 1.3 %.
- **When it saturates:** from about 100 A per winding (1,600 ampere-turns). The
  yoke's middle reaches 1.05 T there and the field is 5 % below the linear
  answer; by 200 A the yoke is near the curve's 1.6 T.
- **What it costs in field:** at 400 A, 14.3 mT above the poles instead of the
  linear 34.5 mT. Above 100 A, four times the current gives 1.75 times the
  field.
- **Solve time:** 6 to 29 solves, about 2.4 times a linear solve with cells; up
  to a minute with finite elements at order 3.

The curve is a stand-in, the same in every solver here: no measurement backs
these numbers. For a real core, enter the steel's datasheet curve.
[Inside saturation](../inside/saturation.md) shows how the solvers follow it.
