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

# A holding magnet on a steel plate

How hard does a disc magnet hold on a steel plate? How fast does the pull fall
with a gap: paint, a plastic cover, a bracket that doesn't sit flat? How fine a
solve does the answer need, and how thin can the plate be? Magpylib alone has no
answer here. Without the plate's response there is no force at all, since a
plate that doesn't magnetize pulls on nothing.

One face-charge solve gives the plate's answer. The result is a Magpylib source,
so `magpy.getFT` reads the force on the magnet off it.

## The model

A 10 mm × 5 mm NdFeB disc, polarized at 1.3 T along its axis, on a 40 × 40 × 6
mm plate of soft steel (χ = 1000). The disc's own susceptibility, about 0.05, is
left out, so that the disc is a fixed source that `getFT` can pull on. The model
is a function of the gap, in
[`holding_magnet_study.py`](holding_magnet_study.py), which also computes the
stored results below:

```{literalinclude} holding_magnet_study.py
:pyobject: holding_magnet
```

```{code-cell} ipython3
import json
import warnings
from pathlib import Path

import magpylib as magpy
from holding_magnet_study import holding_magnet

from magpylib_material_response import solve_face_charge

magnet, model = holding_magnet(gap=0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    solved = solve_face_charge(model, refine=2)
print(caught[0].message)

force, torque = magpy.getFT(solved.children[1], magnet, meshing=2000)
print(f"pull at contact: {-force[2]:.1f} N")
```

`getFT` takes the plate's solved sheet as the source and the magnet as the
target, cut into about 2,000 cells. The warning, printed above, is face charge's
own: a disc touching a block cannot have the block's panels follow its round
footprint, so the field there converges slowly. `refine=2` adds two rounds of
finer panels where the charge varies most. How much that matters is the next
section.

```{code-cell} ipython3
:tags: [hide-input]

magpy.show(solved, backend="widget")
```

## How fast does the pull fall with a gap?

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import plotly.graph_objects as go

stored = json.loads(Path("holding_magnet.json").read_text())
gaps, pulls = stored["gaps"], stored["pull"]

fig = go.Figure(go.Scatter(x=gaps, y=pulls, mode="lines+markers", line={"color": "#eb6834", "width": 3},
                           name="face charge", hovertemplate="%{x} mm: %{y:.1f} N<extra></extra>"))
fig.update_xaxes(title_text="gap between magnet and plate (mm)")
fig.update_yaxes(title_text="pull on the magnet (N)", type="log")
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 20, "b": 50}, showlegend=False)
fig
```

The pull halves by 0.7 mm and falls tenfold by 3.3 mm. A tenth of a millimetre
of paint takes an eighth of it away.

## How fine a solve does a force need?

A force converges more slowly than a field: at 0.5 mm the default level puts the
field beside the magnet 1 % off and the pull 4 %. The three solvers converge on
the same pull from different sides:

```{code-cell} ipython3
:tags: [hide-input]

from plotly.subplots import make_subplots

STYLE = {"face charge": "#eb6834", "cells": "#2a78d6", "finite elements": "#555555"}
fig = make_subplots(rows=1, cols=2, subplot_titles=("at contact", "0.5 mm gap"), horizontal_spacing=0.12)
for col, gap in enumerate(("0", "0.5"), start=1):
    rows = stored["convergence"][gap]
    for method, colour in STYLE.items():
        mine = [r for r in rows if r[0].startswith(method)]
        fig.add_trace(go.Scatter(x=[r[1] for r in mine], y=[r[2] for r in mine], mode="lines+markers",
                                 name=method, legendgroup=method, showlegend=col == 1,
                                 line={"color": colour, "width": 2}, text=[r[0] for r in mine],
                                 hovertemplate="%{text}<br>%{x:,} unknowns: %{y:.2f} N<extra></extra>"), 1, col)
fig.update_xaxes(title_text="unknowns", type="log")
fig.update_yaxes(title_text="pull (N)", col=1)
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 40, "b": 50}, legend={"orientation": "h", "y": -0.25})
fig
```

```{code-cell} ipython3
:tags: [hide-input]

for gap in ("0", "0.5"):
    print(f"gap {gap} mm")
    for label, unknowns, value in stored["convergence"][gap]:
        print(f"  {label:<34} {unknowns:>8,} unknowns  {value:6.2f} N")
```

At 0.5 mm face charge and finite elements agree within 0.6 %, cells within 2 %,
and face charge's default is 4 % low. At contact they bracket the pull: face
charge and cells rise toward it, finite elements falls toward it, and they meet
at about 32.4 N. Face charge's default is 14 % low there, cells with 8 across
the plate 3.5 % low. A block magnet aligned with the plate needs none of this:
the panels follow its footprint: the 10 × 10 × 5 mm block of
[Getting started](../getting_started.md), on its plate, pulls 33.3 N at the
default level, which `refine=2` changes by under 0.1 %. For a disc or any other
shape, give face charge `refine=2`, and check that one more round, or a higher
`level`, moves it by less than you need.

## How thin can the plate be?

The steel here is linear: it carries any flux at the same permeability. A real
steel saturates near 1.6 to 2 T. The magnet's flux closes through the plate,
sideways under the magnet's rim, through a wall of 2π r t for a plate t thick.
Its mean flux density there says whether the linear model holds:

```{code-cell} ipython3
:tags: [hide-input]

print("plate   linear pull   flux density under the rim")
for row in stored["thickness"]:
    print(f"{row['thickness']:>3} mm   {row['pull']:8.1f} N   {row['rim']:10.2f} T")
```

The linear pull barely moves as the plate thins, but at 1 mm the flux density
passes what steel can carry: there the real pull is lower, and the linear one
only an upper bound. Give the plate a `BHCurve` and solve it with `solve_cells`,
which follows saturation ([Saturation](saturation.md)). Face charge refuses a
part on a curve.

## The answers

- **Pull:** about 32 N at contact, 19 N at 0.5 mm; halved by 0.7 mm, a tenth by
  3.3 mm.
- **Solve:** for a disc, `refine=2`; the default is 14 % low at contact. A block
  aligned with the plate is right at the default.
- **Plate:** the linear pull holds down to about 2 mm here; at 1 mm the steel
  saturates and the pull is only an upper bound.

Next:

- [Many solves](../guide/loops.md): a sweep over gaps is a solve per gap, as
  here. A sweep that moves only what drives the plate can keep its factorization
  with `reuse=`, but only at one panel layout: without `refine`.
- [How accurate is a result?](../guide/accuracy.md): each solver's own error
  estimate.
