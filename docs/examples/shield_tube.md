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

# A Magnet in a Steel Tube

A disc magnet sits inside an open steel tube, near one end: a shield, as around
a sensor or a motor's magnet. The model comes from a contribution by Aaditya
Prabhu ([#36](https://github.com/magpylib/magpylib-material-response/pull/36)):
the magnet is 25 mm across and 10 mm tall, polarized along the axis at 1.22 T,
with a susceptibility of 0.05; the tube (χ = 100) has radii of 22.5 and 25 mm
and runs 130 mm, from 15 mm behind the magnet's centre to 115 mm in front of it.

The tube is a `CylinderSegment` that goes all the way round, and `mesh_Cylinder`
cuts it into cells around it, across its wall and along it.

```{code-cell} ipython3
import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cylinder

MM = 1e-3  # lengths in metres, written in millimetres


def model():
    magnet = magpy.magnet.Cylinder(polarization=(0, 0, 1.22), dimension=(25 * MM, 10 * MM))
    magnet.susceptibility = 0.05
    tube = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0), dimension=(22.5 * MM, 25 * MM, 130 * MM, 0, 360), position=(0, 0, 50 * MM)
    )
    tube.susceptibility = 100
    return magnet, tube


def solve(around, along):
    """The model with the tube in cells: `around` x 1 across x `along`."""
    magnet, tube = model()
    cells = mesh_Cylinder(tube, (around, 1, along))
    cells.susceptibility = 100
    return apply_demag(magpy.Collection(magnet, cells))


magnet, tube = model()
magpy.show(magnet, mesh_Cylinder(tube, (12, 1, 8)), backend="widget")
```

+++ {"user_expressions": []}

The field is read on three lines: between the magnet and the wall, 18.45 mm off
the axis, where #36 compared it with FEMM; on the axis in front of the magnet;
and outside the tube, beside the magnet, where the tube has taken most of the
field away. The reference is this package's finite elements at orders 2 to 4,
good to 0.02 % on every line: the stored reference of the repository's
validation case `shield_tube`.

```{code-cell} ipython3
reference = json.loads(Path("../../validation/references/shield_tube.json").read_text())["groups"]
lines = {
    "between magnet and wall": np.linspace((-18.45, 0, 0), (-18.45, 0, 100), 21) * MM,
    "on the axis": np.linspace((0, 0, 10), (0, 0, 100), 10) * MM,
    "outside the tube": np.linspace((30, 0, 0), (60, 0, 0), 7) * MM,
}
magnet_alone = magpy.Collection(model()[0])
solved = {(12, 8): solve(12, 8), (12, 24): solve(12, 24)}

print(f"{'cells':>14}" + "".join(f"{name:>26}" for name in lines))
for (around, along), collection in solved.items():
    errors = []
    for name, points in lines.items():
        exact = np.array(reference[name]["B"])
        errors.append(np.linalg.norm(magpy.getB(collection, points) - exact) / np.linalg.norm(exact))
    print(f"{around} x 1 x {along:<4}" + "".join(f"{e:25.1%} " for e in errors))
```

```{code-cell} ipython3
:tags: [remove-cell]

# the build fails if this page's model or lines drift from the validation case
# whose reference it reads, or if the case changed since that reference
import sys

sys.path.insert(0, str(Path("../..").resolve()))
from validation import references
from validation.cases import CASES

case = CASES["shield_tube"]
references.load(case)
built, groups = case.model()
for mine, theirs in zip((magnet, tube), built.children, strict=True):
    for attribute in ("dimension", "polarization", "position", "susceptibility"):
        np.testing.assert_allclose(getattr(mine, attribute), getattr(theirs, attribute))
assert list(lines) == list(groups)
for name, points in lines.items():
    np.testing.assert_allclose(points, groups[name])
```

+++ {"user_expressions": []}

Inside the tube the method of moments converges: the error between the magnet
and the wall falls from 12 % to 3 % as the cells along the tube triple, and on
the axis from 1.1 % to 0.2 %. Outside, the field is what the tube lets through,
3 mT beside the magnet where the magnet alone gives 21 mT, and the cells are 293
% and 64 % off. Cells along the tube decide it, more than cells around or
across: the flux turns into the wall near the magnet, within a few cells. Cells
of one size fine enough there would number in the thousands, and curved cells
cost more than cuboids. For the field a shield lets through, take face charge
with `refine=1`, within 1.0 % of the finite-element value outside this tube
([face charge](../face_charge.md)), or finite elements.

```{code-cell} ipython3
:tags: [hide-input]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=("between magnet and wall: Bz", "outside the tube: |B|"))
curves = [("magnet alone", magnet_alone, "#8a8984", "dot"), ("96 cells", solved[(12, 8)], "#86b6ef", "solid"),
          ("288 cells", solved[(12, 24)], "#2a78d6", "solid")]
for col, name, coordinate in ((1, "between magnet and wall", 2), (2, "outside the tube", 0)):
    points = lines[name]
    x = points[:, coordinate] / MM

    def shown(B):
        return B[:, 2] * 1e3 if col == 1 else np.linalg.norm(B, axis=-1) * 1e3

    for label, collection, colour, dash in curves:
        fig.add_trace(go.Scatter(x=x, y=shown(magpy.getB(collection, points)), name=label, legendgroup=label,
                                 showlegend=col == 1, mode="lines", line={"color": colour, "dash": dash, "width": 2}), 1, col)
    fig.add_trace(go.Scatter(x=x, y=shown(np.array(reference[name]["B"])), name="finite elements", legendgroup="fem",
                             showlegend=col == 1, mode="markers", marker={"color": "#eb6834", "size": 7}), 1, col)
fig.update_xaxes(title_text="z (mm)", row=1, col=1)
fig.update_xaxes(title_text="x (mm)", row=1, col=2)
fig.update_yaxes(title_text="mT", row=1, col=1)
fig.update_yaxes(title_text="mT", type="log", row=1, col=2)
fig.update_layout(height=400, margin={"l": 60, "r": 20, "t": 40, "b": 40}, legend={"orientation": "h", "y": -0.2})
fig
```
