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

# A magnet in a steel tube

A disc magnet sits inside an open steel tube, near one end: a shield, as around
a sensor or a motor's magnet. The model comes from a contribution by Aaditya
Prabhu ([#36](https://github.com/magpylib/magpylib-material-response/pull/36)):
the magnet is 25 mm across and 10 mm tall, polarized along the axis at 1.22 T,
with a susceptibility of 0.05; the tube (χ = 100) has radii of 22.5 and 25 mm
and runs 130 mm, from 15 mm behind the magnet's centre to 115 mm in front of it.

Two questions: how much field does the tube let through, and which solve can be
trusted there? The field that gets through is small, the difference of the
magnet's and the tube's, and that is where solvers part ways.

## The model

```{literalinclude} shield_tube_study.py
:pyobject: shield_tube
```

```{code-cell} ipython3
import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from shield_tube_study import LINES, shield_tube

from magpylib_material_response import solve_face_charge

magnet, tube = shield_tube()
t0 = time.perf_counter()
solved = solve_face_charge(magpy.Collection(magnet, tube), level=2)
print(f"{solved.face_charge_info['unknowns']:,} unknowns, {time.perf_counter() - t0:.1f} s")
```

```{code-cell} ipython3
:tags: [hide-input]

solved.children[1].style.opacity = 0.35  # see the magnet through the tube
magpy.show(solved, backend="widget")
```

+++ {"user_expressions": []}

## How much gets through?

The field is read on three lines: inside the tube, 18.45 mm off the axis and
along it, where #36 compared it with FEMM; on the axis in front of the magnet;
and outside the tube, beside the magnet. The reference is this package's finite
elements at orders 2 to 4, good to 0.02 % on every line: the stored reference of
the repository's validation case `shield_tube`.

```{code-cell} ipython3
:tags: [hide-input]

reference = json.loads(Path("../../validation/references/shield_tube.json").read_text())["groups"]
for name, coordinate, label in (("outside the tube", 0, "x"), ("on the axis", 2, "z")):
    points = LINES[name]
    alone = np.linalg.norm(magpy.getB(magnet, points), axis=1)
    shielded = np.linalg.norm(np.array(reference[name]["B"]), axis=1)
    print(f"{name}: the tube cuts the field by " + ", ".join(
        f"{a / b:.1f}x at {label} = {p[coordinate] / 1e-3:.0f} mm" for p, a, b in zip(points[::3], alone[::3], shielded[::3])))
```

Beside the magnet the tube cuts the field sevenfold, from 21 to 3 mT at 30 mm
from the axis, and threefold at 60 mm. On the axis in front of the magnet,
nothing changes within 10 mm; the cut grows to 4.5-fold at 70 mm and falls back
toward the open end. Inside the tube, near its wall, the field changes by less
than half.

## Which solve can be trusted there?

The finer face-charge solves take 9 and 22 s here, so
[`shield_tube_study.py`](shield_tube_study.py) stores them; the cells run on
this page.

```{code-cell} ipython3
:tags: [hide-input]

from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_Cylinder

MM = 1e-3
stored = json.loads(Path("shield_tube.json").read_text())


def in_cells(around, along):
    """The model with the tube in cells: `around` x 1 across x `along`."""
    magnet, tube = shield_tube()
    cells = mesh_Cylinder(tube, (around, 1, along))
    cells.susceptibility = 100
    return solve_cells(magpy.Collection(magnet, cells))


fields = {"face charge, level=2": {k: magpy.getB(solved, p) for k, p in LINES.items()}}
for name, label in (("default", "face charge, default"), ("refine=1", "face charge, refine=1")):
    fields[label] = {k: np.array(v) for k, v in stored[name]["B"].items()}
for around, along in ((12, 8), (12, 24)):
    collection = in_cells(around, along)
    fields[f"cells, {around} x 1 x {along}"] = {k: magpy.getB(collection, p) for k, p in LINES.items()}
fields["magnet alone"] = {k: magpy.getB(shield_tube()[0], p) for k, p in LINES.items()}

print(f"{'':>24}{'magnet to wall':>16}{'axis':>8}{'outside':>10}")
for label, by_line in fields.items():
    errors = []
    for name in LINES:
        exact = np.array(reference[name]["B"])
        errors.append(np.linalg.norm(by_line[name] - exact) / np.linalg.norm(exact))
    print(f"{label:>24}{errors[0]:16.1%}{errors[1]:8.1%}{errors[2]:10.1%}")
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
for mine, theirs in zip(shield_tube(), built.children, strict=True):
    for attribute in ("dimension", "polarization", "position", "susceptibility"):
        np.testing.assert_allclose(getattr(mine, attribute), getattr(theirs, attribute))
assert list(LINES) == list(groups)
for name, points in LINES.items():
    np.testing.assert_allclose(points, groups[name])
```

+++ {"user_expressions": []}

Inside the tube face charge is close from level 2 on: within 1.4 % on the line
along the tube and at the reference's own accuracy, 0.02 %, on the axis. Cells
need 288 to get within 2.8 % there. Outside, the field is what the tube lets
through, 3 mT beside the magnet where the magnet alone gives 21 mT. There face
charge needs its finer panels: 15 % off at level 2, 5.6 % at the default level
3, and 1.0 % with `refine=1`, which adds panels where the charge varies most.

Cells are 293 % and 64 % off there. The flux turns into the wall near the
magnet, within a few cells along the tube, and tripling the cells along it
leaves them still 64 % off outside. For the field a shield lets through, take
face charge with `refine=1`, or [finite elements](fem_check.md).

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=("between magnet and wall: Bz", "outside the tube: |B|"))
curves = [("magnet alone", "#8a8984", "dot"), ("cells, 12 x 1 x 24", "#2a78d6", "solid"),
          ("face charge, refine=1", "#eb6834", "solid")]
for col, name, coordinate in ((1, "between magnet and wall", 2), (2, "outside the tube", 0)):
    x = LINES[name][:, coordinate] / MM

    def shown(B):
        return B[:, 2] * 1e3 if col == 1 else np.linalg.norm(B, axis=-1) * 1e3

    for label, colour, dash in curves:
        fig.add_trace(go.Scatter(x=x, y=shown(fields[label][name]), name=label, legendgroup=label,
                                 showlegend=col == 1, mode="lines", line={"color": colour, "dash": dash, "width": 2}), 1, col)
    fig.add_trace(go.Scatter(x=x, y=shown(np.array(reference[name]["B"])), name="finite elements", legendgroup="fem",
                             showlegend=col == 1, mode="markers", marker={"color": "#555555", "size": 7}), 1, col)
fig.update_xaxes(title_text="z (mm)", row=1, col=1)
fig.update_xaxes(title_text="x (mm)", row=1, col=2)
fig.update_yaxes(title_text="mT", row=1, col=1)
fig.update_yaxes(title_text="mT", type="log", row=1, col=2)
fig.update_layout(height=400, margin={"l": 60, "r": 20, "t": 40, "b": 40}, legend={"orientation": "h", "y": -0.2})
fig
```

## The answers

- **What gets through:** beside the magnet, a seventh to a third of its field,
  falling with distance; on the axis in front of it, under a fourth at 70 mm,
  less toward the open end.
- **Which solve:** face charge with `refine=1`, within 1.0 % of finite elements
  outside the tube in 22 s. Cells converge inside the tube and fail outside it.
