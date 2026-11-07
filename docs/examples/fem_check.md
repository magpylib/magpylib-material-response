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

# Checking a result with finite elements

`study_cells` puts an error bar on a cells result
([Error bars for cells](mesh_convergence.md)), but it takes that bar from the
cells' own results. Where more cells close the error only slowly, the bar can be
too short, and nothing in the cells says so. Finite elements solve the same
Magpylib objects by an independent method, with an error bar of their own
([Finite elements](../guide/fem.md) describes the solver; it needs the `fem`
extra, see [Getting started](../getting_started.md)). Three questions follow:

- How do you run the check?
- Where do the two disagree, whose bar is too short, and what then?
- When is a check worth it?

## The model: a soft edge close to the sensors

A test model, made hard on purpose. A magnet, 1 × 1 × 2 mm and 1 T, answers its
own field strongly (χ = 0.5). Beside it, a 1 mm cube of soft iron (χ = 3999,
μr = 4000) is turned by 45°, so that one of its edges points down at a line of
41 sensors. That edge is 0.29 mm above the sensors, and another edge of the cube
is 0.29 mm from the magnet. The field changes fastest at such edges and in such
gaps, and that is where a discretized solver struggles.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response import compare, study_cells, study_fem

MM = 1e-3  # lengths in metres, written in millimetres

magnet = magpy.magnet.Cuboid(
    polarization=(0, 0, 1), dimension=(1 * MM, 1 * MM, 2 * MM), position=(0, 0, 0.5 * MM)
)
magnet.susceptibility = 0.5
soft = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1 * MM,) * 3)
soft.rotate_from_angax(45, "y").move((1.5 * MM, 0, 0))
soft.susceptibility = 3999
model = magpy.Collection(magnet, soft)
line = np.linspace((-4, 0, -1), (6, 0, -1), 41) * MM
```

## How do you run the check?

Run both studies on the same model and sensors, and pass them to `compare`.
`study_cells` takes the cells across the thinnest part, here 2, 4 and 8 across
the 1 mm cube. `study_fem` raises the polynomial order on one mesh, 2, 3 and 4
by default.

```{code-cell} ipython3
cell_study = study_cells(model, line, (2, 4, 8))
fem_study = study_fem(model, line)
comparison = compare(cell_study, fem_study)

print(f"cells: {cell_study.cells[-1]:,} cells in {cell_study.seconds[-1]:.1f} s, "
      f"error {cell_study.error:.2%}")
print(f"finite elements: order {fem_study.orders[-1]}, {fem_study.ndofs[-1]:,} unknowns "
      f"in {fem_study.seconds[-1]:.1f} s, error {fem_study.error:.2%}")
print(f"{comparison.relative_difference:.2%} apart; "
      f"{comparison.agree.sum()} of {comparison.agree.size} sensors agree")
```

Each takes seconds, and over the whole line the two are 0.25 % apart, within the
cells' estimated 0.39 %. Two values agree when they are no farther apart than
their two error bars together; 32 of the 41 sensors do.

## Where do they disagree?

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ORANGE, BLUE = "#eb6834", "#2a78d6"
x = line[:, 0] / MM
size = np.linalg.norm(comparison.b, axis=-1)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=(0.6, 0.4))
for name, value, bar, colour in (
    ("cells, study_cells", comparison.a, comparison.a_errors, BLUE),
    ("finite elements, study_fem", comparison.b, comparison.b_errors, ORANGE),
):
    fig.add_trace(go.Scatter(x=x, y=value[:, 2] * 1e3, name=name, mode="lines+markers", line={"color": colour, "width": 2},
                             error_y={"type": "data", "array": bar * 1e3, "visible": True}), 1, 1)
fig.add_trace(go.Scatter(x=x, y=100 * comparison.difference / size, name="apart", mode="lines", line={"color": "#555555", "width": 2}), 2, 1)
fig.add_trace(go.Scatter(x=x, y=100 * (comparison.a_errors + comparison.b_errors) / size, name="both error bars",
                         mode="lines", line={"color": "#999999", "width": 2, "dash": "dash"}), 2, 1)
off = ~comparison.agree
fig.add_trace(go.Scatter(x=x[off], y=100 * comparison.difference[off] / size[off], name="disagree", mode="markers",
                         marker={"color": "#e34948", "size": 9}), 2, 1)
fig.update_yaxes(title_text="Bz (mT)", row=1, col=1)
fig.update_yaxes(title_text="% of |B|", type="log", row=2, col=1)
fig.update_xaxes(title_text="x (mm)", row=2, col=1)
fig.update_layout(height=520, margin={"l": 60, "r": 20, "t": 20, "b": 40}, legend={"orientation": "h", "y": -0.18})
fig
```

```{code-cell} ipython3
:tags: [hide-input]

labels = ("x (mm)", "|B| (mT)", "apart", "cells' bar", "finite elements' bar")
print(" ".join(f"{label:>{width}}" for label, width in zip(labels, (7, 10, 9, 11, 21), strict=True)))
for k in np.flatnonzero(~comparison.agree):
    print(f"{x[k]:7.2f} {size[k] * 1e3:10.1f} {comparison.difference[k] / size[k]:9.3%} "
          f"{comparison.a_errors[k] / size[k]:11.3%} {comparison.b_errors[k] / size[k]:21.3%}")
```

Nine sensors disagree, in two regions, and both times the cells' bar is the
short one: the finite-element bars there are all below 0.03 %.

- **Under the cube's lower edge**, at x = 1.25 and 1.5 mm, the finest cells mesh
  is 0.25 % and 1.31 % from finite elements, where `study_cells` estimated
  0.14 % and 1.15 %. Both sensors are within 0.4 mm of the edge: three of the
  finest cells, which are 0.125 mm across.
- **Far to the left**, from x = −4 to −2.5 mm, the field is weak, 2 to 7 mT, and
  the two are at most 0.064 % apart.

What to do with a disagreement depends on its size against the accuracy you
need. Below 0.1 %, as on the left, it rarely changes a design. At the edge, take
the finite-element value, or refine the cells and run the study again. In a run
not repeated on this page, 16 cells across the cube (12,288 cells, 38 s on an
Apple M4 Pro) brought the cells' value at x = 1.5 mm within 0.45 % of finite
elements, inside the two bars together. Here finite elements are the cheaper way
to the accurate value.

## When is a check worth it?

- **Sensors close to a soft part's edge or in a narrow gap**, within a few of
  the finest cells: the field changes fast there, and the cells' bar can be
  short, as here.
- **Accuracy you need close to the cells' bar.** Where the bar is far below what
  you need, as on most of this line, the cells' answer stands on its own.
- **Not for the material model.** Both solvers take the same susceptibility, so
  the check covers the numerics only, not whether a linear χ fits your steel.

This model is the case `soft_magnets` of the repository's validation suite: 23
cases, 21 of them with stored finite-element references. On every pull request a
test runs cells and face charge at the settings `validation/pr_check.json`
stores, on 21 of the cases, and fails when an error or the size of a problem
grows. [How accurate is a result?](../guide/accuracy.md) describes the suite.

## The answers

- **How:**
  `compare(study_cells(model, sensors, levels), study_fem(model, sensors))`, on
  the same sensors; each study takes seconds here.
- **Where:** 9 of the 41 sensors disagree, and the cells' bar is the short one.
  Two sit under the soft edge, 1.31 % apart against an estimate of 1.15 %; seven
  sit in the weak field far to the left, at most 0.064 % apart.
- **What then:** below the accuracy you need, nothing. Otherwise take the
  finite-element value, a few seconds away, or refine the cells: at 16 across,
  the sensor under the edge agrees, after 50 s.
- **When:** where sensors read within a few cells of a soft edge or in a narrow
  gap, and the accuracy you need is close to the cells' bar.

This model on the <a href="../inside/map.html#femcheck">solver map</a>: the path
it takes through the solver, box by box.
