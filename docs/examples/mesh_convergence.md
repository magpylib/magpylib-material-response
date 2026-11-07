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

# Error bars for cells: three parts

`solve_cells` gives one answer per mesh and no error bar, and a finer mesh costs
time: eight times the cells take 7 to 13 times as long on the parts below. Two
questions follow for every part:

- How fine a mesh does it need for the accuracy you want?
- Can you trust `study_cells` to tell you?

`study_cells` solves a model on three meshes and estimates the error of the
finest from how the results change between them;
[How accurate is a result?](../guide/accuracy.md) sets out the method. Its last
argument gives the cells across the thinnest part that responds, one value per
mesh: `(4, 8, 16)` cuts a 2 mm bar into cells of 0.5, 0.25 and 0.125 mm, each
mesh with eight times the cells of the one before.

Three parts from sensor and actuator design follow. Each has a finite-element
reference, good to 0.003 % or better, which gives the true error of every mesh.
The references are the stored ones of the repository's validation suite, and
[Checking a result with finite elements](fem_check.md) shows how to make such a
check. The three studies take about 40 s, so `mesh_convergence_studies.py`, next
to this page, ran them once and stored their results with the references in
`mesh_convergence.json`. The page rebuilds each study with
`MeshStudy.from_values`, which gives the estimate `study_cells` gives.

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mesh_convergence_studies import flux_concentrator, halbach_array, steel_shield

from magpylib_material_response import solve_face_charge
from magpylib_material_response.convergence import MeshStudy

MM = 1e-3  # lengths in metres, written in millimetres
STORED = json.loads(Path("mesh_convergence.json").read_text())


def sizes(value):
    """The size of each entry: |B| per sensor for fields, the value itself
    for gains."""
    value = np.asarray(value)
    return np.linalg.norm(value, axis=-1) if value.ndim == 2 else np.abs(value)


def stored(name):
    """The study `mesh_convergence_studies.py` ran and stored, and its
    finite-element reference."""
    run = STORED[name]
    study = MeshStudy.from_values(run["cells"], [np.asarray(v) for v in run["values"]], run["seconds"])
    return study, np.asarray(run["reference"])


def off(value, reference):
    """Relative error of `value` against the reference, per sensor."""
    return sizes(np.asarray(value) - reference) / sizes(reference)


def report(name, labels, unit="", scale=1.0):
    """Each mesh's values and worst error, then the finest mesh's error per
    sensor against what study_cells estimated."""
    study, reference = stored(name)
    print(f"{'cells':>7} {'solve':>7}  " + "".join(f"{label:>10}" for label in labels) + f"{'worst error':>13}")
    for n, t, value in zip(study.cells, study.seconds, study.values, strict=True):
        cols = "".join(f"{v * scale:10.3f}" for v in sizes(value))
        print(f"{n:7,d} {t:6.1f}s  {cols}  {off(value, reference).max():11.2%}")
    print(f"{'finite elements':>17}" + "".join(f"{v * scale:10.3f}" for v in sizes(reference)) + f"  {unit}")
    print(f"order {study.order:.2f}; the finest mesh's error per sensor:")
    print(f"{'true':>17}" + "".join(f"{e:10.2%}" for e in off(study.values[-1], reference)))
    print(f"{'estimated':>17}" + "".join(f"{e:10.2%}" for e in study.errors))
```

## A flux concentrator

A bar of soft nickel–iron (χ = 4000), 16 × 2 × 2 mm, gathers the Earth's field
into a sensor beside its tip. What matters is the gain: the field at the sensor
over the 50 µT around it. How fine must the bar's cells be for the gain, and
does `study_cells` say so?

```{literalinclude} mesh_convergence_studies.py
:pyobject: flux_concentrator
```

```python
study = study_cells(*flux_concentrator(), (4, 8, 16))
```

```{code-cell} ipython3
:tags: [hide-input]

report("concentrator", ["0.5 mm", "1 mm", "2 mm"], "(gains at 0.5, 1 and 2 mm from the tip)")
```

The gain 0.5 mm from the tip is the hardest to get: 2.0 % off with 4 cells
across the bar, 0.81 % with 8 and 0.29 % with 16, which takes 7.5 s. The order,
1.29, is the rate at which the error falls: as the cell size to that power, so
halving the cells' size cuts it by about 2.4. At every sensor the estimate is
above the true error: 0.44 % against 0.29 % at the tip.

## A Halbach array on back iron

Five 6 mm NdFeB cubes, their polarizations turning by 90° from one to the next
(a Halbach array {cite}`halbach1980`), sit on a 3 mm steel plate. Such arrays
drive linear motors and magnetic encoders, and the field profile 2 mm above them
is what the design is judged by. How fine must the plate's cells be for that
profile?

```{literalinclude} mesh_convergence_studies.py
:pyobject: halbach_array
```

```python
study = study_cells(*halbach_array(), (2, 4, 8))
```

```{code-cell} ipython3
:tags: [hide-input]

array, profile = halbach_array()
magpy.show(array, backend="widget")
```

```{code-cell} ipython3
:tags: [hide-input]

halbach, reference = stored("halbach")
true = off(halbach.values[-1], reference)
for n, t, value in zip(halbach.cells, halbach.seconds, halbach.values, strict=True):
    print(f"{n:6,d} cells in {t:4.1f} s: worst error {off(value, reference).max():.2%}")
print(f"order {halbach.order:.2f}; the finest mesh's error over the whole profile: "
      f"{np.linalg.norm(halbach.values[-1] - reference) / np.linalg.norm(reference):.3%} true, "
      f"{halbach.error:.3%} estimated")
print(f"at the worst point: {true.max():.3%} true, {halbach.errors[np.argmax(true)]:.3%} estimated; "
      f"estimate above the true error at {(true <= halbach.errors).sum()} of {true.size} points")
```

```{code-cell} ipython3
:tags: [hide-input]

x = profile[:, 0] / MM
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=(0.55, 0.45))
for k, name in ((0, "Bx"), (2, "Bz")):
    fig.add_trace(go.Scatter(x=x, y=halbach.values[-1][:, k] * 1e3, name=name, mode="lines+markers"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=100 * halbach.errors, name="estimated error", mode="lines+markers",
                         line={"dash": "dash"}), 2, 1)
fig.add_trace(go.Scatter(x=x, y=100 * true, name="true error", mode="lines+markers"), 2, 1)
fig.update_yaxes(title_text="B 2 mm above (mT)", row=1, col=1)
fig.update_yaxes(title_text="error of the finest mesh (%)", row=2, col=1)
fig.update_xaxes(title_text="x (mm)", row=2, col=1)
fig.update_layout(height=520, margin={"l": 60, "r": 20, "t": 20, "b": 40}, legend={"orientation": "h", "y": -0.15})
fig
```

Coarse cells serve here: 2 cells across the plate already put the profile within
0.59 % everywhere, 4 within 0.23 %, and 8, in 4.9 s, within 0.08 %. The largest
errors sit at the ends of the array. The estimate is above the true error at all
21 points: 0.028 % against 0.022 % over the whole profile.

## A steel sheet shielding a sensor

A 1 mm steel sheet (χ = 1000), 24 × 24 mm, sits 3 mm above a 6 mm magnet and
shields three sensors 2 mm behind it. The field there is what the shield lets
through, so it is small, and the coarse meshes overestimate it. How fine must
the sheet's cells be for that residual field?

```{literalinclude} mesh_convergence_studies.py
:pyobject: steel_shield
```

```python
study = study_cells(*steel_shield(), (1, 2, 4))
```

```{code-cell} ipython3
:tags: [hide-input]

report("shield", ["axis", "6 mm out", "12 mm out"], "mT, |B| behind the sheet", 1e3)
```

These meshes are as fine as a few seconds allow, and not fine enough. On the
axis the residual field falls from 12.4 mT on the coarsest mesh to 6.1 mT on the
finest, which is still 9.9 % above finite elements after 11.5 s. The estimates,
13.6 % and 4.7 %, cover the two inner sensors; at the outer one, 0.23 % falls
short of the true 0.30 %. The error is an estimate, not a bound.

Face charge needs no cells through the sheet's thickness: it puts panels on its
faces ([Face charge](../guide/face_charge.md) gives the settings for thin
sheets).

```{code-cell} ipython3
import time

model, behind = steel_shield()
start = time.perf_counter()
solved = solve_face_charge(model, level=1)
seconds = time.perf_counter() - start
_, reference = stored("shield")
errors = off(magpy.getB(solved, behind), reference)
print(f"face charge, level 1: {solved.face_charge_info['unknowns']:,} unknowns in {seconds:.1f} s, "
      f"errors {', '.join(f'{e:.2%}' for e in errors)}")
```

At level 1 face charge is within 0.24 % at all three sensors, in under a second.

## The answers

- **How fine:** for the concentrator's gain, 16 cells across the bar give 0.29 %
  in 7.5 s. For the Halbach profile, 4 cells across the plate give 0.23 % and 8
  give 0.08 %. Behind the steel sheet, 4 cells across leave 9.9 % after 11.5 s;
  face charge gets 0.24 % in under a second.
- **Trust in `study_cells`:** its estimate was above the true error at 26 of the
  27 sensors. The one miss, 0.23 % against 0.30 %, is behind the sheet, where
  the error is smallest. It is an estimate, not a bound:
  [Checking a result with finite elements](fem_check.md) shows how to check it.

These studies on the solver map: the <a href="../inside/map.html#study">flux
concentrator</a>, the <a href="../inside/map.html#halbach">Halbach array</a> and
the <a href="../inside/map.html#sheet">steel sheet</a>, each path box by box.
