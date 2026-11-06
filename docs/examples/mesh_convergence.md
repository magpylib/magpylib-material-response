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

# Mesh convergence

The method of moments gives one answer per mesh and no error bar. `study_cells`
solves a model on meshes four to eight times apart and estimates the error of
the finest from how the results change between them. The estimate is the grid
convergence index {cite}`roache1994`, set out as a procedure by
{cite:t}`celik2008`: Richardson extrapolation {cite}`richardson1911` over the
three finest meshes, the error relative to the finest value, times a factor of
safety of 1.25. It gives one error for the whole quantity and one per sensor,
and none while the results do not yet change steadily.

Three parts from sensor and actuator design follow. The levels passed to
`study_cells`, such as `(4, 8, 16)`, are cells across the thinnest part that
responds: its default mesher, `mesh_uniform`, cuts every such part into cells of
that one size, which the solver multiplies by FFT. Finite elements on the same
models, run separately, show how good the estimate is.

The models and the studies are in `mesh_convergence_studies.py`, next to this
page. The studies take about 40 s on a laptop, so they run once and are stored;
the page rebuilds each with `MeshStudy.from_values`, which gives the estimate
`study_cells` gives.

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go

from mesh_convergence_studies import B0, flux_concentrator, halbach_array, steel_shield

from magpylib_material_response.convergence import MeshStudy

MM = 1e-3  # lengths in metres, written in millimetres
STORED = json.loads(Path("mesh_convergence.json").read_text())


def stored(name):
    """The study `mesh_convergence_studies.py` ran and stored."""
    run = STORED[name]
    return MeshStudy.from_values(run["cells"], [np.asarray(v) for v in run["values"]], run["seconds"])


def report(study, labels, unit, scale=1.0):
    print("  cells  " + "  ".join(f"{label:>12}" for label in labels) + "   solve")
    for n, value, t in zip(study.cells, study.values, study.seconds, strict=True):
        cols = np.linalg.norm(value, axis=-1) if np.ndim(value) == 2 else value
        print(f"{n:7d}  " + "  ".join(f"{v * scale:9.3f} {unit}" for v in cols) + f" {t:6.1f} s")
    if study.order is None:
        print(f"not steady yet: no estimate; last change {study.last_change:.1%}")
    else:
        print(f"order {study.order:.2f}; error of the finest mesh {study.error:.2%}, per entry:")
        print("  " + ", ".join(f"{label} {e:.2%}" for label, e in zip(labels, study.errors, strict=True)))
```

+++ {"user_expressions": []}

## A flux concentrator

A bar of soft nickel–iron (susceptibility 4,000), 16 × 2 × 2 mm, gathers the
Earth's field into a sensor beside its tip. What matters is the gain: the field
at the sensor over the 50 µT around it.

```{literalinclude} mesh_convergence_studies.py
:pyobject: flux_concentrator
```

```{code-cell} ipython3
concentrator = stored("concentrator")  # study_cells(*flux_concentrator(), (4, 8, 16))
report(concentrator, ["0.5 mm", "1 mm", "2 mm"], "", 1)
```

+++ {"user_expressions": []}

The gain 0.5 mm from the tip falls from 4.34 to 4.27 as the cells shrink, while
the gain 2 mm out rises. Finite elements put the gains at 4.259, 3.347 and
2.168, to within 0.001 %. Against these, the finest mesh here is 0.29 %, 0.06 %
and 0.11 % off; `study_cells` said 0.44 %, 0.13 % and 0.17 %.

## A Halbach array on back iron

Five 6 mm NdFeB cubes, their polarizations turning by 90° from one to the next
(a Halbach array {cite}`halbach1980`), sit on a steel plate. Such arrays drive
linear motors and magnetic encoders, and the field profile above them is what
the design is judged by.

```{literalinclude} mesh_convergence_studies.py
:pyobject: halbach_array
```

```{code-cell} ipython3
array, profile = halbach_array()
x = profile[:, 0] / MM
halbach = stored("halbach")  # study_cells(array, profile, (2, 4, 8))
print(f"cells {halbach.cells.tolist()}, order {halbach.order:.2f}, solve of the finest {halbach.seconds[-1]:.1f} s")
print(f"error of the whole profile {halbach.error:.3%}, of the worst point {np.nanmax(halbach.errors):.3%}")
magpy.show(array, backend="widget")
```

```{code-cell} ipython3
:tags: [hide-input]

finest = halbach.values[-1] * 1e3
band = halbach.errors * np.linalg.norm(finest, axis=-1)
fig = go.Figure()
for k, name in ((0, "Bx"), (2, "Bz")):
    fig.add_trace(go.Scatter(x=x, y=finest[:, k], name=name, mode="lines+markers",
                             error_y={"type": "data", "array": band, "visible": True}))
fig.update_layout(xaxis_title="x (mm)", yaxis_title="B 2 mm above the array (mT)", height=360,
                  margin={"l": 60, "r": 20, "t": 20, "b": 40})
fig
```

+++ {"user_expressions": []}

The bars are the estimated error at each point, too small to see on the scale of
the field: 0.03 % for the whole profile, 0.1 % at the worst point, at the end of
the array. Against finite elements, good to 0.001 %, the finest mesh here is
0.02 % off overall and 0.08 % at that point, under the estimate at all 21
points.

## A steel sheet shielding a sensor

A 1 mm steel sheet (susceptibility 1,000), 24 × 24 mm, sits 3 mm above a 6 mm
magnet and shields three sensors 2 mm behind it. The field there is what is left
after the shield, so the coarse meshes overestimate it: the error has to be
measured against the finest value, not against the larger coarse ones.

```{literalinclude} mesh_convergence_studies.py
:pyobject: steel_shield
```

```{code-cell} ipython3
shield = stored("shield")  # study_cells(*steel_shield(), (1, 2, 4))
report(shield, ["axis", "6 mm out", "12 mm out"], "mT", 1e3)
```

+++ {"user_expressions": []}

On the axis the residual field falls from 12.4 mT on the coarsest mesh to 6.1
mT. Finite elements put the three sensors at 5.53, 5.85 and 8.80 mT, to within
0.002 %. Against these, the finest mesh here is 9.9 %, 3.1 % and 0.30 % off, and
`study_cells` said 13.6 %, 4.7 % and 0.23 %. The last falls 0.07 % short: the
error is an estimate, not a bound.

<details>
<summary>The reference values: finite elements at orders 3 to 5, with the
<code>fem</code> extra (under 3 minutes, not run here)</summary>

The same models are cases of the repository's validation suite
(`validation/cases.py`), whose stored references these are.

```python
from magpylib_material_response import compare
from magpylib_material_response import study_fem

bar_model, _ = flux_concentrator()
sensors = np.array([(8.5, 0, 0), (9, 0, 0), (10, 0, 0)]) * MM
fem = [
    study_fem(bar_model, sensors, orders=(3, 4, 5)),
    study_fem(*halbach_array(), orders=(3, 4, 5)),
    study_fem(*steel_shield(), orders=(3, 4, 5)),
]
print(f"gains {fem[0].estimate[:, 0] / B0}, within {fem[0].errors}")
for study, reference in zip((halbach, shield), fem[1:], strict=True):
    comparison = compare(study, reference)
    off = comparison.difference / np.linalg.norm(comparison.b, axis=-1)
    print(f"finest mesh here off by {off}, estimated {study.errors}")
```

</details>

## Notes

- **Choose meshes 4 to 8 times apart.** Meshes closer than that change little
  without being converged. `study_cells` warns when the cell count grows by less
  than 2.2 times, cells 1.3 times smaller, the least {cite:t}`celik2008`
  recommend. The finest solve takes most of the time.
- **No estimate** means the three finest results do not change steadily yet: the
  coarsest is too coarse for the error to follow a power law. Refine once more.
- **What it measures** is the error of the mesh. A faceted surface stays
  faceted, and a linear susceptibility does not saturate: neither shows here.
- **Tetrahedra** follow the surface they are given: on a detailed one, such as a
  64-sided cylinder, TetGen made several thousand cells whatever the target. The
  warning above then says so.
- **One physical quantity per study:** the entries of a function's result share
  one norm, so a force and a field in one array would be weighed against each
  other.
