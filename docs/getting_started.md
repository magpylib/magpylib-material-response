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

# Getting started

Magpylib computes the field of magnets whose polarization is fixed and of
currents. Real parts answer back. An iron plate under a magnet becomes a magnet
itself and pulls the flux in. A permanent magnet loses a little of its
polarization in its own field. This package computes that answer, the **material
response**, and hands back a model that Magpylib reads like any other.

```{include} _install.md

```

````{tip}
**Start with an AI assistant.** The package ships an Agent Skill, and these
pages exist as Markdown for agents ([llms.txt](https://magpylib-material-response.readthedocs.io/en/latest/llms.txt)).
Paste this into Claude Code, Cursor or any coding agent, with your model in
place of the brackets:

```text
I use magpylib-material-response to compute magnetic fields of
magnets near soft-magnetic parts. Install it with `uv add
magpylib-material-response` (or pip), then run `uvx
library-skills` to add its Agent Skill to this project, and
read it. The documentation is at
https://magpylib-material-response.readthedocs.io/en/latest/llms.txt.
My model: [the parts and their materials, the magnets or
coils, where I read the field]. Choose the solver as the skill
says, solve, and report the field at my sensors with its error
estimate. Check one result with solve_fem.
```
````

## A magnet on an iron plate

Every length is in metres and every polarization in tesla, as in Magpylib. A
part responds when it has a `susceptibility`. A magnet without one keeps its
polarization.

```{code-cell} ipython3
import time

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go

from magpylib_material_response import solve_cells, solve_face_charge, solve_fem
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3  # lengths are in metres; this writes them in millimetres

magnet = magpy.magnet.Cuboid(
    dimension=(10 * MM, 10 * MM, 5 * MM),
    position=(0, 0, 5.5 * MM),
    polarization=(0, 0, 1.2),
)
plate = magpy.magnet.Cuboid(
    dimension=(40 * MM, 40 * MM, 6 * MM),
    polarization=(0, 0, 0),
)
plate.susceptibility = 1000  # soft iron below saturation
model = magpy.Collection(magnet, plate)

z = np.linspace(8.5, 30, 44) * MM
line = np.column_stack([0 * z, 0 * z, z])  # on the axis, above the magnet
```

## Solve it

`solve_face_charge` returns a copy of the model in which the plate carries its
answer. The copy is an ordinary Magpylib collection: `getB`, `show()` and
`to_json` work on it.

```{code-cell} ipython3
solved = solve_face_charge(model)

B_alone = magpy.getB(magnet, line[0])[2]
B_solved = magpy.getB(solved, line[0])[2]
print(f"Bz 0.5 mm above the magnet: {B_alone * 1e3:.0f} mT alone, {B_solved * 1e3:.0f} mT on the plate")
```

The plate raises the field above the magnet by about 30 %: the magnet's south
face rests on the iron, which gives its flux a short way back below.

```{code-cell} ipython3
:tags: [hide-input]

solved.children[1].style.label = "iron plate, solved"
magpy.show(solved, backend="widget")
```

The plate is coloured by its surface charge, north poles red and south poles
green. Around the magnet it is a south pole: the flux that leaves the magnet's
top comes back into the plate there. Under the magnet, out of sight, it is a
north pole facing the magnet's south face.

## Three solvers, one answer

The package has three solvers for the same model. They differ in what they can
do and what they cost.

```{code-cell} ipython3
results = {}
t0 = time.perf_counter()
results["face charge"] = solve_face_charge(model)
results["face charge", "s"] = time.perf_counter() - t0

t0 = time.perf_counter()
results["cells"] = solve_cells(mesh_uniform(model, 4))  # 4 cells across the plate's thickness
results["cells", "s"] = time.perf_counter() - t0

t0 = time.perf_counter()
results["finite elements"] = solve_fem(model, order=4, observers=line)
results["finite elements", "s"] = time.perf_counter() - t0

reference = magpy.getB(results["finite elements"], line)[:, 2]
print(f"finite elements: {results['finite elements', 's']:5.2f} s, the reference")
for name in ("face charge", "cells"):
    Bz = magpy.getB(results[name], line)[:, 2]
    error = np.max(np.abs(Bz - reference) / np.abs(reference))
    print(f"{name:>15}: {results[name, 's']:5.2f} s, at most {error:.2%} off it")
```

```{code-cell} ipython3
:tags: [hide-input]

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=(0.6, 0.4))
fig.add_trace(go.Scatter(x=z / MM, y=magpy.getB(magnet, line)[:, 2] * 1e3, name="the magnet alone",
                         line={"color": "#8a8984", "width": 2, "dash": "dot"}), 1, 1)
for name, colour, dash in (("finite elements", "#555555", "solid"), ("face charge", "#eb6834", "dash"), ("cells", "#2a78d6", "dot")):
    Bz = magpy.getB(results[name], line)[:, 2]
    fig.add_trace(go.Scatter(x=z / MM, y=Bz * 1e3, name=name, line={"color": colour, "width": 2, "dash": dash}), 1, 1)
    if name != "finite elements":
        fig.add_trace(go.Scatter(x=z / MM, y=100 * np.abs(Bz - reference) / np.abs(reference), showlegend=False,
                                 line={"color": colour, "width": 2, "dash": dash}), 2, 1)
fig.update_yaxes(title_text="Bz on the axis (mT)", type="log", row=1, col=1)
fig.update_yaxes(title_text="off FEM (%)", type="log", row=2, col=1)
fig.update_xaxes(title_text="height above the plate's centre (mm); the magnet's top is at 8 mm", row=2, col=1)
fig.update_layout(height=480, margin={"l": 60, "r": 20, "t": 20, "b": 40}, legend={"x": 0.98, "y": 0.98, "xanchor": "right"})
fig
```

The three solvers agree. Face charge and finite elements are 0.08 % apart, the
error left in finite elements at order 4: at order 5 the two agree to 0.01 %.
Cells, with 4 cells across the plate's thickness, are within 0.6 %.

- **`solve_face_charge`** is the default choice. A part that is linear and
  uniform inside, like soft iron below saturation, carries its whole response as
  a charge on its surface. Only the surface is solved, with no mesh to build.
- **`solve_cells`** cuts the parts into cells and solves one polarization per
  cell. It covers what face charge does not: saturation on a B–H curve, a
  susceptibility that differs along the axes or varies inside a part, and B
  inside a part. Its error falls slowly with the cell count: here 4 cells across
  the plate leave 0.6 %.
- **`solve_fem`** solves the same model by finite elements, with NGSolve. It is
  slower and needs the `fem` extra, and it is independent of the other two: use
  it to check them.

[Choosing a solver](guide/choosing.md) compares them in detail, and
[Inside the solvers](inside/index.md) shows how each one works.

## Next steps

- [Building a model](guide/model.md): susceptibilities, fixed magnets, coils and
  applied fields, and what a solved model gives back.
- [How accurate is a result?](guide/accuracy.md): each solver's own error
  estimate, and how to check one solver against another.
- [Examples](examples/index.md): worked models, each against a finite-element
  reference.
