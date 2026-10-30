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

# Checking a Result with Finite Elements

`mesh_study` says how far the finest mesh is from what the method of moments
converges to ([Mesh convergence](mesh_convergence.md)). It cannot say whether
its own estimate is too small, or whether the method converges at all where a
thin shell shields a field. The `fem` extra answers both: it solves the same
magpylib objects by finite elements, on their own shapes rather than on cells,
with its own error estimate, and `compare` sets the two side by side, sensor by
sensor.

```bash
pip install "magpylib-material-response[fem]"
```

It installs NGSolve (LGPL-2.1), which has no wheels for Linux on ARM or for
musl.

## A magnet beside a soft cube

The model of [Soft magnets](soft_magnets.md): a magnet (χ = 0.5) beside a cube
of soft iron (χ = 3999) turned by 45°, with 41 sensors on a line 1 mm below
them.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from magpylib_material_response.convergence import compare, mesh_study
from magpylib_material_response.fem import fem_study

MM = 1e-3  # lengths in metres, written in millimetres
ORANGE, BLUE = "#eb6834", "#2a78d6"

magnet = magpy.magnet.Cuboid(
    polarization=(0, 0, 1), dimension=(1 * MM, 1 * MM, 2 * MM), position=(0, 0, 0.5 * MM)
)
magnet.susceptibility = 0.5
soft = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1 * MM,) * 3)
soft.rotate_from_angax(45, "y").move((1.5 * MM, 0, 0))
soft.susceptibility = 3999
model = magpy.Collection(magnet, soft)
line = np.linspace((-4, 0, -1), (6, 0, -1), 41) * MM

mom = mesh_study(model, line, (2, 4, 8))
fem = fem_study(model, line)
comparison = compare(mom, fem)

print(f"MoM: {mom.cells[-1]:,} cells in {mom.seconds[-1]:.1f} s, error {mom.error:.2%}")
print(f"FEM: order {fem.orders[-1]}, {fem.ndofs[-1]:,} unknowns in {fem.seconds[-1]:.1f} s, error {fem.error:.2%}")
print(f"{comparison.relative_difference:.2%} apart; {comparison.agree.sum()} of {comparison.agree.size} sensors agree")
```

+++ {"user_expressions": []}

Each method takes seconds here, and FEM's error is a third of MoM's. Two values
agree when they are no farther apart than their two error bars together; where
they disagree, at least one bar is too short.

```{code-cell} ipython3
:tags: [hide-input]

x = line[:, 0] / MM
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=(0.6, 0.4))
for name, value, bar, colour in (
    ("MoM, mesh_study", comparison.a, comparison.a_errors, BLUE),
    ("FEM, fem_study", comparison.b, comparison.b_errors, ORANGE),
):
    fig.add_trace(go.Scatter(x=x, y=value[:, 2] * 1e3, name=name, mode="lines+markers", line={"color": colour, "width": 2},
                             error_y={"type": "data", "array": bar * 1e3, "visible": True}), 1, 1)
size = np.linalg.norm(comparison.b, axis=-1)
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

+++ {"user_expressions": []}

They disagree in two places, and both times MoM's bar is the short one: FEM at
order 5 puts the sensor just under the soft cube's corner, at 1.5 mm, 1.31 %
from MoM's finest mesh, where `mesh_study` estimated 1.15 %, and FEM's order 4
within 0.002 %. Far to the left the field is weak and both differences are below
0.1 %. A disagreement is the cue to refine the mesh there, or to take FEM's
value.

## What `fem_solve` does

`fem_solve(model, order, observers=...)` solves one model at one polynomial
order and returns a solution with `moment`, and with `getB` and `getH`, which
leave out a uniform `H_ext` as a model solved by `apply_demag` does. `fem_study`
solves it at rising orders on one mesh and takes 1.25 times the last change as
the error of the highest, per sensor and in all.

- **Shapes:** cuboids, cylinders, cylinder segments, spheres, tetrahedra and
  triangular meshes, magnets with remanence and a linear susceptibility, also
  per axis.
- **Open space:** the air around the model is cut off at a sphere and mapped
  onto a second one by a Kelvin transform, so the boundary is exact, not a
  distant box.
- **Edges and sensors:** elements shrink geometrically toward the magnets'
  edges, where the field is singular, and are made small at the observers.
- **Currents:** with current sources the magnets take the total potential and
  the air the reduced one, so a coil around soft iron costs no accuracy; the
  coil is not meshed. A current through a magnet, through the hole of a ring
  magnet, or along an open path is refused.
- **Memory is the limit**, not time: the
  [magnet in a steel tube](shield_tube.md) takes 1.9 million unknowns and 8.4 GB
  at order 4, in 50 s.

Both methods take the same susceptibility, so FEM checks the numerics, not the
material model.

## The validation suite

The repository's `validation/` folder holds 20 cases: the models of these pages
and others that are hard for the method of moments, nearly all with a stored
finite-element reference at their sensors. A test holds the method of moments to
its stored accuracy on every pull request, and every night all methods climb
their settings against references made again, so that a change in either method
shows. Magpylib alone, the model without its material response, is the floor: 18
% off on the three cuboids, 721 % under the soft plate, and 100 % wherever only
soft parts answer. The method of moments beats it on every case, by a factor of
11 to 130 at its finest mesh. From a clone of the repository:

```bash
python -m validation.run --cases plate --methods magpylib MoM FEM  # error against time
python -m validation.run --having CylinderSegment --methods MoM    # cases with curved cells
python -m validation.references --check plate                      # make a reference again
```

The nightly run can be started by hand on some cases only, on any branch, as in
`gh workflow run nightly.yml --ref my-branch -f having=CylinderSegment`.
