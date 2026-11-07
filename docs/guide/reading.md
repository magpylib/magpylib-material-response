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

# Reading a solved model

Every solver returns a Magpylib source. Once you have it, four questions remain:
which fields can you read, outside and inside the parts? What force does a
magnet feel? What does the result tell about its own solve? How do you draw it?
This page answers them in that order, on the magnet and iron plate of
[Getting started](../getting_started.md), solved by all three solvers.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response import solve_cells, solve_face_charge, solve_fem
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3  # lengths are in metres; this writes them in millimetres

magnet = magpy.magnet.Cuboid(
    dimension=(10 * MM, 10 * MM, 5 * MM),
    position=(0, 0, 5.5 * MM),
    polarization=(0, 0, 1.2),
    style_label="magnet",
)
plate = magpy.magnet.Cuboid(
    dimension=(40 * MM, 40 * MM, 6 * MM),
    polarization=(0, 0, 0),
    style_label="plate",
)
plate.susceptibility = 1000
model = magpy.Collection(magnet, plate)

by_face_charge = solve_face_charge(model)
by_cells = solve_cells(mesh_uniform(model, 3))  # 3 cells across the plate
by_fem = solve_fem(model)
```

## Fields outside the parts

`magpy.getB`, `getH`, `getJ` and `getM` read every result, alone or with other
sources. The field includes the drives: the magnets, coils and `UniformField` of
the model, not only the parts' response.

```{code-cell} ipython3
results = {"face charge": by_face_charge, "cells": by_cells, "finite elements": by_fem}
above = (0, 0, 10 * MM)  # 2 mm above the magnet
for name, solved in results.items():
    print(f"{name:>15}: Bz = {magpy.getB(solved, above)[2] * 1e3:.1f} mT")
```

A solved model is a snapshot: moving a magnet of it moves that magnet's field,
not the parts' response. Solve again, as [Many solves](loops.md) shows.

## Fields inside a part

What a result gives inside a solved part depends on the solver:

| Solver              | Result                                                             | Inside a solved part                            |
| ------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| `solve_face_charge` | the model, each part replaced by a `ChargeSheet`                   | H; B reads μ0 H and J, M read 0, with a warning |
| `solve_cells`       | a copy of the meshed model, each cell with its solved polarization | B, H, J, M                                      |
| `solve_fem`         | a `FemSolution`, the field of the whole model                      | B, H, J, M, pointwise                           |

```{code-cell} ipython3
import warnings

inside = (1 * MM, 1 * MM, 0)  # in the plate, at the centre of a cell
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for name, solved in results.items():
        B, J = magpy.getB(solved, inside), magpy.getJ(solved, inside)
        print(f"{name:>15}: Bz = {B[2] * 1e3:6.1f} mT, Jz = {J[2] * 1e3:6.1f} mT")
print(caught[0].message)
```

A sheet carries the part's charge on its surface, not its magnetization, so it
warns and gives μ0 H for B. Its `mean_polarization` is the part's mean
polarization, in T; the cells give the same mean:

```{code-cell} ipython3
sheet = by_face_charge.children[1]  # the plate, solved
cells = by_cells.children[1].sources_all
volumes = np.array([c.volume for c in cells])
J_cells = np.array([c.polarization for c in cells])
print(f"face charge, mean_polarization: {sheet.mean_polarization[2] * 1e3:.2f} mT")
print(f"cells, volume-weighted mean   : {J_cells[:, 2] @ volumes / volumes.sum() * 1e3:.2f} mT")
```

The mean is small here: the flux turns sideways in the plate, and flux in
opposite directions cancels in it. For B or J at a point inside a part, mesh it
and use `solve_cells`, or use `solve_fem`. Inside a part with a high
susceptibility the finite-element values are pointwise and show the elements.

Read off the parts' surfaces and off the boundaries between cells: a point on a
shared surface counts as inside both sides, with no warning.
[Cells](cells.md#read-the-result) shows by how much it can be off.

## Forces and torques

`magpy.getFT(sources, targets, meshing=n)` gives the force and the torque on
each target from the sources. On a solved model:

- **The source** is what the solve made of the parts: the plate's sheet, or the
  plate's cells. Not the whole solved model: Magpylib would add the force of the
  target's own field on itself.
- **The target** is what the solve left as it was: a magnet without a
  susceptibility, or a current. A `ChargeSheet` cannot be a target. Here the
  force on the plate is the opposite of the force on the magnet.
- **`meshing`** is the number of cells the target is cut into. Double it and
  check that the force holds.

```{code-cell} ipython3
for n in (1000, 2000, 4000, 8000):
    force, torque = magpy.getFT(sheet, magnet, meshing=n)
    print(f"face charge, meshing={n:>4}: pull {-force[2]:.2f} N")
force, torque = magpy.getFT(by_cells.children[1], magnet, meshing=2000)
print(f"cells, 3 across the plate:   pull {-force[2]:.2f} N")
print(f"torque about the magnet's centre: {np.linalg.norm(torque):.1e} N m")
```

The magnet rests on the plate, and a force at contact needs a finer solve than a
field. The default face-charge level converges for a block aligned with the
plate; 3 cells across the plate are far from it. A run for this page put 6 cells
across at 32.1 N. The pull also moves with `meshing`, by about 0.4 % per
doubling from 2,000 on.
[A holding magnet on a steel plate](../examples/holding_magnet.md) shows how all
three solvers converge on a disc at contact and with a gap.

The torque is taken about the target's centroid, zero here by symmetry; `pivot=`
sets another point. A `FemSolution` holds the target's own field too: subtract
it, as the holding-magnet example does.

## What the result tells about its solve

Each result names its solver in `solved_by`. A face-charge result keeps its
solve in `face_charge_info`, and each `FemSolution` its size and cost:

```{code-cell} ipython3
print(by_face_charge.solved_by, by_cells.solved_by, by_fem.solved_by)
print(by_face_charge.face_charge_info)
print(f"order {by_fem.order}, {by_fem.ndof:,} unknowns, {by_fem.elements:,} elements")
print({step: f"{s:.2f} s" for step, s in by_fem.seconds.items()})
```

| Result             | Attribute                   | What it holds                                                                            |
| ------------------ | --------------------------- | ---------------------------------------------------------------------------------------- |
| any                | `solved_by`                 | `"face_charge"`, `"cells"` or `"fem"`                                                    |
| face charge        | `face_charge_info`          | `unknowns`, `level`, refinement `rounds`, whether it `reused` a factorization, `seconds` |
| each `ChargeSheet` | `mean_polarization`         | the part's mean polarization in T, the first check for [saturation](saturation.md)       |
|                    | `jump_indicator`            | how far the panels are from converged ([How accurate is a result?](accuracy.md))         |
|                    | `part`                      | a copy of the magnet it was solved from, at the sheet's pose, to [solve again](loops.md) |
| each cell          | `polarization`              | its solved polarization in T, remanence included                                         |
| `FemSolution`      | `order`, `ndof`, `elements` | the polynomial order, the unknowns and the elements of the mesh                          |
|                    | `seconds`                   | wall time in s of the `"mesh"`, `"assemble"` and `"solve"` steps                         |
|                    | `moment`                    | the magnetic moment of all magnetization in A m², remanent and induced                   |

The moment checks one solver against another. The magnet's and the plate's, from
face charge, add up to what finite elements give:

```{code-cell} ipython3
moment = (magnet.polarization * magnet.volume + sheet.mean_polarization * plate.volume) / magpy.mu_0
print(f"face charge    : {moment[2]:.4f} A m²")
print(f"finite elements: {by_fem.moment[2]:.4f} A m²")
```

`to_json` saves the results of `solve_face_charge` and `solve_cells`, with
`solved_by` and `face_charge_info` ([Saving and loading](saving.md)). A
`FemSolution` cannot be saved ([Finite elements](fem.md)).

## Drawing it

`magpy.show()` draws all three results with any Magpylib backend. With plotly
and pyvista a sheet is coloured by its charge, a north pole red and a south pole green; parts solved
together share one colour scale, `style.charge.level`, in A/m.
`style.mesh.grid.show` draws the panels. The `"widget"` backend comes with
magpylib-studio and adds field lines; Magpylib's own backends draw the parts
alone.

```{code-cell} ipython3
sheet.style.mesh.grid.show = True
magpy.show(by_face_charge, backend="widget")
```

## In short

- Read fields outside the parts from any result; inside, from cells or finite
  elements. A sheet gives H inside and its part's `mean_polarization`.
- For a force, pass the solved parts as sources and a magnet without a
  susceptibility, or a current, as the target; refine the solve and `meshing`
  until the force holds.
- `solved_by`, `face_charge_info` and a `FemSolution`'s attributes say how a
  result was made; `moment` compares solvers.
- `show()` draws any result; magpylib-studio adds field lines.
