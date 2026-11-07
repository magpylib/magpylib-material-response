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

# Building a model

A model is a Magpylib `Collection`, and every solver reads the same one. Before
you solve, three questions: which parts respond, and how? What drives them? What
will a solver refuse? This page answers them in that order, then shows what a
solve returns. Units are SI throughout: metres, tesla, amperes, and A/m for H.

## Parts that respond

A magnet responds when it has a `susceptibility` χ, set on itself or on any
collection above it; the nearest one wins. Its relative permeability is
μr = 1 + χ. Its `polarization` is then its remanence $\mathbf{J}_r$, the
polarization it keeps without a field: zero for soft iron.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response import BHCurve, UniformField, solve_face_charge

MM = 1e-3

iron = magpy.magnet.Cuboid(dimension=(10 * MM,) * 3, polarization=(0, 0, 0))
iron.susceptibility = 1000  # μr = 1001: soft iron below saturation

magnet = magpy.magnet.Cylinder(dimension=(8 * MM, 4 * MM), polarization=(0, 0, 1.2))
magnet.susceptibility = 0.05  # a sintered NdFeB magnet answers a little too
```

A susceptibility takes three forms:

| Value                        | Meaning                                                                       | Solvers    |
| ---------------------------- | ----------------------------------------------------------------------------- | ---------- |
| a number, `1000`             | isotropic and linear: $\mathbf{J} = \mathbf{J}_r + \chi\mu_0\mathbf{H}$       | all three  |
| three numbers, `(3, 3, 0.1)` | along the magnet's own x, y and z axes, turning with it like its polarization | cells, FEM |
| a `BHCurve(H=…, B=…)`        | a measured curve: the material saturates                                      | cells, FEM |

Meshing copies a part's susceptibility to its cells.
`solve_cells(model, susceptibility=…)` overrides them all, with one value or one
per cell.

A curve takes H in A/m, from 0 upward, and B in T. Read it back before you
solve: `polarization(H)` gives J on the curve. [Saturation](saturation.md) shows
how to enter a datasheet's curve and which solver follows it. Typical values for
steels, ferrites and magnets are in [Material values](materials.md).

```{code-cell} ipython3
H = np.r_[0.0, np.geomspace(1e-2, 1e6, 200)]  # A/m, from 0, increasing
mu0H = magpy.mu_0 * H
steel = BHCurve(H=H, B=1000 * mu0H / (1 + 1000 * mu0H / 1.6) + mu0H)  # saturates at 1.6 T
print(f"J at H = 1000 A/m: {steel.polarization(1000):.2f} T")
```

## What drives them

Everything in the model without a susceptibility drives the parts, and stays as
it is:

- **magnets** with their polarization;
- **currents**: `magpy.current.Circle`, `Polyline`, `TriangleStrip`,
  `TriangleSheet`, such as the busbar of
  [the Hall current sensor](../examples/hall_current_sensor.md). Magpylib
  computes their field at the parts; they are never meshed;
- **a `UniformField`**, the applied field $\mathbf{B}_\text{ext}$, given as `B`
  in T or `H` in A/m, such as the Earth's field or the field of a large coil far
  away. Several add up, and each turns with its orientation;
- **any other Magpylib source**, such as a `Dipole`, in `solve_face_charge`
  only.

Here a coil drives the iron cube, in the Earth's field:

```{code-cell} ipython3
coil = magpy.current.Circle(current=50, diameter=20 * MM, position=(0, 0, -8 * MM))
earth = UniformField(B=(0, 0, 50e-6))
model = magpy.Collection(iron, coil, earth)
solved = solve_face_charge(model)

centre = (0, 0, -8 * MM)
print(f"B at the centre of the coil, coil alone: {coil.getB(centre)[2] * 1e3:.2f} mT")
print(f"B at the centre of the coil, with iron : {magpy.getB(solved, centre)[2] * 1e3:.2f} mT")
```

## What is refused

- **Paths.** A model whose objects move along a Magpylib path is refused by
  every solver: one solve is one configuration. Loop over the steps instead;
  [Many solves](loops.md) shows how to do that fast.
- **A curve or a per-axis susceptibility**, in `solve_face_charge`, which needs
  each part linear and isotropic. Use `solve_cells` or `solve_fem`.
- **Sources other than magnets, currents and a `UniformField`**, such as a
  `Dipole`, in `solve_cells` and `solve_fem`.
- **Overlapping parts**, in `solve_fem`. The other solvers don't check, so keep
  magnets apart. Parts may touch: face charge couples touching faces exactly.
- **In `solve_fem`:** a current through a part or through the hole of a ring
  magnet, an open current path, and a `TriangleSheet`.

## What comes back

Each solver returns a copy of the model, as a Magpylib source:

| Solver              | Returns                                                                               | Read inside a part                   |
| ------------------- | ------------------------------------------------------------------------------------- | ------------------------------------ |
| `solve_face_charge` | the collection, each solved part replaced by a `ChargeSheet` in its place             | μ0 H, with a warning; J and M read 0 |
| `solve_cells`       | the collection of cells, each with its solved polarization                            | B, H, J, M                           |
| `solve_fem`         | a `FemSolution`: the field of the whole model, its `moment` and the cost of the solve | B, H, J, M, pointwise                |

Each result's `solved_by` names its solver, and `magpy.getB`, `getH` and
`show()` read all three. [Reading a solved model](reading.md) shows how: fields
inside the parts, forces with `magpy.getFT`, and what each result reports.
`to_json` saves the results of `solve_cells` and `solve_face_charge`, not a
`FemSolution` ([Saving and loading](saving.md)).

A solved model is a snapshot. Moving a magnet of it afterwards moves that
magnet's field, not the response of the parts: solve again.

Don't read a field exactly on a part's surface. Magpylib counts a point on a
shared surface as inside both sides, so the value there can be far off, with no
warning; a micrometre off the surface gives the right value. The boundaries
between the cells of a meshed part are such surfaces too:
[Cells](cells.md#read-the-result) shows by how much.

## In short

Give each part that responds a susceptibility, and leave the drives without one.
Solve once per configuration, and read the result with Magpylib, off the parts'
surfaces.
