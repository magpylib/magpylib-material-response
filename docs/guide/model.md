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

A model is a Magpylib `Collection`. Every solver reads the same model; this page
covers what goes in and what comes out. Units are SI throughout: metres, tesla,
amperes, and A/m for H.

## Parts that respond

A magnet responds when it has a `susceptibility`, set on itself or on any
collection above it; the nearest one wins. Its `polarization` is then its
remanence, the polarization it keeps without a field: zero for soft iron.

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
| a number, `1000`             | isotropic and linear: J = J_r + χ μ0 H                                        | all three  |
| three numbers, `(3, 3, 0.1)` | along the magnet's own x, y and z axes, turning with it like its polarization | cells, FEM |
| a `BHCurve(H=…, B=…)`        | a measured curve: the material saturates                                      | cells, FEM |

A part with an anisotropic susceptibility or a curve is refused by
`solve_face_charge`, which needs a part that is linear and isotropic. A meshed
part's cells copy the susceptibility of their magnet when it is meshed.
`solve_cells(model, susceptibility=…)` overrides them all, with one value or one
per cell.

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
  `TriangleSheet`. They are never meshed: their field at the parts joins the
  right-hand side;
- **a `UniformField`**, given as `B` in T or `H` in A/m, such as the Earth's
  field or the field of a large coil far away. Several add up, and each turns
  with its orientation;
- any other Magpylib source, such as a `Dipole`.

```{code-cell} ipython3
coil = magpy.current.Circle(current=50, diameter=20 * MM, position=(0, 0, -8 * MM))
model = magpy.Collection(iron, coil, UniformField(B=(0, 0, 50e-6)))
solved = solve_face_charge(model)
print(f"B at the centre of the coil: {magpy.getB(solved, (0, 0, -8 * MM))[2] * 1e3:.2f} mT")
```

## What comes back

Each solver returns a copy of the model, as a Magpylib source:

| Solver              | Returns                                                                               | Read inside a part                   |
| ------------------- | ------------------------------------------------------------------------------------- | ------------------------------------ |
| `solve_face_charge` | the collection, each solved part replaced by a `ChargeSheet` in its place             | μ0 H, with a warning; J and M read 0 |
| `solve_cells`       | the collection of cells, each with its solved polarization                            | B, H, J, M                           |
| `solve_fem`         | a `FemSolution`: the field of the whole model, its `moment` and the cost of the solve | B, H, J, M, pointwise                |

Each result's `solved_by` names its solver. `magpy.getB`, `getH`, `show()` and
`to_json` read all three. The drives are in the result too: `getB` of a solved
model includes the field of its magnets, coils and `UniformField`.

A solved model is a snapshot. Moving a magnet of it afterwards moves that
magnet's field, not the response of the parts: solve again.
[Many solves](loops.md) shows how to do that fast.

## What is refused

- **Paths.** A model whose objects move along a path is refused: one solve is
  one configuration. Loop over the steps
  ([#184](https://github.com/magpylib/magpylib-material-response/issues/184)
  records what a path solve would take).
- **Overlapping parts.** Magnets must not overlap. Parts may touch: face charge
  couples touching faces exactly.
- **A current through a part**, or through the hole of a ring magnet, in
  `solve_fem`.

## Reading near a boundary

Don't read a field exactly on a part's surface, or on the boundary between two
cells of a meshed part. Magpylib counts a point on a shared surface as inside
both sides, so the value there can be far off, with no warning. A step of a
micrometre off the boundary gives the right value.
