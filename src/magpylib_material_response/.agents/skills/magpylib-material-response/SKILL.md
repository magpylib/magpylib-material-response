---
name: magpylib-material-response
description: >-
  Use when writing, reviewing, or debugging Python code in which magnetic
  materials respond to a field — soft iron, yokes, shields, flux concentrators,
  pole pieces, susceptibility, demagnetization, saturation, B-H curves — on top
  of Magpylib, even when the user never says "material response". Covers
  choosing between solve_face_charge, solve_cells and solve_fem, setting
  susceptibility and BHCurve, UniformField, reading the solved model and error
  bars. Names changed in 0.9: code written from memory with apply_demag, H_ext
  or bh_curve warns or fails, and reading a solved model in the wrong place
  gives a wrong field, so use this skill before writing the code.
license: BSD-3-Clause
---

# Magpylib-Material-Response

Official skill for Magpylib-Material-Response 0.x, shipped inside the package
and versioned with it. Prefer it over recalled API knowledge: the names in this
skill are from 0.9, and the package is in alpha. For the field computation
itself (`getB`, sources, paths, sensors), use Magpylib's own skill.

Check the installed version before writing code:

```python
import magpylib_material_response as mmr

mmr.__version__
```

## Quick reference

- Units are SI, as in Magpylib: metres, tesla, amperes, A/m.
- A model is a `magpy.Collection`. A part with a `susceptibility` responds; a
  magnet without one stays fixed.
- Solve with one of three functions, see [Which solver](#which-solver). Each
  returns a solved model that `magpy.getB` reads.
- A uniform applied field is a [`UniformField`](#a-model) in the collection.
- Check the answer with an error bar, see [How accurate](#how-accurate).
- Read [Gotchas](#gotchas) before reading a field inside a part.

## Names before 0.9

Old names warn for one release, and go in the next. Write the new ones.

| Before 0.9                       | Now                           |
| -------------------------------- | ----------------------------- |
| `demag.apply_demag`              | `solve_cells`                 |
| `face_charge.apply_face_charge`  | `solve_face_charge`           |
| `fem.fem_solve`, `fem.fem_study` | `solve_fem`, `study_fem`      |
| `convergence.mesh_study`         | `study_cells`                 |
| `demag_kwargs=`                  | `solve_kwargs=`               |
| `part.H_ext = (0, 0, 0.1)`       | `UniformField(B=(0, 0, 0.1))` |
| `solved_by == "demag"`           | `solved_by == "cells"`        |

`H_ext` left its field out of `getB` of the result; `UniformField` includes it.
A model with both raises.

## A model

```python
import magpylib as magpy
import numpy as np
from magpylib_material_response import BHCurve, UniformField, solve_face_charge

magnet = magpy.magnet.Cuboid(
    polarization=(0, 0, 1.2), dimension=(0.01, 0.01, 0.005), position=(0, 0, 0.0055)
)
plate = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.04, 0.04, 0.006))
plate.susceptibility = 1000  # soft iron; the magnet, with none, is fixed
model = magpy.Collection(magnet, plate, UniformField(B=(0.01, 0, 0)))

solved = solve_face_charge(model)
B = magpy.getB(solved, [(0, 0, 0.012), (0, 0, 0.02)])  # (2, 3)
```

`susceptibility` is set on a part or on a collection above it, and the nearest
value wins. It takes one of:

| Value               | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| `1000`              | Constant χ, µr = 1 + χ                            |
| `(1000, 1000, 19)`  | Along the part's own axes; turns with it          |
| `BHCurve(H=H, B=B)` | Follows a curve, H in A/m and B in T, from (0, 0) |

A plain table of (H, B) rows raises: name the columns.

```python
H = np.r_[0.0, np.geomspace(1e-2, 1e6, 200)]
mu0H = magpy.mu_0 * H
iron = BHCurve(H=H, B=1000 * mu0H / (1 + 1000 * mu0H / 1.6) + mu0H)  # Js = 1.6 T
```

What drives a part is every other source in the collection: magnets, currents,
and any number of `UniformField(B=...)` or `UniformField(H=...)`, which add up.
A source outside the collection is not seen.

## Which solver

| Use                 | When                                                |
| ------------------- | --------------------------------------------------- |
| `solve_face_charge` | Linear, uniform parts: the default choice. No mesh  |
| `solve_cells`       | A `BHCurve`, per-axis χ, or χ that varies in a part |
| `solve_fem`         | To check the other two, with the `fem` extra        |

```python
from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_uniform

core = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.01, 0.01, 0.01))
core.susceptibility = iron
saturating = magpy.Collection(core, UniformField(B=(0, 0, 1.0)))

solved = solve_cells(mesh_uniform(saturating, 5))  # odd: see Gotchas
J = [c.polarization for c in solved.sources_all if hasattr(c, "polarization")]
```

- `solve_face_charge(model, level=2)`: a higher `level` has finer panels. Start
  magnet arrays and thin parts at `level=2`. It refuses a `BHCurve`, per-axis χ
  and paths, and above `max_unknowns` (20,000 by default). In a loop that only
  changes what drives the parts, `reuse=True` and then `reuse=previous` keep the
  factorization.
- `solve_cells` solves the cells it is given, so mesh first: `mesh_uniform`,
  `mesh_all`, `mesh_Cuboid`, `mesh_Cylinder`, `voxelize`, and
  `mesh_TriangularMesh` with the `tetgen` extra. A magnet left unmeshed is one
  cell.
- `solve_fem(model, order=3)` meshes by itself. Start at order 3: on a curve
  order 2 can stop with "did not settle". It refuses a coil through a hole in a
  magnet, and pass `observers=` to refine the mesh where you read. Its result
  cannot be pickled or saved.

## How accurate

Never report a field from one mesh as if it were exact. Refine, and read the
error.

```python
from magpylib_material_response import compare, study_cells

points = np.array([(0.012, 0, 0), (0, 0, 0.012)])
study = study_cells(saturating, points, (3, 5, 9))
study.errors  # (2,)
```

`study_cells` refines the mesh and `study_fem` the order, and `compare(a, b)`
sets two studies side by side. Face charge has no study function: solve at two
levels and compare. Errors are relative to the field read, which includes a
`UniformField`, so they look small where that dominates: pass a function of the
solved model to study the quantity you need. All of these compare models with
models, so a measurement is still the final check.

## Gotchas

- **Do not read exactly on a boundary between cells.** Touching cells count such
  a point as inside each of them: J comes out 2, 4 or 8 times too large, and B
  can be far off, with no warning. The centre of a part with an even number of
  cells across is such a point: a 10 mm cube with χ = 10 in 0.1 T, 6 cells
  across, reads 0.007 T at its centre and 0.191 T 1 µm away. Use an odd number,
  or read a small step away.
- **Inside a part solved by face charge,** `getB` gives µ0 H, not B, and `getJ`
  and `getM` give 0, each with a warning. No warning comes from a part touching
  another of its material. For B inside, use `solve_cells` or `solve_fem`. The
  mean polarization is on each sheet, `sheet.mean_polarization`.
- **Moving a solved model does not solve it again.** B moves with it, while the
  response stays that of the old position. Solve per position: paths are
  refused.
- **`solved.sources_all` holds the coils and fields too.** Filter before reading
  `.polarization` or `.volume`.
- **Remanence adds to a curve:** a part's `polarization` stays fixed and the
  curve's J adds to it, so 1 T of remanence on a curve saturating at 1.5 T
  reaches 2.5 T. Leave `polarization` at zero on a saturating part.
- **A curve past 5 T warns:** H and B swapped, or the wrong units. H in kA/m is
  not caught, so check that H is in A/m.
- **Thin curved cells:** `solve_cells` on a thin ring of cylinder segments at
  high χ can be far off, and refining does not reliably help. Use face charge if
  it is linear.
- **Parts closer than 0.5 % of their size:** face charge is 1 to 4 % off near
  the gap, and finite elements cannot judge it either.
- **Thin sheets** cost face charge about (size / thickness)² unknowns. Flat
  cuboid cells with at least 3 layers through the thickness are the alternative.
