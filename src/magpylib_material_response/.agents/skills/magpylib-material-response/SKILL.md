---
name: magpylib-material-response
description: >-
  Use when writing, reviewing, or debugging Python code in which magnetic
  materials respond to a field — soft iron, yokes, shields, flux concentrators,
  pole pieces, susceptibility, demagnetization, saturation, B-H curves — on top
  of Magpylib, even when the user never says "material response". Covers
  choosing between solve_face_charge, solve_cells and solve_fem, setting
  susceptibility and BHCurve, UniformField, reading the solved model, error
  bars, and field_matrix for many solves read at the same points. Names changed
  in 0.9: code written from memory with apply_demag, H_ext or bh_curve warns or
  fails, and reading a solved model in the wrong place gives a wrong field, so
  use this skill before writing the code.
license: BSD-3-Clause
---

# Magpylib-Material-Response

Official skill for Magpylib-Material-Response 0.x, shipped inside the package
and versioned with it. Prefer it over recalled API knowledge: the names in this
skill are from 0.13, and the package is in alpha. For the field computation
itself (`getB`, sources, paths, sensors), use Magpylib's own skill, shipped in
the `magpylib` package from 6.2.

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
- Many solves of the same parts, read at the same points: build a
  [`field_matrix`](#many-solves-read-at-the-same-points) once.
- `to_json(*objects)` saves models and results as a JSON string, and
  `from_json(text)` reads them back.
- Read [Gotchas](#gotchas) before reading a field inside a part.

## Names before 0.9

Old names warn for one release, and go in the next, except the last two rows,
which change with no warning. Write the new ones.

| Before 0.9                       | Now                                                      |
| -------------------------------- | -------------------------------------------------------- |
| `demag.apply_demag`              | `solve_cells`                                            |
| `face_charge.apply_face_charge`  | `solve_face_charge`                                      |
| `fem.fem_solve`, `fem.fem_study` | `solve_fem`, `study_fem`                                 |
| `convergence.mesh_study`         | `study_cells`                                            |
| `demag_kwargs=`                  | `solve_kwargs=`                                          |
| `part.H_ext = (0, 0, 0.1)`       | `UniformField(B=(0, 0, 0.1))`                            |
| `solved_by == "demag"`           | `solved_by == "cells"`                                   |
| `fem_solution.getB(one_row)[0]`  | `fem_solution.getB(one_row)`: now `(3,)`, so `[0]` is Bx |

`H_ext` left its field out of `getB` of the result; `UniformField` includes it.
It also stayed in the global frame, where a `UniformField` turns with its
parent. A model with both raises.

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

A coil of several windings comes from one turn: `move_grid_along_polyline`
sweeps a winding pack along it, one polyline per wire. The pack is `(u, v)` per
wire, relative to the turn's start, in the frame `cross_section_frame(turn)`
returns: `u` along the turn's vector area made perpendicular to its first
segment, `v = t x u`. For a coil turn, `u` is its axis and `v` points outward,
wherever the turn sits. A straight polyline needs `axis=`.

```python
from magpylib_material_response.polyline import (
    create_polyline_fillet,
    move_grid_along_polyline,
)

h = 0.014  # a square turn round a 20 mm core, starting mid-edge
square = [(0, h, 0), (0, h, h), (0, -h, h), (0, -h, -h), (0, h, -h), (0, h, 0)]
turn = create_polyline_fillet(square, max_radius=0.006, N=6)
pack = [(a, b) for a in (-0.0022, 0, 0.0022) for b in (0.0011, 0.0033)]
wires = move_grid_along_polyline(turn, pack)  # (6, n, 3)
coil = magpy.Collection(
    *(magpy.current.Polyline(current=25, vertices=w) for w in wires)
)
```

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
  factorization. For a CAD export (`TriangularMesh`) with bores or fillets, pass
  `widen_curves=True`: its facets lie inside the curves, an error no level
  removes; it is off by default because a mesh may be the real shape.
- `solve_cells` solves the cells it is given, so mesh first: `mesh_uniform`,
  `mesh_all`, `mesh_Cuboid`, `mesh_Cylinder`, `voxelize`, and
  `mesh_TriangularMesh` with the `tetgen` extra. A magnet left unmeshed is one
  cell.
- `solve_fem(model, order=3)` meshes by itself. Start at order 3: on a curve
  order 2 can stop with "did not settle". It refuses a coil through a hole in a
  magnet, and pass `observers=` to refine the mesh where you read. Its result
  cannot be pickled or saved. A CAD part from a STEP file:
  `fem.from_step("part.step", polarization=(0, 0, 0))` (millimetres to metres)
  is a `TriangularMesh` whose exact solid `solve_fem` meshes.

## Many solves, read at the same points

In a loop that solves the same parts again with only their drive changed, a
`field_matrix` gives the sheets' field at fixed points by one product instead of
a `getB`: 0.4 ms against 25 ms for a plate at 2,601 points. Build the
`FieldMatrix` once, from the first solve's sheets, and add the other sources'
field with `getB`.

```python
from magpylib_material_response.face_charge import ChargeSheet, field_matrix


def sheets_of(solved):
    return [s for s in solved.sources_all if isinstance(s, ChargeSheet)]


points = np.column_stack([np.linspace(-0.02, 0.02, 9), np.zeros(9), np.full(9, 0.015)])
first = solve_face_charge(model, reuse=True)
fm = field_matrix(sheets_of(first), points)
magnet.polarization = (0, 0, 1.0)  # only a drive changes
later = solve_face_charge(model, reuse=first)
sheets = sheets_of(later)
others = [s for s in later.sources_all if not any(s is k for k in sheets)]
B = fm(sheets) + magpy.getB(others, points, sumup=True)  # (9, 3)
```

- It refuses sheets with another mesh or pose, never giving a wrong field: a
  part that moved, another `level`, `refine`, or a magnet closer than about 20 %
  of a part's half-side, whose panels then follow it. Build a new one then.
- It holds `3 × points × unknowns` numbers: 62 MB for 2,601 points around a
  plate at level 1. `dtype=np.float32` halves that, within 1e-6 of the peak near
  the sheets but up to 4e-3 of the field far away.
- It keeps no sheets and no solve: `fm()` reads the build's sheets only while
  they exist, so pass the sheets.

## How accurate

Never report a field from one mesh as if it were exact. Refine, and read the
error.

```python
from magpylib_material_response import compare, study_cells

points = np.array([(0.012, 0, 0), (0, 0, 0.012)])
study = study_cells(saturating, points, (3, 5, 9))
study.errors  # (2,)
```

`study_cells` refines the mesh, `study_face_charge` the panel level (2, 3, 4 by
default) and `study_fem` the order, and `compare(a, b)` sets two studies side by
side. Errors are relative to the field read, which includes a `UniformField`, so
they look small where that dominates: pass a function of the solved model to
study the quantity you need. All of these compare models with models, so a
measurement is still the final check.

## Gotchas

- **Do not read exactly on a boundary between cells.** Touching cells count such
  a point as inside each of them: J comes out 2, 4 or 8 times too large, and B
  can be far off, with no warning. The centre of a part with an even number of
  cells across is such a point: a 10 mm cube with χ = 10 in 0.1 T, 6 cells
  across, reads 0.107 T at its centre and 0.29 T 1 µm off it along each axis.
  Use an odd number of cells, and read off every boundary: a step along one axis
  can stay on another, as (0, 0, 1e-6) does.
- **Inside a part solved by face charge,** `getB` gives µ0 H, not B, and `getJ`
  and `getM` give 0, each with a warning. For B inside, use `solve_cells` or
  `solve_fem`. The mean polarization is on each sheet,
  `sheet.mean_polarization`.
- **A magnet's copy leaves its parent behind,** and what it inherits from it: a
  part under a parent with a `susceptibility` becomes fixed, with a warning only
  if it has no polarization. `collection.copy()` and `sheet.part` keep it.
- **Solve the original model, not a result.** A result's polarizations hold the
  response, which a new solve counts as remanence. It warns, unless the cells
  were copied out of it one by one. A sheet's `part` holds the magnet it was
  solved from, at the sheet's pose, to rebuild a model from a face-charge
  result.
- **Changing a solved model does not solve it again,** moving it or setting a
  `UniformField` on it. B moves with it, while the response stays that of the
  old position. Solve per position: paths are refused.
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
- **Thin sheets** cost face charge about 10 area / thickness² unknowns at level
  2, so a square sheet 40 times wider than thick is near the 20,000 limit. Flat
  cuboid cells with at least 3 layers through the thickness are the alternative.
- **A sharp knee can stop the iteration:** a curve rising within about 100 A/m
  can raise "did not converge" or "did not settle", at any order. Give it more
  points or a gentler rise, or beside a strong magnet try `solver="direct"`.
- **`move_grid_along_polyline` took an absolute grid before 0.12.** A grid with
  as many columns as the polyline now raises: give the pack as `(u, v)`,
  relative to the polyline's start, as in [A model](#a-model).
- **magpylib-studio does not draw a `UniformField` yet:** its field lines leave
  it out, and warn, and there is no arrow.
