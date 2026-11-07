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

# Saving and loading a solved model

A solve can take seconds. A field map, a report or a colleague may need the
solved model later, without solving again. This page answers four questions. How
do I save a solved model? Does it give the same field once loaded? What comes
back with it? And what cannot be saved?

## Save and load

`to_json` writes Magpylib objects to a JSON string, and `from_json` reads them
back as a list. The model is the magnet on an iron plate from
[Getting started](../getting_started.md), solved by face charge.

```{code-cell} ipython3
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from magpylib_material_response import from_json, solve_cells, solve_face_charge, to_json
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3  # lengths are in metres; this writes them in millimetres

magnet = magpy.magnet.Cuboid(
    dimension=(10 * MM, 10 * MM, 5 * MM),
    position=(0, 0, 5.5 * MM),
    polarization=(0, 0, 1.2),
)
plate = magpy.magnet.Cuboid(dimension=(40 * MM, 40 * MM, 6 * MM), polarization=(0, 0, 0))
plate.susceptibility = 1000
model = magpy.Collection(magnet, plate)
solved = solve_face_charge(model)

Path("plate.json").write_text(to_json(solved))
loaded = from_json(Path("plate.json").read_text())[0]
```

## The same field

```{code-cell} ipython3
points = np.array([(0, 0, 8.5), (15, 0, 3.5), (0, 0, -4)]) * MM  # above, beside, below
difference = np.max(np.abs(loaded.getB(points) - solved.getB(points)))
print(f"largest difference: {difference:.1e} T")
```

The loaded model gives the same field to the last bit: the file holds the solved
charge, not a recipe to solve again.

## What comes back

```{code-cell} ipython3
sheet = loaded.children[1]
print(loaded.solved_by, loaded.face_charge_info)
print(type(sheet).__name__, "solved from a", type(sheet.part).__name__, "with χ =", sheet.part.susceptibility)
print(f"mean polarization, z: {sheet.mean_polarization[2] * 1e3:.1f} mT, jump indicator {sheet.jump_indicator:.3f}")
```

- **Objects:** the magnets (`Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`,
  `Tetrahedron`, `TriangularMesh`), the currents (`Polyline`, `Circle`,
  `TriangleStrip`, `TriangleSheet`), `Sensor`, `UniformField`, `ChargeSheet` and
  nested collections, so the results of `solve_face_charge` and `solve_cells`
  alike.
- **Their state:** geometry, poses, polarizations and susceptibilities, a
  `BHCurve` included, and styles.
- **The solve's record:** `solved_by`, `face_charge_info`, and each sheet's
  `part`, the magnet it was solved from. A loaded face-charge model can
  therefore be solved again: see
  [Solve a solved model again](loops.md#solve-a-solved-model-again).

A factorization kept with `reuse=True` is not saved: the first solve after
loading factors afresh.

## How large the file is

A sheet holds triangles and their charges, a solved cells model one magnet per
cell. Here the plate solved both ways, with 6 cells across its thickness for
`solve_cells`:

```{code-cell} ipython3
for name, solve in (
    ("face charge", lambda: solve_face_charge(model)),
    ("cells", lambda: solve_cells(mesh_uniform(model, 6))),
):
    start = time.perf_counter()
    result = solve()
    solve_time = time.perf_counter() - start
    text = to_json(result)
    start = time.perf_counter()
    from_json(text)
    load_time = time.perf_counter() - start
    print(f"{name:>11}: {len(text) / 1e6:4.1f} MB, solved in {solve_time:4.2f} s, loaded in {load_time:4.2f} s")
```

```{code-cell} ipython3
:tags: [remove-cell]

Path("plate.json").unlink()
```

Loading is several times faster than solving for both, and the face-charge file
is far smaller. A cells file grows with the cell count, so a fine mesh makes a
large file.

## What cannot be saved

A `FemSolution`, the result of `solve_fem`, cannot be saved: save the values you
read from it ([Finite elements](fem.md#what-it-returns)).

## In short

- `to_json` and `from_json` save and load a solved model, and the loaded one
  gives the same field.
- The file keeps what the solve recorded and each sheet's part, so the model can
  be solved again; it does not keep a factorization.
- A face-charge file is small; a cells file grows with the cells. A
  `FemSolution` cannot be saved.
