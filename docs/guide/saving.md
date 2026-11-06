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

A solve can take seconds to minutes. Save the solved model, and read fields from
it later without solving again.

`to_json` writes a Magpylib collection to a JSON string, and `from_json` reads
it back. It takes the magnets (`Cuboid`, `Cylinder`, `CylinderSegment`,
`Sphere`, `Tetrahedron`, `TriangularMesh`), the currents (`Polyline`, `Circle`,
`TriangleStrip`, `TriangleSheet`), `Sensor`, `UniformField`, the `ChargeSheet`
of `solve_face_charge`, and nested collections: solved cells and solved sheets
alike. Geometry, paths, polarizations and susceptibilities come back exactly.

A `FemSolution` cannot be saved: save the values you read from it.

## Solve and save a model

```{code-cell} ipython3
from pathlib import Path
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response import to_json
from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_all

magpy.defaults.display.backend = "plotly"

# soft magnet in the centre
cube = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.002, 0.002, 0.004))
cube.susceptibility = 3999
cube.style.label = "Iron core"

# current loop around it
loop = magpy.current.Circle(current=10, diameter=0.003)
loop.style.label = "Current loop"

coll = magpy.Collection(cube, loop, style_label="setup")

# mesh and apply demagnetization — clock this
t0 = time.perf_counter()
coll_demag = mesh_all(coll, target_elems=200)
solve_cells(coll_demag, inplace=True)
t_demag = time.perf_counter() - t0
print(f"Demag computation : {t_demag * 1e3:.0f} ms")
```

## Save to disk

```{code-cell} ipython3
json_path = Path("demag_collection.json")
json_path.write_text(to_json(coll_demag, indent=2))
print(f"Saved {json_path.stat().st_size / 1024:.1f} kB → {json_path.name}")
```

## Reload and display

```{code-cell} ipython3
from magpylib_material_response import from_json

# reload — no simulation needed
t0 = time.perf_counter()
coll_loaded = from_json(json_path.read_text())[0]
t_load = time.perf_counter() - t0

print(f"Load from JSON : {t_load * 1e3:.0f} ms")
print(f"Full demag     : {t_demag * 1e3:.0f} ms  ({t_demag / t_load:.0f}× slower)")

magpy.show(*coll_loaded)
```
