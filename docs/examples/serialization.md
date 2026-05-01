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

# Saving and loading a demagnetized collection

Demagnetization can be expensive to compute. Once the result is obtained it is
useful to persist the demagnetized state to disk so that further field analyses
can be run later without repeating the simulation.

`to_json` serializes any supported magpylib collection to a JSON string;
`from_json` reconstructs the objects from that string. All geometry
(position, orientation, paths), magnetic polarizations and susceptibilities
are preserved exactly.

## Define sources, mesh and apply demagnetization

```{code-cell} ipython3
from pathlib import Path
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response import to_json
from magpylib_material_response.demag import apply_demag
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
apply_demag(coll_demag, inplace=True)
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
import time

from magpylib_material_response import from_json

# reload — no simulation needed
t0 = time.perf_counter()
coll_loaded = from_json(json_path.read_text())[0]
t_load = time.perf_counter() - t0

print(f"Load from JSON : {t_load * 1e3:.0f} ms")
print(f"Full demag     : {t_demag * 1e3:.0f} ms  ({t_demag / t_load:.0f}× slower)")

magpy.show(*coll_loaded)
```
