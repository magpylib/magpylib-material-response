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

# Soft Magnets with Mesh Import

+++

This code demonstrates demagnetization calculations using the Magpylib library,
focusing on the analysis of different meshing strategies.

In particular, it compares the built-in _autoMesh_ functionality provided by
magpylib-material-response with custom meshes generated from external CAD
software.

The study evaluates how mesh structure influence the demagnetization results.
Custom meshes are imported in `.msh` and `.inp` formats and are restricted to
cuboid (hexahedral) elements, which allows direct translation into
Magpylib-compatible cuboids.

The computed magnetic fields from these meshing approaches are compared to
baseline Magpylib results without demagnetization, enabling an assessment of
accuracy and consistency across mesh generation methods.

+++ {"user_expressions": []}

## Define magnetic sources with their susceptibilities - Custom Mesh

```{code-cell} ipython3
import json

import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_all
from magpylib_material_response.customMesh import import_mesh

magpy.defaults.display.backend = "plotly"

# hard magnet
cube1 = import_mesh("cuboid1.inp",scaling=1e-3, polarization=(0,0,1), succeptibility=0.5)
cube1.move((0, 0,  -0.0005))
cube1.style.label = f"Hard cuboid magnet, susceptibility=0.5"

# soft magnet
cube2 = import_mesh("cuboid2.inp",scaling=1e-3, polarization=(0,0,0), succeptibility=3999)
cube2.style.label = f"Soft cuboid magnet, susceptibility=3999"

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-0.004, 0, z), (0.006, 0, z), 1001),
        style_label=f"Sensor, z={z}m",
    )
    for z in (-0.001, -0.003, -0.005)
]

# collection of all cells
coll_customMesh = magpy.Collection(*cube1, *cube2, *sensors,override_parent=True)

magpy.show(coll_customMesh)
```

+++ {"user_expressions": []}

## Compute material response - demagnetization - Custom Mesh

```{code-cell} ipython3
# apply demagnetization
colls = []
coll_customMesh_demag = apply_demag(
            coll_customMesh,
            style={"label": f"Coll_demag customMesh ({len(coll_customMesh.sources_all):3d} cells)"},
        )
colls.append(coll_customMesh_demag)
```

+++ {"user_expressions": []}

## Define magnetic sources with their susceptibilities - Custom Mesh

```{code-cell} ipython3
# hard magnet
cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.002))
cube1.move((0, 0, 0.0005))
cube1.susceptibility = 0.5  # µr=1.5
cube1.style.label = f"Hard cuboid magnet, susceptibility={cube1.susceptibility}"

# soft magnet
cube2 = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.001, 0.001, 0.001))
cube2.rotate_from_angax(angle=45, axis="y").move((0.0015, 0, 0))
cube2.susceptibility = 3999  # µr=4000
cube2.style.label = f"Soft cuboid magnet, susceptibility={cube2.susceptibility}"

# collection of all cells
coll_autoMesh = magpy.Collection(cube1, cube2, style_label="No demag")

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-0.004, 0, z), (0.006, 0, z), 1001),
        style_label=f"Sensor, z={z}m",
    )
    for z in (-0.001, -0.003, -0.005)
]

# Mesh Generation using automesh
coll_autoMesh_Meshed = mesh_all(
    coll_autoMesh, target_elems=128, per_child_elems=False, style_label="No demag - meshed"
)
magpy.show(*coll_autoMesh_Meshed)
```

+++ {"user_expressions": []}

## Compute material response - demagnetization - Auto Mesh

```{code-cell} ipython3
# apply demagnetization
coll_autoMesh_demag = apply_demag(
            coll_autoMesh,
            style={"label": f"Coll_demag autoMesh ({len(coll_customMesh.sources_all):3d} cells)"},
        )
colls.append(coll_autoMesh_demag)
```

+++ {"user_expressions": []}

## Compare with FEM analysis

```{code-cell} ipython3
B_no_demag_df = magpy.getB(coll_autoMesh_Meshed, sensors, output="dataframe")
B_cols = ["Bx", "Bz"]


def get_FEM_dataframe(sim):
    res = sim["results"][0]
    df = B_no_demag_df.copy()
    for Bk in B_cols:
        df[Bk] = res["value"].get(Bk, np.nan)
    df["computation"] = res["computation"]
    return df


def get_magpylib_dataframe(collection, sensors):
    df = magpy.getB(collection, sensors, output="dataframe")
    df["computation"] = collection.style.label
    return df


from magpylib_material_response import get_dataset

sim_ANSYS = get_dataset("FEMdata_test_softmag")  # FEM dataset has only Bx and Bz

df = pd.concat(
    [
        get_FEM_dataframe(sim_ANSYS),
        *[get_magpylib_dataframe(c, sensors) for c in colls],
    ]
).sort_values(["computation", "path"])


df["Distance [m]"] = sensors[0].position[df["path"]][:, 0]
df["Distance [m]"] -= df["Distance [m]"].min()
```

```{code-cell} ipython3
px_kwargs = dict(
    x="path",
    y=B_cols,
    facet_row="variable",
    facet_col="sensor",
    color="computation",
    line_dash="computation",
    height=600,
    facet_col_spacing=0.05,
    labels={**{Bk: f"{Bk} [T]" for Bk in B_cols}, "value": "value [T]"},
)
fig1 = px.line(
    df,
    title="Methods comparison",
    **px_kwargs,
)
fig1.update_yaxes(matches=None, showticklabels=True)

df_diff = df.copy()
ref = sim_ANSYS["results"][0]["computation"]

for st in df_diff["computation"].unique():
    df_diff.loc[df_diff["computation"] == st, B_cols] -= df.loc[
        df["computation"] == ref, B_cols
    ].values

fig2 = px.line(
    df_diff,
    title=f"Methods comparison - diff vs {ref}",
    **px_kwargs,
)
fig2.update_yaxes(matches=None, showticklabels=True)
display(fig1, fig2)
```

+++ {"user_expressions": []}

As shown above, the demagnetized collection outputs approach the reference FEM
values. Notably, the custom mesh provides more accurate results compared to the
autoMesh approach, even with the same number of total primitives.
