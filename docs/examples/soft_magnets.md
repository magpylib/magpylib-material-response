---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# Soft Magnets

+++

This code demonstrates demagnetization calculations for a hard and a soft cuboid magnet using the Magpylib library.
Demagnetization is applied using varying numbers of cells for the mesh and compared to the computed magnetic fields from Magpylib withoug demagnetization and with FEM analysis data obtained from an external dataset.

+++ {"user_expressions": []}

## Define magnetic sources with their susceptibilities

```{code-cell} ipython3
import json

import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_all

magpy.defaults.display.backend = "plotly"

# hard magnet
cube1 = magpy.magnet.Cuboid(magnetization=(0, 0, 1000), dimension=(1, 1, 2))
cube1.move((0, 0, 0.5))
cube1.xi = 0.5  # µr=1.5
cube1.style.label = f"Hard cuboid magnet, xi={cube1.xi}"

# soft magnet
cube2 = magpy.magnet.Cuboid(magnetization=(0, 0, 0), dimension=(1, 1, 1))
cube2.rotate_from_angax(angle=45, axis="y").move((1.5, 0, 0))
cube2.xi = 3999  # µr=4000
cube2.style.label = f"Soft cuboid magnet, xi={cube2.xi}"

# collection of all cells
coll = magpy.Collection(cube1, cube2, style_label="No demag")

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-4, 0, z), (6, 0, z), 1001),
        style_label=f"Sensor, z={z}mm",
    )
    for z in (-1, -3, -5)
]

magpy.show(*coll, *sensors)
```

```{code-cell} ipython3
# example of meshed Collection
coll_meshed = mesh_all(
    coll, target_elems=50, per_child_elems=False, style_label="No demag - meshed"
)
magpy.show(*coll_meshed)
```

+++ {"user_expressions": []}

## Compute material response - demagnetization

```{code-cell} ipython3
# apply demagnetization with varying number of cells
colls = [coll]
for target_elems in [1, 2, 8, 16, 32, 64, 128, 256]:
    with logger.contextualize(target_elems=target_elems):
        coll_meshed = mesh_all(
            coll, target_elems=target_elems, per_child_elems=True, min_elems=1
        )
        coll_demag = apply_demag(
            coll_meshed,
            style={"label": f"Coll_demag ({len(coll_meshed.sources_all):3d} cells)"},
        )
        colls.append(coll_demag)
```

+++ {"user_expressions": []}

## Compare with FEM analysis

```{code-cell} ipython3
# compute field before demag
B_no_demag_df = magpy.getB(coll_meshed, sensors, output="dataframe")

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


from magpylib_material_response.data import get_dataset

sim_ANSYS = get_dataset("FEMdata_test_softmag")  # FEM dataset has only Bx and Bz

df = pd.concat(
    [
        get_FEM_dataframe(sim_ANSYS),
        *[get_magpylib_dataframe(c, sensors) for c in colls],
    ]
).sort_values(["computation", "path"])


df["Distance [mm]"] = sensors[0].position[df["path"]][:, 0]
df["Distance [mm]"] -= df["Distance [mm]"].min()
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
    labels={Bk: f"{Bk} [mT]" for Bk in B_cols},
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

As shown above, the demagnetized collection outputs are approaching the reference FEM values.
