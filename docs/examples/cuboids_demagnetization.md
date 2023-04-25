---
jupytext:
  formats: md:myst
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

# Cuboids demagnetization

The following example demonstrates how to create magnetic sources with different susceptibilities using the Magpylib library. It defines three cuboid magnets with varying susceptibilities and positions, creates a collection of these magnets, and computes their magnetic field responses using different levels of meshing. The results are then compared to a Finite Element Method (FEM) analysis to evaluate the performance of the Magpylib-Response approach. The comparison is presented in two separate plots, one showing the magnetic field values and the other showing the difference between the Magpylib results and the FEM reference data. The code demonstrates that even with a low number of mesh elements, the Magpylib results quickly approach the reference FEM values.

+++ {"user_expressions": []}

## Define magnetic sources with their susceptibilities

```{code-cell} ipython3
import json
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from magpylib_response.demag import apply_demag
from magpylib_response.meshing import mesh_all

magpy.defaults.display.backend = "plotly"

# some low quality magnets with different susceptibilities
cube1 = magpy.magnet.Cuboid(magnetization=(0, 0, 1000), dimension=(1, 1, 1))
cube1.move((-1.5, 0, 0))
cube1.xi = 0.3  # µr=1.3
cube1.style.label = f"Cuboid, xi={cube1.xi}"

cube2 = magpy.magnet.Cuboid(magnetization=(900, 0, 0), dimension=(1, 1, 1))
cube2.rotate_from_angax(-45, "y").move((0, 0, 0.2))
cube2.xi = 1.0  # µr=2.0
cube2.style.label = f"Cuboid, xi={cube2.xi}"

mx, my = 600 * np.sin(30 / 180 * np.pi), 600 * np.cos(30 / 180 * np.pi)
cube3 = magpy.magnet.Cuboid(magnetization=(mx, my, 0), dimension=(1, 1, 2))
cube3.move((1.6, 0, 0.5)).rotate_from_angax(30, "z")
cube3.xi = 0.5  # µr=1.5
cube3.style.label = f"Cuboid, xi={cube3.xi}"

# collection of all cells
coll = magpy.Collection(cube1, cube2, cube3, style_label="No demag")

sensor = magpy.Sensor(position=np.linspace((-4, 0, -1), (4, 0, -1), 301))

magpy.show(*coll, sensor)
```

```{code-cell} ipython3
# example of meshed Collection
coll_meshed = mesh_all(coll, target_elems=50, per_child_elems=False, style_label="No demag - meshed")
coll_meshed.show()
```

+++ {"user_expressions": []}

## Compute material response - demagnetization

```{code-cell} ipython3
# apply demagnetization with varying number of cells
colls = [coll]
for target_elems in [1, 2, 8, 16, 32, 64, 128, 256]:
    with logger.contextualize(target_elems=target_elems):
        coll_meshed = mesh_all(coll, target_elems=target_elems, per_child_elems=True, min_elems=1)
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
B_no_demag = sensor.getB(coll_meshed)

B_cols = ["Bx [mT]", "By [mT]", "Bz [mT]"]


def get_FEM_dataframe(sim):
    res = sim["results"][0]
    df = pd.DataFrame(data=res["values"], columns=B_cols)
    df["Distance [mm]"] = sensor.position[:, 0]
    df["computation"] = res["computation"]
    return df


def get_magpylib_dataframe(collection, sensor):
    B_no_demag = collection.getB(sensor)
    df = pd.DataFrame(
        data=B_no_demag,
        columns=B_cols,
    )
    df["Distance [mm]"] = sensor.position[:, 0]
    df["computation"] = collection.style.label
    return df


from magpylib_response.data import get_dataset
sim_ANSYS = get_dataset("FEMdata_test_cuboids.json")

df = pd.concat(
    [
        get_FEM_dataframe(sim_ANSYS),
        *[get_magpylib_dataframe(c, sensor) for c in colls],
    ]
).sort_values(["computation", "Distance [mm]"])
```

```{code-cell} ipython3
px_kwargs = dict(
    x="Distance [mm]",
    y=B_cols,
    facet_col="variable",
    color="computation",
    line_dash="computation",
    height=400,
    facet_col_spacing=0.05,
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

```{code-cell} ipython3
px_kwargs = dict(
    x="Distance [mm]",
    y=B_cols,
    facet_col="variable",
    color="computation",
    line_dash="computation",
    height=400,
    facet_col_spacing=0.05,
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

As shown above, already with a low number of mesh elements, the result is approaching the reference FEM values.
