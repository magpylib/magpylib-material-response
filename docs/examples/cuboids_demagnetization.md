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

# Cuboid Demagnetization

Three hard cuboid magnets with different susceptibilities, positions, and
orientations are meshed at increasing resolution and solved with `apply_demag`.
The field along a sensor line is compared against a Finite Element Method (FEM)
reference solution — both directly and as the difference to the reference.
Already coarse meshes land close to the FEM values, and refining the mesh
converges toward them.

+++

## Define magnetic sources with their susceptibilities

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from magpylib_material_response import get_dataset
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_all

# Uncomment to enable logging output from the package (silent by default).
# from magpylib_material_response import configure_logging
# configure_logging(min_log_time=5)  # log steps taking longer than 5 s

magpy.defaults.display.backend = "plotly"

# some low quality magnets with different susceptibilities, SI units (m, T)
cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
cube1.move((-0.0015, 0, 0))
cube1.susceptibility = 0.3  # µr=1.3
cube1.style.label = f"Cuboid, susceptibility={cube1.susceptibility}"

cube2 = magpy.magnet.Cuboid(polarization=(0.9, 0, 0), dimension=(0.001, 0.001, 0.001))
cube2.rotate_from_angax(-45, "y").move((0, 0, 0.0002))
cube2.susceptibility = 1.0  # µr=2.0
cube2.style.label = f"Cuboid, susceptibility={cube2.susceptibility}"

mx = 0.6 * np.sin(np.deg2rad(30))
my = 0.6 * np.cos(np.deg2rad(30))
cube3 = magpy.magnet.Cuboid(polarization=(mx, my, 0), dimension=(0.001, 0.001, 0.002))
cube3.move((0.0016, 0, 0.0005)).rotate_from_angax(30, "z")
cube3.susceptibility = 0.5  # µr=1.5
cube3.style.label = f"Cuboid, susceptibility={cube3.susceptibility}"

# collection of all cells
coll = magpy.Collection(cube1, cube2, cube3, style_label="No demag")

sensor = magpy.Sensor(
    position=np.linspace((-0.004, 0, -0.001), (0.004, 0, -0.001), 301)
)

magpy.show(*coll, sensor)
```

```{code-cell} ipython3
# example of meshed Collection
coll_meshed = mesh_all(
    coll, target_elems=50, per_child_elems=False, style_label="No demag - meshed"
)
magpy.show(coll_meshed)
```

## Compute material response - demagnetization

```{code-cell} ipython3
# apply demagnetization with varying number of cells
colls = [coll]
for target_elems in [1, 2, 8, 16, 32, 64, 128, 256]:
    print(f"Processing demagnetization with {target_elems} target elements")

    coll_meshed = mesh_all(
        coll, target_elems=target_elems, per_child_elems=True, min_elems=1
    )

    coll_demag = apply_demag(
        coll_meshed,
        style={"label": f"Coll_demag ({len(coll_meshed.sources_all):3d} cells)"},
    )
    colls.append(coll_demag)

    print(f"Completed demagnetization: {len(coll_meshed.sources_all)} cells created")
```

## Compare with FEM analysis

```{code-cell} ipython3
# template dataframe with the right index/path structure; the FEM values
# from the reference dataset are filled into a copy of it below
B_template_df = magpy.getB(coll_meshed, sensor, output="dataframe")

B_cols = ["Bx", "By", "Bz"]


def get_FEM_dataframe(sim):
    res = sim["results"][0]
    df = B_template_df.copy()
    for Bk in B_cols:
        df[Bk] = res["value"].get(Bk, np.nan)
    df["computation"] = res["computation"]
    return df


def get_magpylib_dataframe(collection, sensors):
    df = magpy.getB(collection, sensors, output="dataframe")
    df["computation"] = collection.style.label
    return df


sim_ANSYS = get_dataset("FEMdata_test_cuboids")

df = pd.concat(
    [
        get_FEM_dataframe(sim_ANSYS),
        *[get_magpylib_dataframe(c, sensor) for c in colls],
    ]
).sort_values(["computation", "path"])

df["Distance [m]"] = sensor.position[df["path"]][:, 0]
df["Distance [m]"] -= df["Distance [m]"].min()
```

```{code-cell} ipython3
px_kwargs = dict(
    x="Distance [m]",
    y=B_cols,
    facet_col="variable",
    color="computation",
    line_dash="computation",
    height=400,
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

As shown above, already with a low number of mesh elements, the result is
approaching the reference FEM values and improves while refining the mesh.

For larger meshes the default dense solver becomes the bottleneck — see
[solvers and performance](solver_performance.md) for the FFT-accelerated
iterative solver and guidance on choosing between them.
