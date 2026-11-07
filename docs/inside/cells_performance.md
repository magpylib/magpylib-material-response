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

# Cells: direct and iterative solvers, measured

`solve_cells` solves one linear system, set out in
[the cell equations](cells_equations.md), in one of two ways. `solver="direct"`
stores the dense matrix and factors it: exact, with memory that grows as the
square of the cells. `solver="iterative"` runs GMRES and never stores the
matrix; on a grid of identical cuboids its products are FFT convolutions.
`solver="auto"`, the default, picks one ([Cells](../guide/cells.md) gives its
rule). Do the two give the same answer? How do their time and memory grow on one
grid? How does the layout of the cells shift the balance? And when is it worth
choosing by hand?

```{code-cell} ipython3
:tags: [hide-input]

import json
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cells_performance_bench import max_rel_diff, topologies
from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_Cuboid

# studio says that objects made in a loop are copies: it matters only for editing
warnings.filterwarnings("ignore", message="magpylib-studio", category=UserWarning)
bench = json.loads(Path("cells_performance.json").read_text())
```

+++ {"user_expressions": []}

## Do they give the same answer?

Three models take every kind of block the solvers build: one meshed magnet, a
single grid; two meshed bodies of different cell sizes beside a lone magnet,
with blocks between grids and a cell off any grid; and a turned mesh with an
anisotropic susceptibility.

```{code-cell} ipython3
cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
single = magpy.Collection(mesh_Cuboid(cube, 343))

body_a = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
body_b = magpy.magnet.Cuboid(
    polarization=(1, 0, 0), dimension=(2e-3, 1e-3, 1e-3), position=(2e-3, 0, 0)
)
lone = magpy.magnet.Cuboid(
    polarization=(0, 0, 1), dimension=(0.5e-3, 0.5e-3, 0.5e-3), position=(0, 2e-3, 0)
)
mixed = magpy.Collection(mesh_Cuboid(body_a, 125), mesh_Cuboid(body_b, 125), lone)

turned_cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
turned_cube.rotate_from_angax(35, (1, 1, 0))
turned = magpy.Collection(mesh_Cuboid(turned_cube, 125))

rows = []
for label, model, chi in (
    ("one meshed magnet", single, 3.0),
    ("two bodies and a lone magnet", mixed, 3.0),
    ("turned, anisotropic", turned, (0.3, 0.1, 0.5)),
):
    n = len(model.sources_all)
    direct = solve_cells(model, susceptibility=[chi] * n, solver="direct")
    iterative = solve_cells(model, susceptibility=[chi] * n, solver="iterative", solver_tol=1e-8)
    rows.append({"model": label, "cells": n, "largest difference": f"{max_rel_diff(direct, iterative):.1e}"})
pd.DataFrame(rows)
```

+++ {"user_expressions": []}

The largest difference is that of any cell's polarization, relative to the
largest polarization. At `solver_tol=1e-8` it stays within about twice the
tolerance. For soft iron the gap grows with the susceptibility and the cell
count ([the cell equations](cells_equations.md) give the numbers), and `"auto"`
tightens the tolerance a hundredfold for that reason. Either way the choice
between the two is one of time and memory.

## How do time and memory grow on one grid?

A meshed cube of 216 to 8,000 cells, χ = 3. The docs are built on 2 shared CPUs,
whose timings would say little, so the times were measured once on a quiet
machine by `cells_performance_bench.py`, next to this page, and stored. Times
are relative to the direct solve of 216 cells; memory is the peak that
`tracemalloc` traces, which counts NumPy's arrays, in a second run.

```{code-cell} ipython3
:tags: [hide-input]

print(
    f"Measured on {bench['machine']}, {bench['date']}, "
    f"magpylib-material-response {bench['versions']['magpylib-material-response']}"
)
scaling = pd.DataFrame(bench["scaling"])
t_ref = scaling.query("solver == 'direct'").sort_values("cells")["time"].iloc[0]
scaling["relative time"] = scaling["time"] / t_ref
scaling.pivot(index="cells", columns="solver", values=["relative time", "peak_mb"]).round(2)
```

```{code-cell} ipython3
:tags: [hide-input]

colours = {"direct": "#2a78d6", "iterative": "#1baf7a"}
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
for solver, colour in colours.items():
    df = scaling[scaling["solver"] == solver]
    style = {"mode": "lines+markers", "line": {"color": colour, "width": 2}, "marker": {"color": colour, "size": 8}}
    fig.add_trace(go.Scatter(x=df["cells"], y=df["relative time"], name=solver, hovertemplate="%{x} cells: %{y:.1f}×<extra>" + solver + "</extra>", **style), 1, 1)
    fig.add_trace(go.Scatter(x=df["cells"], y=df["peak_mb"], name=solver, showlegend=False, hovertemplate="%{x} cells: %{y:.0f} MB<extra>" + solver + "</extra>", **style), 1, 2)
fig.update_xaxes(title_text="cells", type="log")
fig.update_yaxes(type="log")
fig.update_yaxes(title_text="time (× direct at 216 cells)", row=1, col=1)
fig.update_yaxes(title_text="peak memory (MB)", row=1, col=2)
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 20, "b": 40}, legend={"orientation": "h", "y": 1.12})
fig
```

+++ {"user_expressions": []}

At 216 cells the two take the same time. From there the iterative solver pulls
ahead: 1.8 times faster at 512 cells, 2.7 times at 1,000 and 4.7 times at 1,728.
Its memory stays low from the start: at 1,728 cells it peaks at 17 MB, against
433 MB for the direct solver. It then solves 8,000 cells in 0.5 s and 70 MB,
where the direct solver's matrix alone would take 4.6 GB. On one grid, then, the
iterative solver wins from a few hundred cells on.

## How does the layout shift the balance?

FFT products cover only part of the matrix: the cells of each grid among
themselves, and large grids of one spacing between each other. Every other
block, between small or turned bodies or for cells that are not cuboids, is
computed pair by pair, the same work for both solvers. Five layouts show it,
four of 864 to 1,536 cuboids and a meshed cylinder of 110 cells:

```{code-cell} ipython3
:tags: [hide-input]

layouts = topologies()
shown = []
for (label, model), x, colour in zip(
    layouts.items(), (0, 2, 6, 16, 28), ("#2a78d6", "#1baf7a", "#eb6834", "#c94dbf", "#8a8984"), strict=True
):
    copy = model.copy(style_label=label).move((x * 1e-3, 0, 0))
    for cell in copy.sources_all:
        cell.style.color, cell.style.magnetization.show = colour, False
    shown.append(copy)
magpy.show(*shown, backend="widget")
```

```{code-cell} ipython3
:tags: [hide-input]

pd.DataFrame(bench["topology"]).rename(
    columns={"topology": "layout", "structure": "clusters", "agreement": "largest difference"}
)
```

+++ {"user_expressions": []}

These are single runs at χ = 0.5, `solver_tol=1e-8`. On one body, or a few
parallel ones, the iterative solver takes 0.6 to 0.7 of the direct time, and 24
bodies of 64 cells still 0.7. Turn each of those bodies differently and the
blocks between them share nothing: 0.9, near parity. The meshed cylinder has no
grid at all, and both solvers spend their time on the same field evaluations:
1.0. Its cells cost far more each than cuboids on a grid, but at a high
susceptibility curved cells reach a given error with fewer cells
([Inside cells](cells.md) measures both on a soft cube).

At χ = 1000, on larger models, the balance tips the other way where FFT covers
little. The share of the matrix that FFT covers decides it:

| Model                         | Cells | FFT   | Direct         | Iterative      | `"auto"`  |
| ----------------------------- | ----- | ----- | -------------- | -------------- | --------- |
| one cube, identical cells     | 4,096 | 100 % | 3.1 s, 1.4 GB  | 0.41 s, 0.2 GB | iterative |
| three cubes, three cell sizes | 3,240 | 40 %  | 9.0 s, 1.4 GB  | 8.3 s, 1.0 GB  | iterative |
| a grid and 3,900 tetrahedra   | 8,031 | 26 %  | 112 s, 5.7 GB  | 99 s, 4.4 GB   | iterative |
| 100 cubes of 27 cells         | 2,700 | 1 %   | 5.1 s, 0.6 GB  | 6.2 s, 0.7 GB  | direct    |
| voxelized sphere              | 1,304 | 100 % | 0.30 s, 0.3 GB | 0.20 s, 0.1 GB | iterative |
| tetrahedra                    | 1,937 | 0     | 3.3 s, 0.9 GB  | 3.3 s, 1.0 GB  | direct    |
| cuboids of 216 sizes          | 1,728 | 0     | 34 s, 0.3 GB   | 61 s, 0.4 GB   | direct    |

Seven of the 39 models on which `"auto"`'s rule was set, single runs with
magpylib 6.0; the row of tetrahedra with magpylib 6.1, whose threads make such a
solve about 3 times faster (1,842 tetrahedra: 3.1 s on 8 threads, 9.5 s on one).
The rows with one cube, three cubes and the sphere are measured as the direct
solver now builds its matrix; in the other rows with grids it now peaks somewhat
lower. Where FFT covers a tenth of the matrix or more, the iterative solver is
as fast or faster and lighter; below, many small bodies or cells of many sizes
make it slower.

## When is it worth choosing by hand?

Rarely. `"auto"` follows the layout, not the cell count, and the layout is what
decides. Set `solver="direct"` for an exact answer with no tolerance, while its
matrix fits: 72 bytes times the square of the cells, 1 GB at 3,700 cells. Set
`solver="iterative"` to choose the tolerance yourself, for instance a tighter
one for soft iron, since `"auto"` divides `solver_tol` by 100 and `"iterative"`
uses it as given.

## In short

Both solvers give the same polarizations, to within the iterative tolerance. On
one grid the iterative solver is faster from a few hundred cells on and needs a
small fraction of the memory; it solves 8,000 cells in half a second. Where few
cells lie on grids, many small or differently turned bodies, cells of many sizes
or tetrahedra, the direct solver is as fast or faster. `"auto"` measures that
share for each model, so leave it unless you need the direct solver's exact
answer or your own tolerance.
