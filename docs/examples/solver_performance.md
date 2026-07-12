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

# Solvers and Performance

+++

`apply_demag` offers two solvers that share the **same interaction model**
(the analytical volume-averaged Newell tensor for parallel cuboid cells,
point matching otherwise) and therefore agree to solver tolerance for any
input:

- `solver="direct"` (default) — assembles the dense system matrix and solves
  it exactly with LAPACK. Memory grows as (3N)² and time as N³, so it
  is the right choice up to a few thousand cells.
- `solver="iterative"` — solves the same system matrix-free with GMRES.
  Cells on uniform grids (the output of `mesh_Cuboid`) are handled by an
  O(N log N) FFT convolution, so meshes with tens of thousands of cells
  stay fast and memory-light. If GMRES cannot reach `solver_tol` within
  `max_iter` iterations, a `RuntimeError` is raised — a partially converged
  result is never returned silently.

This example verifies the agreement on typical use cases and measures the
performance crossover.

```{code-cell} ipython3
import time
import tracemalloc

import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

magpy.defaults.display.backend = "plotly"


def max_rel_diff(coll_a, coll_b):
    """Maximum relative polarization difference between two solved collections."""
    pol_a = np.array([s.polarization for s in coll_a.sources_all])
    pol_b = np.array([s.polarization for s in coll_b.sources_all])
    return np.abs(pol_a - pol_b).max() / np.abs(pol_a).max()
```

+++ {"user_expressions": []}

## Both solvers agree — typical use cases

Three configurations that exercise all interaction paths: a single meshed
magnet (FFT grid), several bodies with different cell sizes plus a standalone
magnet (grid + cross + generic blocks), and a rotated mesh with anisotropic
susceptibility (rotation handling).

```{code-cell} ipython3
# 1. single meshed cuboid magnet (SI units: m, T)
cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
case_single = magpy.Collection(mesh_Cuboid(cube, target_elems=343))

# 2. two meshed bodies with different cell sizes + one standalone magnet
body_a = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
body_b = magpy.magnet.Cuboid(
    polarization=(1, 0, 0), dimension=(2e-3, 1e-3, 1e-3), position=(2e-3, 0, 0)
)
lone = magpy.magnet.Cuboid(
    polarization=(0, 0, 1), dimension=(0.5e-3, 0.5e-3, 0.5e-3), position=(0, 2e-3, 0)
)
case_mixed = magpy.Collection(mesh_Cuboid(body_a, 125), mesh_Cuboid(body_b, 125), lone)

# 3. rotated meshed cuboid with anisotropic susceptibility
cube_rot = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
cube_rot.rotate_from_angax(35, (1, 1, 0))
case_rotated = magpy.Collection(mesh_Cuboid(cube_rot, 125))

magpy.show(
    {"objects": case_single, "col": 1},
    {"objects": case_mixed, "col": 2},
    {"objects": case_rotated, "col": 3},
)
```

```{code-cell} ipython3
cases = {
    "single meshed magnet": (case_single, 3.0),
    "two bodies + standalone": (case_mixed, 3.0),
    "rotated, anisotropic": (case_rotated, (0.3, 0.1, 0.5)),
}

rows = []
for label, (coll, sus) in cases.items():
    n = len(coll.sources_all)
    sus_list = [sus] * n
    coll_direct = apply_demag(coll, susceptibility=sus_list, solver="direct")
    coll_iter = apply_demag(
        coll, susceptibility=sus_list, solver="iterative", solver_tol=1e-8
    )
    rows.append(
        {
            "use case": label,
            "cells": n,
            "max rel. difference": f"{max_rel_diff(coll_direct, coll_iter):.1e}",
        }
    )

pd.DataFrame(rows)
```

+++ {"user_expressions": []}

The two solvers match to the requested `solver_tol` in every configuration —
the choice between them is purely a performance trade-off.

## Performance scaling

Wall time and peak memory for a single meshed cuboid magnet of growing cell
count. Absolute seconds depend heavily on the machine, so times are reported
**relative to the direct solve of the smallest mesh** — the ratios are much
more portable. Peak memory is measured with `tracemalloc` (which tracks NumPy
allocations) in a separate run, since tracing skews wall time.

```{code-cell} ipython3
def measure_solve(target_elems, solver):
    def build():
        cube = magpy.magnet.Cuboid(
            polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3)
        )
        return magpy.Collection(mesh_Cuboid(cube, target_elems=target_elems))

    coll = build()
    n = len(coll.sources_all)
    t0 = time.perf_counter()
    apply_demag(coll, susceptibility=3.0, solver=solver, solver_tol=1e-8)
    dt = time.perf_counter() - t0

    tracemalloc.start()
    apply_demag(build(), susceptibility=3.0, solver=solver, solver_tol=1e-8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"solver": solver, "cells": n, "time": dt, "peak_mb": peak / 1e6}


sizes_direct = [216, 512, 1000, 1728]
sizes_iterative = [216, 512, 1000, 1728, 4096, 8000]

records = [measure_solve(s, "direct") for s in sizes_direct]
```

```{code-cell} ipython3
records += [measure_solve(s, "iterative") for s in sizes_iterative]

perf_df = pd.DataFrame(records)
t_ref = perf_df.query("solver == 'direct'")["time"].iloc[0]  # smallest direct solve
perf_df["rel_time"] = perf_df["time"] / t_ref
perf_df.pivot(index="cells", columns="solver", values=["rel_time", "peak_mb"]).round(2)
```

```{code-cell} ipython3
series_style = {
    "direct": {"color": "#2a78d6", "symbol": "circle"},
    "iterative": {"color": "#1baf7a", "symbol": "square"},
}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("wall time (relative)", "peak memory"),
    horizontal_spacing=0.12,
)
for solver, style in series_style.items():
    df = perf_df[perf_df["solver"] == solver]
    line = {"color": style["color"], "width": 2}
    marker = {"color": style["color"], "symbol": style["symbol"], "size": 9}
    fig.add_trace(
        go.Scatter(
            x=df["cells"],
            y=df["rel_time"],
            name=solver,
            mode="lines+markers",
            line=line,
            marker=marker,
            hovertemplate="%{x} cells: %{y:.1f}×<extra>" + solver + "</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["cells"],
            y=df["peak_mb"],
            name=solver,
            showlegend=False,
            mode="lines+markers",
            line=line,
            marker=marker,
            hovertemplate="%{x} cells: %{y:.0f} MB<extra>" + solver + "</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_annotation(
        x=np.log10(df["cells"].iloc[-1]),
        y=np.log10(df["rel_time"].iloc[-1]),
        text=solver,
        font={"color": style["color"]},
        xanchor="left",
        xshift=12,
        showarrow=False,
        xref="x",
        yref="y",
    )

fig.update_xaxes(title_text="number of cells", type="log")
fig.update_yaxes(type="log")
fig.update_yaxes(title_text=f"wall time (× direct at {sizes_direct[0]} cells)", row=1, col=1)
fig.update_yaxes(title_text="peak memory (MB)", row=1, col=2)
fig.update_layout(
    title="apply_demag scaling — single meshed cuboid",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.08},
    template="plotly_white",
)
fig.show()
```

+++ {"user_expressions": []}

The direct solver's N³ time slope and (3N)² memory slope take over in the low
thousands of cells, while the FFT-accelerated iterative solver stays almost
flat in both panels. Beyond the crossover the gap widens rapidly — the dense
matrix becomes the hard limit (extrapolating the right panel, N = 27 000
would already need a ~47 GB matrix, while the iterative solver handles it in
seconds within a few hundred MB).

## Model topology matters

The solvers assemble the interaction operator from *structure clusters*
(uniform grids of identical parallel cells → FFT / analytical blocks,
everything else → point-matched `magpy.getH`), so the model topology decides
which paths do the work — and how much the solver choice matters. Here the
same comparison runs on characteristic topologies of similar total cell
count.

```{code-cell} ipython3
from collections import Counter

from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag_fft import analyze_structure
from magpylib_material_response.meshing import mesh_Cylinder


def structure_kinds(coll):
    """Summarize the structure clusters the solvers will work with."""
    cells = coll.sources_all
    pos = np.array([getattr(s, "barycenter", s.position) for s in cells])
    isc = np.array([isinstance(s, magpy.magnet.Cuboid) for s in cells])
    dims = np.array(
        [s.dimension if isinstance(s, magpy.magnet.Cuboid) else (1, 1, 1) for s in cells]
    )
    rots = R.from_quat([s.orientation.as_quat() for s in cells])
    kinds = Counter(c["kind"] for c in analyze_structure(pos, dims, rots, isc))
    return " + ".join(f"{v} {k}" for k, v in sorted(kinds.items()))


def bench_topology(label, coll):
    n = len(coll.sources_all)
    sus = [0.5] * n
    t0 = time.perf_counter()
    coll_direct = apply_demag(coll, susceptibility=sus, solver="direct")
    t_direct = time.perf_counter() - t0
    t0 = time.perf_counter()
    coll_iter = apply_demag(coll, susceptibility=sus, solver="iterative", solver_tol=1e-8)
    t_iter = time.perf_counter() - t0
    return {
        "topology": label,
        "cells": n,
        "structure": structure_kinds(coll),
        "iter / direct time": round(t_iter / t_direct, 2),
        "agreement": f"{max_rel_diff(coll_direct, coll_iter):.0e}",
    }


def block(pos=(0, 0, 0)):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3), position=pos
    )


def grid_positions(k, pitch=1.8e-3):
    side = int(np.ceil(np.sqrt(k)))
    return [(i * pitch, j * pitch, 0) for i in range(side) for j in range(side)][:k]


topologies = {
    "single body": magpy.Collection(mesh_Cuboid(block(), 1000)),
    "4 bodies": magpy.Collection(
        *(mesh_Cuboid(block(p), 250) for p in grid_positions(4))
    ),
    "24 bodies": magpy.Collection(
        *(mesh_Cuboid(block(p), 48) for p in grid_positions(24))
    ),
}

meshes = []
for i, p in enumerate(grid_positions(24, pitch=2.2e-3)):
    m = mesh_Cuboid(block(p), 48)
    m.rotate_from_angax(i * 15.0, (0, 0, 1), anchor=p)
    meshes.append(m)
topologies["24 bodies, individually rotated"] = magpy.Collection(*meshes)

cyl = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1e-3, 1e-3))
topologies["meshed cylinder"] = magpy.Collection(mesh_Cylinder(cyl, 300))

for label, coll in topologies.items():
    coll.style.label = label

magpy.show(
    *(
        {"objects": coll, "row": i // 3 + 1, "col": i % 3 + 1}
        for i, coll in enumerate(topologies.values())
    ),
)
```

```{code-cell} ipython3
topo_rows = [
    bench_topology(label, coll)
    for label, coll in topologies.items()
    if label != "meshed cylinder"
]
```

```{code-cell} ipython3
topo_rows.append(bench_topology("meshed cylinder", topologies["meshed cylinder"]))

pd.DataFrame(topo_rows)
```

+++ {"user_expressions": []}

What the rows show:

- **Single body / a few parallel bodies** — everything runs on the FFT and
  analytical Newell paths; the iterative solver is ahead and scales far
  better (previous section).
- **Dozens of bodies** — the pairwise cross-blocks (one per body pair)
  start to dominate the operator build. The iterative solver still wins,
  but its advantage shrinks as the body count grows at fixed total cells.
- **Bodies rotated individually** — cross-blocks between differently
  oriented bodies fall back to point matching, which both solvers share:
  expect near parity. The self-blocks of each body still use their own
  rotated FFT grid.
- **Meshed cylinder** — non-cuboid cells (here CylinderSegment, likewise
  tetrahedra from `mesh_TriangularMesh`) take the point-matched generic
  path. Both solvers spend nearly all time evaluating the cells' analytical
  fields, so the solver choice barely matters — the cell *type* is the cost
  driver. Prefer cuboid meshes when the geometry allows it.

## Choosing a solver

| Model topology | Interaction paths | Recommendation |
| --- | --- | --- |
| One meshed body, up to ~2000 cells | FFT / analytical | either; `direct` (default) is exact and tuning-free |
| One meshed body, large | FFT self-block | `iterative` — O(N log N); only option for N ≳ 10⁴ (memory) |
| Few parallel bodies | FFT + analytical cross-blocks | `iterative` — cross-blocks are analytical and cheap |
| Dozens of bodies | cross-blocks dominate the build | `iterative`, but advantage shrinks with body count |
| Bodies rotated differently | point-matched cross-blocks | either — build cost is shared, expect parity |
| Non-cuboid cells (cylinder, tetrahedra) | point-matched, dense | either — cell field evaluation dominates; keep counts moderate |

Two knobs control the iterative solver: `solver_tol` (relative residual,
default `1e-6`) and `max_iter` (default 50). Non-convergence raises a
`RuntimeError` with guidance rather than returning an inaccurate result. To
see the detected structure for your own model, enable the package logging
(`magpylib_material_response.configure_logging()`) — the cluster summary is
logged at the start of every solve.
