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

# Arbitrary Shapes with Tetrahedral Meshes

+++

Shapes that cannot be meshed into cuboid or cylinder cells are handled through
`magpylib.magnet.TriangularMesh`: any closed triangular surface can be
discretized into conforming `Tetrahedron` cells with `mesh_TriangularMesh`,
which drives [TetGen](https://tetgen.org) under the hood. TetGen is an optional
dependency:

```bash
pip install magpylib-material-response[tetgen]
```

Tetrahedral cells take the point-matched _generic_ interaction path (see the
[solver performance example](solver_performance.md)): both solvers share it and
agree to solver tolerance, and the computation cost is dominated by the cells'
analytical field evaluation rather than by the solver choice.

This example validates the whole chain on a case with an exact analytical
answer: a **soft magnetic sphere**, for which the demagnetizing factor is
exactly 1/3, so a remanent polarization $J_0$ relaxes to

$$J = \frac{J_0}{1 + \chi/3}$$

## Build and mesh a spherical TriangularMesh

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.demag_fft import analyze_collection
from magpylib_material_response.meshing import mesh_TriangularMesh

magpy.defaults.display.backend = "plotly"


def fibonacci_sphere(n, radius):
    """n near-uniform points on a sphere (SI units: m)."""
    k = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    z = 1.0 - 2.0 * (k + 0.5) / n
    r = np.sqrt(1.0 - z**2)
    return radius * np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


sphere = magpy.magnet.TriangularMesh.from_ConvexHull(
    points=fibonacci_sphere(200, radius=1e-3),
    polarization=(0, 0, 1),
)

mesh = mesh_TriangularMesh(sphere, target_elems=300)
cells = mesh.sources_all
print(f"{len(cells)} tetrahedra")

magpy.show(
    {"objects": sphere, "col": 1},
    {"objects": mesh, "col": 2},
)
```

+++ {"user_expressions": []}

The structure analysis confirms the cells land in the point-matched generic
cluster:

```{code-cell} ipython3
_, clusters = analyze_collection(cells)
[(c["kind"], len(c["indices"])) for c in clusters]
```

## Apply demagnetization and compare with the exact sphere solution

```{code-cell} ipython3
chi = 0.5
n = len(cells)

mesh_direct = apply_demag(mesh, susceptibility=[chi] * n, solver="direct")
mesh_iter = apply_demag(
    mesh, susceptibility=[chi] * n, solver="iterative", solver_tol=1e-8
)

pol_direct = np.array([s.polarization for s in mesh_direct.sources_all])
pol_iter = np.array([s.polarization for s in mesh_iter.sources_all])
print(
    "solver agreement:"
    f" {np.abs(pol_direct - pol_iter).max() / np.abs(pol_direct).max():.1e}"
)
```

```{code-cell} ipython3
# volume-weighted mean polarization vs the analytical sphere result
volumes = np.array(
    [
        abs(np.linalg.det(np.array(c.vertices[1:]) - np.array(c.vertices[0]))) / 6
        for c in cells
    ]
)
Jz_mean = (volumes * pol_direct[:, 2]).sum() / volumes.sum()
J_exact = 1.0 / (1.0 + chi / 3.0)

print(f"volume-averaged Jz : {Jz_mean:.4f} T")
print(f"analytical sphere  : {J_exact:.4f} T")
print(f"relative error     : {abs(Jz_mean - J_exact) / J_exact:.2%}")
```

+++ {"user_expressions": []}

The volume-averaged polarization of the meshed sphere reproduces the exact
demagnetizing-factor result to a fraction of a percent — the residual is the
polyhedral approximation of the sphere surface and shrinks with finer surface
and volume meshes.

## Practical notes

- The generic path builds a dense point-matched interaction matrix, so cost and
  memory grow as N² regardless of solver — keep tetrahedra counts moderate
  (thousands, not tens of thousands), and prefer cuboid meshes when the geometry
  allows them.
- `target_elems` is a lower-bound hint: TetGen inserts Steiner points to meet
  its quality constraints, so the resulting cell count is typically larger. The
  `minratio` and `mindihedral` arguments trade cell count against element
  quality.
- Everything shown in the [solver performance example](solver_performance.md)
  (agreement guarantee, `solver_tol`, `max_iter`, non-convergence raising)
  applies unchanged.
