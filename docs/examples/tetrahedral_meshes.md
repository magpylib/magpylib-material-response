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

Each tetrahedron acts on the others through magpylib's closed-form `Tetrahedron`
field, taken at their centroids (point matching), like any other cell type.

This example validates the whole chain on a case with an exact analytical
answer: a **soft magnetic sphere**, for which the demagnetizing factor is
exactly 1/3, so a remanent polarization $J_0$ relaxes to

$$J = \frac{J_0}{1 + \chi/3}$$

## Build and mesh a spherical TriangularMesh

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response.demag import apply_demag
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

## Apply demagnetization and compare with the exact sphere solution

```{code-cell} ipython3
chi = 0.5

mesh_demag = apply_demag(mesh, susceptibility=chi)
pol = np.array([s.polarization for s in mesh_demag.sources_all])

# volume-weighted mean polarization vs the analytical sphere result
volumes = np.array(
    [
        abs(np.linalg.det(np.array(c.vertices[1:]) - np.array(c.vertices[0]))) / 6
        for c in cells
    ]
)
Jz_mean = (volumes * pol[:, 2]).sum() / volumes.sum()
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

- The solver stores a dense interaction matrix of (3N)² entries, so memory grows
  as N² and solve time faster: keep tetrahedra counts to a few thousand.
- `target_elems` is a lower-bound hint: TetGen inserts Steiner points to meet
  its quality constraints, so the resulting cell count is typically larger. The
  `minratio` and `mindihedral` arguments trade cell count against element
  quality.
- For a cuboid-vs-tetrahedral comparison on the same solid — accuracy per cell
  and solve-time trade-offs at matched refinement levels — see the
  [U-core electromagnet example](ucore_electromagnet.md).
