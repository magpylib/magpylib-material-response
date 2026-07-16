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

# Quickstart

Magpylib treats magnet polarization as fixed. This package adds the **material
response**: mesh the magnets into cells, assign a magnetic susceptibility, and
`apply_demag` computes the self-consistent polarization of every cell —
including the demagnetization of the magnets themselves and the response of soft
magnetic parts nearby.

The minimal workflow has three steps: **mesh → susceptibility → `apply_demag`**.
All quantities are SI (meters, Tesla).

```{code-cell} ipython3
import magpylib as magpy

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

# a hard magnet with finite susceptibility, SI units (m, T)
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
magnet.susceptibility = 0.3  # µr = 1.3

# 1. mesh it into cells, 2. apply the material response
mesh = mesh_Cuboid(magnet, target_elems=125)
mesh_demag = apply_demag(mesh)

# the demagnetized collection is a drop-in field source
observer = (0, 0, 1.5e-3)
print("B ignoring material response:", magpy.getB(magnet, observer))
print("B with demagnetization      :", magpy.getB(mesh_demag, observer))
```

+++ {"user_expressions": []}

The z-field drops by several percent — that is the magnet demagnetizing itself.
Refining the mesh (`target_elems`) converges the result; the
[cuboid example](examples/cuboids_demagnetization.md) compares against FEM.

## Setting material properties

Susceptibility can be attached to objects — searched up the parent `Collection`
tree when not set on the object itself — or passed explicitly to `apply_demag`,
which then takes precedence:

```{code-cell} ipython3
magnet.susceptibility = 0.3  # isotropic
magnet.susceptibility = (0.3, 0.1, 0.0)  # anisotropic, global frame

# explicit values override object attributes: one scalar/vector for all
# cells, or one entry per cell
coll = apply_demag(mesh, susceptibility=0.3)
```

Anisotropic 3-vectors act component-wise as a diagonal tensor in the **global
frame** — this works for arbitrarily rotated cells (see the
[method page](method.md) for how rotations are handled).

A uniform external field can be applied through the `H_ext` attribute, given as
flux density in Tesla units (i.e. $\mu_0 H_\text{ext}$):

```{code-cell} ipython3
soft = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1e-3, 1e-3, 1e-3))
soft.susceptibility = 3999  # µr = 4000
soft.H_ext = (0, 0, 0.1)  # 0.1 T applied along z

soft_demag = apply_demag(mesh_Cuboid(soft, target_elems=125))
print("induced polarization of the first cell:")
print(soft_demag.sources_all[0].polarization)
```

Current sources (`magpy.current.*`) placed in the collection also drive the
material response — their field at the cells adds to the applied field. The
[U-core electromagnet example](examples/ucore_electromagnet.md) builds a
coil-driven soft core this way.

## Meshing helpers

- `mesh_Cuboid` / `slice_Cuboid` — uniform grids of cuboid cells (fastest solver
  path),
- `mesh_Cylinder` — cylinder / cylinder-segment cells,
- `mesh_TriangularMesh` — tetrahedral cells for arbitrary closed surfaces
  (optional [TetGen](https://tetgen.pyvista.org/) dependency, see the
  [tetrahedral example](examples/tetrahedral_meshes.md)),
- `mesh_all` — walk a `Collection` and mesh every supported child.

## Where to go next

- [Solvers and performance](examples/solver_performance.md) — choosing between
  the exact dense solver and the FFT-accelerated iterative solver as models
  grow.
- [Method of Moments](method.md) — the physics and the numerical method.
- [API reference](api.md) — all public functions.
