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
response**: mesh the magnets into cells, give them a susceptibility, and
`apply_demag` computes the self-consistent polarization of every cell — the
demagnetization of the magnets themselves and the magnetization of soft parts
nearby. All quantities are SI (m, T).

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

# the demagnetized collection is a field source like any other
observer = (0, 0, 1.5e-3)
print(f"Bz without material response: {magpy.getB(magnet, observer)[2] * 1e3:.2f} mT")
print(f"Bz with demagnetization     : {magpy.getB(mesh_demag, observer)[2] * 1e3:.2f} mT")
```

+++ {"user_expressions": []}

The field drops by about 10 %: the magnet demagnetizes itself. Refining the mesh
(`target_elems`) converges the result; the
[cuboid example](examples/cuboids_demagnetization.md) compares against FEM.

## Setting material properties

Susceptibility is an attribute of a magnet, or of a parent `Collection` for all
the magnets in it; the cells of a mesh copy their magnet's value when it is
meshed. A value passed to `apply_demag` takes precedence:

```{code-cell} ipython3
magnet.susceptibility = 0.3  # isotropic
magnet.susceptibility = (0.3, 0.1, 0.0)  # anisotropic, along the magnet's axes

# overrides the attributes: one value for all cells, or one per cell
coll = apply_demag(mesh, susceptibility=0.3)
```

An anisotropic value acts along the **magnet's own axes** and turns with it,
like its polarization (see the [method page](method.md)).

A uniform applied field is set the same way, with the `H_ext` attribute, as
$\mu_0 \mathbf{H}$ in tesla:

```{code-cell} ipython3
soft = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1e-3, 1e-3, 1e-3))
soft.susceptibility = 3999  # µr = 4000
soft.H_ext = (0, 0, 0.1)  # 0.1 T applied along z

soft_demag = apply_demag(mesh_Cuboid(soft, target_elems=125))
Jz = [cell.polarization[2] for cell in soft_demag.sources_all]
print(f"mean induced polarization: {sum(Jz) / len(Jz):.3f} T along z")
```

Current sources (`magpy.current.*`) placed in the collection also drive the
material response: their field at each cell's centroid adds to the applied
field. The [U-core electromagnet example](examples/ucore_electromagnet.md)
builds a coil-driven soft core this way.

## Large meshes and the two interaction rules

Large meshes need no setting. The default `solver="auto"` solves meshes made
mostly of grids of identical cuboids, as `mesh_Cuboid` and `mesh_all` make them,
matrix-free with GMRES and FFT products, and other meshes with the dense direct
solver. Pass `solver="direct"` or `solver="iterative"` to choose yourself (see
[solvers and performance](examples/solver_performance.md)).

By default each cell's field is taken at the centroids of the others (point
matching). With `interaction="newell"` it is averaged over each cell instead:

```{code-cell} ipython3
coll_point = apply_demag(mesh, susceptibility=0.3)
coll_newell = apply_demag(mesh, susceptibility=0.3, interaction="newell")
```

Both converge to the same answer, and at equal cell count neither is more
accurate in general. The Newell rule needs cuboid cells with one orientation. In
return, for a soft body with positive susceptibility in a uniform applied field,
its induced moment along the field never exceeds the converged value and rises
each time every cell is split into smaller ones, so a convergence study knows
which side the answer lies on (see the [method page](method.md)).

## Meshing helpers

- `mesh_Cuboid` — a grid of cuboid cells; `slice_Cuboid` cuts a cuboid in two,
- `mesh_Cylinder` — cylinder and cylinder-segment cells,
- `mesh_TriangularMesh` — tetrahedral cells for arbitrary closed surfaces
  (optional [TetGen](https://tetgen.pyvista.org/) dependency, see the
  [tetrahedral example](examples/tetrahedral_meshes.md)),
- `mesh_all` — walk a `Collection` and mesh every supported child.
