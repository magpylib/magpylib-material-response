# Examples

Executable examples, in suggested reading order:

- [Cuboid demagnetization](cuboids_demagnetization.md) — two hard cuboid
  magnets; mesh-refinement convergence against FEM reference data.
- [Soft magnets](soft_magnets.md) — a hard magnet next to a high-permeability
  soft cuboid; field comparison against FEM.
- [U-core electromagnet](ucore_electromagnet.md) — current sources driving a
  soft core: a swept multi-turn coil built with the polyline helpers, the field
  on a plane above the poles, and a cuboid- vs tetrahedral-mesh comparison.
- [Solvers and performance](solver_performance.md) — the direct and the
  iterative solver: their agreement, their time and memory as meshes grow, and
  which interactions take the fast paths.
- [Mesh convergence](mesh_convergence.md) — how fine a mesh must be, and the
  error of the finest, with `mesh_study` on a flux concentrator, a Halbach array
  on back iron and a steel shield.
- [Arbitrary shapes with tetrahedral meshes](tetrahedral_meshes.md) — meshing a
  `TriangularMesh` magnet with TetGen, checked against the exact solution for a
  magnetized sphere.

```{toctree}
:maxdepth: 1
:hidden:

cuboids_demagnetization
soft_magnets
ucore_electromagnet
solver_performance
mesh_convergence
tetrahedral_meshes
```
