# Examples

Executable examples, in suggested reading order:

- [Cuboid demagnetization](cuboids_demagnetization.md) — two hard cuboid
  magnets; mesh-refinement convergence against FEM reference data.
- [Soft magnets](soft_magnets.md) — a hard magnet next to a high-permeability
  soft cuboid; field comparison against FEM.
- [U-core electromagnet](ucore_electromagnet.md) — current sources driving a
  soft core: a swept multi-turn coil built with the polyline helpers, the field
  on a plane above the poles, and a cuboid- vs tetrahedral-mesh comparison.
- [Arbitrary shapes with tetrahedral meshes](tetrahedral_meshes.md) — meshing a
  `TriangularMesh` magnet with TetGen, checked against the exact solution for a
  magnetized sphere.

```{toctree}
:maxdepth: 1
:hidden:

cuboids_demagnetization
soft_magnets
ucore_electromagnet
tetrahedral_meshes
```
