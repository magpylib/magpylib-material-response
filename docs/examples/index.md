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
- [Checking a result with finite elements](fem_check.md) — `fem_study` and
  `compare` against `mesh_study` on a magnet beside a soft cube: where the two
  methods disagree, and whose error bar is too short.
- [A magnet in a steel tube](shield_tube.md) — a tube meshed into cylinder
  segments: inside it the method of moments converges, behind the shield it
  takes finite elements.
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
fem_check
shield_tube
tetrahedral_meshes
```
