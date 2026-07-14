# Examples

Worked, executable examples — in suggested reading order:

- [Cuboid demagnetization](cuboids_demagnetization.md) — two hard cuboid
  magnets; mesh-refinement convergence against FEM reference data.
- [Soft magnets](soft_magnets.md) — a hard magnet next to a high-permeability
  soft cuboid; field comparison against FEM.
- [Solvers and performance](solver_performance.md) — direct vs. iterative
  solver: agreement guarantee, scaling of wall time and memory, and how the
  model topology decides which interaction paths do the work.
- [Arbitrary shapes with tetrahedral meshes](tetrahedral_meshes.md) —
  meshing a `TriangularMesh` magnet with TetGen, validated against the exact
  soft-sphere solution.

```{toctree}
:maxdepth: 1
:hidden:

cuboids_demagnetization
soft_magnets
solver_performance
tetrahedral_meshes
```
