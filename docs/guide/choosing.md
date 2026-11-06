# Choosing a solver

All three solvers take the same Magpylib model and return a model that Magpylib
reads. Start from what your parts are made of and what you need to read.

| Your model                                                          | Solver                            |
| ------------------------------------------------------------------- | --------------------------------- |
| Soft iron below saturation, ferrite, magnets with a small χ         | `solve_face_charge`               |
| The same, read many times while something moves                     | `solve_face_charge` with `reuse=` |
| A part that saturates, on a B–H curve                               | `solve_cells`                     |
| A susceptibility that differs along a part's axes, or varies inside | `solve_cells`                     |
| B, J or M **inside** a part, such as the flux in a yoke             | `solve_cells`                     |
| An independent check of either, with an error bar                   | `solve_fem` and `study_fem`       |

## What each one is

|                     | `solve_face_charge`                                                                | `solve_cells`                                                                                                                     | `solve_fem`                                    |
| ------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Unknown             | a charge on the parts' surfaces                                                    | one polarization per cell of the parts' volume                                                                                    | a scalar potential in the parts and the air    |
| Mesh                | made inside, from `level`                                                          | you mesh the parts (`mesh_uniform`, `mesh_*`)                                                                                     | made inside by Netgen, from `order` and `maxh` |
| Shapes solved       | cuboids, cylinders, cylinder segments, spheres, tetrahedra, closed triangle meshes | cells of any shape: cuboids, cylinder segments, tetrahedra; `mesh_TriangularMesh` (TetGen) and `voxelize` mesh any closed surface | the same as face charge, unmeshed              |
| Material            | linear, isotropic, uniform in each part                                            | linear or a `BHCurve`; per axis; per cell                                                                                         | linear or a `BHCurve`; per axis                |
| Field inside a part | μ0 H only, with a warning                                                          | B, H, J, M                                                                                                                        | B, H, J, M                                     |
| Error estimate      | `jump_indicator`, free with each solve                                             | `study_cells`, several meshes                                                                                                     | `study_fem`, several orders                    |
| Extra install       | none                                                                               | none; `[tetgen]` for `mesh_TriangularMesh`                                                                                        | `[fem]`, NGSolve                               |

## How they compare

The repository's validation suite solves its cases with each method against
stored finite-element references. The time to bring the worst group of sensors
within 1 % of the reference, solve and `getB` together, one run each on an Apple
M4 Pro:

| Case                       | Face charge | Finite elements |
| -------------------------- | ----------: | --------------: |
| Magnet beside a soft cube  |      0.02 s |           2.3 s |
| Iron cube in 10 mT         |      0.04 s |          0.52 s |
| Flux concentrator          |      0.04 s |          0.83 s |
| Halbach array on back iron |      0.06 s |           1.5 s |
| U-core driven by a coil    |      0.07 s |           4.5 s |
| Magnet on a soft plate     |      0.09 s |           3.1 s |
| CAD part, a triangle mesh  |      0.20 s |            11 s |
| Thin shielding sheet       |      0.28 s |           2.7 s |

Face charge reaches 1 % faster than finite elements on all fourteen cases where
both reach it, 13 to 104 times faster on nine of them. On soft iron it is 20 to
40 times more accurate than `solve_cells` in the same time. The full table, and
where finite elements come closest, is in
[the face-charge guide](face_charge.md#against-finite-elements).

## Where face charge is not the fastest

- **A few weak magnets on their own**, χ around 0.05: `solve_cells` reaches the
  same accuracy faster, because their polarization hardly varies.
- **Many separate parts**: face charge solves one dense system for all of them,
  so 100 small cubes take 7,200 unknowns at level 1 and four times as many at
  level 2.
- **Thin sheets**: the panels are sized to the thickness over the whole sheet.
  Start at `level=2`.
- **More than `max_unknowns`**, 20,000 by default, a 3.2 GB system: without a
  `level`, face charge takes the highest level that fits, and warns; with one,
  or if even level 1 is too large, it refuses before it assembles. Then mesh the
  parts for `solve_cells`.

## What none of them does

- **Paths.** Each solve is one configuration: a model whose objects move along a
  path is refused. Loop over the steps; [Many solves](loops.md) shows how to
  keep the factorization between them.
- **Hysteresis and eddy currents.** The materials are static and single-valued:
  a remanence plus a susceptibility, or a B–H curve.
