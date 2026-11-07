# Choosing a solver

All three solvers take the same Magpylib model and return a model that Magpylib
reads. Which one fits depends on two questions: what are your parts made of, and
where do you need the field? The terms below, χ and `BHCurve` among them, are
set out in [Building a model](model.md).

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
| Field inside a part | not B: `getB` gives μ0 H, with a warning                                           | B, H, J, M                                                                                                                        | B, H, J, M                                     |
| Error               | `jump_indicator` with each solve; a second `level` to check                        | `study_cells`, three meshes or more                                                                                               | `study_fem`, three orders or more              |
| Extra install       | none                                                                               | none; `[tetgen]` for `mesh_TriangularMesh`                                                                                        | `[fem]`, NGSolve                               |

The pages [Face charge](face_charge.md), [Cells](cells.md) and
[Finite elements](fem.md) give each solver's settings and limits;
[How accurate is a result?](accuracy.md) gives the error estimates;
[Inside the solvers](../inside/index.md) sets out the three methods.

## How fast each one is

The repository's validation suite solves its cases against stored finite-element
references. Its sensors are grouped by region, such as above the magnet or under
the plate, and each group's error is taken against its own field. The time to
bring every group within 1 % of the reference, solve and `getB` together, one
run each on an Apple M4 Pro:

| Case                        | Face charge | Finite elements |
| --------------------------- | ----------: | --------------: |
| Magnet beside a soft cube   |      0.02 s |           2.3 s |
| Iron cube in 10 mT          |      0.04 s |          0.52 s |
| Flux concentrator           |      0.04 s |          0.83 s |
| Halbach array on back iron  |      0.06 s |           1.5 s |
| U-core driven by a coil     |      0.07 s |           4.5 s |
| Magnet 4 mm tall on a plate |      0.09 s |           3.1 s |
| CAD part, a triangle mesh   |      0.20 s |            11 s |
| Thin shielding sheet        |      0.28 s |           2.7 s |

Face charge reaches 1 % first on all 14 cases where both methods reach it: 13 to
over 100 times sooner on nine of them, 3 to 12 times on the other five. The
table shows eight; [Face charge](face_charge.md) has all 14, and the times to
0.1 %.

## Where face charge costs more

Face charge solves one dense system for all the parts. Many separate parts and
thin sheets make that system large, and a few weak magnets on their own reach
the same accuracy sooner with `solve_cells`. Above `max_unknowns`, 20,000 by
default, face charge takes a lower level, or refuses.
[Face charge](face_charge.md) gives the numbers and the settings for each case.

## What none of them does

- **Hysteresis and eddy currents.** The materials are static and single-valued:
  a remanence plus a susceptibility, or a B–H curve.
- **Paths.** One solve is one configuration: see
  [what each solver refuses](model.md#what-is-refused).

## In short

Start with `solve_face_charge`. Take `solve_cells` when a part saturates, when
its susceptibility differs along its axes or inside it, or when you need B
inside a part. Check either with `solve_fem` and `study_fem`.
