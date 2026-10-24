# Face charge on triangle meshes (CAD parts)

A homogeneous, isotropic, linear part is fully described by the charge on its
surface, so a CAD part needs only its surface triangles (a magpylib
`TriangularMesh`), no tetrahedra. This folder builds that solver step by step.

**Status: step 1 of 4 done.** Next: step 2, grading toward sharp edges.

| Step | What                                                                                                                                      | Status |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1    | Linear charge per triangle; checked on a sphere and on a cube given as triangles                                                          | done   |
| 2    | Grade the mesh toward sharp edges (by the angle between neighbours); cube as triangles, then the U-core as one closed mesh against FEM    | todo   |
| 3    | A CAD-like part (fillets, a hole) against the tetrahedral MoM; a cylinder as `Cylinder` with curved panels vs as 32, 64, 128-sided meshes | todo   |
| 4    | Compression for large meshes (fast multipole or hierarchical matrices)                                                                    | todo   |

## Step 1: the solver

`tri.py`: the field of a flat triangle whose charge is 1, u or v (linear across
it), in closed form: Wilton's per-edge potential and solid angle, and the linear
part reduced to edge integrals, as in explore/cells-and-fem's `lintet.py`.
`tri_solver.py`: the same Galerkin scheme as the cuboid panels, on any closed
mesh. `meshes.py`: icosphere, and cuboid panels cut into triangles.

- **Kernel** (`t_tri.py`, `results/kernel.txt`): within 1e-9 of a quadrature
  refined toward each observer (0.003 to 3 away, in the plane), and a rectangle
  cut in two matches the rectangle closed form to 2.6e-13. A sign in the edge's
  foot point was wrong at first and gave O(1) errors; the check caught it.
- **Sphere** (χ 1 and 1000, uniform field; `step1.py`, `results/step1.*`): the
  moment's error follows the faceted volume's, −8.56e-3 against −8.61e-3 at
  1,280 triangles. Per unit of the faceted volume, the magnetization is within
  5e-5 of the exact sphere's. The faceting is the error; the solver adds little.
- **Cube as triangles** (χ → ∞): −2.9e-3, −9.9e-4, −4.6e-4 against Helsing &
  Perfekt's 3.644305, as the cuboid solver on the same panels (−3.5e-3, −1.1e-3,
  −4.5e-4), with twice the unknowns: each panel is two triangles.

![Step 1: icosphere errors against the faceting; cube polarizability for both solvers](results/step1.png)

Cost so far: dense, 3,840 unknowns in 10 s (1,280 triangles). The pairs closer
than twice their size are re-integrated on 16 sub-triangles each, one test
triangle at a time in Python.

## How to resume

From this folder, with material-response's environment: `python step1.py` (about
a minute). Each step writes `results/step<k>.*`.
