# Face charge: a cuboid's full linear response in milliseconds, no user mesh

A study from 2026-10-23/24. This folder is a study, not package code: ruff and
codespell skip it, nothing here is tested in CI, and nothing here is imported by
`magpylib_material_response`.

**Question:** compute a cuboid with susceptibility χ (and remanence) in the
field of other magpylib sources, without asking the user for a mesh, in
milliseconds.

**Answer, for one isotropic linear cuboid:** yes. The solve takes 0.08 ms for a
magnet and under 1 ms for a soft part. Accuracy is 3e-5 for the magnet and about
1e-3 for the soft part, at sensors 0.5 mm or more from a 10 mm body.
material-response's MoM reaches 2.5 % on the same soft-part case with 32,768
cells in 6 s.

## The idea: no new physics, a different representation

1. **Faces only.** With a uniform isotropic χ and a uniform remanence M0,
   `M = M0 + χH` and `div B = 0` give `div M = 0` inside the body. The whole
   body is then a surface charge σ = M·n on its 6 faces, solving

   ```
   (1/χ + 1/2) σ − K'σ = H_ext·n + M0·n / χ
   ```

   K' is the adjoint double-layer (Neumann–Poincaré) operator of the surface. It
   depends on the shape only.

2. **χ is a scalar in front of a shape-only operator.** K' has a real spectrum.
   The "demagnetization spectrum" n = 1/2 − λ lies in [0.17, 0.74] for a cube
   and [0.06, 0.78] for a 20×10×2.5 plate, away from 0 and 1. The response for
   any χ is therefore a short sum of poles χ/(1 + χ nₖ).
   - Six poles give the moment in a uniform field for every χ from 1e-3 to 1e6,
     to 2e-6 (cube) and 4e-6 (plate); each pole adds about one digit
     (`spectrum.txt`).
   - The weights sum to 1, and their first moment is the magnetometric demag
     factor.
   - The moment of one uniformly magnetized cell is a lower bound (Jensen's
     inequality on that spectrum).

3. **Do the shape work once.** Per shape: the face mesh, the operator, and one
   Schur decomposition per mirror-parity block. The cuboid's three mirror planes
   split the operator into 8 independent blocks. Per χ: triangular solves. Per
   external field: H at the load points, then mat-vecs.

What makes it accurate:

- a linear charge per panel, i.e. P1 discontinuous Galerkin with densities 1, s,
  t;
- closed-form panel fields;
- the Galerkin flux computed as the integral of the bounded weighted solid angle
  of the test panel;
- panels graded toward the edges (power 2.5), in layers sized by the body's
  smallest side.

Thin plates need the layers sized by thickness.

## Findings

Numbers are from `results/`, produced by the scripts listed under each finding.

1. **Kernels** (`kernels.txt`): the constant-density rectangle field matches
   magpylib's `Cuboid` to 1e-10. The linear-density fields match brute-force
   quadrature to 6e-14.

2. **Cube polarizability** (`cube_polarizability.txt`), against Helsing &
   Perfekt (2013), α(χ→∞) = 3.644305:

   | Discretization                    | Unknowns |  Error  | Order |
   | --------------------------------- | -------: | :-----: | :---: |
   | constant panels, collocation      |    6,144 | −1.75 % |  1.4  |
   | constant panels, Galerkin         |    3,456 | −0.38 % |  1.9  |
   | same, Richardson on n = 16, 24    |        — |  +3e-5  |   —   |
   | constant panels, geometric layers |    4,704 | −0.63 % |  1.3  |
   | linear panels (P1), Galerkin      |    1,152 | −4.4e-4 |  3.4  |
   | linear panels (P1), Galerkin      |    4,608 | −4.1e-5 |  3.4  |

   More geometric layers change nothing for constant panels (L = 3, 6, 9 agree
   to 6 digits). Each layer is too coarse for its distance from the edge.

3. **Near field: references and convergence** (`convergence.txt`). Test cases
   are in `cases.py`:
   - a 10 mm cube or a 20×10×2.5 mm plate;
   - χ = 1000, 10, 1, or a magnet (χ = 0.05, J = 1.2 T);
   - a tilted 5 mm NdFeB cube 2 mm below, or a uniform field;
   - sensors 0.1–5 mm off a face, an edge and a corner.

   The references are Richardson limits of P1 on n = 16, 20, 24 (cube) and of
   the layered mesh on m = 4, 6, 8 (plate). The constant-panel limit (n up
   to 48) agrees with the cube reference to 1e-3 at ≥ 0.5 mm and 2e-3 at 0.25
   mm, within its own uncertainty. At 0.1 mm off the face they differ by 1e-2.
   There the constant panels are wrong, since their near field is first order.

4. **Accuracy of the engine** (`grading.txt`, maximum relative error of the
   body's B):

   | Case                    | Mesh  | Unknowns | ≥ 1 mm | ≥ 0.5 mm | < 0.5 mm |
   | ----------------------- | ----- | -------: | :----: | :------: | :------: |
   | cube χ = 1000 + magnet  | m = 4 |    1,152 | 3.7e-3 |  8.0e-3  |  3.5e-2  |
   | cube χ = 1000 + magnet  | m = 6 |    2,592 | 6.6e-4 |  1.4e-3  |  1.2e-2  |
   | magnet χ = 0.05, 1.2 T  | m = 3 |      648 | 3.2e-5 |  5.2e-5  |  2.5e-4  |
   | plate χ = 1000 + magnet | m = 4 |    2,976 | 1.9e-3 |  3.5e-3  |  1.7e-2  |
   | plate χ = 1000 + magnet | m = 6 |    6,516 | 5.2e-4 |  9.9e-4  |  5.4e-3  |

5. **Against material-response's MoM** (`mr_soft_cube.txt`, `mr_magnet.txt`).
   The MoM uses the point rule and the solver chosen by `auto`, on the same
   case. The magnet is rotated, so the Newell rule refuses it.

   | Soft cube χ = 1000 | Unknowns | Solve  | ≥ 1 mm | ≥ 0.5 mm |
   | ------------------ | -------: | :----: | :----: | :------: |
   | MoM, 4,096 cells   |   12,288 | 0.63 s |  12 %  |   17 %   |
   | MoM, 32,768 cells  |   98,304 | 6.2 s  | 2.5 %  |  3.3 %   |
   | face charge, m = 6 |    2,592 | 0.7 ms | 6.6e-4 |  1.4e-3  |
   - **Moments:** both methods agree to 1 %.
   - **Convergence:** the MoM near field converges toward the face-charge values
     as its cells are refined.
   - **Magnet:** the MoM at 32,768 cells (4.1 s) agrees with the 648-unknown
     face-charge result to 1.1e-4 (≥ 1 mm), and the gap shrinks with the cells.
   - **Field output:** MoM getB at 18 sensors takes about 370 ms at 32,768
     cells; face charge takes about 2 ms.

6. **Timing** (`timing.txt`, a 14-core Mac, NumPy, one process):

   | Body          | Unknowns | Build once | Solve, any χ | Solve, fixed χ | Load points |
   | ------------- | -------: | :--------: | :----------: | :------------: | :---------: |
   | magnet (cube) |      648 |   0.1 s    |   0.28 ms    |    0.08 ms     |   0.9 ms    |
   | soft cube     |    2,592 |   1.5 s    |    2.1 ms    |    0.67 ms     |   2.7 ms    |
   | soft plate    |    2,976 |   1.9 s    |    2.7 ms    |    0.88 ms     |   3.1 ms    |
   - "Load points" is magpylib's `getH` of one Cuboid source at the load points
     (3×3 per panel). It now dominates the solve.
   - Field output costs 30–120 µs per sensor.
   - Splitting into the 8 mirror blocks is exact (1e-11 against the full
     operator). It cuts the Schur decomposition about 10× and the solve about
     8×.

## Not done, and limits

- **Anisotropic χ and saturation** bring back volume charge (div M ≠ 0). The
  faces-only shortcut does not cover them.
- **Several bodies with χ** (magnet on a yoke, arrays) are not coupled yet.
  Touching bodies need interface charges.
- **Sensors closer than about 0.25 mm** to a soft body's edge or corner see
  about 1e-2 error. Those are the singular edge zones.
- **Evaluation** costs 30–120 µs per sensor. Many sensors need a far-field
  compression (multipole or treecode).
- **Build per shape** (0.1–2 s) could be cached by aspect ratio. Assembly could
  also be 8× cheaper using the symmetry; today the full operator is assembled.
- **Other shapes:**
  - polyhedra (`TriangularMesh`, `Tetrahedron`) would use linear-density
    triangle closed forms;
  - cylinders need faceting or curved-panel quadrature;
  - spheres and ellipsoids are closed form already.
- **Novelty is not checked.** The pieces are known (see below); whether this
  combination exists as a tool is not.

## Files

| File                                                                  | What                                                                                        |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `bem.py`                                                              | constant-panel solver (collocation and Galerkin), rectangle kernel, meshes (`layer_panels`) |
| `p1.py`                                                               | linear-density closed forms, P1 Galerkin assembly, load, moment, field                      |
| `sym.py`                                                              | mirror-parity bases (Z₂³) that split the operator into 8 blocks                             |
| `engine.py`                                                           | `CuboidResponse` (one Schur) and `SymCuboidResponse` (8 blocks, `fix_chi`)                  |
| `cases.py`                                                            | test shapes, the magnet source, the sensors                                                 |
| `ref.py`, `ref_p1.py`, `ref_lay.py`                                   | reference runs (constant, P1 power grading, P1 layered); write `*.npy`                      |
| `conv.py`, `conv2.py`                                                 | convergence and cross-checks of the references                                              |
| `t_rect*.py`, `t_lin.py`                                              | kernel checks                                                                               |
| `t_cube*.py`, `t_geo.py`, `t_nq.py`, `t_gal_speed.py`, `t_p1cube.py`  | cube polarizability by discretization, quadrature checks                                    |
| `t_spec.py`                                                           | spectrum, sum rules, few-pole fits in χ                                                     |
| `t_engine.py`, `t_nqrhs.py`, `t_grade.py`, `t_sym.py`, `t_symeng*.py` | engine accuracy, grading, symmetry split, timing                                            |
| `t_mr.py`, `t_mr2.py`, `t_mr_magnet.py`                               | material-response MoM on the same cases                                                     |
| `*.npy`                                                               | reference fields (the reference runs take minutes and up to 6 GB)                           |

Run from this folder with the package's environment, e.g. `python t_grade.py`.
The `ref*.py` runs are heavy: `ref.py 40 48` peaks at about 6 GB. The `.npy`
files hold their output.

## Sources

- J. Helsing and K.-M. Perfekt (2013), "On the polarizability and capacitance of
  the cube", Appl. Comput. Harmon. Anal. 34, 445–468: cube polarizability
  3.644305 (checked in an earlier study).
- Background, not re-checked for this study:
  - Bergman's spectral representation of effective permittivity;
  - Mayergoyz, Fredkin and Zhang on Neumann–Poincaré spectra (electrostatic
    resonances);
  - Ammari and Kang on polarization tensors;
  - Chen, Pardo and Sanchez on χ-dependent demagnetizing factors of prisms;
  - Aharoni on the magnetometric factors of prisms.
