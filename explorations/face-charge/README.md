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

**Coupled bodies work too.** A magnet glued on an iron plate is within 2.4e-3 at
every sensor with 3,888 unknowns, including the shielded side under the plate.
There the MoM is off by 70 % with 17,152 cells. Moving a part takes a rebuild of
about 3 s; a new polarization, χ or external field takes about 6 ms. Bodies may
be placed and turned freely.

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

7. **Coupled bodies** (`multibody.py`; `contact.txt`, `magnet_on_plate.txt`).
   Each body's equation gets the other bodies' fields on its faces, computed
   with the same weighted-solid-angle flux.
   - **Contact.** Where two faces share a plane, the coplanar coupling is exact:
     no principal value, and the one-sided limit gives −1/2 of the overlap
     integral of the two charges. The overlap of a test panel with the other
     body's panel is clipped as a polygon (a turned rectangle when the body is
     turned in the plane) and integrated exactly, so the meshes need not match.
   - **Rotation.** Each body has an orientation. The other body's Gauss points
     and normals are mapped into the test panel's frame; loads and fields are
     rotated in and out.
   - **Footprint lines.** Each body's mesh is also graded toward the edges of
     every body touching it (`add_contact_lines`). These are exact when the two
     bodies are turned by multiples of 90° against each other. Otherwise they
     only bound the footprint, because the mesh lines follow the body's axes.

   Checks:
   - **Two touching cubes, equal χ, matching meshes:** they reproduce the merged
     20×10×10 block to 1e-13 (B and moment). Their interface charges cancel to
     1e-12.
   - **Halves meshed differently:** they converge to the merged block (1.3e-2,
     9.6e-3, 3.7e-3), slower than matching meshes (1.1e-2, 4.2e-3, 9e-4). An
     interface that should carry no charge is the worst case for non-matching
     meshes.
   - **Magnet on plate** (`yoke_case.py`): a 6×6×4 mm magnet (1.2 T, χ = 0.05)
     off-centre on a 20×20×5 mm plate (χ = 1000). Sensors are 0.5–5 mm above the
     magnet, beside it on the plate, under the plate (about 5 mT, shielded) and
     off the plate's side. The face-charge levels m = 3, 4, 6 converge with
     order 3.6; errors are against their Richardson limit.

   | Method                       | Unknowns | Above magnet | Beside | Under plate | Plate side |
   | ---------------------------- | -------: | :----------: | :----: | :---------: | :--------: |
   | face charge, m = 3           |    3,888 |     2e-5     |  6e-4  |    2e-3     |    2e-3    |
   | face charge, m = 4           |    6,912 |     9e-6     |  2e-4  |    8e-4     |    8e-4    |
   | MoM, 1 mm cells (0.9 s)      |    6,432 |     4e-3     |  2e-2  |    300 %    |    20 %    |
   | MoM, 0.5 mm cells (7.0 s)    |   51,456 |     1e-3     |  4e-3  |    70 %     |    4 %     |
   | MoM, Richardson on 2, 1, 0.5 |        — |     2e-4     |  1e-3  |    10 %     |    8e-3    |
   - **Independent check:** where the MoM converges, its own extrapolation lands
     on the face-charge limit to 2e-4–1e-3. Under the plate it is still falling
     toward it.
   - **MoM at 0.4 mm cells** (32,250 cells) ran over 12 minutes and reached 10.8
     GB before I stopped it, where 0.5 mm took 7.0 s; see finding 8. With the
     fix of finding 8 it takes 2 min and 10 GB. It is still 40 % off under the
     plate, and its extrapolation over 1, 0.5 and 0.4 mm lands on the
     face-charge limit to 3e-5 above the magnet and 3e-2 under the plate
     (`mom_jump.txt`).
   - **Timing at m = 3:** plate self-block 1.7 s, cross blocks 0.8 s, LU 0.1 s.
     After that, a repeat solve takes 6.3 ms and field output 0.2 ms per sensor.
   - **Footprint lines are needed.** Without them the plate's mesh would depend
     on its shape only (cacheable), but the error under the plate is 6e-2 at m =
     4 and 1.6e-2 at m = 6.

   Rotated bodies (`t_rot.py`, `t_yoke_turn.py`; `rotation.txt`):
   - **Exact checks** (machine precision, 5e-15 to 7e-13):
     - the axis-aligned results are unchanged;
     - one body turned together with its field gives B turned;
     - the whole magnet-on-plate turned by a random rotation gives every
       sensor's B turned;
     - the magnet turned 90° equals an unturned magnet with its x and y sides
       swapped.
   - **Magnet turned 30° on the plate:** it agrees with the MoM's extrapolation
     to 3e-4 above the magnet and 3e-3 beside it. But the order falls from 3.6
     to 1.85: m = 3 gives 5e-2 under the plate and 2e-2 beside the magnet, and m
     = 6 gives 1e-2. The turned footprint edges cut across panels that follow
     the plate's axes. Panels that follow the footprint (triangles) would fix
     that.

8. **Why the MoM jumps from 7 s to over 12 minutes** (`run_04.py`;
   `mom_jump.txt`, `mom_jump_patched_run.txt`). `mom_jump.txt` is the output of
   two scripts that probed 0.6.0's internals, `probe_jump.py` and
   `csr_bytes.py`. They and the one-line patch were removed after the fix was
   merged; they are in commit 6c96cdf. Two causes, on main's code (0.6.0):
   1. **The grids stop matching.** At 0.5 mm the plate and the magnet share one
      0.5 mm lattice, and the block between them is an FFT. At 0.4 mm the
      plate's 5 mm side gets 12 cells of 0.4167 mm while the magnet keeps 0.4
      mm. The block is then explicit: 608 M entries each way. `mesh_uniform` has
      the same exposure, since it cuts each side into `round(side / size)`
      cells.
   2. **Above 2 GB the block is sparsified regardless,** without the first-chunk
      check that it pays. At χ = 1000 the row budget is 5e-13, so the CSR keeps
      99 % of the entries:
      - 7.2 GB per direction stored (12 bytes per entry, int32 indices, measured
        on the real block);
      - about 29 GB while built: 24-byte triplets per entry, copied once more by
        the concatenation before the CSR is made;
      - the reverse direction is built again;
      - against 4.9 GB per direction dense.

      The run I stopped had reached 10.8 GB after 12 minutes, still building.

      The 2 GB warning counts dense blocks only, so it stays silent. The point
      rule's reverse block is not a transpose here, because the two bodies'
      cells differ in shape.

   **Fix:** drop the `72 * n_a * n_b <= DENSE_BLOCK_MAX_BYTES` condition, so the
   first chunk decides at every size (one line). The 0.4 mm case then took 2 min
   2 s: 1 min 39 s building 9.7 GB of dense blocks, 22 s of GMRES, peak 10.0 GB.
   The existing warning fired.

   **Merged as 8cd79ea**
   ([#86](https://github.com/magpylib/magpylib-material-response/pull/86),
   figure from `figures/pr86_dense_cross.py`). The PR got more than the one line
   in review:
   - whether to sparsify is judged on at least 64 observers drawn at random
     across the body, not on the first slab;
   - a block goes sparse only from 90 % dropped, because the sparse product runs
     on one thread, or from 60 % when the explicit blocks would pass 2 GB in
     all;
   - the CSR is built per chunk, with int32 columns, at about 20 bytes per entry
     at peak;
   - a reverse block given by reciprocity is a transposed view with a scale, not
     a copy;
   - the warning counts sparse blocks too and names the cause (bodies whose
     grids differ in spacing), and NaN entries are kept.

   **Checked on merged main** (`mom_jump_main86_run.txt`): the 0.4 mm case takes
   2 min 10 s (beside the test suite), peak 9.9 GB, and warns about 9.7 GB of
   explicit blocks. Its B at the 16 sensors equals the patched run's to 3e-14.
   - **Still worth doing:** warn before building, from the predicted size, and
     name a cell size that every body's sides are multiples of. The warning now
     comes after the build.
   - **Not done:** a matrix-free or FMM cross block.

## Solved bodies as magpylib sources

`source.py`, `LinearChargeTriangle`: triangles whose magnetic surface charge
(M·n, in A/m, given at each corner) varies linearly across each one, in closed
form. It is a `magpylib.misc.CustomSource`, so `getB`, `getH`, paths,
collections and `show()` work with magpylib as it is; a field function and class
in magpylib core could take its place later with the same parameters. It gives H
everywhere and B outside the bodies: inside a magnetized body B also holds the
magnetization, which the surface charge does not carry.

```python
import fc  # validation/fc.py
solved = fc.apply_face_charge(collection, m=3)  # a copy: each soft body is now its surface charge
B = solved.getB(sensors)  # like getB of a collection solved by material-response
```

`from_cuboid` and `from_triangles` turn a solved cuboid body or triangle body
(TriBody, MacroBody, HBody) into a source; `fc.apply_face_charge` does it for a
whole magpylib collection, keeping coils and magnets without susceptibility as
they are. Checks (`t_source.py`, a few seconds):

- uniform charge against magpylib's own `misc.Triangle`: 5e-15;
- linear charge against uniform pieces: 1.9e-3, 4.7e-4, 1.2e-4 at 64, 256 and
  1,024 pieces, the second-order convergence of the pieces to it;
- the solvers' own fields, as sources: 1.3e-10 (two values of μ0 differ by
  5e-10); a collection through `apply_face_charge` against `fc.solve`: 1e-11;
- a path of two positions turned 30°, against the same by hand: 5e-16; a copy
  with doubled charge, and two sources in one `getB`: exact;
- `show()` draws it in plotly and matplotlib; 192 triangles at 1,000 observers
  take 0.08 s.

magpylib checks a custom field function's arguments with `getfullargspec`, which
counts `self`, so a bound method is refused through `field_func`. The class sets
it past the setter: unlike a closure, a bound method follows `copy()` to the
copy, and `getB` groups sources by it, so each source keeps its own charge.

## Not done, and limits

- **Anisotropic χ and saturation** bring back volume charge (div M ≠ 0). The
  faces-only shortcut does not cover them.
- **Coupled bodies:** the contact footprint sets the mesh, so moving a part
  means a rebuild of seconds. The NumPy closed-form assembly dominates that
  rebuild; compiled kernels would cut it. A footprint turned against the other
  body's axes converges at order 1.85 instead of 3.6; panels that follow the
  footprint would need triangles.
- **Sensors closer than about 0.25 mm** to a soft body's edge or corner see
  about 1e-2 error. Those are the singular edge zones.
- **Evaluation** costs 30–120 µs per sensor. Many sensors need a far-field
  compression (multipole or treecode).
- **Build per shape** (0.1–2 s) could be cached by aspect ratio. Assembly could
  also be 8× cheaper using the symmetry; today the full operator is assembled.
- **Other shapes:**
  - any closed triangle mesh (a `TriangularMesh`, a CAD export) is solved in
    `triangles/`, graded automatically toward its sharp edges; see
    `triangles/PROGRESS.md`. Cylinders work as exported facets, with the error
    falling as 1/N²; narrow facets share panels, so a fine export costs few more
    unknowns, but assembly still grows with the square of the triangles until
    compression;
  - spheres and ellipsoids are closed form already.
- **Novelty is not checked.** The pieces are known (see below); whether this
  combination exists as a tool is not.

## Files

| File                                                                              | What                                                                                        |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `bem.py`                                                                          | constant-panel solver (collocation and Galerkin), rectangle kernel, meshes (`layer_panels`) |
| `p1.py`                                                                           | linear-density closed forms, P1 Galerkin assembly, load, moment, field                      |
| `sym.py`                                                                          | mirror-parity bases (Z₂³) that split the operator into 8 blocks                             |
| `engine.py`                                                                       | `CuboidResponse` (one Schur) and `SymCuboidResponse` (8 blocks, `fix_chi`)                  |
| `cases.py`                                                                        | test shapes, the magnet source, the sensors                                                 |
| `ref.py`, `ref_p1.py`, `ref_lay.py`                                               | reference runs (constant, P1 power grading, P1 layered); write `*.npy`                      |
| `conv.py`, `conv2.py`                                                             | convergence and cross-checks of the references                                              |
| `t_rect*.py`, `t_lin.py`                                                          | kernel checks                                                                               |
| `t_cube*.py`, `t_geo.py`, `t_nq.py`, `t_gal_speed.py`, `t_p1cube.py`              | cube polarizability by discretization, quadrature checks                                    |
| `t_spec.py`                                                                       | spectrum, sum rules, few-pole fits in χ                                                     |
| `t_engine.py`, `t_nqrhs.py`, `t_grade.py`, `t_sym.py`, `t_symeng*.py`             | engine accuracy, grading, symmetry split, timing                                            |
| `t_mr.py`, `t_mr2.py`, `t_mr_magnet.py`                                           | material-response MoM on the same cases                                                     |
| `multibody.py`                                                                    | coupled axis-aligned bodies: contact coupling, footprint lines, coupled solve               |
| `source.py`, `t_source.py`                                                        | solved bodies as magpylib sources (`LinearChargeTriangle`), and its checks                  |
| `t_split.py`, `t_split_nc.py`                                                     | touching halves vs the merged block, matching and non-matching meshes                       |
| `t_rot.py`, `t_yoke_turn.py`                                                      | rotated bodies: exact checks, magnet turned 30° on the plate                                |
| `run_04.py`, `figures/`                                                           | the MoM on the 0.4 mm magnet-on-plate case; the source of the #86 figure                    |
| `yoke_case.py`, `t_yoke.py`, `t_yoke_mr.py`, `yoke_compare.py`, `t_yoke_speed.py` | magnet on plate: face-charge levels, MoM levels, comparison, timing                         |
| `*.npy`                                                                           | reference fields (the reference runs take minutes and up to 6 GB)                           |

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
