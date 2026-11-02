# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A solved `ChargeSheet` keeps the part it was solved from, in `part`, at the
  sheet's pose now and through `to_json`, so a solved model can be solved again
  from what it holds
  ([#167](https://github.com/magpylib/magpylib-material-response/pull/167))

## [0.9.0] - 2026-11-02

### Changed

- **Breaking, with warnings for one release:** the solvers take one form and are
  importable from the top level: `solve_cells` (`apply_demag`),
  `solve_face_charge` (`apply_face_charge`), `solve_fem` (`fem_solve`),
  `study_cells` (`mesh_study`), `study_fem` (`fem_study`). The module `demag` is
  now `cells` and `demag_kwargs` is `solve_kwargs`. With no warning: `solved_by`
  says `"cells"`, also for files saved by 0.8.0
  ([#154](https://github.com/magpylib/magpylib-material-response/pull/154))
- **Breaking, with a warning for one release:** a uniform applied field is a
  source, `UniformField(B=...)` or `(H=...)`, in the model's collection, in
  place of the attribute `H_ext`. `getB` of a solved model now includes it, as
  it does a coil's field, it acts on every part, and several add up. A model
  with both raises
  ([#71](https://github.com/magpylib/magpylib-material-response/issues/71),
  [#163](https://github.com/magpylib/magpylib-material-response/pull/163))
- **Breaking, with no warning:** `FemSolution.getB` squeezes a lone observer as
  magpylib does, `(3,)`, not `(1, 3)`, so `getB(one_row)[0]` is now Bx. It is a
  magpylib source: `magpy.getB`, `getH`, `getJ` and `getM` read it, alone or
  with other sources, it moves and copies like one, and it keeps its own copy of
  the coils, so changing one after the solve no longer changes the solution.
  Pickling one raises, where reading it back crashed
  ([#152](https://github.com/magpylib/magpylib-material-response/pull/152),
  [#164](https://github.com/magpylib/magpylib-material-response/pull/164))
- Warnings for what was silent: `getJ` and `getM` inside a solved face-charge
  part give 0, and solving the collection `solve_cells` returned counts its
  polarizations as remanence. The docs say not to read a cell model exactly on a
  boundary between cells, where the field is wrong with no warning
  ([#157](https://github.com/magpylib/magpylib-material-response/pull/157),
  [#158](https://github.com/magpylib/magpylib-material-response/issues/158),
  [#164](https://github.com/magpylib/magpylib-material-response/pull/164))

### Added

- Saturation: `susceptibility` takes a `BHCurve(H=H, B=B)`, in A/m and T, on a
  part or a collection, in place of a constant, in `solve_cells`. Each cell
  takes the susceptibility its own field gives on the curve, the linear solve
  repeated until none changes, with the matrix built once and Anderson
  acceleration: about 3 to 40 solves, more on a sharp knee. A sphere comes out
  within 1 % of its exact answer along the curve. A curve past any material
  warns, as with H and B swapped. Meshers and `to_json` carry it;
  `solve_face_charge` refuses it
  ([#150](https://github.com/magpylib/magpylib-material-response/pull/150),
  [#159](https://github.com/magpylib/magpylib-material-response/pull/159),
  [#162](https://github.com/magpylib/magpylib-material-response/pull/162),
  [#164](https://github.com/magpylib/magpylib-material-response/pull/164))
- `solve_fem` follows such a curve too, so a saturated part has an independent
  check: a sphere's moment is within 0.08 % of its exact value at order 4 from
  0.05 to 2 T, and B at its centre within 0.3 %. Start at order 3: order 2 can
  fail to settle, and says so
  ([#151](https://github.com/magpylib/magpylib-material-response/pull/151),
  [#160](https://github.com/magpylib/magpylib-material-response/pull/160))
- An Agent Skill ships inside the package, so AI coding assistants write the
  current names and know where a solved model can be read. Add it to a project
  with `uvx library-skills`
  ([#164](https://github.com/magpylib/magpylib-material-response/pull/164))

### Fixed

- `mesh_thin_CylinderSegment_with_cuboids` refuses an anisotropic susceptibility
  set on a parent too: it read only the part's own, and came out 11 times off
  with no error
  ([#162](https://github.com/magpylib/magpylib-material-response/pull/162))
- `import *` from an old module gives the old names, and a list of numbers and
  vectors as the `susceptibility` argument works
  ([#162](https://github.com/magpylib/magpylib-material-response/pull/162),
  [#164](https://github.com/magpylib/magpylib-material-response/pull/164))

## [0.8.0] - 2026-11-01

Parts close to each other are solved right, and loops get cheap. Face charge
treated a turned part up to 1 % of its size from another as touching it, and
stopped grading toward another part past that same 1 %; near such parts the
field was off by 10 to 50 %, and by far more with high susceptibilities on both
sides, with a finer level no help. Both are fixed, and three validation cases
with finite-element references guard them. The cost: a solve with parts in that
range takes more unknowns at a level, up to 30 times for magnet arrays. A loop
that moves only the magnets can now reuse the factorization, 4 to 13 times
faster. No change to existing code is needed.

### Added

- Every solved result names its method in `solved_by`: `"demag"`,
  `"face_charge"` or `"fem"`, and `to_json` keeps it
  ([#145](https://github.com/magpylib/magpylib-material-response/pull/145))
- `getB` of a solved face-charge part warns when an observer lies inside it,
  where it gives μ0 H and not B
  ([#144](https://github.com/magpylib/magpylib-material-response/pull/144))
- `apply_face_charge(model, reuse=previous)` keeps the factorization for a loop
  that changes only what drives the solved parts: 72 to 18 ms a call at 1,518
  unknowns, 1.15 s to 0.12 s at 9,558. It compares the meshed parts with the
  previous result's and otherwise solves afresh, so a poor match costs time, not
  accuracy
  ([#135](https://github.com/magpylib/magpylib-material-response/pull/135))

### Changed

- Large face-charge solves of cuboids are up to 1.55x faster on 8 threads and 2x
  on one: panel pairs far apart take Gauss points in place of the exact field.
  Answers move by 7e-7 at most, far below what a finer mesh changes; smaller
  solves are unchanged
  ([#133](https://github.com/magpylib/magpylib-material-response/pull/133))

### Fixed

- Face charge grades a part's panels toward another part out to a gap of 20 % of
  the smaller's half-side, not 1 %. Just past 1 %, the field near the parts was
  up to 48 % off finite elements at level 3 and 7 % at level 6, aligned parts
  too; it is now 0.5 %. It costs more at a level: 1.6 times as long for one part
  beside another, and it grows with the magnets on a part, 30 times for 25
  magnets on a yoke (648 to 13,608 unknowns), where the default level may drop,
  with a warning, and a model near `max_unknowns` may be refused
  ([#140](https://github.com/magpylib/magpylib-material-response/pull/140))
- Face charge no longer treats a turned part as touching another up to 1 % of
  its size away: at a gap of 0.5 % the field near the parts was 9 to 19 % off
  finite elements at level 6, and far more with both at chi 1000, and a finer
  level did not help. It is now about 1 % off at level 6
  ([#137](https://github.com/magpylib/magpylib-material-response/pull/137))
- The field of a solved face-charge model is right far away: it lost accuracy
  from about 10 times the model's size and grew with distance, 14 % off at 100
  times, where it now falls as a dipole's
  ([#134](https://github.com/magpylib/magpylib-material-response/pull/134))

## [0.7.0] - 2026-10-31

Two new ways to solve a model join `apply_demag`: face charge, which needs no
volume mesh, and finite elements, an independent check of any result. This
release requires magpylib 6.1. The entries marked **Breaking** are the ones that
can change or stop existing code.

### Added

- **Face charge solver**, `face_charge.apply_face_charge`: solves a model on the
  surfaces of its parts, without a volume mesh. On the U-core of the docs it
  reaches 1 % in 0.07 s, against 4.5 s for finite elements, and on the six
  documented examples it lands within each reference's uncertainty, 0.64 % at
  worst. It takes cuboids, cylinders, cylinder segments, spheres, tetrahedra and
  closed triangle meshes such as CAD exports, each with a uniform, isotropic
  susceptibility, coupled to each other and to every other source. Each solved
  part becomes a `ChargeSheet`, an ordinary magpylib source that `to_json`
  saves. `refine` adds panels where the charge jumps most, and a solve above
  `max_unknowns`, 20,000 by default (3.2 GB), is refused before it starts
  ([#89](https://github.com/magpylib/magpylib-material-response/pull/89),
  [#104](https://github.com/magpylib/magpylib-material-response/pull/104),
  [#110](https://github.com/magpylib/magpylib-material-response/pull/110),
  [#113](https://github.com/magpylib/magpylib-material-response/pull/113),
  [#114](https://github.com/magpylib/magpylib-material-response/pull/114),
  [#115](https://github.com/magpylib/magpylib-material-response/pull/115),
  [#116](https://github.com/magpylib/magpylib-material-response/pull/116),
  [#123](https://github.com/magpylib/magpylib-material-response/pull/123),
  [#126](https://github.com/magpylib/magpylib-material-response/pull/126)).
- **Finite elements**, `fem` module in a new `fem` extra (NGSolve): `fem_solve`
  and `fem_study` solve the model `apply_demag` takes, with magnets, a uniform
  applied field and current sources, and estimate the error by raising the
  element order. `compare` puts a `mesh_study` and a `fem_study` side by side
  for each observer: both values, both error bars, and whether they agree. The
  reference data of the docs and tests now come from it, in place of exports
  from a commercial code that no one could rerun; `get_dataset` labels them
  "FEM" ([#90](https://github.com/magpylib/magpylib-material-response/pull/90),
  [#93](https://github.com/magpylib/magpylib-material-response/pull/93),
  [#95](https://github.com/magpylib/magpylib-material-response/pull/95),
  [#99](https://github.com/magpylib/magpylib-material-response/pull/99)).
- **Validation in the repository**, `validation/`: 20 cases, 17 with a stored
  reference. Every PR checks the method of moments against them and reports time
  and memory against its base branch. Every night all methods climb to finer
  settings against references made again, and must beat magpylib alone, the
  model without its material response
  ([#99](https://github.com/magpylib/magpylib-material-response/pull/99),
  [#101](https://github.com/magpylib/magpylib-material-response/pull/101),
  [#125](https://github.com/magpylib/magpylib-material-response/pull/125)).
- Documentation: pages on face charge, on checking a result with finite
  elements, and on a magnet in a steel tube, where the method of moments
  converges inside but needs far more cells behind the shield
  ([#101](https://github.com/magpylib/magpylib-material-response/pull/101),
  [#102](https://github.com/magpylib/magpylib-material-response/pull/102),
  [#122](https://github.com/magpylib/magpylib-material-response/pull/122)).

### Changed

- **Breaking:** requires magpylib 6.1, NumPy 2.0.1 and scipy 1.17.1. magpylib
  6.1 computes fields in chunks, on threads: a coil of 2,400 segments around a
  soft core now solves by `apply_face_charge` in 0.28 GB instead of 4.0 GB and
  by `apply_demag` in 0.26 GB instead of 2.0 GB, both 2 to 4 times faster. CI
  tests the package at its lowest pins
  ([#108](https://github.com/magpylib/magpylib-material-response/issues/108),
  [#117](https://github.com/magpylib/magpylib-material-response/pull/117),
  [#119](https://github.com/magpylib/magpylib-material-response/pull/119)).
- **Breaking:** `voxelize` keeps the cells whose centre lies inside the object,
  as its docstring said, and tunes the grid spacing so that they hold the
  object's exact volume, close to `target_elems`. It kept only the cells wholly
  inside, which left a sphere of 1,000 cells 31 % short of its volume and
  moment; `strict_inside=True` still does that. It warns when the volume can be
  held only within 5 %, and refuses beyond that
  ([#96](https://github.com/magpylib/magpylib-material-response/pull/96),
  [#97](https://github.com/magpylib/magpylib-material-response/pull/97)).
- A magnet without a susceptibility, on itself or a parent, stays fixed in
  `apply_demag`, `mesh_study` and `fem_solve`, as in `apply_face_charge`,
  instead of raising an error. A model in which no magnet has one warns that
  nothing responds and skips the solve; when only some are fixed, the log says
  how many
  ([#121](https://github.com/magpylib/magpylib-material-response/pull/121),
  [#128](https://github.com/magpylib/magpylib-material-response/pull/128)).
- `apply_demag` runs bodies on one lattice that do not fill a box, such as a
  voxelized sphere, as one FFT grid. A sphere of 4,529 cells at χ = 1000 took
  8.7 s and 1.5 GB, and now takes 0.8 s and 84 MB. In the clusters of
  `analyze_structure`, `flat` gives each cell's place in its grid's box, and
  only grids that fill their box keep `order`
  ([#97](https://github.com/magpylib/magpylib-material-response/pull/97)).
- Meshers set `sealed = True`, so magpylib-studio shows a mesh as one object
  rather than one per cell, and a mesh draws as the magnet it came from, with
  its opacity and magnetization style
  ([#127](https://github.com/magpylib/magpylib-material-response/pull/127)).

### Removed

- **Breaking:** the `pairs_matching`, `max_dist` and `split` options of
  `apply_demag` and `demag_tensor`, and the helpers `filter_distance` and
  `match_pairs`. On a cube of 2,744 cells the default takes 0.2 s and 0.14 GB,
  where `pairs_matching` took 10 s and over 3 GB and `max_dist` was 11 to 23 %
  off. The arguments after `inplace` and after `src_list` are now keyword-only,
  so an old positional call raises instead of shifting
  ([#87](https://github.com/magpylib/magpylib-material-response/pull/87)).

### Fixed

- A part no longer stops responding, without a word, when it is meshed: every
  mesher keeps the susceptibility and applied field it inherits from the model
  around it
  ([#121](https://github.com/magpylib/magpylib-material-response/pull/121),
  [#128](https://github.com/magpylib/magpylib-material-response/pull/128)).
- `apply_demag`: a thin ring in cylinder-segment cells at high susceptibility no
  longer swings with the cells across its wall. A tube with a 1 mm wall at χ =
  1000 came out −10 %, −68 % and +2 % from its moment with 1, 2 and 3 cells
  across, and now −12 %, −11 % and −11 %, closer as cells are added around the
  ring. Close pairs with a susceptibility from 1 take each other's field
  averaged over the cell, which makes these solves 2 to 7 times slower.
  `demag_tensor` takes `susceptibility`, so it returns the tensor `apply_demag`
  solves with
  ([#92](https://github.com/magpylib/magpylib-material-response/issues/92),
  [#100](https://github.com/magpylib/magpylib-material-response/pull/100),
  [#121](https://github.com/magpylib/magpylib-material-response/pull/121),
  [#128](https://github.com/magpylib/magpylib-material-response/pull/128)).
- `H_ext` is checked before every solve: anything but three finite numbers
  raises an error that says what is expected, and fields above 10 T warn that
  they are most likely H in A/m, not μ0·H in tesla. The Earth's 40 A/m was taken
  as 40 T, silently
  ([#119](https://github.com/magpylib/magpylib-material-response/pull/119)).
- `apply_demag`: a large block between two bodies stays dense unless sparsifying
  pays. A magnet on an iron plate meshed at 0.4 mm ran for more than 12 minutes,
  and now takes 2 minutes and 10 GB, with a memory warning that shows
  ([#86](https://github.com/magpylib/magpylib-material-response/pull/86)).
- `to_json` saves magpylib's `current.TriangleStrip` and
  `current.TriangleSheet`, and `mesh_study` warns when it gives no error
  estimate instead of returning NaN silently
  ([#110](https://github.com/magpylib/magpylib-material-response/pull/110)).

## [0.6.0] - 2026-10-23

### Added

- `convergence.mesh_study`: solves a model on meshes of increasing resolution
  and estimates the error of the finest, as a whole and per sensor, by the grid
  convergence index (Richardson extrapolation over the three finest meshes, the
  error relative to the finest value, a safety factor of 1.25). It gives no
  estimate while the results do not change steadily, and warns when the meshes
  barely grow. Its default mesher, `meshing.mesh_uniform`, gives every part that
  responds cells of one size; `MeshStudy.from_values` gives the same estimate
  for results from another solver
  ([#67](https://github.com/magpylib/magpylib-material-response/pull/67)).
- Documentation: a "How it works" page for engineers, with 3D views from
  magpylib-studio and interactive charts: the cells, field lines through a
  magnet and an iron plate, how two cells act on each other, how many cells are
  needed, and how the solver is chosen
  ([#78](https://github.com/magpylib/magpylib-material-response/pull/78)).
- `apply_demag(..., solver="iterative")`: a matrix-free GMRES solver next to the
  dense direct solver (`solver="direct"`), with `solver_tol` and `max_iter`
  (iterations). Both solvers apply the same interaction rule. GMRES stops at a
  residual below `solver_tol`; measured against the direct solver, the
  polarization agrees to about 2 × `solver_tol` for hard magnets; for soft iron
  at χ = 1000 the gap grows with the cell count, to 160 × `solver_tol` at 4,096
  cells. GMRES that does not converge raises. Its block-Jacobi preconditioner
  scales cells other than cuboids as cubes, so a grid with tetrahedra at χ =
  1000 converges in 80 iterations instead of failing at 1,000
  ([#77](https://github.com/magpylib/magpylib-material-response/pull/77)). Boxes
  of cuboid cells on a uniform grid use FFT products (`demag_fft` module):
  32,768 cells of soft iron solve in about 4 s. Each meshed body is a grid of
  its own, told apart by the Collection its mesher made or, when the cells were
  flattened into one, by the boxes they fill on their lattice; a yoke with its
  legs, meshed with `mesh_all`, then gets FFT products within each body and, for
  pairs large enough to pay for it, between bodies of one cell size. Grids with
  holes, such as voxelized round bodies, and other cell types use dense blocks;
  a dense block is only sparsified when that drops at least half of its entries,
  which at high susceptibility it rarely does. The direct solver assembles its
  matrix in Fortran order and factors it in place, in 1.1 times the matrix's
  memory where `scipy.linalg.solve` on a C-ordered copy took 4 (3,773 cells: 1.9
  GB instead of 4.8)
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- `apply_demag(..., interaction="newell")`: cells act on each other through the
  field averaged over each cell, the volume-averaged Newell tensor (`newell`
  module, generalized to cells of different sizes), instead of the field at each
  cell's centroid, which stays the default (`interaction="point"`). It is the
  Galerkin form of the problem: for a soft body with positive susceptibility in
  a uniform field, the induced moment along the field never exceeds the
  converged value and rises each time every cell is split, so a mesh study knows
  which side the answer is on. At equal cell count neither rule is more accurate
  in general. It needs every cell to be a cuboid, all with one orientation, and
  raises otherwise instead of mixing the two rules, which would make results
  jump when a body turns slightly. `demag_tensor` takes the same argument
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- Cuboid cells sharing a shape and orientation are fast under either rule: each
  distinct displacement is computed once (between two grids of one spacing by
  index arithmetic, without a sort, which took most of a dense assembly),
  reverse blocks follow by reciprocity, and uniform grids use FFT products. Far
  apart, where the closed forms lose precision, both rules switch to a far-field
  expansion
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- `demag_fft.analyze_collection(sources)`: splits a collection into grid, loose
  and generic clusters covering every cell once
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- `scipy>=1.12` dependency
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).

### Changed

- `apply_demag` picks its solver by default (`solver="auto"`): the iterative
  solver when FFT products cover at least a tenth of the interaction matrix, run
  to `solver_tol / 100` so that it matches the direct solver to about
  `solver_tol`, and the direct solver otherwise or when GMRES does not converge
  ([#77](https://github.com/magpylib/magpylib-material-response/pull/77)).
- `get_H_ext(*sources)` no longer takes the `H_ext=` keyword, which was read and
  immediately shadowed by the attribute lookup
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- The direct solver assembles in place (about 2-3x less peak memory), derives
  reverse blocks by reciprocity, and every block is built in chunks of at most
  200,000 (observer, source) pairs. 1,757 tetrahedra: 16.8 s and 7.7 GB before,
  7.5 s and 1.7 GB after
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- `timelog` no longer blocks on exit and skips its watchdog thread while package
  logging is silent (a 27-cell solve went from about 0.5 s to 5 ms)
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).

### Fixed

- `solver="auto"` and `"iterative"` converge on thin sheets of very high
  susceptibility, where GMRES ran out of iterations (a 1 mm mu-metal sheet, χ =
  10,000): GMRES restarts every 100 iterations instead of 20, which also takes
  15 to 55 % fewer iterations on soft iron, for about 2 kB more memory per cell.
  `mesh_TriangularMesh` warns when the surface, not `target_elems`, sets the
  cell count, and its docstring says that the target caps the cell volume
  ([#80](https://github.com/magpylib/magpylib-material-response/pull/80)).
- Meshing edge cases (thin segments, empty voxel grids), scale-relative
  tolerances in inside-masks and in `trimesh_from_model3d`, and a latent import
  error in `get_dataset`
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).

## [0.5.0] - 2026-10-19

### Added

- Serialization (`to_json` / `from_json`) now supports `magnet.Sphere`,
  `magnet.Tetrahedron`, and `magnet.TriangularMesh` — demagnetized tetrahedral
  meshes produced by `mesh_TriangularMesh()` can be persisted and reloaded
  ([#55](https://github.com/magpylib/magpylib-material-response/pull/55)).
- `mesh_TriangularMesh()` — meshes a `magnet.TriangularMesh` into conforming
  tetrahedral cells using [TetGen](https://tetgen.pyvista.org/). Quality and
  cell count are controlled via `target_elems`, `minratio`, and `mindihedral`.
  TetGen is an optional dependency:
  `pip install magpylib-material-response[tetgen]`.
- `trimesh_from_model3d()` helper in `meshing_utils` — builds a closed
  `magnet.TriangularMesh` from a magpylib `model3d` shape trace (cuboid,
  cylinder_segment, ellipsoid, prism, pyramid, tetrahedron).
- `mesh_all()` now supports `magnet.CylinderSegment` and `magnet.TriangularMesh`
  in addition to `magnet.Cuboid` and `magnet.Cylinder`.
- Optional dependency group `tetgen` (`tetgen>=0.8.3`).
  ([#41](https://github.com/magpylib/magpylib-material-response/pull/41)).
- Documentation: a page on the method, a quickstart, an API reference and new
  examples
  ([#53](https://github.com/magpylib/magpylib-material-response/pull/53),
  [#65](https://github.com/magpylib/magpylib-material-response/pull/65)).

### Changed

- **Breaking:** requires magpylib 6. Meshed and demagnetized collections are
  force targets with one point per cell: pass `meshing=1` to `magpylib.getFT`,
  which takes the cell count on the call since magpylib 6
  ([#62](https://github.com/magpylib/magpylib-material-response/pull/62)).
- **Breaking:** an anisotropic susceptibility `(χx, χy, χz)` acts along each
  cell's own axes and turns with its orientation, like its polarization, instead
  of along the global axes. Results change for anisotropic susceptibilities on
  rotated magnets, which before depended on how the magnet was turned in space.
  `mesh_thin_CylinderSegment_with_cuboids`, whose cells are turned about z,
  refuses an anisotropic susceptibility that differs along x and y; use
  `mesh_Cylinder`. Scalar values, and unrotated magnets meshed otherwise, give
  the same results as before
  ([#50](https://github.com/magpylib/magpylib-material-response/issues/50)).
- `mesh_Cuboid` and `mesh_Cylinder` cut an integer `target_elems` into
  near-cubic cells: the cell edge comes from the volume and the count, and each
  side is cut into a whole number of cells. `mesh_Cylinder` also cuts each ring
  in proportion to its middle radius instead of its outer radius, so a cylinder
  no longer gets up to twice the requested cells. The count follows the request
  less closely; a triple still sets the divisions. The cells are the ones
  magpylib's force mesher produces
  ([#56](https://github.com/magpylib/magpylib-material-response/pull/56),
  [#57](https://github.com/magpylib/magpylib-material-response/pull/57),
  [magpylib#1023](https://github.com/magpylib/magpylib/pull/1023)).

### Fixed

- `mesh_Cylinder` gives thin rings enough sectors that every cell's centroid
  lies inside it. Demagnetization matches the field at the centroid, and for
  wide sectors of a thin ring that point lay outside the material: in a
  transverse field, a 4-5 mm tube meshed with 12 cells came out with the wrong
  sign. Thin rings can now get more cells than `target_elems`
  ([#51](https://github.com/magpylib/magpylib-material-response/pull/51)).
- `apply_demag` refuses objects whose dimension, polarization, current or other
  property varies along a path, not only position paths, instead of failing
  inside the solver
  ([#51](https://github.com/magpylib/magpylib-material-response/pull/51)).
- `to_json` writes a path-valued `current` or `diameter`, and a susceptibility
  given as a numpy array, instead of raising
  ([#51](https://github.com/magpylib/magpylib-material-response/pull/51)).
- The legacy options run again: `split` no longer crashes when a chunk holds a
  single cell; `max_dist` no longer crashes, and refuses negative values and the
  `pairs_matching` or `split` it silently ignored; `pairs_matching=True`
  computes its fields with `magpy.func`, since magpylib 6 removed the
  string-based `getH("Cuboid", ...)` it called, and no longer merges pairs of
  cells of different sizes, or every pair of a mesh with cells below 1e-8 m;
  `demag_tensor(..., pairs_matching=True)` no longer refuses a split nobody
  asked for
  ([#62](https://github.com/magpylib/magpylib-material-response/pull/62)).
- Current sources act on each cell at its centroid, where the cells' own fields
  are matched, instead of at its `position`, which for a tetrahedron built from
  vertices is the origin and for a `mesh_Cylinder` ring segment lies on the
  axis: the field above a tetrahedral U-core driven by a coil came out 2.2 times
  too strong. A current placed before a magnet in the collection no longer moves
  the solved polarizations onto the wrong objects
  ([#62](https://github.com/magpylib/magpylib-material-response/pull/62)).
- Material attributes are no longer lost or misread: `H_ext` set on a magnet
  before meshing moves to the meshed collection, where its cells ignored it
  before and stayed unmagnetized; a collection solved as a copy
  (`inplace=False`) keeps the `H_ext` and susceptibility it inherits from a
  parent; three magnets with their own scalar susceptibilities get one value
  each, not one anisotropic value shared by all; `to_json` keeps `H_ext` and a
  Collection's susceptibility. `mesh_all` on a magnet inside a collection
  returns its mesh, not the unmeshed magnet
  ([#63](https://github.com/magpylib/magpylib-material-response/pull/63)).

## [0.4.0] - 2026-05-01

### Added

- Serialization API for magpylib objects: `to_json`, `from_json`. Supports
  `magnet.Cuboid`, `magnet.Cylinder`, `magnet.CylinderSegment`,
  `current.Polyline`, `current.Circle`, `Sensor`, and nested `Collection`. Uses
  namespaced type discriminators (`"magnet.Cuboid"`, `"current.Polyline"`, …)
  and is fully JSON round-trippable
  ([#38](https://github.com/magpylib/magpylib-material-response/pull/38)).
- Added serialization example to docs
  ([#39](https://github.com/magpylib/magpylib-material-response/pull/39)).

### Changed

- **Breaking:** logging is now silent by default. Call
  `magpylib_material_response.configure_logging()` (or
  `loguru.logger.enable("magpylib_material_response")`) to opt in to log output
  ([#30](https://github.com/magpylib/magpylib-material-response/pull/30)).
- Refactored logging to follow the
  [loguru library recipe](https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application):
  the package no longer calls `logger.remove()` and never overrides sinks
  configured by the host application.
- Replaced f-string log calls with structured logging
  (`logger.info("msg {x}", x=...)`).
- Added `MAGPYLIB_LOG_LEVEL`, `MAGPYLIB_LOG_COLORS`, `MAGPYLIB_LOG_TIME`,
  `MAGPYLIB_LOG_MIN_TIME` environment variables.
- Added a `min_log_time` argument to `configure_logging()` that sets the default
  duration threshold used by `timelog` (and by the demag functions when their
  own `min_log_time` argument is left at its default).
- Added `docs/logging.md`.

### Fixed

- `timelog` no longer raises `TypeError` on the failure path when the wrapped
  block raises an exception.

## [0.3.1] - 2025-10-30

### Fixed

- Fixed susceptibility tree traversal
  ([#26](https://github.com/magpylib/magpylib-material-response/pull/26),
  [#28](https://github.com/magpylib/magpylib-material-response/pull/28))

### Changed

- Improved clarity and error handling in `get_susceptibilities` function
- Enhanced logging configuration
- Refactored demag.py for better organization
- Improved documentation theme and layout
  ([#29](https://github.com/magpylib/magpylib-material-response/pull/29))
  - Switched from Furo to PyData Sphinx theme
  - Added custom CSS for full-width content and smaller sidebar
  - Enhanced navigation and branding with Magpylib logo
  - Reorganized examples section structure

### Maintenance

- Updated pre-commit hooks to latest versions
  ([#21](https://github.com/magpylib/magpylib-material-response/pull/21))
- Updated GitHub Actions dependencies
  ([#24](https://github.com/magpylib/magpylib-material-response/pull/24))
- Removed test PyPI repository URL from publish step
  ([#27](https://github.com/magpylib/magpylib-material-response/pull/27))
- Added VSCode settings for pytest configuration

## [0.3.0] - 2024-10-16

### Added

- Support for anisotropic susceptibilities
- FE tests for anisotropic and negative susceptibility values
- Comprehensive tests for interface and computation validation
- Susceptibility input to parent collections
- Support for both scalar (isotropic) and 3-vector (anisotropic) susceptibility
  inputs

### Changed

- Improved interface for susceptibility (chi) input
- Enhanced susceptibility input possibilities:
  - Scalar input now assumes isotropic susceptibility
  - 3-vector input enables anisotropic susceptibility
- Improved internal architecture and code organization

### Fixed

- Various computation fixes confirmed by isotropic and anisotropic tests

## [0.2.1a0]

### Fixed

- Fixed null polarization for rotated objects
  ([#7](https://github.com/magpylib/magpylib-material-response/pull/7))
- Fixed documentation build issues
  ([#6](https://github.com/magpylib/magpylib-material-response/pull/6))

## [0.2.0a0]

### Changed

- Renamed `xi` parameter to `susceptibility` for better clarity
  ([#5](https://github.com/magpylib/magpylib-material-response/pull/5))
- Updated to support magpylib v5
  ([#4](https://github.com/magpylib/magpylib-material-response/pull/4))

[Unreleased]:
  https://github.com/magpylib/magpylib-material-response/compare/0.9.0...HEAD
[0.9.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.8.0...0.9.0
[0.8.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.7.0...0.8.0
[0.7.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.6.0...0.7.0
[0.6.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.5.0...0.6.0
[0.5.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.4.0...0.5.0
[0.4.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.3.1...0.4.0
[0.3.1]:
  https://github.com/magpylib/magpylib-material-response/compare/0.3.0...0.3.1
[0.3.0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.2.1a0...0.3.0
[0.2.1a0]:
  https://github.com/magpylib/magpylib-material-response/compare/0.2.0a0...0.2.1a0
[0.2.0a0]:
  https://github.com/magpylib/magpylib-material-response/releases/tag/0.2.0a0
