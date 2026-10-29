# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `magpylib_material_response.fem` in a new `fem` extra (NGSolve): `fem_solve`
  and `fem_study` solve the model `apply_demag` takes by finite elements, on the
  magnets' own shapes, and estimate the error by raising the element order, or
  warn when the orders give none. An independent check of the method of moments:
  with an exact open boundary and layers refined toward the edges, it puts a
  cube's polarizability within 2e-5 of Helsing and Perfekt's in seconds.
  Magnets, a uniform applied field and current sources: with currents the
  magnets take the total potential, so a coil around soft iron costs no
  accuracy, and on the U-core of the docs it agrees with an independent FEM
  within 0.25 %. The reference data of the docs and the tests are now its
  results, which `scripts/fem_references.py` recomputes, in place of exports
  from a commercial code that no one could rerun. `compare` puts a `mesh_study`
  and a `fem_study` side by side, per observer: both values, both error bars,
  and whether they agree. In the repository, `validation/` holds 19 cases, 16
  with a stored reference; every PR checks the method of moments against them
  and is timed against its base branch in a comment, and every night all methods
  climb their settings against references made again, their records kept on a
  `benchmarks` branch, and must beat magpylib alone, the model without its
  material response. It found a mesh NGSolve could not assemble, which
  `fem_solve` now makes again at another grading. Two docs pages: checking a
  result with finite elements, and a magnet in a steel tube, where the method of
  moments converges inside but behind the shield needs far more cells
  ([#90](https://github.com/magpylib/magpylib-material-response/pull/90),
  [#93](https://github.com/magpylib/magpylib-material-response/pull/93),
  [#95](https://github.com/magpylib/magpylib-material-response/pull/95),
  [#99](https://github.com/magpylib/magpylib-material-response/pull/99),
  [#101](https://github.com/magpylib/magpylib-material-response/pull/101),
  [#110](https://github.com/magpylib/magpylib-material-response/pull/110)).
- `face_charge.apply_face_charge` solves cuboids, cylinders, cylinder segments,
  spheres, tetrahedra and closed triangle meshes, such as CAD exports, with a
  uniform, isotropic susceptibility on their surfaces, without a volume mesh,
  coupled to each other and to every other source; each solved part becomes a
  `ChargeSheet`, an ordinary magpylib source. On the six documented examples it
  lands within each reference's uncertainty, 0.64 % at worst. `refine` splits
  the panels where the charge jumps most, and each sheet reports
  `jump_indicator` and `mean_polarization`, hints for accuracy and saturation;
  the solved collection's `face_charge_info` holds the unknowns, level,
  refinement rounds and time, and the unknowns are logged before the solve. All
  bodies form one dense system, factorized in single precision and refined to
  double's accuracy: `max_unknowns`, 20,000 (3.2 GB, 4.8 GB while solved) by
  default, refuses a larger one before it is assembled, a triangle mesh's before
  it is cut; without a `level`, the highest level up to 3 that fits is taken. It
  runs on magpylib's `defaults.threads` threads where magpylib has that setting,
  otherwise on up to 8. Parts of one material that touch are solved as one, and
  parts a gap below their panels apart are graded toward each other. `to_json`
  saves a solved collection
  ([#89](https://github.com/magpylib/magpylib-material-response/pull/89),
  [#104](https://github.com/magpylib/magpylib-material-response/pull/104),
  [#110](https://github.com/magpylib/magpylib-material-response/pull/110),
  [#113](https://github.com/magpylib/magpylib-material-response/pull/113),
  [#114](https://github.com/magpylib/magpylib-material-response/pull/114),
  [#115](https://github.com/magpylib/magpylib-material-response/pull/115),
  [#116](https://github.com/magpylib/magpylib-material-response/pull/116)).

### Changed

- `voxelize` keeps the cells whose centre lies inside the object, as its
  docstring said, and tunes the grid spacing so that these touching cells hold
  the object's exact volume, and a magnet its moment, close to `target_elems`.
  It kept only the cells lying wholly inside by default, which left a sphere a
  quarter short of its volume at 1,000 cells, and its moment with it;
  `strict_inside=True` still does that. Where no spacing near the target holds
  the volume, it takes the count that does, as a flat body holds it only with
  whole layers; failing that, the closest within 5 %, with a warning; beyond, it
  refuses. It tries a lattice with a cell centred at the object's origin and one
  with a cell corner there, which splits no cell along faces through the origin,
  such as a cylinder segment's sides. A triangle mesh's lattice is anchored at
  the middle of its box: off its own origin, the mesh lost cells, in both modes
  ([#96](https://github.com/magpylib/magpylib-material-response/pull/96),
  [#97](https://github.com/magpylib/magpylib-material-response/pull/97)).
- `apply_demag` runs cells on one lattice that do not fill a box, such as a
  voxelized sphere, as one grid by FFT, with zero polarization in the holes,
  when they fill at least an eighth of their box and are four cells deep. Such
  bodies ran as many small boxes and a rim of loose cells, which FFT products
  barely covered. A voxelized sphere of 4,529 cells at χ = 1000 took 8.7 s and
  1.5 GB, and now takes 0.8 s and 84 MB; one of 1,304 cells took 0.76 s, and now
  0.24 s. Grid clusters from `analyze_structure` give each cell's place in the
  grid's box as `flat`, and only grids that fill their box keep `order`. The
  direct solver gathers the blocks of grids in smaller chunks: a cube of 4,096
  cells peaks at 1.4 GB instead of 2.1 GB, in the same time. A cell on another's
  site, or positions noisy just above the lattice tolerance, made the structure
  analysis fail or ask for terabytes; such cells now run loose. The inside test
  of a triangle mesh runs on chunks of points: a 792-face tube voxelized into
  3,000 cells needed over 3 GB and now takes 1.6 GB
  ([#97](https://github.com/magpylib/magpylib-material-response/pull/97)).
- Requires the magpylib release with
  [magpylib#1059](https://github.com/magpylib/magpylib/pull/1059), which
  computes fields in chunks, on threads. A drive of many current segments no
  longer takes gigabytes: #108's coil of 2,400 segments around a soft core
  solves by `apply_face_charge` in 0.28 GB instead of 4.0 GB, by `apply_demag`
  in 0.26 GB instead of 2.0 GB, both 2 to 4 times faster. Face charge no longer
  evaluates its drive in chunks of its own
  ([#108](https://github.com/magpylib/magpylib-material-response/issues/108)).

### Removed

- **Breaking:** the `pairs_matching`, `max_dist` and `split` options of
  `apply_demag` and `demag_tensor`, with their helpers `filter_distance` and
  `match_pairs`. Each forced the old dense, directly solved path. On a cube
  meshed into 2,744 cells the default path takes 0.2 s and 0.14 GB, where
  `pairs_matching` took 10 s and `split=4` 8.5 s, both with over 3 GB, and
  `max_dist` was 11 to 23 % off. The arguments after `inplace` and after
  `src_list` are now keyword-only, so an old positional call raises instead of
  shifting
  ([#87](https://github.com/magpylib/magpylib-material-response/pull/87)).

### Fixed

- `to_json` saves magpylib's `current.TriangleStrip` and
  `current.TriangleSheet`, and `mesh_study` warns when it gives no error
  estimate, from two meshes or from meshes that do not change steadily, instead
  of returning NaN silently
  ([#110](https://github.com/magpylib/magpylib-material-response/pull/110)).
- `apply_demag`: a block between two bodies whose cells share no grid was always
  stored sparse above 2 GB. At high susceptibility that kept nearly every entry,
  in more memory than the dense block and far more while it was built, and the
  memory warning missed it: a magnet on an iron plate meshed at 0.4 mm ran for
  more than 12 minutes. The block now stays dense unless sparsifying pays, at
  any size, and the warning shows: 2 minutes and 10 GB. Whether it pays is
  judged on cells drawn at random across the body, not on its first slab, and
  only from 90 % dropped, as the sparse product is single-threaded; from 60 %
  when the blocks would pass 2 GB in all. A sparse block peaks at a third of the
  memory while built, and a reverse block given by reciprocity is no longer
  copied
  ([#86](https://github.com/magpylib/magpylib-material-response/pull/86)).
- `apply_demag`: a thin ring in cylinder-segment cells at high susceptibility no
  longer swings with the cells across its wall. On curved cells point matching
  gave the system positive eigenvalues: a tube with a 1 mm wall at χ = 1000 came
  out −10 %, −68 % and +2 % from its moment with 1, 2 and 3 cells across.
  Segment cells close together now take each other's field averaged over the
  cell, which keeps every eigenvalue negative: −12 %, −11 % and −11 %, and
  closer as cells are added around the ring. Solves with segment cells take 2 to
  3 times longer
  ([#92](https://github.com/magpylib/magpylib-material-response/issues/92),
  [#100](https://github.com/magpylib/magpylib-material-response/pull/100)).

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
  https://github.com/magpylib/magpylib-material-response/compare/0.6.0...HEAD
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
