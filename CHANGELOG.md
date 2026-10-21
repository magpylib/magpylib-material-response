# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `get_H_ext(*sources)` no longer takes the `H_ext=` keyword, which was read and
  immediately shadowed by the attribute lookup
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).

### Added

- `apply_demag(..., solver="iterative")`: a matrix-free GMRES solver next to the
  dense direct solver (`solver="direct"`, the default), with `solver_tol` and
  `max_iter` (iterations). Both solvers apply the same interaction rule. GMRES
  stops at a residual below `solver_tol`; measured against the direct solver,
  the polarization agrees to about 2 × `solver_tol` for hard magnets and 35 ×
  `solver_tol` for soft iron at χ = 1000. GMRES that does not converge raises.
  Boxes of cuboid cells on a uniform grid use FFT products (`demag_fft` module):
  32,768 cells of soft iron solve in about 4 s. Each meshed body is a grid of
  its own, told apart by the Collection its mesher made or, when the cells were
  flattened into one, by the boxes they fill on their lattice; a yoke with its
  legs, meshed with `mesh_all`, then gets FFT products within each body and
  between bodies of one cell size. Grids with holes, such as voxelized round
  bodies, and other cell types use dense blocks; a dense block is only
  sparsified when that drops at least half of its entries, which at high
  susceptibility it rarely does. The direct solver assembles its matrix in
  Fortran order and factors it in place, in 1.1 times the matrix's memory where
  `scipy.linalg.solve` on a C-ordered copy took 4 (3,773 cells: 1.9 GB instead
  of 4.8)
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

- The direct solver assembles in place (about 2-3x less peak memory), derives
  reverse blocks by reciprocity, and every block is built in chunks of at most
  200,000 (observer, source) pairs. 1,757 tetrahedra: 16.8 s and 7.7 GB before,
  7.5 s and 1.7 GB after
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).
- `timelog` no longer blocks on exit and skips its watchdog thread while package
  logging is silent (a 27-cell solve went from about 0.5 s to 5 ms)
  ([#52](https://github.com/magpylib/magpylib-material-response/pull/52)).

### Fixed

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
  https://github.com/magpylib/magpylib-material-response/compare/v0.4.0...HEAD
[0.4.0]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.3.1...v0.4.0
[0.3.1]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.3.0...v0.3.1
[0.3.0]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.2.1a0...v0.3.0
[0.2.1a0]:
  https://github.com/magpylib/magpylib-material-response/compare/v0.2.0a0...v0.2.1a0
[0.2.0a0]:
  https://github.com/magpylib/magpylib-material-response/releases/tag/v0.2.0a0
