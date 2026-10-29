# magpylib-material-response

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/magpylib/magpylib-material-response/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/magpylib/magpylib-material-response/actions
[pypi-link]:                https://pypi.org/project/magpylib-material-response/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/magpylib-material-response
[pypi-version]:             https://img.shields.io/pypi/v/magpylib-material-response
[rtd-badge]:                https://readthedocs.org/projects/magpylib-material-response/badge/?version=latest
[rtd-link]:                 https://magpylib-material-response.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/magpylib/magpylib-material-response/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/magpylib/magpylib-material-response

<!-- prettier-ignore-end -->

> **Warning** This package is experimental and in development phase, breaking
> API changes may happen at any time.

Magpylib-Material-Response adds the response of magnetic materials to
[Magpylib](https://magpylib.readthedocs.io/): the magnetization of soft-magnetic
parts and the demagnetization of hard magnets. A solved model is a magpylib
source like any other.

- **`face_charge.apply_face_charge`** solves linear, uniform parts on their
  surface, with no volume mesh: cuboids, cylinders, cylinder segments, spheres
  and closed triangle meshes, such as CAD exports. It gets within 1 % of a
  finite-element reference faster than finite elements on every validation case
  where both get there.
- **`demag.apply_demag`** cuts the parts into cells and solves for the
  polarization of every cell at once with the method of moments: for an
  anisotropic susceptibility, or one that varies inside a part. A dense direct
  solver, or matrix-free GMRES with FFT products on grids of identical cells;
  meshing helpers cut cuboids, cylinders and, through TetGen, any closed
  triangular surface into cells.
- **`fem.fem_solve`** (the `fem` extra, NGSolve) solves the same model by finite
  elements, to check the other two; `convergence.mesh_study`, `fem_study` and
  `compare` give error bars by refining the mesh or the element order.

## Installation

```bash
pip install magpylib-material-response
```

Tetrahedral meshing of `TriangularMesh` magnets needs the optional
[TetGen](https://tetgen.pyvista.org/), finite elements
[NGSolve](https://ngsolve.org/):

```bash
pip install "magpylib-material-response[tetgen]"
pip install "magpylib-material-response[fem]"
```

## Minimal examples

```python
import magpylib as magpy
from magpylib_material_response.face_charge import apply_face_charge

# a magnet on an iron plate, SI units (m, T)
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1.2), dimension=(0.01, 0.01, 0.005))
magnet.position = (0, 0, 0.0055)
plate = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.04, 0.04, 0.006))
plate.susceptibility = 1000  # soft iron; the magnet, with none, keeps its polarization

solved = apply_face_charge(magpy.Collection(magnet, plate))
print(
    magpy.getB(solved, (0, 0, 0.012))
)  # 205 mT above the magnet, 160 without the plate
```

```python
import magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

# a magnet with finite susceptibility, demagnetizing itself
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
magnet.susceptibility = 0.3  # µr = 1.3

mesh_demag = apply_demag(mesh_Cuboid(magnet, target_elems=125))
print(magpy.getB(mesh_demag, (0, 0, 1.5e-3)))
```

See the [documentation](https://magpylib-material-response.readthedocs.io/) for
a quickstart, worked examples, and the method description.
