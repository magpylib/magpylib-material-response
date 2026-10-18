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
[Magpylib](https://magpylib.readthedocs.io/): the demagnetization of hard
magnets and the magnetization of soft-magnetic parts. It cuts the magnets into
cells and solves for the polarization of every cell at once with the method of
moments.

- **`apply_demag`** solves a magpylib `Collection` for a scalar or anisotropic
  susceptibility per cell, driven by remanent polarizations, a uniform applied
  field and the current sources in the collection.
- **Meshing helpers** cut cuboids, cylinders and, through TetGen, any closed
  triangular surface into cells.

## Installation

```bash
pip install magpylib-material-response
```

Tetrahedral meshing of `TriangularMesh` magnets requires the optional
[TetGen](https://tetgen.pyvista.org/) dependency:

```bash
pip install magpylib-material-response[tetgen]
```

## Minimal example

```python
import magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

# a magnet with finite susceptibility, SI units (m, T)
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
magnet.susceptibility = 0.3  # µr = 1.3

# mesh into cells and solve; the result is a field source like any other
mesh_demag = apply_demag(mesh_Cuboid(magnet, target_elems=125))
print(magpy.getB(mesh_demag, (0, 0, 1.5e-3)))
```

See the [documentation](https://magpylib-material-response.readthedocs.io/) for
a quickstart, worked examples, and the method description.
