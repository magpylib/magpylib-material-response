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

- **`solve_face_charge`** solves linear, uniform parts on their surface, with no
  volume mesh: cuboids, cylinders, cylinder segments, spheres and closed
  triangle meshes, such as CAD exports. It gets within 1 % of a finite-element
  reference faster than finite elements on every validation case where both get
  there.
- **`solve_cells`** cuts the parts into cells and solves for the polarization of
  every cell at once with the method of moments: for a part that saturates on a
  `BHCurve`, an anisotropic susceptibility, or one that varies inside a part. A
  dense direct solver, or matrix-free GMRES with FFT products on grids of
  identical cells; meshing helpers cut cuboids, cylinders and, through TetGen,
  any closed triangular surface into cells.
- **`solve_fem`** (the `fem` extra, NGSolve) solves the same model by finite
  elements, to check the other two, a saturating part too; `study_cells`,
  `study_fem` and `compare` give error bars by refining the mesh or the element
  order.

What drives a part is any other source in the model: magnets, currents and a
`UniformField`. All of these are importable from the package itself.

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
from magpylib_material_response import solve_face_charge

# a magnet on an iron plate, SI units (m, T)
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1.2), dimension=(0.01, 0.01, 0.005))
magnet.position = (0, 0, 0.0055)
plate = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.04, 0.04, 0.006))
plate.susceptibility = 1000  # soft iron; the magnet, with none, keeps its polarization

solved = solve_face_charge(magpy.Collection(magnet, plate))
print(
    magpy.getB(solved, (0, 0, 0.012))
)  # 205 mT above the magnet, 160 without the plate
```

```python
import magpylib as magpy
from magpylib_material_response import solve_cells
from magpylib_material_response.meshing import mesh_Cuboid

# a magnet with finite susceptibility, demagnetizing itself
magnet = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3))
magnet.susceptibility = 0.3  # µr = 1.3

mesh_demag = solve_cells(mesh_Cuboid(magnet, target_elems=125))
print(magpy.getB(mesh_demag, (0, 0, 1.5e-3)))
```

```python
import magpylib as magpy
import numpy as np
from magpylib_material_response import BHCurve, UniformField, solve_cells
from magpylib_material_response.meshing import mesh_Cuboid

# an iron cube saturating in an applied field, with a B-H curve in A/m and T
H = np.r_[0.0, np.geomspace(1e-2, 1e6, 200)]
mu0H = magpy.mu_0 * H
cube = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.01, 0.01, 0.01))
cube.susceptibility = BHCurve(H=H, B=1000 * mu0H / (1 + 1000 * mu0H / 1.6) + mu0H)

model = magpy.Collection(mesh_Cuboid(cube, (4, 4, 4)), UniformField(B=(0, 0, 1)))
solved = solve_cells(model)
print(magpy.getB(solved, (0, 0, 0.02)))  # the applied 1 T included
```

AI coding assistants: the package ships an Agent Skill, which
`uvx library-skills` adds to a project.

See the [documentation](https://magpylib-material-response.readthedocs.io/) for
a quickstart, worked examples, and the method description.
