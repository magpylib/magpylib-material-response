# magpylib-material-response

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/magpylib/magpylib-material-response/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/magpylib/magpylib-material-response/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/magpylib-material-response
[conda-link]:               https://github.com/conda-forge/magpylib-material-response-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/magpylib/magpylib-material-response/discussions
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

Magpylib-Material-Response is an extension to the
[Magpylib](https://magpylib.readthedocs.io/) library, providing magnetic field
analysis for soft materials and demagnetization of hard magnets. Leveraging the
Method of Moments, it calculates the self-consistent magnetic material response
by meshing sources into an arbitrary number of unit cells.

Key features:

- **`apply_demag`** — self-consistent demagnetization of magpylib `Collection`
  objects, with per-cell scalar or anisotropic susceptibility and external field
  support.
- **Two solvers, one physics** — an exact dense solver and an FFT-accelerated
  iterative solver that agree to solver tolerance for any input; meshes with
  tens of thousands of cells solve in seconds.
- **Analytical interactions** — volume-averaged Newell tensors for cuboid cells
  (generalized to different cell sizes), point matching for everything else.
- **Meshing helpers** — cuboids, cylinders, and arbitrary closed surfaces via
  tetrahedral meshing (`mesh_TriangularMesh`, using TetGen).

## Installation

```bash
pip install magpylib-material-response
```

or with
[conda](https://github.com/conda-forge/magpylib-material-response-feedstock):

```bash
conda install -c conda-forge magpylib-material-response
```

Tetrahedral meshing of `TriangularMesh` magnets requires the optional
[TetGen](https://tetgen.pyvista.org/) dependency:

```bash
pip install magpylib-material-response[tetgen]
```

See the [documentation](https://magpylib-material-response.readthedocs.io/) for
a quickstart, worked examples, and the method description.
