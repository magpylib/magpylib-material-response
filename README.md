# magpylib-material-response

[![Code style: black][black-badge]][black-link]

> **Warning**
> **This package is experimental and in a very dynamic development phase. Breaking API changes may happen at any time.**

Magpylib-Material-Response is an extension to the Magpylib library, providing magnetic field analysis for soft materials and demagnetization of hard magnets. Leveraging the Method of Moments, it calculates magnetic material response by meshing sources in an arbitrary number of unit elements.

## Installation

Install from PyPi (not yet available)

```console
$ pip install magpylib-material-response
```

or locally:

```
$ pip install -e .[code_style,testing]
```

## Testing

Enter created folder then run tests:

```
$ flake8 .
$ black .
$ pytest
```

To use pre-commit:

```
$ git add *
# to apply to staged files
$ pre-commit run
# restage if changes
$ git add *
# to run on commits
$ pre-commit install
$ git commit -m 'Initial commit'
```

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/ambv/black

(package originally created by [python-pkg-cookiecutter](https://github.com/executablebooks/python-pkg-cookiecutter))
