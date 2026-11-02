"""Copyright (c) 2026 Alexandre Boisselet. All rights reserved.

magpylib-material-response: Python package extending the Magpylib library by providing magnetic field analysis for soft materials and demagnetization of hard magnets.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

from magpylib_material_response._data import get_dataset
from magpylib_material_response.logging_config import configure_logging, disable_logging
from magpylib_material_response.utils import from_json, to_json

from ._version import version as __version__

if TYPE_CHECKING:  # for editors and linters; at run time, __getattr__ below
    from magpylib_material_response._saturation import BHCurve
    from magpylib_material_response.cells import solve_cells
    from magpylib_material_response.convergence import compare, study_cells
    from magpylib_material_response.face_charge import solve_face_charge
    from magpylib_material_response.fem import solve_fem, study_fem  # noqa: F401

# loaded when first asked for: import stays fast, and NGSolve optional
_FUNCTIONS = {
    "BHCurve": "_saturation",
    "compare": "convergence",
    "solve_cells": "cells",
    "solve_face_charge": "face_charge",
    "solve_fem": "fem",
    "study_cells": "convergence",
    "study_fem": "fem",
}
_MODULES = (
    "cells",
    "convergence",
    "demag",
    "demag_fft",
    "face_charge",
    "fem",
    "meshing",
    "meshing_utils",
    "newell",
    "polyline",
    "utils",
)

# not solve_fem and study_fem: `import *` must not need NGSolve, nor load it
__all__ = [
    "BHCurve",
    "__version__",
    "compare",
    "configure_logging",
    "disable_logging",
    "from_json",
    "get_dataset",
    "solve_cells",
    "solve_face_charge",
    "study_cells",
    "to_json",
]


def __getattr__(name):
    if name in _FUNCTIONS:
        return getattr(import_module(f"{__name__}.{_FUNCTIONS[name]}"), name)
    if name in _MODULES:
        return import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    names = {*globals(), *_FUNCTIONS, *_MODULES}
    # what lists a module's names reads each one, and only NGSolve can fail
    return sorted(
        names if find_spec("ngsolve") else names - {"fem", "solve_fem", "study_fem"}
    )
