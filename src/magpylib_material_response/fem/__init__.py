"""Finite elements on a magpylib model: an independent check of the method of
moments.

:func:`solve_fem` solves the same objects that
:func:`~magpylib_material_response.cells.solve_cells` takes, with NGSolve, on
their own shapes rather than on cells; :func:`study_fem` raises the element
order on one mesh and estimates the error of the highest, as
:func:`~magpylib_material_response.convergence.study_cells` does for the method
of moments. Install with ``pip install "magpylib-material-response[fem]"``.

The formulation is the reduced scalar potential with an exact open boundary (a
Kelvin transform), on meshes refined geometrically toward the magnets' edges;
with current sources the magnets take the total potential, so that a coil
around soft iron costs no accuracy, and the currents are not meshed.
On a soft sphere it reproduces the exact field, and on a cube of infinite
susceptibility the polarizability of Helsing and Perfekt (2013), 3.644305, to
2e-5 at order 5.

References
----------
- J. Helsing, K.-M. Perfekt (2013) "On the polarizability and capacitance of
  the cube", Appl. Comput. Harmon. Anal. 34(3), 445-468.
  doi:10.1016/j.acha.2012.07.006
"""

from __future__ import annotations

import importlib.util

# only a missing NGSolve gets the install hint; one that fails to load (a
# missing library, say) raises its own error below
if importlib.util.find_spec("ngsolve") is None:  # pragma: no cover
    msg = (
        "magpylib_material_response.fem needs NGSolve: "
        'pip install "magpylib-material-response[fem]"'
    )
    raise ImportError(msg)

from magpylib_material_response._renamed import old_names
from magpylib_material_response.fem._solve import FemSolution, solve_fem
from magpylib_material_response.fem._study import FemStudy, study_fem

__all__ = ["FemSolution", "FemStudy", "solve_fem", "study_fem"]

__getattr__ = old_names(__name__, {"fem_solve": "solve_fem", "fem_study": "study_fem"})
