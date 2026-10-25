"""Finite elements on a magpylib model: an independent check of the method of
moments.

:func:`fem_solve` solves the same objects that
:func:`~magpylib_material_response.demag.apply_demag` takes, with NGSolve, on
their own shapes rather than on cells; :func:`fem_study` raises the element
order on one mesh and estimates the error of the highest, as
:func:`~magpylib_material_response.convergence.mesh_study` does for the method
of moments. Install with ``pip install "magpylib-material-response[fem]"``.

The formulation is the reduced scalar potential with an exact open boundary (a
Kelvin transform), on meshes refined geometrically toward the magnets' edges.
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

try:
    import ngsolve  # noqa: F401
except ImportError as err:  # pragma: no cover
    msg = (
        "magpylib_material_response.fem needs NGSolve: "
        'pip install "magpylib-material-response[fem]"'
    )
    raise ImportError(msg) from err

from magpylib_material_response.fem._solve import FemSolution, fem_solve
from magpylib_material_response.fem._study import FemStudy, fem_study

__all__ = ["FemSolution", "FemStudy", "fem_solve", "fem_study"]
