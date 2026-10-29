"""Face charge: linear, uniform parts solved on their surfaces, without a volume mesh.

A body with a uniform, isotropic susceptibility ``chi`` and a uniform remanence
``M0`` has ``M = M0 + chi H`` and ``div B = 0``, so ``div M = 0`` inside: it
carries no magnetic charge in its volume. Its whole response is the surface
charge ``sigma = M.n`` on its faces, which solves

    (1/chi + 1/2) sigma - K' sigma = H0.n + M0.n / chi,

with ``H0`` the applied field (``H_ext`` is ``mu0 H0``, in tesla) and ``K'``
the adjoint double-layer operator of the surface, which depends on the shape
only. The faces are cut into panels graded toward the edges, each
carrying a charge linear across it, with closed-form panel fields; touching
bodies are coupled exactly. A solved body becomes a :class:`ChargeSheet`, an
ordinary magpylib source.

An anisotropic susceptibility and one varying inside a body bring back volume
charge; they stay with :func:`~magpylib_material_response.demag.apply_demag`.
Neither models saturation yet: a solved sheet's
:attr:`ChargeSheet.mean_polarization` tells whether it matters, and its
:attr:`ChargeSheet.jump_indicator` how accurate the fields near its surface are.

On a cube the method reproduces the polarizability of Helsing and Perfekt:

    J. Helsing, K.-M. Perfekt (2013) "On the polarizability and capacitance of
    the cube", Appl. Comput. Harmon. Anal. 34(3), 445-468.
"""

from __future__ import annotations

from magpylib_material_response.face_charge._apply import apply_face_charge
from magpylib_material_response.face_charge._sheet import ChargeSheet

__all__ = ["ChargeSheet", "apply_face_charge"]
