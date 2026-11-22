"""The fields of a magnetic dipole and of a homogeneously polarized sphere
(magpylib's ``Dipole`` and ``Sphere``), written against the Python array API
standard: one implementation for NumPy, any other array library, and the
50-digit namespace that writes the reference values of
``kernels/reference/sphere.json`` (#208). Only its test and
``kernels/reference/fill.py`` use it; the solvers keep magpylib's.

As magpylib: a dipole's ``H = (3 (m.r) r / r^5 - m / r^3) / (4 pi)``; a
sphere of diameter ``d`` and polarization ``J`` is, outside, the dipole of
moment ``J V / mu_0`` (``V`` its volume), and inside ``B = 2 J / 3``
("Theoretical Physics", Bertelmann), the surface counted inside.
"""

from magpylib_material_response._array_api import namespace, quiet
from magpylib_material_response._array_api.cuboid import MU0, _promote


def dipole_field(field, observers, moment):
    """B in T or H in A/m (``field`` 'B' or 'H') of magnetic dipoles at the
    origin, one per observer: ``moment`` (i, 3) in A m^2. NaN at the dipole,
    as magpylib's ``getB`` and ``func.dipole_field`` (test_Dipole_zero_position)."""
    if field not in ("B", "H"):
        msg = f"field must be 'B' or 'H'; instead received {field!r}."
        raise ValueError(msg)
    xp = namespace(observers, moment)
    observers, moment = _promote(xp, observers, moment)
    with quiet("divide", "invalid", "over"):
        r2 = xp.sum(observers * observers, axis=-1)
        r = xp.sqrt(r2)
        mr = xp.sum(moment * observers, axis=-1)
        H = (3 * (mr / r**5)[:, None] * observers - moment / (r**3)[:, None]) / (
            4 * xp.pi
        )
    H = xp.where((r2 == 0)[:, None], float("nan"), H)
    return H if field == "H" else MU0 * H


def sphere_field(field, observers, diameter, polarization):
    """B in T, H in A/m, J in T or M in A/m (``field`` 'B', 'H', 'J' or 'M')
    of homogeneously polarized spheres at the origin, one per observer:
    ``diameter`` (i,) in m, ``polarization`` (i, 3) in T. Inside, the surface
    included, ``B = 2 J / 3`` and ``J`` the polarization; outside, the dipole
    of moment ``J V / mu_0``, and ``J`` 0. ``H = (B - J) / mu_0``,
    ``M = J / mu_0``."""
    if field not in ("B", "H", "J", "M"):
        msg = f"field must be 'B', 'H', 'J' or 'M'; instead received {field!r}."
        raise ValueError(msg)
    xp = namespace(observers, diameter, polarization)
    observers, diameter, polarization = _promote(xp, observers, diameter, polarization)
    radius = xp.abs(diameter) / 2
    r2 = xp.sum(observers * observers, axis=-1)
    r = xp.sqrt(r2)
    out = (r > radius)[:, None]
    J = xp.where(out, 0.0, polarization)
    if field in ("J", "M"):
        return J if field == "J" else J / MU0
    with quiet("divide", "invalid", "over"):
        Jr = xp.sum(polarization * observers, axis=-1)
        outside = (
            ((3 * Jr)[:, None] * observers - polarization * r2[:, None])
            / (r**5)[:, None]
            * (radius**3 / 3)[:, None]
        )
    B = xp.where(out, outside, 2 * polarization / 3)
    return B if field == "B" else (B - J) / MU0
