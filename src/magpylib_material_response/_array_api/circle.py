"""The field of a circular current loop (magpylib's ``Circle``), written
against the Python array API standard: one implementation for NumPy, any
other array library, and the 50-digit namespace that writes the reference
values of ``kernels/reference/current.json`` (#208). Only its test and
``kernels/reference/fill.py`` use it; the solvers keep magpylib's.

As magpylib: the loop of radius ``r0`` in the ``z = 0`` plane, centered on
the origin, its current counter-clockwise about ``+z``. In cylindrical
coordinates scaled by ``r0`` (Ortner et al., "Numerically stable and
computationally efficient expression for the magnetic field of a current
loop", Magnetism 2023), ``H_r`` and ``H_z`` are each one complete elliptic
integral ``cel``, by Bulirsch's iteration with its first step done.
"""

from magpylib_material_response._array_api import namespace, quiet
from magpylib_material_response._array_api.cuboid import MU0, _promote

# distance from the wire in the loop's plane, in radii, taken as on it
RTOL_WIRE = 1e-15
# Bulirsch's stop: the iteration converges quadratically, so a relative step
# below the square root of the precision leaves the integral exact to it.
# Magpylib's 1e-8 in doubles; the 50-digit namespace's 5e-26
ERRTOL = 1e-8


def _cel_iter(xp, qc, p, g, cc, ss, em, kk, errtol):
    """Bulirsch's ``cel`` from the state after its first step (magpylib's
    ``_cel_iter``); every entry iterates until the slowest converges. A NaN
    entry counts as converged."""
    while xp.any(xp.abs(g - qc) > g * errtol):
        qc = 2 * xp.sqrt(kk)
        kk = em * qc
        g = kk / p
        cc, ss = cc + ss / p, 2 * (ss + cc * g)
        p = p + g
        g = em
        em = em + qc
    return (xp.pi / 2) * (ss + cc * em) / (em * (em + p))


def circle_field(field, observers, diameter, current):
    """B in T or H in A/m (``field`` 'B' or 'H') of circular current loops
    centered on the origin in the ``z = 0`` plane, one per observer:
    ``diameter`` (i,) in m (its sign ignored), ``current`` (i,) in A. As
    magpylib: 0 for a loop without diameter, and within ``RTOL_WIRE`` radii
    of the wire in its plane or so near it that the distance squares below the
    smallest normal number."""
    if field not in ("B", "H"):
        msg = f"field must be 'B' or 'H'; instead received {field!r}."
        raise ValueError(msg)
    xp = namespace(observers, diameter, current)
    observers, diameter, current = _promote(xp, observers, diameter, current)
    fi = xp.finfo(observers.dtype)
    errtol = min(ERRTOL, fi.eps**0.5)
    x, y, z = xp.unstack(observers, axis=-1)
    with quiet("divide", "invalid", "over"):
        r0 = xp.abs(diameter) / 2
        r = xp.sqrt(x * x + y * y)
        general = ~((r0 == 0) | ((xp.abs(r - r0) < RTOL_WIRE * r0) & (z == 0)))
        # the special cases are evaluated on the axis of a unit loop, then set to 0
        r0 = xp.where(general, r0, 1.0)
        rho = xp.where(general, r, 0.0) / r0
        zeta = xp.where(general, z, 1.0) / r0
        z2 = zeta * zeta
        x0 = z2 + (rho + 1) ** 2
        k2 = 4 * rho / x0
        q2 = (z2 + (rho - 1) ** 2) / x0
        # an observer whose distance from the wire squares below the smallest
        # normal number (1e-154 radii off it) is on it to the precision, and
        # gets its 0: Bulirsch's iteration never stops at q = 0 (magpylib
        # #1100), and 1 / q2 overflows above it
        general = general & ~(q2 < fi.smallest_normal)
        q2 = xp.where(general, q2, 1.0)
        q = xp.sqrt(q2)
        p = 1 + q
        # 1 / q2 applied to the integrals, which scale as q: no overflow
        pf = current / (4 * xp.pi * r0 * xp.sqrt(x0))
        inv_q2 = 1 / q2
        one = xp.ones_like(q)
        cc = k2 * 4 * zeta / x0
        ss = 2 * cc * q / p
        Hr = pf * (_cel_iter(xp, q, p, one, cc, ss, p, q, errtol) * inv_q2)
        k4 = k2 * k2
        cc = k4 - (q2 + 1) * (4 / x0)
        ss = 2 * q * (k4 / p - (4 / x0) * p)
        Hz = -pf * (_cel_iter(xp, q, p, one, cc, ss, p, q, errtol) * inv_q2)
        # back to Cartesian: x / r and y / r; on the axis H_r is 0, and 0 * H_r
        # keeps a NaN
        on_axis = r == 0
        rr = xp.where(on_axis, 1.0, r)
        Hx = xp.where(on_axis, 0 * Hr, Hr * x / rr)
        Hy = xp.where(on_axis, 0 * Hr, Hr * y / rr)
    H = xp.where(general[:, None], xp.stack((Hx, Hy, Hz), axis=-1), 0.0)
    return H if field == "H" else MU0 * H
