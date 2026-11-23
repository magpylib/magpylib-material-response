"""The field of a homogeneously polarized cylinder (magpylib's ``Cylinder``),
written against the Python array API standard: one implementation for NumPy,
any other array library, and the 50-digit namespace that writes the reference
values of ``kernels/reference/cylinder.json`` (#208). Only its test and
``kernels/reference/fill.py`` use it; the solvers keep magpylib's.

As magpylib: the cylinder of diameter ``d`` and height ``h`` centered on the
origin, its axis along ``z``. Lengths are scaled by the radius ``r0 = d / 2``:
the observer at radius ``r`` and height ``z``, the bases at ``z = +-z0``.

- The axial polarization ``J_z`` gives B by Derby's formula (American Journal
  of Physics 78, 2010): ``B_r`` and ``B_z`` are each two complete elliptic
  integrals ``cel``.
- The transverse polarization ``J_xy`` gives ``mu_0 H`` (Caciagli, JMMM 456,
  2018; Ortner, Leitner, Rauber) by ``K``, ``E`` and ``Pi`` of a negative
  parameter, each a ``cel``.
- Both instead by one series in the radius wherever ``r`` is within
  ``SERIES_RADIUS`` of the distance from the axis point to the hull: near the
  axis, where magpylib keeps the transverse series to three terms (``r <
  0.05``, off up to 1.5e-9 of the transverse field) and this module to the
  precision, and in a cone along the axis far away, where the elliptic
  integrals cancel (magpylib's transverse field off 6e-5 at 100 radii up, 0.1
  out).

With ``rho = (x, y) / r0``, the transverse field is
``mu_0 H_xy = -(S_f J_xy + D (J . rho) rho) / 4`` and
``mu_0 H_z = -S_z (J . rho) / 4``: no angle, and no division by ``r`` on the
axis. Every ``cel`` is Bulirsch's, stopped as in ``circle.py``.
"""

import math

from magpylib_material_response._array_api import cel_errtol, quiet
from magpylib_material_response._array_api.cuboid import MU0, _magnet_inputs

# within this share of the radius of the hull's radius and of the half
# height of a base's height, a point is on the edge, as magpylib (isclose)
RTOL_EDGE = 1e-15
# the transverse field is the series within this distance of the axis, in
# radii: the elliptic integrals there lose eps / r^2 to cancellation (4e-15
# at the switch), while each term of the series is about r^2 times the last
SERIES_RADIUS = 0.25


def series_terms(eps):
    """The number of terms that leave the series exact to ``eps`` at
    ``SERIES_RADIUS``: 15 in doubles, 44 at 50 digits."""
    # 13.0 in doubles: rounded first, so that its last bit cannot add a term
    return math.ceil(round(math.log(float(eps)) / math.log(SERIES_RADIUS**2), 9)) + 2


def _cel(xp, kc, p, pairs, errtol):
    """Bulirsch's ``cel(kc, p, c, s)`` for each ``(c, s)`` of ``pairs`` at
    one ``kc`` and ``p``, either branch (``p > 0`` or not): the iterations
    share every variable but ``c`` and ``s``, and the stop. A NaN entry counts
    as converged; ``kc`` must not be 0, where the iteration never stops."""
    positive = p > 0
    f = kc * kc
    g = xp.where(positive, 1.0, 1 - p)
    p_new = xp.where(
        positive, xp.sqrt(xp.where(positive, p, 1.0)), xp.sqrt((f - p) / g)
    )
    cs = []
    for c, s in pairs:
        c_neg = (c - s) / g
        s_neg = -(1 - f) * (s - c * p) / (g * g * p_new) + c_neg * p_new
        cs.append([xp.where(positive, c, c_neg), xp.where(positive, s / p_new, s_neg)])
    p = p_new
    em = xp.ones_like(kc)
    qc = xp.abs(kc)
    kk = qc
    while True:
        g = kk / p
        for pair in cs:
            c, s = pair
            pair[0], pair[1] = c + s / p, 2 * (s + c * g)
        p = p + g
        g = em
        em = em + qc
        if not xp.any(xp.abs(g - qc) > g * errtol):
            break
        qc = 2 * xp.sqrt(kk)
        kk = em * qc
    return [(xp.pi / 2) * (s + c * em) / (em * (em + p)) for c, s in cs]


def _axial(xp, z0, r, z, errtol):
    """``B_r`` and ``B_z`` per unit ``J_z`` (Derby)."""
    zph, zmh = z + z0, z - z0
    dpr, dmr = 1 + r, 1 - r
    sq0 = xp.sqrt(zmh * zmh + dpr * dpr)
    sq1 = xp.sqrt(zph * zph + dpr * dpr)
    k0 = xp.sqrt(zmh * zmh + dmr * dmr) / sq0
    k1 = xp.sqrt(zph * zph + dmr * dmr) / sq1
    gamma = dmr / dpr
    one = xp.ones_like(r)
    (br1,) = _cel(xp, k1, one, [(one, -one)], errtol)
    (br0,) = _cel(xp, k0, one, [(one, -one)], errtol)
    (bz1,) = _cel(xp, k1, gamma * gamma, [(one, gamma)], errtol)
    (bz0,) = _cel(xp, k0, gamma * gamma, [(one, gamma)], errtol)
    Br = (br1 / sq1 - br0 / sq0) / xp.pi
    Bz = (zph * bz1 / sq1 - zmh * bz0 / sq0) / dpr / xp.pi
    return Br, Bz


def _transverse_series(xp, z0, r, z, n):
    """``S_f``, ``D`` and ``S_z / r`` of the transverse field near the axis.

    The field is that of the scalar potential ``cos(phi)`` times a series in
    ``r`` whose coefficients are the derivatives of ``f(z) = z / sqrt(1 +
    z^2)`` at the two bases: with ``F_m = f^(m)(z + z0) - f^(m)(z - z0)`` and
    ``c_k = (-1)^k / (4^k k! (k+1)!)``, ``S_f = sum c_k F_2k r^2k``, ``S_r =
    sum (2k+1) c_k F_2k r^2k``, ``D = (S_r - S_f) / r^2`` and ``S_z = sum c_k
    F_(2k+1) r^(2k+1)``. ``f' = (1 + z^2)^(-3/2)``, whose derivatives are
    Gegenbauer polynomials: ``f^(m+1)(z) = m! rho^(-3-m) C_m(-z / rho)``,
    ``C = C^(3/2)``, ``rho = sqrt(1 + z^2)``. The axial field follows from
    the same sums: ``B_z = (2 S_f + r^2 D) / 4``, ``B_r = -S_z / 4``.

    The series is the field within the distance from the axis point to the
    hull (each side's alone reaches its edge, but beside the hull their sum is
    not the field), each term about ``(r / rho)^2`` times the last."""

    def side(zs):
        rho2 = 1 + zs * zs
        rho = xp.sqrt(rho2)
        x = -zs / rho
        u2 = r * r / rho2
        c_prev, c_odd = xp.ones_like(x), 3 * x  # C_(2k-2), C_(2k-1)
        w = -xp.ones_like(x) / 8  # c_k (2k-1)! / rho^2k, without r^2k
        sf, dr, sz = 0 * x, 0 * x, 0 * x
        for k in range(1, n):
            c_even = ((4 * k + 1) * x * c_odd - (2 * k + 1) * c_prev) / (2 * k)
            sf = sf + w * u2 * c_odd
            dr = dr + (2 * k) * w * c_odd
            sz = sz + (2 * k) * w * u2 * c_even
            w = w * (-(2 * k) * (2 * k + 1) * u2) / (4 * (k + 1) * (k + 2))
            c_prev = c_even
            c_odd = ((4 * k + 3) * x * c_even - (2 * k + 2) * c_odd) / (2 * k + 1)
        return rho, sf / rho2, dr / (rho2 * rho2), sz / (rho2 * rho)

    zp, zm = z + z0, z - z0
    rp, fp, dp, sp = side(zp)
    rm, fm, dm, sm = side(zm)
    # the leading terms, f(zp) - f(zm) and rho^-3 at zp minus at zm, without
    # the cancellation far along the axis: zp^2 - zm^2 = 4 z z0
    d2 = 4 * z * z0
    same = zp * zm > 0
    f0 = xp.where(
        same,
        d2 / (rp * rm * xp.where(same, zp * rm + zm * rp, 1.0)),
        zp / rp - zm / rm,
    )
    s0 = -d2 / (rm + rp) * (rm * rm + rm * rp + rp * rp) / (rp * rp * rp * rm * rm * rm)
    return f0 + fp - fm, dp - dm, s0 + sp - sm


def _transverse_general(xp, z0, r, z, errtol):
    """``S_f``, ``D`` and ``S_z / r`` of the transverse field off the axis,
    from magpylib's ``magnet_cylinder_diametral_Hfield``: ``K``, ``E`` and
    ``Pi`` of the parameter ``-4 r / a^2`` at each base, as ``cel``."""
    zp, zm = z + z0, z - z0
    rp, rm = r + 1, r - 1
    r2 = r * r
    # on the hull's radius (r = 1), Pi of an infinite characteristic is 0
    # and the term with 1 / rm vanishes: their limits
    hull = rm == 0
    rm_s = xp.where(hull, 1.0, rm)
    p_pi = 1 + 4 * r / (rm_s * rm_s)
    one = xp.ones_like(r)

    def side(zs):
        a2 = zs * zs + rm * rm
        a = xp.sqrt(a2)
        kc = xp.sqrt(1 + 4 * r / a2)
        K, E = _cel(xp, kc, one, [(one, one), (one, kc * kc)], errtol)
        (P,) = _cel(xp, kc, p_pi, [(one, one)], errtol)
        return a, K, E, xp.where(hull, 0.0, P)

    ap, Kp, Ep, Pp = side(zp)
    am, Km, Em, Pm = side(zm)
    Xr = (
        -zm * am * Em
        + zp * ap * Ep
        + zm / am * (2 + zm * zm) * Km
        - zp / ap * (2 + zp * zp) * Kp
        + (zm / am * Pm - zp / ap * Pp) * rp * (r2 + 1) / rm_s
    )
    Xf = (
        zm * am * Em
        - zp * ap * Ep
        - zm / am * (2 + zm * zm + 2 * r2) * Km
        + zp / ap * (2 + zp * zp + 2 * r2) * Kp
        + zm / am * rp * rp * Pm
        - zp / ap * rp * rp * Pp
    )
    Xz = am * Em - ap * Ep - (1 + zm * zm + r2) / am * Km + (1 + zp * zp + r2) / ap * Kp
    return Xf / (xp.pi * r2), (Xr - Xf) / (xp.pi * r2 * r2), 2 * Xz / (xp.pi * r2)


def cylinder_field(field, observers, dimension, polarization):
    """B in T, H in A/m, J in T or M in A/m (``field`` 'B', 'H', 'J' or
    'M') of cylinders in their own frames, one per observer: ``dimension``
    (i, 2) diameter and height in m (signs ignored), ``polarization`` (i, 3)
    in T. As magpylib: J the polarization inside, the surface included, and
    0 outside; B and H 0 on the edge (within ``RTOL_EDGE``) and for a zero
    polarization; ``H = (B - J) / mu_0``, ``M = J / mu_0``. Also 0, where
    magpylib gives NaN or 0: for a zero diameter or height, and so near the
    edge that the distance squares below the smallest normal number."""
    xp, (observers, dimension, polarization) = _magnet_inputs(
        field, observers, dimension, polarization
    )
    fi = xp.finfo(observers.dtype)
    errtol = cel_errtol(fi.eps)
    n = series_terms(fi.eps)
    x, y, z = xp.unstack(observers, axis=-1)
    diameter, height = xp.unstack(dimension, axis=-1)
    with quiet("divide", "invalid", "over"):
        radius = xp.abs(diameter) / 2
        half = xp.abs(height) / 2
        rad = xp.sqrt(x * x + y * y)
        # no volume, no inside (magpylib: 0 / 0 for J)
        dim_not_null = (radius != 0) & (half != 0)
        inside = (xp.abs(z) <= half) & (rad <= radius) & dim_not_null
        J = xp.where(inside[:, None], polarization, 0.0)
        if field in ("J", "M"):
            return J if field == "J" else J / MU0
        # scaled by the radius; the special cases are evaluated at the center
        # of a unit cylinder, then set to 0
        r0 = xp.where(dim_not_null, radius, 1.0)
        r = rad / r0
        zz = z / r0
        z0 = xp.where(dim_not_null, half, 1.0) / r0
        on_edge = (xp.abs(r - 1) <= RTOL_EDGE) & (
            xp.abs(xp.abs(zz) - z0) <= RTOL_EDGE * z0
        )
        to_edge = (r - 1) * (r - 1) + (xp.abs(zz) - z0) * (xp.abs(zz) - z0)
        pol_not_null = xp.any(polarization != 0, axis=-1)
        general = (
            pol_not_null & dim_not_null & ~on_edge & ~(to_edge < fi.smallest_normal)
        )
        xs = xp.where(general, x / r0, 0.0)
        ys = xp.where(general, y / r0, 0.0)
        zz = xp.where(general, zz, 0.0)
        r = xp.where(general, r, 0.0)

        # the series within SERIES_RADIUS of the distance from the axis point
        # to the hull (1 between the bases, to the edge beyond them): near the
        # axis, and in a cone along it far away, where the elliptic integrals
        # lose to cancellation; elsewhere Derby's and the elliptic integrals
        gap = xp.abs(zz) - z0
        to_hull = 1 + xp.where(gap > 0, gap * gap, 0.0)
        near = r * r < SERIES_RADIUS**2 * to_hull
        rs = xp.where(near, r, 0.0)
        rg = xp.where(near, 1 + SERIES_RADIUS, r)
        Sf_s, D_s, Sz_s = _transverse_series(xp, z0, rs, zz, n)
        Sf_g, D_g, Sz_g = _transverse_general(xp, z0, rg, zz, errtol)
        Br_g, Bz_g = _axial(xp, z0, rg, zz, errtol)
        Sf = xp.where(near, Sf_s, Sf_g)
        D = xp.where(near, D_s, D_g)
        Sz = xp.where(near, Sz_s, Sz_g)
        # axial: B_r along (x, y) / r; in the series B_r / r = -S_z / 4
        br_r = xp.where(near, -Sz_s / 4, Br_g / rg)
        Bz = xp.where(near, (2 * Sf_s + rs * rs * D_s) / 4, Bz_g)
        bx, by = br_r * xs, br_r * ys

        jx, jy, jz = xp.unstack(polarization, axis=-1)
        jr = jx * xs + jy * ys
        inside_g = inside & general
        mu0H = xp.stack(
            (
                jz * bx - (Sf * jx + D * jr * xs) / 4,
                jz * by - (Sf * jy + D * jr * ys) / 4,
                jz * Bz - xp.where(inside_g, jz, 0.0) - Sz * jr / 4,
            ),
            axis=-1,
        )
    mu0H = xp.where(general[:, None], mu0H, 0.0)
    if field == "H":
        return mu0H / MU0
    return mu0H + xp.where(inside_g[:, None], polarization, 0.0)
