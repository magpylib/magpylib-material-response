"""The field of a homogeneously polarized cuboid, written against the Python
array API standard: one implementation for NumPy, any other array library,
and the 50-digit namespace that writes the reference values (#208).

The reference implementation of ``kernels/reference/cuboid.json``: only its
test and ``kernels/reference/fill.py`` use it, the solvers keep magpylib's.
Kept here until magpylib's own array API rewrite takes over. It is Jack
Smith's rewrite of magpylib's ``field_BH_cuboid`` (branch ``array-api`` of
jack-smithy/magpylib, 81ddc1e4), with the stable edge logarithms of magpylib
#1066 carried into it. Magpylib is BSD-2-Clause, (c) 2022 Silicon Austria
Labs, Magpylib Developers.

Field computations use magnetic surface charge density; similar expressions
were published repeatedly, e.g. Yang (Supercond. Sci. Technol. 3(12):591,
1990), Engel-Herbert (J. Appl. Phys. 97(7):074504-4, 2005), and Camacho
(Rev. Mex. Fís. E 59 (2013) 8-17). Positions are mapped to one octant,
where no indeterminate form arises along the extensions of edges (Cichon,
IEEE Sensors Journal 19(7):2509, 2019).
"""

from magpylib_material_response._array_api import namespace, quiet

MU0 = 1.25663706127e-06  # scipy's mu_0 (CODATA 2022), as magpylib uses it
RTOL_SURFACE = 1e-15  # relative distance to a face within which a point is on it


def _promote(xp, *arrays):
    """The arrays in their common real floating dtype; integers become the
    namespace's default floating dtype."""
    floats = [a.dtype for a in arrays if xp.isdtype(a.dtype, "real floating")]
    dtype = xp.result_type(*floats) if floats else xp.asarray(1.0).dtype
    return tuple(xp.astype(a, dtype, copy=False) for a in arrays)


def _sqrt_plus_signed(xp, q, radius, transverse_squared):
    """``sqrt(q**2 + transverse_squared) + q`` without cancellation (#1066): a
    radius can round to ``|q|`` near an edge, although the transverse distance
    is not zero; for ``q < 0`` the sum is rationalized."""
    return xp.where(q < 0, transverse_squared / (radius - q), radius + q)


def cuboid_B_core(observers, dimensions, polarizations):
    """B of cuboids in units of the polarization, one per observer, (i, 3)
    each: centered, sides along the axes, no special cases."""
    xp = namespace(observers, dimensions, polarizations)
    observers, dimensions, polarizations = _promote(
        xp, observers, dimensions, polarizations
    )
    pol_x, pol_y, pol_z = xp.unstack(polarizations, axis=-1)
    a, b, c = xp.unstack(dimensions / 2, axis=-1)
    x, y, z = xp.unstack(observers, axis=-1)

    # every position to the octant x >= 0, y <= 0, z <= 0
    maskx, masky, maskz = x < 0, y > 0, z > 0
    x = xp.where(maskx, -x, x)
    y = xp.where(masky, -y, y)
    z = xp.where(maskz, -z, z)
    # a field component changes sign with every flipped axis that is either
    # its own or that of the polarization
    # (an array, not two scalars: the standard's where takes at most one)
    one = xp.ones_like(x)
    sign_x = xp.where(maskx, -one, one)
    sign_y = xp.where(masky, -one, one)
    sign_z = xp.where(maskz, -one, one)
    sign_xy, sign_xz, sign_yz = sign_x * sign_y, sign_x * sign_z, sign_y * sign_z

    with quiet("divide", "invalid", "over"):
        xma, xpa = x - a, x + a
        ymb, ypb = y - b, y + b
        zmc, zpc = z - c, z + c
        xma2, xpa2 = xma**2, xpa**2
        ymb2, ypb2 = ymb**2, ypb**2
        zmc2, zpc2 = zmc**2, zpc**2

        mmm = xp.sqrt(xma2 + ymb2 + zmc2)
        pmp = xp.sqrt(xpa2 + ymb2 + zpc2)
        pmm = xp.sqrt(xpa2 + ymb2 + zmc2)
        mmp = xp.sqrt(xma2 + ymb2 + zpc2)
        mpm = xp.sqrt(xma2 + ypb2 + zmc2)
        ppp = xp.sqrt(xpa2 + ypb2 + zpc2)
        ppm = xp.sqrt(xpa2 + ypb2 + zmc2)
        mpp = xp.sqrt(xma2 + ypb2 + zpc2)

        def sp(q, radius, t2):
            return _sqrt_plus_signed(xp, q, radius, t2)

        ff2x = xp.log(
            sp(xma, mmm, ymb2 + zmc2)
            * (xpa + ppm)
            * (xpa + pmp)
            * sp(xma, mpp, ypb2 + zpc2)
        ) - xp.log(
            (xpa + pmm)
            * sp(xma, mpm, ypb2 + zmc2)
            * sp(xma, mmp, ymb2 + zpc2)
            * (xpa + ppp)
        )
        ff2y = xp.log(
            (-ymb + mmm)
            * sp(-ypb, ppm, xpa2 + zmc2)
            * (-ymb + pmp)
            * sp(-ypb, mpp, xma2 + zpc2)
        ) - xp.log(
            (-ymb + pmm)
            * sp(-ypb, mpm, xma2 + zmc2)
            * (-ymb + mmp)
            * sp(-ypb, ppp, xpa2 + zpc2)
        )
        ff2z = xp.log(
            (-zmc + mmm) * (-zmc + ppm) * sp(-zpc, pmp, xpa2 + ymb2) * sp(-zpc, mpp, xma2 + ypb2)
        ) - xp.log(
            (-zmc + pmm) * (-zmc + mpm) * sp(-zpc, mmp, xma2 + ymb2) * sp(-zpc, ppp, xpa2 + ypb2)
        )  # fmt: skip

        at = xp.atan2
        ff1x = (
            at(ymb * zmc, xma * mmm) - at(ymb * zmc, xpa * pmm) - at(ypb * zmc, xma * mpm)
            + at(ypb * zmc, xpa * ppm) - at(ymb * zpc, xma * mmp) + at(ymb * zpc, xpa * pmp)
            + at(ypb * zpc, xma * mpp) - at(ypb * zpc, xpa * ppp)
        )  # fmt: skip
        ff1y = (
            at(xma * zmc, ymb * mmm) - at(xpa * zmc, ymb * pmm) - at(xma * zmc, ypb * mpm)
            + at(xpa * zmc, ypb * ppm) - at(xma * zpc, ymb * mmp) + at(xpa * zpc, ymb * pmp)
            + at(xma * zpc, ypb * mpp) - at(xpa * zpc, ypb * ppp)
        )  # fmt: skip
        ff1z = (
            at(xma * ymb, zmc * mmm) - at(xpa * ymb, zmc * pmm) - at(xma * ypb, zmc * mpm)
            + at(xpa * ypb, zmc * ppm) - at(xma * ymb, zpc * mmp) + at(xpa * ymb, zpc * pmp)
            + at(xma * ypb, zpc * mpp) - at(xpa * ypb, zpc * ppp)
        )  # fmt: skip

        # the 'missing' third sign is hidden in ff1x, ff1y, ff1z
        bx = pol_x * ff1x + pol_y * ff2z * sign_xy + pol_z * ff2y * sign_xz
        by = pol_x * ff2z * sign_xy + pol_y * ff1y - pol_z * ff2x * sign_yz
        bz = pol_x * ff2y * sign_xz - pol_y * ff2x * sign_yz + pol_z * ff1z
        return xp.stack((bx, by, bz), axis=-1) / (4 * xp.pi)


def cuboid_field(field, observers, dimension, polarization):
    """B in T, H in A/m, J in T or M in A/m (``field`` 'B', 'H', 'J' or
    'M') of cuboids in their own frames, one per observer, (i, 3) each, with
    magpylib's special cases: B and H 0 on edges and corners and for a zero
    dimension or polarization; J the polarization inside, faces included, and
    0 outside; H = (B - J) / mu_0, M = J / mu_0."""
    if field not in ("B", "H", "J", "M"):
        msg = f"field must be 'B', 'H', 'J' or 'M'; instead received {field!r}."
        raise ValueError(msg)
    xp = namespace(observers, dimension, polarization)
    observers, dimension, polarization = _promote(
        xp, observers, dimension, polarization
    )
    half = xp.abs(dimension) / 2
    dist = xp.abs(observers) - half  # distance to the faces, per axis

    pol_not_null = xp.any(polarization != 0, axis=-1)
    # a product, as magpylib: sizes whose product underflows count as zero
    hx, hy, hz = xp.unstack(half, axis=-1)
    dim_not_null = hx * hy * hz != 0
    surf_x, surf_y, surf_z = xp.unstack(xp.abs(dist) < RTOL_SURFACE * half, axis=-1)
    inside_xyz = dist < RTOL_SURFACE * half
    in_x, in_y, in_z = xp.unstack(inside_xyz, axis=-1)
    inside = xp.all(inside_xyz, axis=-1, keepdims=True)
    J = xp.where(inside, polarization, 0.0)
    if field in ("J", "M"):
        return J if field == "J" else J / MU0
    on_edge = (
        (surf_y & surf_z & in_x) | (surf_x & surf_z & in_y) | (surf_x & surf_y & in_z)
    )
    general = (pol_not_null & dim_not_null & ~on_edge)[:, None]

    # the special cases are evaluated at the center of a unit cube, clear of
    # every singularity, and then replaced by 0
    B = xp.where(
        general,
        cuboid_B_core(
            xp.where(general, observers, 0.0),
            xp.where(general, dimension, 1.0),
            polarization,
        ),
        0.0,
    )
    if field == "B":
        return B
    return (B - J) / MU0
