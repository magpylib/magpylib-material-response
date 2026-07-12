"""Analytical demagnetization tensor for rectangular prisms (Newell 1993).

Implements the volume-averaged demagnetization tensor between two identical
axis-aligned rectangular prisms, following:

    A. J. Newell, W. Williams, D. J. Dunlop (1993)
    "A generalization of the demagnetizing tensor for nonuniform magnetization"
    J. Geophys. Res. 98(B6), 9551-9555.

The 27-point second-difference form (used by OOMMF / MuMax3) is employed:

    N_xx(R; a,b,c) = -1/(4*pi*a*b*c) * sum_{i,j,k=-1,0,1} w(i)w(j)w(k)
                                       * f(R + (i*a, j*b, k*c))

with weights w(-1)=w(1)=1, w(0)=-2. ``N_xy`` follows the same form with the
auxiliary function ``g``. The remaining four components are obtained by
coordinate / dimension permutations.

Sign convention: the tensor returned by :func:`demag_tensor_newell` matches
the layout produced by :func:`magpylib_material_response.demag.demag_tensor`
*before* the ``*= mu_0`` step in ``apply_demag``. That is

    T[k, i, j, m] = -(1/mu_0) * N_mk(disp = pos[j] - pos[i], dim)

so that, for a uniformly magnetised cube cell at zero displacement, the
diagonal is exactly ``-1 / (3 * mu_0)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "demag_block",
    "demag_block_general",
    "demag_tensor_newell",
    "newell_f",
    "newell_g",
    "self_demag_factors",
]


# 27-point second-difference weights and offsets.
_W = (1, -2, 1)
_OFF = (-1, 0, 1)


def _safe(arr):
    """Replace NaN/+-inf with 0; used to neutralise removable singularities
    where a vanishing prefactor multiplies a logarithmic divergence."""
    return np.where(np.isfinite(arr), arr, 0.0)


def _atan(num, den):
    """arctan(num/den) with the original Newell sign convention.

    Returns 0 when both are 0 (removable singularity in the prefactor).
    Uses single-argument arctan (which is odd in its argument) rather than
    atan2; this is required so that the auxiliary functions retain the
    symmetries quoted in Newell 1993 eq. A1 / A4.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe(np.arctan(num / den))


def newell_f(x, y, z):
    """Newell auxiliary function ``f`` (Newell 1993, eq. A1).

    Vectorised; ``x``, ``y``, ``z`` may be arrays of any broadcastable shape.
    Singular intermediate values (caused by zero denominators) are replaced
    by their analytical limit, which is zero for every relevant call site.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x2, y2, z2 = x * x, y * y, z * z
        R = np.sqrt(x2 + y2 + z2)
        t1 = _safe(0.5 * y * (z2 - x2) * np.arcsinh(y / np.sqrt(x2 + z2)))
        t2 = _safe(0.5 * z * (y2 - x2) * np.arcsinh(z / np.sqrt(x2 + y2)))
        t3 = -x * y * z * _atan(y * z, x * R)
        t4 = (2.0 * x2 - y2 - z2) * R / 6.0
    return t1 + t2 + t3 + t4


def newell_g(x, y, z):
    """Newell auxiliary function ``g`` (Newell 1993, eq. A4)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x2, y2, z2 = x * x, y * y, z * z
        R = np.sqrt(x2 + y2 + z2)
        t1 = _safe(x * y * z * np.arcsinh(z / np.sqrt(x2 + y2)))
        t2 = _safe((y / 6.0) * (3.0 * z2 - y2) * np.arcsinh(x / np.sqrt(y2 + z2)))
        t3 = _safe((x / 6.0) * (3.0 * z2 - x2) * np.arcsinh(y / np.sqrt(x2 + z2)))
        t4 = -(z**3 / 6.0) * _atan(x * y, z * R)
        t5 = -(z * y2 / 2.0) * _atan(x * z, y * R)
        t6 = -(z * x2 / 2.0) * _atan(y * z, x * R)
        t7 = -x * y * R / 3.0
    return t1 + t2 + t3 + t4 + t5 + t6 + t7


def _double_diff_27(func, X, Y, Z, a, b, c):
    """Apply the 27-point second-difference operator to ``func``.

    Equivalent to (1/4pi/(a*b*c)) * sum w_i w_j w_k * func(X+i*a, Y+j*b, Z+k*c).
    Returns the *negative* of that sum divided by 4*pi*a*b*c, which is the
    demag-tensor scalar element.
    """
    shape = np.broadcast_shapes(np.shape(X), np.shape(Y), np.shape(Z))
    acc = np.zeros(shape, dtype=float)
    for wi, oi in zip(_W, _OFF, strict=True):
        for wj, oj in zip(_W, _OFF, strict=True):
            for wk, ok in zip(_W, _OFF, strict=True):
                w = wi * wj * wk
                if w == 0:
                    continue
                acc = acc + w * func(X + oi * a, Y + oj * b, Z + ok * c)
    return -acc / (4.0 * np.pi * a * b * c)


def _N_xx(X, Y, Z, a, b, c):
    return _double_diff_27(newell_f, X, Y, Z, a, b, c)


def _N_xy(X, Y, Z, a, b, c):
    return _double_diff_27(newell_g, X, Y, Z, a, b, c)


def _axis_terms(h1, h2):
    """Offsets/weights of the 4-point double difference for one axis.

    The double integral over source extent ``h1`` and observer extent ``h2``
    of ``G''`` collapses to::

        G(X+(h1+h2)/2) - G(X+(h1-h2)/2) - G(X-(h1-h2)/2) + G(X-(h1+h2)/2)

    For ``h1 == h2`` this degenerates to the (1, -2, 1) second difference,
    saving one function evaluation on that axis.
    """
    if abs(h1 - h2) <= 1e-12 * max(abs(h1), abs(h2)):
        return ((+h1, 1.0), (0.0, -2.0), (-h1, 1.0))
    return (
        (+(h1 + h2) / 2.0, 1.0),
        (+(h1 - h2) / 2.0, -1.0),
        (-(h1 - h2) / 2.0, -1.0),
        (-(h1 + h2) / 2.0, 1.0),
    )


def _double_diff_64(func, X, Y, Z, dim_src, dim_obs):
    """64-point double-difference operator for two different-size prisms.

    Generalization of :func:`_double_diff_27` to a source prism with side
    lengths ``dim_src`` and an observer prism with side lengths ``dim_obs``.
    Normalised by the *observer* volume so that the far field reduces to a
    point dipole with moment ``M * V_src``.
    """
    a1, b1, c1 = (float(v) for v in dim_src)
    a2, b2, c2 = (float(v) for v in dim_obs)
    shape = np.broadcast_shapes(np.shape(X), np.shape(Y), np.shape(Z))
    acc = np.zeros(shape, dtype=float)
    for ox, wx in _axis_terms(a1, a2):
        for oy, wy in _axis_terms(b1, b2):
            for oz, wz in _axis_terms(c1, c2):
                acc = acc + (wx * wy * wz) * func(X + ox, Y + oy, Z + oz)
    return -acc / (4.0 * np.pi * a2 * b2 * c2)


def demag_block_general(disp, dim_src, dim_obs):
    """3x3 volume-averaged demag tensor between two parallel rectangular prisms.

    Generalizes :func:`demag_block` to source and observer prisms of
    *different* side lengths (both axis-aligned in the same frame).

    Parameters
    ----------
    disp : array_like, shape (..., 3)
        Centre-to-centre displacement ``obs - src``.
    dim_src : array_like, shape (3,)
        Source prism side lengths ``(a1, b1, c1)``.
    dim_obs : array_like, shape (3,)
        Observer prism side lengths ``(a2, b2, c2)``.

    Returns
    -------
    N : ndarray, shape (..., 3, 3)
        ``N_mk`` such that the H-field volume-averaged over the observer
        prism per unit magnetization of the source prism is ``H = -N @ M``.
        Satisfies the volume-weighted reciprocity
        ``V_obs * N(d; s, o) = V_src * N(d; o, s).T``.
    """
    dim_src = np.asarray(dim_src, dtype=float)
    dim_obs = np.asarray(dim_obs, dtype=float)
    if np.array_equal(dim_src, dim_obs):
        return demag_block(disp, dim_src)

    disp = np.asarray(disp, dtype=float)
    X = disp[..., 0]
    Y = disp[..., 1]
    Z = disp[..., 2]
    a1, b1, c1 = dim_src
    a2, b2, c2 = dim_obs

    Nxx = _double_diff_64(newell_f, X, Y, Z, (a1, b1, c1), (a2, b2, c2))
    # Same permutation identities as in demag_block, applied to both dims.
    Nyy = _double_diff_64(newell_f, Y, X, Z, (b1, a1, c1), (b2, a2, c2))
    Nzz = _double_diff_64(newell_f, Z, Y, X, (c1, b1, a1), (c2, b2, a2))
    Nxy = _double_diff_64(newell_g, X, Y, Z, (a1, b1, c1), (a2, b2, c2))
    Nxz = _double_diff_64(newell_g, X, Z, Y, (a1, c1, b1), (a2, c2, b2))
    Nyz = _double_diff_64(newell_g, Y, Z, X, (b1, c1, a1), (b2, c2, a2))

    out = np.empty((*np.shape(Nxx), 3, 3), dtype=float)
    out[..., 0, 0] = Nxx
    out[..., 1, 1] = Nyy
    out[..., 2, 2] = Nzz
    out[..., 0, 1] = out[..., 1, 0] = Nxy
    out[..., 0, 2] = out[..., 2, 0] = Nxz
    out[..., 1, 2] = out[..., 2, 1] = Nyz
    return out


def demag_block(disp, dim):
    """Compute the 3x3 demag tensor block ``N`` for given displacement and dimension.

    Parameters
    ----------
    disp : array_like, shape (..., 3)
        Centre-to-centre displacement ``obs - src`` (observer minus source).
    dim : array_like, shape (3,)
        Common cell side lengths ``(a, b, c)``.

    Returns
    -------
    N : ndarray, shape (..., 3, 3)
        Symmetric demag tensor block. ``N_ii`` are diagonal demag factors
        (sum to 1 for ``disp = 0``); off-diagonal terms are mutual coupling.
    """
    disp = np.asarray(disp, dtype=float)
    a, b, c = (float(v) for v in dim)
    X = disp[..., 0]
    Y = disp[..., 1]
    Z = disp[..., 2]

    Nxx = _N_xx(X, Y, Z, a, b, c)
    # Permutation identities (verified e.g. against OOMMF):
    #   N_yy(X,Y,Z; a,b,c) = N_xx(Y,X,Z; b,a,c)
    #   N_zz(X,Y,Z; a,b,c) = N_xx(Z,Y,X; c,b,a)
    Nyy = _N_xx(Y, X, Z, b, a, c)  # pylint: disable=arguments-out-of-order
    Nzz = _N_xx(Z, Y, X, c, b, a)  # pylint: disable=arguments-out-of-order
    Nxy = _N_xy(X, Y, Z, a, b, c)
    #   N_xz(X,Y,Z; a,b,c) = N_xy(X,Z,Y; a,c,b)
    #   N_yz(X,Y,Z; a,b,c) = N_xy(Y,Z,X; b,c,a)
    Nxz = _N_xy(X, Z, Y, a, c, b)  # pylint: disable=arguments-out-of-order
    Nyz = _N_xy(Y, Z, X, b, c, a)  # pylint: disable=arguments-out-of-order

    out = np.empty((*X.shape, 3, 3), dtype=float)
    out[..., 0, 0] = Nxx
    out[..., 1, 1] = Nyy
    out[..., 2, 2] = Nzz
    out[..., 0, 1] = out[..., 1, 0] = Nxy
    out[..., 0, 2] = out[..., 2, 0] = Nxz
    out[..., 1, 2] = out[..., 2, 1] = Nyz
    return out


def self_demag_factors(dim):
    """Return ``(Nxx, Nyy, Nzz)`` for an isolated rectangular prism.

    Sum to 1 by Brown's identity; equal to ``1/3`` each for a cube.
    """
    N = demag_block(np.zeros(3), dim)
    return np.array([N[0, 0], N[1, 1], N[2, 2]])


def demag_tensor_newell(positions, dim, mu_0):
    """Volume-averaged demag tensor for ``n`` identical axis-aligned cuboids.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Centre positions of the cells (world frame).
    dim : array_like, shape (3,)
        Common side lengths ``(a, b, c)``.
    mu_0 : float
        Magnetic constant (``magpylib.mu_0``).

    Returns
    -------
    T : ndarray, shape (3, n, n, 3)
        Same layout as :func:`magpylib_material_response.demag.demag_tensor`:
        ``T[k, i, j, m]`` is the world-frame component ``m`` of the volume
        averaged H-field over observer cell ``j`` per unit polarisation along
        world axis ``k`` of source cell ``i``, divided by ``mu_0`` is *not*
        applied here -- the caller multiplies by ``mu_0`` afterwards.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        msg = f"positions must have shape (n, 3); got {pos.shape}"
        raise ValueError(msg)
    # disp[j, i] = pos[j] - pos[i]  (observer minus source)
    disp = pos[:, None, :] - pos[None, :, :]
    N = demag_block(disp, dim)  # shape (n, n, 3, 3); axes (j, i, m, k)
    # T[k, i, j, m] = -(1/mu_0) * N[j, i, m, k]
    return np.transpose(N, (3, 1, 0, 2)) * (-1.0 / mu_0)
