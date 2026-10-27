"""Field of a flat triangle whose surface charge varies linearly across it.

Triangle ``a, b, c``, observer ``x`` at height ``w = (x - a).n``. Per edge
``p -> q``, with unit direction ``t`` and in-plane normal ``m = t x n``
(Wilton et al. 1984; Graglia 1993):

    f_e    = ln((R+ + l+) / (R- + l-))                         l = (vertex - x).t
    beta_e = atan(P0 l+ / (R0^2 + |w| R+)) - atan(P0 l- / (R0^2 + |w| R-))
    Phi    = int 1/R dS          = sum P0 f - |w| sum beta     P0 = (p - x).m
    E0     = int (x - y)/R^3 dS  = sum m f + n sign(w) sum beta

and for a linear density ``psi(y) = psi(rho) + g.(y - rho)``, ``rho`` the
projection of ``x`` on the plane, by the surface divergence theorem:

    int psi (x - y)/R^3 dS = psi(rho) E0 - sum (g.m) L_e - g Phi,
    L_e = int_e (x - y)/R dl = d_e f_e - t (R+ - R-),

with ``d_e`` the vector from the foot of ``x`` on the edge line to ``x``. The
densities are ``1, u, v``: ``u, v`` the in-plane coordinates along ``e1, e2``
from the centroid, divided by the triangle's scale ``sqrt(area)``.

    D. R. Wilton, S. M. Rao, A. W. Glisson, D. H. Schaubert, O. M. Al-Bundak,
    C. M. Butler (1984) "Potential integrals for uniform and linear source
    distributions on polygonal and polyhedral domains", IEEE Trans. Antennas
    Propag. 32(3), 276-281.

    R. D. Graglia (1993) "On the numerical integration of the linear shape
    functions times the 3-D Green's function or its gradient on a plane
    triangle", IEEE Trans. Antennas Propag. 41(10), 1448-1455.
"""

from __future__ import annotations

import numpy as np


def frames(V):
    """Per triangle of ``V`` (N, 3, 3): centroid, normal, ``e1``, ``e2``, scale
    ``sqrt(area)`` and area."""
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    n = cross / (2 * area)[:, None]
    e1 = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
    e2 = np.cross(n, e1)
    return V.mean(axis=1), n, e1, e2, np.sqrt(area), area


def _log_sum(R, l, R02):
    """``ln(R + l)``, stable where ``l < 0``: there ``R + l = R0^2 / (R - l)``."""
    pos = l >= 0
    return np.where(
        pos,
        np.log(np.where(pos, R + l, 1.0)),
        np.log(np.where(pos, 1.0, R02)) - np.log(np.where(pos, 1.0, R - l)),
    )


def field(X, V, fr):
    """``int phi (x - y) / R^3 dS`` for the densities ``phi = 1, u, v``.

    Parameters
    ----------
    X : ndarray, shape (M, 3)
        Observers.
    V : ndarray, shape (N, 3, 3)
        Triangles.
    fr : tuple
        Their :func:`frames`.

    Returns
    -------
    ndarray, shape (3 densities, M, N, 3)
    """
    cen, n, e1, e2, s, _ = fr
    x = X[:, None, :]
    w = np.sum((x - V[None, :, 0]) * n[None], axis=-1)  # (M, N)
    rho = x - w[..., None] * n[None]  # the projection onto the plane
    # An observer exactly in the plane, on an edge's extension or at a vertex,
    # makes edge logarithms infinite: lift it 1e-12 of the triangle off, and
    # take every distance below from there.
    w = np.where(w == 0, 1e-12 * s[None], w)
    x = rho + w[..., None] * n[None]
    sign_w = np.sign(w)
    abs_w = np.abs(w)
    E0 = np.zeros((*w.shape, 3))
    Phi = np.zeros(w.shape)
    sum_L = [
        np.zeros((*w.shape, 3)),
        np.zeros((*w.shape, 3)),
    ]  # sum (g.m) L_e for g = e1, e2
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = V[None, :, i], V[None, :, j]
            t = (q - p) / np.linalg.norm(q - p, axis=-1)[..., None]
            m = np.cross(t, n[None])
            l_minus = np.sum((p - x) * t, axis=-1)
            l_plus = np.sum((q - x) * t, axis=-1)
            P0 = np.sum((p - rho) * m, axis=-1)
            R_minus = np.linalg.norm(x - p, axis=-1)
            R_plus = np.linalg.norm(x - q, axis=-1)
            R02 = P0**2 + w**2
            f = _log_sum(R_plus, l_plus, R02) - _log_sum(R_minus, l_minus, R02)
            beta = np.arctan2(P0 * l_plus, R02 + abs_w * R_plus) - np.arctan2(
                P0 * l_minus, R02 + abs_w * R_minus
            )
            E0 += m * f[..., None] + n[None] * (sign_w * beta)[..., None]
            Phi += P0 * f - abs_w * beta
            d = (x - p) + l_minus[..., None] * t  # x minus its foot on the edge line
            L = d * f[..., None] - t * (R_plus - R_minus)[..., None]
            for k, e in enumerate((e1, e2)):
                sum_L[k] += np.sum(e[None] * m, axis=-1)[..., None] * L
    out = [E0]
    for k, e in enumerate((e1, e2)):
        psi_rho = (
            np.sum((rho - cen[None]) * e[None], axis=-1) / s[None]
        )  # density at the projection
        g = e[None] / s[None, :, None]  # gradient of u (or v) in the plane
        out.append(
            psi_rho[..., None] * E0 - sum_L[k] / s[None, :, None] - g * Phi[..., None]
        )
    return np.stack(out)


def normal_field(X, V, fr, pairs=False):
    """``n . int phi (x - y) / R^3 dS``: only the component along each
    triangle's normal, which is all a Galerkin flux needs. With ``n . L_e =
    w f_e`` and ``g`` in the plane it is ``psi(rho) sign(w) sum beta - w sum
    (g.m) f``.

    ``X`` (M, 3) observers, ``V`` (N, 3, 3) triangles and ``fr`` their
    :func:`frames`: returns (3 densities, M, N). With ``pairs``, ``X`` and ``V``
    have the same length and observer ``i`` goes with triangle ``i`` only:
    returns (3 densities, M).
    """
    cen, n, e1, e2, s, _ = fr
    if pairs:
        x, Vb, nb, cb, e1b, e2b, sb = X, V, n, cen, e1, e2, s
    else:
        x = X[:, None, :]
        Vb, nb, cb, e1b, e2b, sb = (
            V[None],
            n[None],
            cen[None],
            e1[None],
            e2[None],
            s[None],
        )
    w = np.sum((x - Vb[..., 0, :]) * nb, axis=-1)
    rho = x - w[..., None] * nb
    # lifted off the plane as in `field`, every distance taken from there
    w = np.where(w == 0, 1e-12 * sb, w)
    x = rho + w[..., None] * nb
    abs_w = np.abs(w)
    sum_beta = 0.0
    sum_mf = [0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = Vb[..., i, :], Vb[..., j, :]
            t = (q - p) / np.linalg.norm(q - p, axis=-1)[..., None]
            m = np.cross(t, nb)
            l_minus = np.sum((p - x) * t, axis=-1)
            l_plus = np.sum((q - x) * t, axis=-1)
            P0 = np.sum((p - rho) * m, axis=-1)
            R_minus = np.linalg.norm(x - p, axis=-1)
            R_plus = np.linalg.norm(x - q, axis=-1)
            R02 = P0**2 + w**2
            f = _log_sum(R_plus, l_plus, R02) - _log_sum(R_minus, l_minus, R02)
            sum_beta = (
                sum_beta
                + np.arctan2(P0 * l_plus, R02 + abs_w * R_plus)
                - np.arctan2(P0 * l_minus, R02 + abs_w * R_minus)
            )
            for k, e in enumerate((e1b, e2b)):
                sum_mf[k] = sum_mf[k] + np.sum(e * m, axis=-1) * f
    En0 = np.sign(w) * sum_beta
    out = [En0]
    for k, e in enumerate((e1b, e2b)):
        psi_rho = np.sum((rho - cb) * e, axis=-1) / sb
        out.append(psi_rho * En0 - w * sum_mf[k] / sb)
    return np.stack(out)


def normal_field_far(X, V, fr):
    """:func:`normal_field` for all observers against all triangles, (3, M, N),
    with every dot product one matrix product: ``|x - p|^2 = |x|^2 - 2 x.p +
    |p|^2``, only logarithms, arctangents and roots per pair. For pairs farther
    apart than a triangle's size: the cancellation in ``|x - p|^2`` grows as
    ``(size / distance)^2`` times the rounding."""
    cen, n, e1, e2, s, _ = fr
    xx = np.sum(X * X, axis=1)[:, None]
    w = X @ n.T - np.sum(V[:, 0] * n, axis=1)[None]
    w = np.where(w == 0, 1e-12 * s[None], w)
    abs_w = np.abs(w)
    R = [
        np.sqrt(
            np.maximum(xx - 2 * X @ V[:, k].T + np.sum(V[:, k] ** 2, axis=1)[None], 0.0)
        )
        for k in range(3)
    ]
    sum_beta = 0.0
    sum_mf = [0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = V[:, i], V[:, j]
            t = (q - p) / np.linalg.norm(q - p, axis=1)[:, None]
            m = np.cross(t, n)
            xt, xm = X @ t.T, X @ m.T
            l_minus = np.sum(p * t, axis=1)[None] - xt
            l_plus = np.sum(q * t, axis=1)[None] - xt
            P0 = np.sum(p * m, axis=1)[None] - xm
            R02 = P0**2 + w**2
            f = _log_sum(R[j], l_plus, R02) - _log_sum(R[i], l_minus, R02)
            sum_beta = (
                sum_beta
                + np.arctan2(P0 * l_plus, R02 + abs_w * R[j])
                - np.arctan2(P0 * l_minus, R02 + abs_w * R[i])
            )
            for k, e in enumerate((e1, e2)):
                sum_mf[k] = sum_mf[k] + np.sum(e * m, axis=1)[None] * f
    En0 = np.sign(w) * sum_beta
    out = [En0]
    for k, e in enumerate((e1, e2)):
        psi_rho = (X @ e.T - np.sum(cen * e, axis=1)[None]) / s[None]
        out.append(psi_rho * En0 - w * sum_mf[k] / s[None])
    return np.stack(out)
