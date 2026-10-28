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

    A. van Oosterom, J. Strackee (1983) "The solid angle of a plane
    triangle", IEEE Trans. Biomed. Eng. 30(2), 125-126.
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


_EDGES = ((0, 1), (1, 2), (2, 0))


def _r_plus_l(R, l, R02):
    """``R + l``, stable where ``l < 0``: there it is ``R0^2 / (R - l)``."""
    return np.where(l >= 0, R + l, R02 / (R - l))


def _dot(a, b):
    """Dot products along the last axis, broadcast."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def charge_field(X, V, fr, c):
    """``sum_n sum_k c[n, k] int phi_k (x - y) / R^3 dS`` over the triangles:
    the field, times ``4 pi``, of charge ``c0 + c1 u + c2 v`` on each, (M, 3).

    In each triangle's plane coordinates ``u, v, w``, with ``m = t x n = (tv,
    -tu)`` and ``d_e = -P0 m + w n``, every term is a product of scalars, the
    coefficients taken in per triangle; only the sum leaves the plane, by
    three matrix products. ``sign(w) sum beta`` is the solid angle, taken as in
    :func:`normal_field`.

    Parameters
    ----------
    X : ndarray, shape (M, 3)
        Observers.
    V : ndarray, shape (N, 3, 3)
        Triangles.
    fr : tuple
        Their :func:`frames`.
    c : ndarray, shape (N, 3)
        Each triangle's coefficients of ``1, u, v``.
    """
    _, n, e1, e2, s, area = fr
    a = V[:, 0]
    ub = _dot(V[:, 1] - a, e1)
    uc, vc = _dot(V[:, 2] - a, e1), _dot(V[:, 2] - a, e2)
    corners = ((0.0, 0.0), (ub, 0.0), (uc, vc))
    u, v, w = (
        np.matmul(X, np.stack([e1, e2, n]).transpose(0, 2, 1))
        - np.stack([_dot(a, e1), _dot(a, e2), _dot(a, n)])[:, None]
    )
    # lifted off the plane as in `normal_field`
    w = np.where(w == 0, 1e-12 * s, w)
    w2, abs_w = w * w, np.abs(w)
    du = [pu - u for pu, _ in corners]
    dv = [pv - v for _, pv in corners]
    R = [np.sqrt(du[k] * du[k] + dv[k] * dv[k] + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(_EDGES, (2, 0, 1), strict=True):
        den = den + (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    omega = 2 * np.arctan2(2 * area * abs_w, den)  # sum beta
    c0, c1, c2 = c[:, 0], c[:, 1], c[:, 2]
    # E0 = sum m f + n sign(w) sum beta, Phi = sum P0 f - |w| sum beta, and
    # sum (g.m) L_e with g = c1 e1 + c2 e2, L_e = d_e f - t (R+ - R-)
    E0u = E0v = Phi = SLu = SLv = SLw = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in _EDGES:
            (pu, pv), (qu, qv) = corners[i], corners[j]
            L = np.sqrt((qu - pu) ** 2 + (qv - pv) ** 2)
            tu, tv = (qu - pu) / L, (qv - pv) / L
            l_minus = du[i] * tu + dv[i] * tv
            l_plus = du[j] * tu + dv[j] * tv
            P0 = du[i] * tv - dv[i] * tu
            R02 = P0 * P0 + w2
            f = np.log(_r_plus_l(R[j], l_plus, R02) / _r_plus_l(R[i], l_minus, R02))
            P0f, dR, gm = P0 * f, R[j] - R[i], c1 * tv - c2 * tu
            E0u = E0u + tv * f
            E0v = E0v - tu * f
            Phi = Phi + P0f
            SLu = SLu - gm * (P0f * tv + dR * tu)
            SLv = SLv + gm * (P0f * tu - dR * tv)
            SLw = SLw + gm * f
    Phi = Phi - abs_w * omega
    # the density at the projection, from the centroid (ub + uc, vc) / 3
    psi = c0 + (c1 * (u - (ub + uc) / 3) + c2 * (v - vc / 3)) / s
    Fu = psi * E0u - (SLu + c1 * Phi) / s
    Fv = psi * E0v - (SLv + c2 * Phi) / s
    Fw = psi * np.sign(w) * omega - w * SLw / s
    return Fu @ e1 + Fv @ e2 + Fw @ n


def normal_field(X, V, fr, pairs=False):
    """``n . int phi (x - y) / R^3 dS``: only the component along each
    triangle's normal, which is all a Galerkin flux needs. With ``n . L_e =
    w f_e`` and ``g`` in the plane it is ``psi(rho) sign(w) sum beta - w sum
    (g.m) f``, ``sign(w) sum beta`` the solid angle, taken as one arctangent
    (van Oosterom and Strackee 1983, see :func:`normal_field_far`).

    ``X`` (M, 3) observers, ``V`` (N, 3, 3) triangles and ``fr`` their
    :func:`frames`: returns (3 densities, M, N). With ``pairs``, observers and
    triangles broadcast against each other instead, ``X`` (..., 3) against
    ``V`` (..., 3, 3) and ``fr`` likewise: ``X`` (K, Q, 3) against ``V`` (K, 1,
    3, 3) gives (3, K, Q), each triangle's own terms once for its Q observers.

    In each triangle's plane coordinates ``u, v`` along ``e1, e2`` from its
    first corner, and ``w``, every term is a product of scalars.
    """
    _, n, e1, e2, s, area = fr
    if not pairs:
        X = X[:, None]
        V, n, e1, e2, s, area = (a[None] for a in (V, n, e1, e2, s, area))
    a = V[..., 0, :]
    # the corners: (0, 0), (ub, 0), (uc, vc)
    ub = _dot(V[..., 1, :] - a, e1)
    uc, vc = _dot(V[..., 2, :] - a, e1), _dot(V[..., 2, :] - a, e2)
    corners = ((0.0, 0.0), (ub, 0.0), (uc, vc))
    d = X - a
    u, v, w = _dot(d, e1), _dot(d, e2), _dot(d, n)
    # lifted off the plane as in `field`
    w = np.where(w == 0, 1e-12 * s, w)
    w2 = w * w
    du = [pu - u for pu, _ in corners]  # from the observer to each corner
    dv = [pv - v for _, pv in corners]
    R = [np.sqrt(du[k] * du[k] + dv[k] * dv[k] + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(_EDGES, (2, 0, 1), strict=True):
        den = den + (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    En0 = 2 * np.sign(w) * np.arctan2(2 * area * np.abs(w), den)
    sum_mu = sum_mv = 0.0  # sum (e1.m) f and sum (e2.m) f, m = t x n = (tv, -tu)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in _EDGES:
            (pu, pv), (qu, qv) = corners[i], corners[j]
            L = np.sqrt((qu - pu) ** 2 + (qv - pv) ** 2)
            tu, tv = (qu - pu) / L, (qv - pv) / L
            l_minus = du[i] * tu + dv[i] * tv
            l_plus = du[j] * tu + dv[j] * tv
            P0 = du[i] * tv - dv[i] * tu
            R02 = P0 * P0 + w2
            f = np.log(_r_plus_l(R[j], l_plus, R02) / _r_plus_l(R[i], l_minus, R02))
            sum_mu = sum_mu + tv * f
            sum_mv = sum_mv - tu * f
    # the densities at the projection, from the centroid (ub + uc, vc) / 3
    psi_u, psi_v = (u - (ub + uc) / 3) / s, (v - vc / 3) / s
    return np.stack([En0, psi_u * En0 - w * sum_mu / s, psi_v * En0 - w * sum_mv / s])


def normal_field_far(X, V, fr):
    """:func:`normal_field` for all observers against all triangles, (3, M, N),
    with every dot product one matrix product: ``|x - p|^2 = |x|^2 - 2 x.p +
    |p|^2``. For pairs farther apart than a triangle's size: the cancellation in
    ``|x - p|^2`` grows as ``(|x| / distance)^2`` times the rounding, ``|x|``
    from the frame's origin: the callers centre it on the part.

    Two exact forms halve the work. ``sign(w) sum beta`` is the solid angle,
    one arctangent from the corners (van Oosterom and Strackee 1983), and
    accurate far away where the edge terms cancel. ``f_e = ln((S + L) / (S -
    L))``, ``S = R+ + R-``, ``L`` the edge's length, stable off the edge itself;
    near it, the few pairs closer than ``S - L < L / 100`` are taken by the
    stable branch.
    """
    cen, n, e1, e2, s, area = fr
    # every dot product with an observer: -2 x.p for the corners, x.n, and the
    # densities' x.e / s
    XD = np.matmul(
        X,
        np.stack(
            [
                -2 * V[:, 0],
                -2 * V[:, 1],
                -2 * V[:, 2],
                n,
                e1 / s[:, None],
                e2 / s[:, None],
            ]
        ).transpose(0, 2, 1),
    )
    xx = np.sum(X * X, axis=1)[:, None]
    R2 = [np.maximum(XD[k] + np.sum(V[:, k] ** 2, axis=1) + xx, 0.0) for k in range(3)]
    R = [np.sqrt(r2) for r2 in R2]
    w = XD[3] - np.sum(V[:, 0] * n, axis=1)
    w = np.where(w == 0, 1e-12 * s, w)
    edge = [V[:, j] - V[:, i] for i, j in _EDGES]
    L2 = [np.sum(d * d, axis=1) for d in edge]
    # the corners' r_i.r_j = (R_i^2 + R_j^2 - L_ij^2) / 2, each times the third R
    den = R[0] * R[1] * R[2]
    for (i, j), l2, k in zip(_EDGES, L2, (2, 0, 1), strict=True):
        den = den + 0.5 * (R2[i] + R2[j] - l2) * R[k]
    out = np.empty((3, len(X), len(V)))
    out[0] = 2 * np.sign(w) * np.arctan2(2 * area * np.abs(w), den)
    sum_mf = [0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for (i, j), d, l2 in zip(_EDGES, edge, L2, strict=True):
            L = np.sqrt(l2)
            t = d / L[:, None]
            S = R[i] + R[j]
            gap = S - L
            f = np.log((S + L) / gap)
            a, b = np.nonzero(gap < 1e-2 * L)
            if len(a):
                x, p, q, tb = X[a], V[b, i], V[b, j], t[b]
                l_minus = np.sum((p - x) * tb, axis=1)
                l_plus = np.sum((q - x) * tb, axis=1)
                foot = (x - p) + l_minus[:, None] * tb  # from the edge line
                # at least w², lifted off the plane: finite on the edge line
                R02 = np.maximum(np.sum(foot * foot, axis=1), w[a, b] ** 2)
                R_minus = np.linalg.norm(x - p, axis=1)
                R_plus = np.linalg.norm(x - q, axis=1)
                f[a, b] = np.log(
                    _r_plus_l(R_plus, l_plus, R02) / _r_plus_l(R_minus, l_minus, R02)
                )
            m = np.cross(t, n)
            for k, e in enumerate((e1, e2)):
                sum_mf[k] = sum_mf[k] + (np.sum(e * m, axis=1) / s) * f
    for k, e in enumerate((e1, e2)):
        psi_rho = XD[4 + k] - np.sum(cen * e, axis=1) / s  # density at the projection
        out[1 + k] = psi_rho * out[0] - w * sum_mf[k]
    return out
