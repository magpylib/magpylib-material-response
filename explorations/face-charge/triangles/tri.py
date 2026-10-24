"""Field of a flat triangle whose surface charge varies linearly over it.

Triangle a, b, c (counter-clockwise about its outward normal n), observer x,
height w = (x - a).n. Per edge p -> q with unit direction t and outward in-plane
normal m = t x n (Wilton et al. 1984; Graglia 1993):

    f_e    = ln((R+ + l+) / (R- + l-))                       l = (vertex - x).t
    beta_e = atan(P0 l+ / (R0^2 + |w| R+)) - atan(P0 l- / (R0^2 + |w| R-))
    Phi    = int 1/R dS        = sum P0 f - |w| sum beta     P0 = (p - x).m
    E0     = int (x - y)/R^3 dS = sum m f + n sign(w) sum beta

and for a linear density psi(y) = psi(rho) + g.(y - rho), rho the projection of
x, by the surface divergence theorem (as in explore/cells-and-fem, lintet.py):

    int psi (x - y)/R^3 dS = psi(rho) E0 - sum (g.m) L_e - g Phi,
    L_e = int_e (x - y)/R dl = d_e f_e - t (R+ - R-),  d_e = x minus its foot on the edge line.

The densities are 1, u, v: u, v the in-plane coordinates along e1, e2 from the
centroid, divided by the panel scale s.
"""

import numpy as np

FOUR_PI = 4 * np.pi


def frames(V):
    """Per triangle (N, 3, 3): centroid, normal, e1, e2, scale, area."""
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / (2 * area)[:, None]
    e1 = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
    e2 = np.cross(n, e1)
    cen = V.mean(axis=1)
    s = np.sqrt(area)
    return cen, n, e1, e2, s, area


def _log_sum(R, l, R02):
    """ln(R + l), stable for l < 0 where R + l = R0^2 / (R - l)."""
    pos = l >= 0
    return np.where(pos, np.log(np.where(pos, R + l, 1.0)), np.log(np.where(pos, 1.0, R02)) - np.log(np.where(pos, 1.0, R - l)))


def field(X, V, fr=None):
    """E_k(x) = int phi_k (x - y)/R^3 dS for densities phi = (1, u, v).

    X (M, 3) observers, V (N, 3, 3) triangles. Returns (3 densities, M, N, 3)."""
    cen, n, e1, e2, s, _ = frames(V) if fr is None else fr
    x = X[:, None, :]  # (M, 1, 3)
    w = np.sum((x - V[None, :, 0]) * n[None], axis=-1)  # (M, N)
    # An observer exactly in the plane on an edge's extension makes two edge
    # logs infinite where they cancel: lift it 1e-12 of the triangle off the plane.
    w = np.where(w == 0, 1e-12 * s[None], w)
    rho = x - w[..., None] * n[None]
    sw = np.sign(w)
    aw = np.abs(w)
    E0 = np.zeros(w.shape + (3,))
    Phi = np.zeros(w.shape)
    sumL = [np.zeros(w.shape + (3,)), np.zeros(w.shape + (3,))]  # sum (g.m) L_e for g = e1, e2 (per unit g)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p, q = V[None, :, i], V[None, :, j]
        L = np.linalg.norm(q - p, axis=-1)
        t = (q - p) / L[..., None]
        m = np.cross(t, n[None])
        lm = np.sum((p - x) * t, axis=-1)
        lp = np.sum((q - x) * t, axis=-1)
        P0 = np.sum((p - rho) * m, axis=-1)
        Rm = np.linalg.norm(x - p, axis=-1)
        Rp = np.linalg.norm(x - q, axis=-1)
        R02 = P0**2 + w**2
        f = _log_sum(Rp, lp, R02) - _log_sum(Rm, lm, R02)
        beta = np.arctan2(P0 * lp, R02 + aw * Rp) - np.arctan2(P0 * lm, R02 + aw * Rm)
        E0 += m * f[..., None] + n[None] * (sw * beta)[..., None]
        Phi += P0 * f - aw * beta
        d = (x - p) + lm[..., None] * t  # x minus its foot on the edge line
        Le = d * f[..., None] - t * (Rp - Rm)[..., None]
        for k, e in enumerate((e1, e2)):
            sumL[k] += np.sum(e[None] * m, axis=-1)[..., None] * Le
    out = [E0]
    for k, e in enumerate((e1, e2)):
        psi_rho = np.sum((rho - cen[None]) * e[None], axis=-1) / s[None]  # density at the projection
        g = e[None] / s[None, :, None]  # gradient of u (or v) in the plane
        out.append(psi_rho[..., None] * E0 - sumL[k] / s[None, :, None] - g * Phi[..., None])
    return np.stack(out)


def normal_field(X, V, fr=None, pairs=False):
    """n . E_k for densities (1, u, v): only the component along each triangle's
    normal, which is all the Galerkin flux needs. With n.L_e = w f_e and g in the
    plane, n . int psi (x - y)/R^3 dS = psi(rho) sign(w) sum beta - w sum (g.m) f.

    X (M, 3), V (N, 3, 3): returns (3, M, N). With pairs=True, X and V have the
    same length and observer i goes with triangle i only: returns (3, M)."""
    cen, n, e1, e2, s, _ = frames(V) if fr is None else fr
    if pairs:
        x, Vb, nb, cb, e1b, e2b, sb = X, V, n, cen, e1, e2, s
    else:
        x = X[:, None, :]
        Vb, nb, cb, e1b, e2b, sb = V[None], n[None], cen[None], e1[None], e2[None], s[None]
    w = np.sum((x - Vb[..., 0, :]) * nb, axis=-1)
    w = np.where(w == 0, 1e-12 * sb, w)
    rho = x - w[..., None] * nb
    aw = np.abs(w)
    sum_beta = 0.0
    sum_mf = [0.0, 0.0]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p, q = Vb[..., i, :], Vb[..., j, :]
        t = (q - p) / np.linalg.norm(q - p, axis=-1)[..., None]
        m = np.cross(t, nb)
        lm = np.sum((p - x) * t, axis=-1)
        lp = np.sum((q - x) * t, axis=-1)
        P0 = np.sum((p - rho) * m, axis=-1)
        Rm = np.linalg.norm(x - p, axis=-1)
        Rp = np.linalg.norm(x - q, axis=-1)
        R02 = P0**2 + w**2
        f = _log_sum(Rp, lp, R02) - _log_sum(Rm, lm, R02)
        sum_beta = sum_beta + np.arctan2(P0 * lp, R02 + aw * Rp) - np.arctan2(P0 * lm, R02 + aw * Rm)
        for k, e in enumerate((e1b, e2b)):
            sum_mf[k] = sum_mf[k] + np.sum(e * m, axis=-1) * f
    En0 = np.sign(w) * sum_beta
    out = [En0]
    for k, e in enumerate((e1b, e2b)):
        psi_rho = np.sum((rho - cb) * e, axis=-1) / sb
        out.append(psi_rho * En0 - w * sum_mf[k] / sb)
    return np.stack(out)


def normal_field_mm(X, V, fr=None):
    """normal_field for all observers against all triangles (3, M, N), with
    every dot product as one matrix product: x.t, x.m, x.n, x.e and |x - p|^2 =
    |x|^2 - 2 x.p + |p|^2. Only logs, arctangents and roots are per pair. Use it
    for pairs farther apart than a triangle's size; the cancellation in |x - p|^2
    grows as (size / distance)^2 times the rounding."""
    cen, n, e1, e2, s, _ = frames(V) if fr is None else fr
    xx = np.sum(X * X, axis=1)[:, None]
    w = X @ n.T - np.sum(V[:, 0] * n, axis=1)[None]
    w = np.where(w == 0, 1e-12 * s[None], w)
    aw = np.abs(w)
    xp = [X @ V[:, k].T for k in range(3)]
    pp = [np.sum(V[:, k] ** 2, axis=1)[None] for k in range(3)]
    R = [np.sqrt(np.maximum(xx - 2 * xp[k] + pp[k], 0.0)) for k in range(3)]
    sum_beta = 0.0
    sum_mf = [0.0, 0.0]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p, q = V[:, i], V[:, j]
        t = (q - p) / np.linalg.norm(q - p, axis=1)[:, None]
        m = np.cross(t, n)
        xt, xm = X @ t.T, X @ m.T
        lm = np.sum(p * t, axis=1)[None] - xt
        lp = np.sum(q * t, axis=1)[None] - xt
        P0 = np.sum(p * m, axis=1)[None] - xm
        R02 = P0**2 + w**2
        f = _log_sum(R[j], lp, R02) - _log_sum(R[i], lm, R02)
        sum_beta = sum_beta + np.arctan2(P0 * lp, R02 + aw * R[j]) - np.arctan2(P0 * lm, R02 + aw * R[i])
        for k, e in enumerate((e1, e2)):
            sum_mf[k] = sum_mf[k] + np.sum(e * m, axis=1)[None] * f
    En0 = np.sign(w) * sum_beta
    out = [En0]
    for k, e in enumerate((e1, e2)):
        psi_rho = (X @ e.T - np.sum(cen * e, axis=1)[None]) / s[None]
        out.append(psi_rho * En0 - w * sum_mf[k] / s[None])
    return np.stack(out)
