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
