"""Field of a tetrahedron whose polarization varies linearly inside it.

J(x') = a + G xi, xi = (x' - c) / l (c centroid, l scale). Unknowns u = [a (3), G row-major (9)].
mu0 H = 1/(4 pi) [ sum_f int sigma_f r/R^3 dS - div(J) sum_f n_f P_f ],  P_f = int_f 1/R dS,  r = x - x'.
Per face: P_f and E_f = int r/R^3 dS in closed form (Wilton et al.); the linear charge reduces, by the
surface divergence theorem, to edge integrals L_e = int_e r/R dl = d Lambda_e - t (R+ - R-).
"""

import numpy as np

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FACES = [(o, [v for v in range(4) if v != o]) for o in range(4)]  # (opposite vertex, face vertices)


def _safe_sum(l, R, R02, xp):
    """R + l, stable for l < 0 where it equals R0^2 / (R - l)."""
    pos = l >= 0
    return xp.where(pos, R + l, R02 / xp.where(pos, 1.0, R - l))


def tensor(X, V, scale, linear=True, xp=np):
    """mu0 H (T) at X (n, 3) from tetrahedra V (n, 4, 3), per unit coefficient: (n, 3, 12) or (n, 3, 3)."""
    dot = lambda a, b: xp.sum(a * b, axis=-1)  # noqa: E731
    unit = lambda a: a / xp.sqrt(dot(a, a))[..., None]  # noqa: E731
    c = xp.mean(V, axis=1)
    D = V - X[:, None, :]  # vertex - observer
    Rv = xp.sqrt(dot(D, D))  # (n, 4)
    edge = {}
    for i, j in EDGES:
        t = unit(V[:, j] - V[:, i])
        li, lj = dot(D[:, i], t), dot(D[:, j], t)
        R02 = Rv[:, i] ** 2 - li**2
        Lam = xp.log(_safe_sum(lj, Rv[:, j], R02, xp)) - xp.log(_safe_sum(li, Rv[:, i], R02, xp))
        d = -D[:, i] + li[:, None] * t  # observer minus its foot on the edge line
        Le = d * Lam[:, None] - t * (Rv[:, j] - Rv[:, i])[:, None]
        edge[(i, j)] = (t, li, lj, R02, Lam, Le)
    P, E, nh, w, Ls = [], [], [], [], []
    for o, (a, b, cc) in FACES:
        n_ = unit(xp.cross(V[:, b] - V[:, a], V[:, cc] - V[:, a]))
        n_ = n_ * xp.sign(dot(n_, V[:, a] - V[:, o]))[:, None]  # outward
        wf = -dot(D[:, a], n_)  # (x - a).n
        Pf, Ef, Lf = 0.0, 0.0, []
        for i, j in ((a, b), (b, cc), (a, cc)):
            third = ({a, b, cc} - {i, j}).pop()
            t, li, lj, R02, Lam, Le = edge[(i, j)]
            m = unit((V[:, i] - V[:, third]) - dot(V[:, i] - V[:, third], t)[:, None] * t)  # in-plane, outward
            P0 = dot(D[:, i], m)
            aw = xp.abs(wf)
            beta = xp.arctan(P0 * lj / (R02 + aw * Rv[:, j])) - xp.arctan(P0 * li / (R02 + aw * Rv[:, i]))
            Pf = Pf + P0 * Lam - aw * beta
            Ef = Ef + m * Lam[:, None] + n_ * (xp.sign(wf) * beta)[:, None]
            Lf.append((m, Le))
        P.append(Pf), E.append(Ef), nh.append(n_), w.append(wf), Ls.append(Lf)
    cols = []
    for mm in range(3):  # uniform columns: sigma = n_m
        cols.append(sum(nh[f][:, mm][:, None] * E[f] for f in range(4)))
    if linear:
        Vol = sum(nh[f] * P[f][:, None] for f in range(4))  # int_V r/R^3 dV
        for mm in range(3):
            for k in range(3):
                ek = xp.zeros(3, dtype=X.dtype)
                ek = xp.where(xp.arange(3) == k, 1.0, ek)
                acc = 0.0
                for f in range(4):
                    nk = nh[f][:, k]
                    ekt = ek[None, :] - nk[:, None] * nh[f]
                    Irk = sum(dot(ek[None, :], m)[:, None] * Le for m, Le in Ls[f]) + ekt * P[f][:, None] + (nk * w[f])[:, None] * E[f]
                    xk = (X[:, k] - c[:, k])[:, None]
                    acc = acc + nh[f][:, mm][:, None] * (xk * E[f] - Irk)
                if mm == k:
                    acc = acc - Vol
                cols.append(acc / scale[:, None])
    return xp.stack(cols, axis=-1) / (4 * np.pi)
