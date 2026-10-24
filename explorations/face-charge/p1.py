"""Linear-charge (P1, discontinuous) Galerkin face-charge solver for a cuboid.

Each panel carries sigma = c0 + c1 s + c2 t, with (s, t) in [-1, 1]^2 its
local coordinates. All panel fields are closed forms (corner sums); the
Galerkin flux between panels is the integral over the source panel of the
(bounded) weighted solid angle of the test panel, by Gauss quadrature.
"""

import numpy as np

from bem import FOUR_PI, Panels, _ln_y_plus_r  # noqa: F401


def lin_rect(P, Q, W, p1, p2, q1, q2, want_inplane=True):
    """Field per unit density of a rectangle with densities 1, s, t.

    s = (p - pc)/hp, t = (q - qc)/hq. Returns array (3 densities, 3 comps
    p q w, ...broadcast shape). With want_inplane=False only the w comps."""
    pc, hp = 0.5 * (p1 + p2), 0.5 * (p2 - p1)
    qc, hq = 0.5 * (q1 + q2), 0.5 * (q2 - q1)
    sW = np.where(W < 0, -1.0, 1.0)
    aW = np.abs(W)
    W2 = W * W
    Sx = Sy = Sw = Sxx = Syy = Sxy = 0.0
    for ix, x in enumerate((P - p1, P - p2)):
        for iy, y in enumerate((Q - q1, Q - q2)):
            s = 1.0 if ix == iy else -1.0
            r = np.sqrt(x * x + y * y + W2)
            at = np.arctan2(x * y, aW * r)
            Sw = Sw + s * sW * at
            lyr = _ln_y_plus_r(x * x + W2, y, r)
            lxr = _ln_y_plus_r(y * y + W2, x, r)
            Sx = Sx - s * lyr  # int X/R^3
            Sy = Sy - s * lxr  # int Y/R^3
            if want_inplane:
                Sxx = Sxx + s * (y * lxr - aW * at)  # int X^2/R^3
                Syy = Syy + s * (x * lyr - aW * at)  # int Y^2/R^3
                Sxy = Sxy - s * r  # int XY/R^3
    # density s = (P - pc - X)/hp, t = (Q - qc - Y)/hq ; field (X, Y, W)/R^3
    a, b = (P - pc) / hp, (Q - qc) / hq
    w0 = Sw
    w1 = a * Sw - W * Sx / hp
    w2 = b * Sw - W * Sy / hq
    if not want_inplane:
        return np.stack([w0, w1, w2]) / FOUR_PI
    out = np.stack(
        [
            np.stack([Sx, Sy, Sw]),
            np.stack([a * Sx - Sxx / hp, a * Sy - Sxy / hp, w1]),
            np.stack([b * Sx - Sxy / hq, b * Sy - Syy / hq, w2]),
        ]
    )
    return out / FOUR_PI


class P1:
    def __init__(self, pan: Panels, nq=4):
        self.pan = pan
        self.nq = nq
        p = pan
        self.hp = 0.5 * (p.p2 - p.p1)
        self.hq = 0.5 * (p.q2 - p.q1)
        self.pc = 0.5 * (p.p1 + p.p2)
        self.qc = 0.5 * (p.q1 + p.q2)
        N = p.N
        self.mass = np.concatenate([p.area[:, None] * np.array([1.0, 1 / 3, 1 / 3])], 1).ravel()
        self.Nd = 3 * N

    def gauss_points(self, nq):
        """Gauss points of every panel: pts (nq*nq, N, 3), wts (nq*nq, N), s, t."""
        p = self.pan
        g, w = np.polynomial.legendre.leggauss(nq)
        S, T = np.meshgrid(g, g, indexing="ij")
        Wt = np.outer(w, w).ravel() * 0.25
        S, T = S.ravel(), T.ravel()
        pts = np.repeat(p.center[None], len(S), 0)
        idx = np.arange(p.N)
        pts[:, idx, p.i] = self.pc + S[:, None] * self.hp
        pts[:, idx, p.j] = self.qc + T[:, None] * self.hq
        return pts, Wt[:, None] * p.area[None, :], S, T

    def assemble(self):
        """Galerkin matrix G (3N x 3N) of K' and mass M (diag)."""
        p = self.pan
        N = p.N
        G = np.zeros((N, 3, N, 3))
        pts, wts, S, T = self.gauss_points(self.nq)  # source quadrature
        for fa in range(6):
            ta = np.flatnonzero(p.face == fa)
            ka, ia, ja, sa = p.k[ta[0]], p.i[ta[0]], p.j[ta[0]], p.sign[ta[0]]
            P1_, P2_ = p.p1[ta][:, None], p.p2[ta][:, None]
            Q1_, Q2_ = p.q1[ta][:, None], p.q2[ta][:, None]
            W0 = p.w0[ta][:, None]
            for fb in range(6):
                if fb == fa:
                    continue
                sb = np.flatnonzero(p.face == fb)
                acc = np.zeros((3, len(ta), len(sb), 3))
                for g in range(len(S)):
                    y = pts[g, sb]
                    w3 = lin_rect(
                        y[None, :, ia], y[None, :, ja], y[None, :, ka] - W0,
                        P1_, P2_, Q1_, Q2_, want_inplane=False,
                    )  # (3 test dens, ta, sb)
                    phi = np.array([1.0, S[g], T[g]])
                    acc += (w3[..., None] * (wts[g, sb][None, None, :, None] * phi))
                G[np.ix_(ta, [0, 1, 2], sb, [0, 1, 2])] = -sa * acc.transpose(1, 0, 2, 3)
        return G.reshape(3 * N, 3 * N)

    def rhs(self, Hfun, nq=4):
        """Galerkin load: int psi_alpha H_ext.n over each panel."""
        p = self.pan
        pts, wts, S, T = self.gauss_points(nq)
        H = Hfun(pts.reshape(-1, 3)).reshape(len(S), p.N, 3)
        Hn = np.einsum("gnd,nd->gn", H, p.normal)
        F = np.stack([(wts * Hn).sum(0), (wts * Hn * S[:, None]).sum(0), (wts * Hn * T[:, None]).sum(0)], 1)
        return F.ravel()

    def rhs_const_normal(self, m0n):
        """Load of a panelwise-constant normal value (remanence M0.n)."""
        F = np.zeros((self.pan.N, 3))
        F[:, 0] = m0n * self.pan.area
        return F.ravel()

    def system(self, chi, G):
        Nd = self.Nd
        A = np.empty((Nd + 1, Nd + 1))
        A[:Nd, :Nd] = -G
        A[np.arange(Nd), np.arange(Nd)] += (1 / chi + 0.5) * self.mass
        e = np.zeros(Nd)
        e[0::3] = self.pan.area / self.pan.area.sum()
        A[:Nd, Nd] = e * self.pan.area.sum()  # border on the mean mode
        A[Nd, :Nd] = e
        A[Nd, Nd] = 0
        return A

    def solve(self, chi, G, F, F0=None):
        rhs = F if F0 is None else F + F0 / chi
        rhs = np.concatenate([rhs, [0.0]])
        return np.linalg.solve(self.system(chi, G), rhs)[: self.Nd]

    def moment(self, c):
        p = self.pan
        c = c.reshape(p.N, 3)
        m = (c[:, 0] * p.area) @ p.center
        idx = np.arange(p.N)
        add = np.zeros((p.N, 3))
        add[idx, p.i] = c[:, 1] * p.area * self.hp / 3
        add[idx, p.j] = c[:, 2] * p.area * self.hq / 3
        return m + add.sum(0)

    def field(self, obs, c, chunk=400_000):
        """H at observers (M,3) from coefficients c (3N,)."""
        p = self.pan
        c = c.reshape(p.N, 3)
        obs = np.atleast_2d(obs)
        out = np.zeros((len(obs), 3))
        idx = np.arange(p.N)
        step = max(1, chunk // p.N)
        for a in range(0, len(obs), step):
            o = obs[a : a + step]
            F = lin_rect(o[:, p.i], o[:, p.j], o[:, p.k] - p.w0, p.p1, p.p2, p.q1, p.q2)
            # F: (3 dens, 3 comps, m, N) -> weight by c
            Fc = np.einsum("dcmn,nd->cmn", F, c)
            H = np.zeros((len(o), 3))
            for comp_local in range(3):
                axes = [p.i, p.j, p.k][comp_local]
                for ax in range(3):
                    sel = axes == ax
                    H[:, ax] += Fc[comp_local][:, sel].sum(1)
            out[a : a + step] = H
        return out
