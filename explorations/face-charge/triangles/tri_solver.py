"""Face-charge solve on a closed triangle mesh: linear charge per triangle.

The same scheme as the cuboid panels (p1.py): unknowns are the coefficients of
sigma = c0 + c1 u + c2 v on each triangle; the Galerkin flux of a source
triangle through a test triangle is the integral over the source of the test
triangle's weighted solid angle, -n_i . E_i[psi](y) / 4 pi, by Gauss quadrature,
refined for pairs closer than twice their size. Triangles in one plane give no
flux to each other, as the closed form gives exactly.
"""

import numpy as np

from tri import FOUR_PI, field, frames

MU0 = 4e-7 * np.pi

# 7-point degree-5 rule (Dunavant) and the 3-point degree-2 rule (edge midpoints)
_a1, _b1 = 0.059715871789770, 0.470142064105115
_a2, _b2 = 0.797426985353087, 0.101286507323456
BARY7 = np.array([[1 / 3] * 3, [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
                  [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2]])
W7 = np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3])
BARY3 = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
W3 = np.full(3, 1 / 3)


def _subdivide_bary(k):
    """Barycentric corners of the 4^k sub-triangles of the reference triangle."""
    T = np.eye(3)[None]
    for _ in range(k):
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack(s, 1) for s in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return T


def rule(k=0):
    """Barycentric points and weights (summing to 1) of the 7-point rule on 4^k sub-triangles."""
    if k == 0:
        return BARY7, W7
    sub = _subdivide_bary(k)
    pts = np.einsum("qk,tkl->tql", BARY7, sub).reshape(-1, 3)
    return pts, np.tile(W7, len(sub)) / len(sub)


class TriBody:
    """A closed triangle mesh (outward faces) with susceptibility chi and
    remanence J0 (T, body frame), at position, turned by orientation."""

    def __init__(self, vertices, faces, position=(0, 0, 0), orientation=None, chi=1.0, J0=None, near_level=2):
        self.V = np.asarray(vertices, float)[np.asarray(faces)]
        self.position = np.asarray(position, float)
        self.R = np.eye(3) if orientation is None else orientation.as_matrix()
        self.chi = float(chi)
        self.J0 = None if J0 is None else np.asarray(J0, float)
        self.fr = frames(self.V)
        self.F = len(self.V)
        self.Nd = 3 * self.F
        self.near_level = near_level
        cen, n, e1, e2, s, area = self.fr
        # mass blocks and first moments, exact with the degree-2 rule
        p3 = np.einsum("qk,fkd->fqd", BARY3, self.V)
        phi3 = self.basis(p3)  # (F, 3q, 3)
        self.mass = np.einsum("fq,fqa,fqb->fab", area[:, None] * W3[None], phi3, phi3)
        self.first = np.einsum("fq,fqa,fqd->fad", area[:, None] * W3[None], phi3, p3)  # int phi_a y dS

    def basis(self, P):
        """(1, u, v) at points P (F, q, 3) of each triangle."""
        cen, n, e1, e2, s, _ = self.fr
        d = P - cen[:, None]
        return np.stack([np.ones(P.shape[:2]), (d @ e1[..., None])[..., 0] / s[:, None],
                         np.einsum("fqd,fd->fq", d, e2) / s[:, None]], axis=-1)

    def quad(self, k=0):
        b, w = rule(k)
        P = np.einsum("qk,fkd->fqd", b, self.V)
        return P, self.fr[5][:, None] * w[None], self.basis(P)

    def to_local(self, x):
        return (np.asarray(x) - self.position) @ self.R

    def to_global(self, x):
        return np.asarray(x) @ self.R.T + self.position

    # ── operator ────────────────────────────────────────────────────────────
    def assemble(self, chunk=2_000_000):
        """G (F, 3, F, 3): flux of source j's densities through test i's."""
        F = self.F
        n = self.fr[1]
        G = np.zeros((F, 3, F, 3))
        P, W, B = self.quad(0)
        nq = P.shape[1]
        step = max(1, chunk // (3 * nq * F))
        for j0 in range(0, F, step):
            js = np.arange(j0, min(j0 + step, F))
            E = field(P[js].reshape(-1, 3), self.V, self.fr)  # (3, nj*nq, F, 3)
            flux = -np.einsum("aqid,id->aqi", E, n) / FOUR_PI  # (3 test densities, points, test i)
            flux = flux.reshape(3, len(js), nq, F)
            G[:, :, js, :] = np.einsum("ajqi,jq,jqb->iajb", flux, W[js], B[js])
        # pairs closer than twice their size: again with sub-divided source triangles
        cen, size = self.fr[0], np.sqrt(self.fr[5])
        Pk, Wk, Bk = self.quad(self.near_level)
        for i in range(F):
            near = np.flatnonzero(np.linalg.norm(cen - cen[i], axis=1) < 2 * (size + size[i]))
            near = near[near != i]
            if not len(near):
                continue
            E = field(Pk[near].reshape(-1, 3), self.V[i : i + 1], tuple(a[i : i + 1] for a in self.fr))
            flux = (-np.einsum("aqd,d->aq", E[:, :, 0], n[i]) / FOUR_PI).reshape(3, len(near), -1)
            G[i, :, near, :] = np.einsum("ajq,jq,jqb->jab", flux, Wk[near], Bk[near])
        G[np.arange(F), :, np.arange(F), :] = 0.0  # own triangle: principal value 0, the jump is in the mass term
        return G

    def load(self, H_fun):
        """int psi_a H.n over each triangle, H_fun on global points -> global H."""
        P, W, B = self.quad(1)
        H = H_fun(self.to_global(P.reshape(-1, 3))) @ self.R  # into the body frame
        Hn = np.einsum("fqd,fd->fq", H.reshape(P.shape), self.fr[1])
        return np.einsum("fq,fq,fqa->fa", W, Hn, B).ravel()

    def solve(self, H_fun=None, G=None):
        G = self.assemble() if G is None else G
        F, Nd = self.F, self.Nd
        A = np.zeros((Nd + 1, Nd + 1))
        A[:Nd, :Nd] = -G.reshape(Nd, Nd)
        mass = np.zeros((F, 3, F, 3))
        mass[np.arange(F), :, np.arange(F), :] = self.mass
        A[:Nd, :Nd] += (1 / self.chi + 0.5) * mass.reshape(Nd, Nd)
        e = np.zeros(Nd)
        e[0::3] = self.fr[5]
        A[:Nd, Nd], A[Nd, :Nd] = e, e / e.sum()
        rhs = np.zeros(Nd + 1)
        if H_fun is not None:
            rhs[:Nd] += self.load(H_fun)
        if self.J0 is not None:
            m0n = self.fr[1] @ (self.J0 / MU0)
            rhs[:Nd] += (np.einsum("f,fa->fa", m0n, self.mass[:, 0, :])).ravel() / self.chi
        return np.linalg.solve(A, rhs)[:Nd]

    def B(self, X, c, chunk=4_000_000):
        """B (T) at observers X (global) outside the body."""
        Xl = self.to_local(np.atleast_2d(X))
        cc = c.reshape(self.F, 3)
        out = np.zeros((len(Xl), 3))
        step = max(1, chunk // (9 * self.F))
        for k in range(0, len(Xl), step):
            E = field(Xl[k : k + step], self.V, self.fr)
            out[k : k + step] = np.einsum("amfd,fa->md", E, cc) / FOUR_PI
        return MU0 * out @ self.R.T

    def moment(self, c):
        return np.einsum("fa,fad->d", c.reshape(self.F, 3), self.first) @ self.R.T
