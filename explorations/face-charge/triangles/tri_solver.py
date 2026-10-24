"""Face-charge solve on a closed triangle mesh: linear charge per triangle.

The same scheme as the cuboid panels (p1.py): unknowns are the coefficients of
sigma = c0 + c1 u + c2 v on each triangle; the Galerkin flux of a source
triangle through a test triangle is the integral over the source of the test
triangle's weighted solid angle, -n_i . E_i[psi](y) / 4 pi, by Gauss quadrature,
refined for pairs about as close as they are long. Triangles in one plane give
no flux to each other, as the closed form gives exactly.
"""

import numpy as np
import scipy.linalg
import scipy.sparse

from tri import FOUR_PI, field, frames, normal_field, normal_field_mm

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

    def __init__(self, vertices, faces, position=(0, 0, 0), orientation=None, chi=1.0, J0=None, near_level=1):
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
    def assemble(self, chunk=2_000_000, far_k=0, near_factor=1.0):
        """G (F, 3, F, 3): flux of source j's densities through test i's.

        All pairs with the 7-point rule on the source (matrix-product kernel),
        then the pairs whose centroids are closer than near_factor times the sum
        of their longest edges again, on 4^near_level sub-triangles, in batches.
        Sizing pairs by sqrt(area) instead missed the long thin layer strips: the
        U-core's peak then wandered by 0.1 % between mesh levels."""
        F = self.F
        G = np.zeros((F, 3, F, 3))
        P, W, B = self.quad(far_k)
        nq = P.shape[1]
        step = max(1, chunk // (nq * F))
        for j0 in range(0, F, step):
            js = np.arange(j0, min(j0 + step, F))
            flux = -normal_field_mm(P[js].reshape(-1, 3), self.V, self.fr) / FOUR_PI  # (3, nj*nq, F)
            G[:, :, js, :] = np.einsum("ajqi,jq,jqb->iajb", flux.reshape(3, len(js), nq, F), W[js], B[js])
        I, J = self.near_pairs(near_factor)
        Pk, Wk, Bk = self.quad(self.near_level)
        Q = Pk.shape[1]
        kstep = max(1, 200_000 // Q)
        for k0 in range(0, len(I), kstep):
            i, j = I[k0 : k0 + kstep], J[k0 : k0 + kstep]
            ii = np.repeat(i, Q)
            fr = tuple(a[ii] for a in self.fr)
            En = normal_field(Pk[j].reshape(-1, 3), self.V[ii], fr, pairs=True)  # (3, K*Q)
            G[i, :, j, :] = np.einsum("akq,kq,kqb->kab", -En.reshape(3, len(i), Q) / FOUR_PI, Wk[j], Bk[j])
        G[np.arange(F), :, np.arange(F), :] = 0.0  # own triangle: principal value 0, the jump is in the mass term
        return G

    def near_pairs(self, factor=1.0):
        """Ordered pairs (i, j), i != j, with centroids closer than factor times
        the sum of their longest edges."""
        from scipy.spatial import cKDTree

        cen = self.fr[0]
        size = np.linalg.norm(np.roll(self.V, -1, axis=1) - self.V, axis=-1).max(axis=1)
        tree = cKDTree(cen)
        pairs = np.array(sorted(tree.query_pairs(2 * factor * size.max())), dtype=int).reshape(-1, 2)
        d = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
        pairs = pairs[d < factor * (size[pairs[:, 0]] + size[pairs[:, 1]])]
        return np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])

    def load(self, H_fun, k=0, chunk=5000):
        """int psi_a H.n over each triangle, H_fun on global points -> global H.
        Evaluated in chunks: a field function over all points at once can need
        far more memory than the solve (16 polylines at 70,000 points: 9.4 GB)."""
        P, W, B = self.quad(k)
        X = self.to_global(P.reshape(-1, 3))
        H = np.concatenate([H_fun(X[i : i + chunk]) for i in range(0, len(X), chunk)]) @ self.R
        Hn = np.einsum("fqd,fd->fq", H.reshape(P.shape), self.fr[1])
        return np.einsum("fq,fq,fqa->fa", W, Hn, B).ravel()

    def solve(self, H_fun=None, G=None, load=None):
        """Coefficients. The system is built in G's memory when G is not given."""
        G = self.assemble() if G is None else G.copy()
        F, Nd = self.F, self.Nd
        A = np.zeros((Nd + 1, Nd + 1))
        A[:Nd, :Nd] = -G.reshape(Nd, Nd)
        del G
        blocks = A[:Nd, :Nd].reshape(F, 3, F, 3)
        blocks[np.arange(F), :, np.arange(F), :] += (1 / self.chi + 0.5) * self.mass
        e = np.zeros(Nd)
        e[0::3] = self.fr[5]
        A[:Nd, Nd], A[Nd, :Nd] = e, e / e.sum()
        rhs = np.zeros(Nd + 1)
        if load is not None:
            rhs[:Nd] += load
        elif H_fun is not None:
            rhs[:Nd] += self.load(H_fun)
        if self.J0 is not None:
            m0n = self.fr[1] @ (self.J0 / MU0)
            rhs[:Nd] += (np.einsum("f,fa->fa", m0n, self.mass[:, 0, :])).ravel() / self.chi
        return scipy.linalg.solve(A, rhs, overwrite_a=True, overwrite_b=True, check_finite=False)[:Nd]

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


class MacroBody(TriBody):
    """A TriBody whose charge is linear across groups of triangles (macro panels):
    the Galerkin projection of the triangle basis onto one density a + b xi + c eta
    per group. A band of narrow facets on a curved face then carries 3 unknowns
    instead of 3 per facet, while the triangles keep the exact geometry. xi and eta
    are the band's own coordinates, arc length along its rim and distance from it,
    where bands gives them (refine_rings(groups=True)): a fixed direction in the
    mean tangent plane turns away from the rim's normal along a curved band, about
    30 deg on a 5 mm cylinder's cap, and missed the steep rise of the charge toward
    the rim (2e-3 1 mm from it). Elsewhere, the mean tangent plane. The operator
    is projected chunk by chunk as it is assembled, so the triangles' matrix is
    never stored."""

    def __init__(self, vertices, faces, groups, bands=None, **kw):
        V = np.asarray(vertices, float)[np.asarray(faces)]
        order = np.argsort(np.asarray(groups), kind="stable")  # a group's triangles together
        super().__init__(V[order].reshape(-1, 3), np.arange(3 * len(V)).reshape(-1, 3), **kw)
        _, self.g = np.unique(np.asarray(groups)[order], return_inverse=True)
        self.Ng = int(self.g.max()) + 1
        self.Nm = 3 * self.Ng
        cen, n, e1, e2, s, area = self.fr
        A = np.bincount(self.g, area, self.Ng)
        o = np.stack([np.bincount(self.g, area * cen[:, d], self.Ng) for d in range(3)], 1) / A[:, None]
        ng = np.stack([np.bincount(self.g, area * n[:, d], self.Ng) for d in range(3)], 1)
        ng /= np.linalg.norm(ng, axis=1)[:, None]
        first = np.searchsorted(self.g, np.arange(self.Ng))
        t1 = e1[first] - np.einsum("gd,gd->g", e1[first], ng)[:, None] * ng
        weak = np.linalg.norm(t1, axis=1) < 0.1
        t1[weak] = (e2[first] - np.einsum("gd,gd->g", e2[first], ng)[:, None] * ng)[weak]
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t2 = np.cross(ng, t1)
        sg = np.sqrt(A)
        gi = self.g
        # per triangle: directions of xi and eta, and their values at its centroid
        dir1, dir2 = t1[gi], t2[gi]
        x1 = np.einsum("fd,fd->f", cen - o[gi], dir1)
        x2 = np.einsum("fd,fd->f", cen - o[gi], dir2)
        if bands is not None:
            bd = np.asarray(bands, float)[order]
            inband = np.isfinite(bd[:, 0])
            whole = np.bincount(gi, ~inband, self.Ng) == 0  # every triangle of the group in a band
            use = whole[gi]
            a, tt, nu, s0, Lr = bd[:, 0:3], bd[:, 3:6], bd[:, 6:9], bd[:, 9], bd[:, 10]
            sc = s0 + np.einsum("fd,fd->f", cen - a, tt)
            ref = sc[first][gi]  # unwrap closed rims about the group's first triangle
            sc = np.where(use & (Lr > 0), sc - Lr * np.round((sc - ref) / np.where(Lr > 0, Lr, 1)), sc)
            dc = np.einsum("fd,fd->f", cen - a, nu)
            sm = np.bincount(gi, np.where(use, area * sc, 0), self.Ng) / A
            dm = np.bincount(gi, np.where(use, area * dc, 0), self.Ng) / A
            dir1, dir2 = np.where(use[:, None], tt, dir1), np.where(use[:, None], nu, dir2)
            x1, x2 = np.where(use, sc - sm[gi], x1), np.where(use, dc - dm[gi], x2)
        P = np.zeros((self.F, 3, 3))  # triangle (c0, c1, c2) from group (a, b, c)
        P[:, 0, 0] = 1.0
        for col, t, x in ((1, dir1, x1), (2, dir2, x2)):
            P[:, 0, col] = x / sg[gi]
            P[:, 1, col] = s * np.einsum("fd,fd->f", e1, t) / sg[gi]
            P[:, 2, col] = s * np.einsum("fd,fd->f", e2, t) / sg[gi]
        self.P = P
        rows = (3 * np.arange(self.F)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
        cols = (3 * gi[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
        self.Ps = scipy.sparse.csr_matrix((P.ravel(), (rows.ravel(), cols.ravel())), shape=(self.Nd, self.Nm))

    def fine(self, c):
        """Triangle coefficients from group coefficients."""
        return self.Ps @ c

    def assemble(self, chunk=2_000_000, far_k=0, near_factor=1.0):
        """G (Nm, Nm) in the group basis: TriBody.assemble's two passes, each
        projected as it is computed; the far pass's values for near and own pairs
        are dropped before projecting, so nothing is subtracted."""
        F, Nm = self.F, self.Nm
        Gm = np.zeros((Nm, Nm))
        PsT = self.Ps.T.tocsr()
        I, J = self.near_pairs(near_factor)
        by_j = np.argsort(J, kind="stable")
        I, J = I[by_j], J[by_j]
        P, W, B = self.quad(far_k)
        nq = P.shape[1]
        step = max(1, chunk // (nq * F))
        for j0 in range(0, F, step):
            j1 = min(j0 + step, F)
            js = np.arange(j0, j1)
            flux = -normal_field_mm(P[js].reshape(-1, 3), self.V, self.fr) / FOUR_PI
            Gc = np.einsum("ajqi,jq,jqb->iajb", flux.reshape(3, len(js), nq, F), W[js], B[js])
            k0, k1 = np.searchsorted(J, [j0, j1])
            Gc[I[k0:k1], :, J[k0:k1] - j0, :] = 0.0
            Gc[js, :, js - j0, :] = 0.0
            c0, c1 = 3 * self.g[j0], 3 * self.g[j1 - 1] + 3
            X = PsT @ Gc.reshape(3 * F, 3 * len(js))
            Gm[:, c0:c1] += X @ self.Ps[3 * j0 : 3 * j1, c0:c1].toarray()
        Pk, Wk, Bk = self.quad(self.near_level)
        Q = Pk.shape[1]
        kstep = max(1, 200_000 // Q)
        a3 = np.arange(3)
        for k0 in range(0, len(I), kstep):
            i, j = I[k0 : k0 + kstep], J[k0 : k0 + kstep]
            ii = np.repeat(i, Q)
            fr = tuple(a[ii] for a in self.fr)
            En = normal_field(Pk[j].reshape(-1, 3), self.V[ii], fr, pairs=True)
            D = np.einsum("akq,kq,kqb->kab", -En.reshape(3, len(i), Q) / FOUR_PI, Wk[j], Bk[j])
            D = np.einsum("kra,krs,ksb->kab", self.P[i], D, self.P[j])
            idx = (3 * self.g[i][:, None, None] + a3[None, :, None]) * Nm + 3 * self.g[j][:, None, None] + a3[None, None, :]
            np.add.at(Gm.reshape(-1), idx.ravel(), D.ravel())
        return Gm

    def solve(self, H_fun=None, G=None, load=None):
        """Group coefficients; TriBody.fine turns them into triangle ones."""
        Gm = self.assemble() if G is None else G.copy()
        Nm = self.Nm
        A = np.zeros((Nm + 1, Nm + 1))
        A[:Nm, :Nm] = -Gm
        del Gm
        Mf = np.einsum("fra,frs,fsb->fab", self.P, (1 / self.chi + 0.5) * self.mass, self.P)
        idx = (3 * self.g[:, None, None] + np.arange(3)[None, :, None]) * (Nm + 1) + 3 * self.g[:, None, None] + np.arange(3)[None, None, :]
        np.add.at(A.reshape(-1), idx.ravel(), Mf.ravel())  # A's rows are Nm + 1 long
        e = np.zeros(self.Nd)
        e[0::3] = self.fr[5]
        em = self.Ps.T @ e
        A[:Nm, Nm], A[Nm, :Nm] = em, em / e.sum()
        rhs = np.zeros(Nm + 1)
        if load is not None:
            rhs[:Nm] += self.Ps.T @ load
        elif H_fun is not None:
            rhs[:Nm] += self.Ps.T @ self.load(H_fun)
        if self.J0 is not None:
            m0n = self.fr[1] @ (self.J0 / MU0)
            rhs[:Nm] += self.Ps.T @ (np.einsum("f,fa->fa", m0n, self.mass[:, 0, :])).ravel() / self.chi
        return scipy.linalg.solve(A, rhs, overwrite_a=True, overwrite_b=True, check_finite=False)[:Nm]

    def B(self, X, c, chunk=4_000_000):
        return super().B(X, self.fine(c), chunk)

    def moment(self, c):
        return super().moment(self.fine(c))
