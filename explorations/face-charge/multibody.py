"""Several linear cuboids, each placed and rotated freely, coupled through
their face charges.

Body a's equation on its faces, tested with its P1 functions:

    (1/chi_a + 1/2) M_a c_a - G_aa c_a - sum_b G_ab c_b = F_a + F0_a / chi_a

G_ab is the flux through a's test panels of b's charges, computed like the
self block: the closed-form weighted solid angle of the test panel (in a's
frame) at b's Gauss points (mapped into a's frame). Where a face of b lies in
a face plane of a (contact), the coplanar part is exact: no principal value,
and the one-sided limit from a's inside gives -1/2 of the overlap integral of
the two charges. The overlap of a test rectangle with b's panel (a rectangle,
turned in the plane when b is) is clipped as a polygon and integrated exactly,
so the two meshes need not match.
"""

import itertools

import numpy as np

from bem import Panels
from p1 import P1, lin_rect

MU0 = 4e-7 * np.pi


def multi_layer_edges(a, lines, ell, m, g, tol=1e-9):
    """Edges on [-a, a], graded (m power-g panels in a layer of length ell)
    toward both ends and toward every interior line, uniform in between."""
    pts = np.unique(np.concatenate([[-a, a], [x for x in lines if -a + tol * a < x < a - tol * a]]))
    t = np.linspace(0, 1, m + 1)
    out = []
    for x0, x1 in zip(pts[:-1], pts[1:]):
        L = x1 - x0
        lay = min(ell, L / 2) * t**g
        hmax = lay[-1] - lay[-2]
        gap = L - 2 * lay[-1]
        k = int(np.ceil(gap / hmax - 1e-9)) if gap > tol * a else 0
        mid = np.linspace(x0 + lay[-1], x1 - lay[-1], k + 1) if k else [x0 + lay[-1]]
        out.extend([x0 + lay, mid, x1 - lay[::-1]])
    return np.unique(np.round(np.concatenate(out) / a, 12)) * a


class Body:
    """A cuboid with susceptibility chi and remanence J0 (T, in the body's
    frame, like magpylib's polarization), at `position`, turned by
    `orientation` (a scipy Rotation from the body frame to the global one)."""

    def __init__(self, dimension, position=(0, 0, 0), orientation=None, chi=1.0, J0=None,
                 m=4, g=2.5, kappa=2.0, lines=None, nq=4):
        self.dimension = np.asarray(dimension, float)
        self.half = self.dimension / 2
        self.position = np.asarray(position, float)
        self.R = np.eye(3) if orientation is None else orientation.as_matrix()
        self.chi = float(chi)
        self.J0 = None if J0 is None else np.asarray(J0, float)
        self.mesh_args = (m, g, kappa)
        self.lines = [[] for _ in range(3)] if lines is None else lines
        self.extra = [[] for _ in range(3)]  # edges added as they are, per axis: bisections
        self.nq = nq
        self.G = None

    def build(self):
        m, g, kappa = self.mesh_args
        c = np.min(self.half)
        edges = [np.unique(np.round(np.concatenate([multi_layer_edges(h, self.lines[k], kappa * c, m, g),
                                                    self.extra[k]]) / h, 12)) * h for k, h in enumerate(self.half)]
        self.pan = Panels(self.half, 0, axis_edges=edges)
        self.s = P1(self.pan, nq=self.nq)
        self.G = self.s.assemble()
        return self

    def bisect(self, panels):
        """Split these panels in two along both of their in-plane axes: in the
        tensor mesh, the whole strips through them. Rebuild to apply."""
        p = self.pan
        for n in panels:
            self.extra[p.i[n]].append(0.5 * (p.p1[n] + p.p2[n]))
            self.extra[p.j[n]].append(0.5 * (p.q1[n] + p.q2[n]))

    @property
    def Nd(self):
        return self.s.Nd

    def to_local(self, x):
        return (np.asarray(x) - self.position) @ self.R

    def to_global(self, x):
        return np.asarray(x) @ self.R.T + self.position


def _relative(a, b):
    """b's frame seen from a's: rotation (b local -> a local) and b's centre."""
    return a.R.T @ b.R, (b.position - a.position) @ a.R


def add_contact_lines(bodies, reach=0.0):
    """Grade each body's mesh toward the faces of every other body that touches
    it or comes within `reach` (the footprint edges of a contact). Exact when
    the two are turned by multiples of 90 degrees against each other; otherwise
    the lines bound the footprint (its box in this body's frame)."""
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    for a in bodies:
        for b in bodies:
            if a is b:
                continue
            Rab, cb = _relative(a, b)
            corners = cb + (signs * b.half) @ Rab.T
            lo, hi = corners.min(0), corners.max(0)
            gap = np.maximum(0, np.maximum(lo - a.half, -a.half - hi))
            if np.linalg.norm(gap) > reach * np.min(a.half) + 1e-12:
                continue
            for k in range(3):
                a.lines[k] = sorted(set(a.lines[k]) | {lo[k], hi[k]})


def _clip(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman: a convex polygon (list of (x, y)) clipped to a box."""
    for axis, bound, keep_above in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        out = []
        n = len(poly)
        for k in range(n):
            p, q = poly[k], poly[(k + 1) % n]
            pin = p[axis] >= bound if keep_above else p[axis] <= bound
            qin = q[axis] >= bound if keep_above else q[axis] <= bound
            if pin:
                out.append(p)
            if pin != qin:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if not poly:
            break
    return poly


def _overlap_rule(poly):
    """Points and weights exact for quadratics over a convex polygon:
    edge midpoints of a triangle fan, weight area / 3 each."""
    pts, wts = [], []
    p0 = np.array(poly[0])
    for k in range(1, len(poly) - 1):
        p1, p2 = np.array(poly[k]), np.array(poly[k + 1])
        area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        if area == 0:
            continue
        pts += [(p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2]
        wts += [area / 3] * 3
    return np.array(pts).reshape(-1, 2), np.array(wts)


def cross_block(a, b, tol=1e-9):
    """G_ab (a.Nd x b.Nd): flux through a's test functions of b's charges."""
    pa, pb = a.pan, b.pan
    Rab, cb = _relative(a, b)
    pts, wts, S, T = b.s.gauss_points(b.nq)
    ng = len(S)
    y = pts @ Rab.T + cb  # b's Gauss points in a's frame, (g, Nb, 3)
    n_b = pb.normal @ Rab.T  # b's panel normals in a's frame
    c_b = pb.center @ Rab.T + cb
    phi = np.stack([np.ones(ng), S, T], 1)  # (g, 3)
    G = np.zeros((pa.N, 3, pb.N, 3))
    scale = np.max(a.half)
    for fa in range(6):
        ta = np.flatnonzero(pa.face == fa)
        ka, ia, ja, sa = pa.k[ta[0]], pa.i[ta[0]], pa.j[ta[0]], pa.sign[ta[0]]
        w0 = pa.w0[ta[0]]
        # source panels of b lying in this face plane: handled exactly below
        cop = (np.abs(n_b[:, ka]) > 1 - 1e-9) & (np.abs(c_b[:, ka] - w0) < tol * scale)
        src = np.flatnonzero(~cop)
        if len(src):
            yy = y[:, src]
            # test panels in chunks: about 2e6 kernel points per call (~0.5 GB)
            step = max(1, int(2e6 // (ng * len(src))))
            for c0 in range(0, len(ta), step):
                tc = ta[c0 : c0 + step]
                w3 = lin_rect(
                    yy[None, ..., ia], yy[None, ..., ja], yy[None, ..., ka] - w0,
                    pa.p1[tc][:, None, None], pa.p2[tc][:, None, None],
                    pa.q1[tc][:, None, None], pa.q2[tc][:, None, None],
                    want_inplane=False,
                )  # (3 test dens, tc, g, ns)
                blk = np.einsum("atgn,gn,gb->tanb", w3, wts[:, src], phi)
                G[np.ix_(tc, [0, 1, 2], src, [0, 1, 2])] = -sa * blk
        for j in np.flatnonzero(cop):
            # coplanar panel j of b: -1/2 of the overlap integral of psi_alpha phi_beta
            i_b, j_b, k_b = pb.i[j], pb.j[j], pb.k[j]
            corners = np.zeros((4, 3))
            for r, (u, v) in enumerate(((pb.p1[j], pb.q1[j]), (pb.p2[j], pb.q1[j]),
                                        (pb.p2[j], pb.q2[j]), (pb.p1[j], pb.q2[j]))):
                corners[r, [i_b, j_b, k_b]] = u, v, pb.w0[j]
            ca = corners @ Rab.T + cb  # in a's frame
            quad = [(c[ia], c[ja]) for c in ca]
            lo, hi = ca.min(0), ca.max(0)
            hit = np.flatnonzero((pa.p2[ta] > lo[ia]) & (pa.p1[ta] < hi[ia])
                                 & (pa.q2[ta] > lo[ja]) & (pa.q1[ta] < hi[ja]))
            pcb, hpb = 0.5 * (pb.p1[j] + pb.p2[j]), 0.5 * (pb.p2[j] - pb.p1[j])
            qcb, hqb = 0.5 * (pb.q1[j] + pb.q2[j]), 0.5 * (pb.q2[j] - pb.q1[j])
            for t_idx in hit:
                i = ta[t_idx]
                poly = _clip(quad, pa.p1[i], pa.p2[i], pa.q1[i], pa.q2[i])
                if len(poly) < 3:
                    continue
                xy, w = _overlap_rule(poly)
                if not len(w):
                    continue
                psi = np.stack([np.ones(len(w)),
                                (xy[:, 0] - 0.5 * (pa.p1[i] + pa.p2[i])) / (0.5 * (pa.p2[i] - pa.p1[i])),
                                (xy[:, 1] - 0.5 * (pa.q1[i] + pa.q2[i])) / (0.5 * (pa.q2[i] - pa.q1[i]))])
                x3 = np.zeros((len(w), 3))
                x3[:, ia], x3[:, ja], x3[:, ka] = xy[:, 0], xy[:, 1], w0
                xb = (x3 - cb) @ Rab  # into b's frame
                ph = np.stack([np.ones(len(w)), (xb[:, i_b] - pcb) / hpb, (xb[:, j_b] - qcb) / hqb])
                G[i, :, j, :] += -0.5 * np.einsum("ap,bp,p->ab", psi, ph, w)
    return G.reshape(a.Nd, b.Nd)


def solve_system(bodies, H_ext=None, cross=None):
    """Coupled solve. H_ext: callable, global points (M,3) -> global H (A/m).
    Returns the list of coefficient vectors, one per body."""
    nb = len(bodies)
    offs = np.cumsum([0] + [b.Nd for b in bodies])
    N = offs[-1]
    A = np.zeros((N + nb, N + nb))
    rhs = np.zeros(N + nb)
    for ia_, a in enumerate(bodies):
        sl = slice(offs[ia_], offs[ia_ + 1])
        A[sl, sl] = -a.G
        idx = np.arange(offs[ia_], offs[ia_ + 1])
        A[idx, idx] += (1 / a.chi + 0.5) * a.s.mass
        e = np.zeros(a.Nd)
        e[0::3] = a.pan.area
        A[sl, N + ia_] = e
        A[N + ia_, sl] = e / a.pan.area.sum()
        if H_ext is not None:
            rhs[sl] += a.s.rhs(lambda x, a=a: H_ext(a.to_global(x)) @ a.R)
        if a.J0 is not None:
            rhs[sl] += a.s.rhs_const_normal(a.pan.normal @ (a.J0 / MU0)) / a.chi
        for ib_, b in enumerate(bodies):
            if ib_ == ia_:
                continue
            Gab = cross[(ia_, ib_)] if cross is not None else cross_block(a, b)
            A[sl, offs[ib_] : offs[ib_ + 1]] = -Gab
    x = np.linalg.solve(A, rhs)
    return [x[offs[k] : offs[k + 1]] for k in range(nb)]


def field_B(bodies, coefs, obs):
    """B (T) of all bodies at observers outside them (global frame)."""
    obs = np.atleast_2d(obs)
    H = sum(b.s.field(b.to_local(obs), c) @ b.R.T for b, c in zip(bodies, coefs))
    return MU0 * H


def moment(bodies, coefs):
    """Total magnetic moment (A m^2, global frame)."""
    return sum(b.s.moment(c) @ b.R.T for b, c in zip(bodies, coefs))
