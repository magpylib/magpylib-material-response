"""Panels on a cuboid's faces, and the Galerkin form of the face-charge equation.

Each face is cut into rectangular panels by a list of edges along each of its
two axes, graded toward the cuboid's edges (where a soft body's charge is
singular) and toward any interior line of that face, such as the footprint of a
body resting on it. Each panel carries
the charge ``c0 + c1 s + c2 t``, linear in its local coordinates ``s, t`` in
``[-1, 1]``, discontinuous from one panel to the next (P1 discontinuous
Galerkin).

The equation tested with those same functions is

    (1/chi + 1/2) M c - G c = F + F0 / chi,

with ``M`` the (diagonal) mass matrix, ``F`` the load of the applied field and
``F0`` that of the remanence. ``G`` holds the flux of each panel's charge
through each test panel: the integral, over the source panel, of the test
panel's weighted solid angle, which stays bounded where the field itself is
logarithmically singular (a shared edge). Panels on one face carry no flux
through each other.
"""

from __future__ import annotations

import itertools

import numpy as np

from magpylib_material_response.face_charge._parallel import CHUNK, THREADS, run
from magpylib_material_response.face_charge._rectangle import rectangle_field


def contract(w3, wts, phi):
    """``sum_g w3[a, t, g, n] wts[g, n] phi[g, b]``, as ``(t, a, n, b)``: the
    test fluxes at the Gauss points into the coupling of each pair of
    functions, by matrix products."""
    return np.matmul((w3 * wts).transpose(1, 0, 3, 2), phi)


def graded_edges(a, lines, layer, m, g, tol=1e-9):
    """Edges on ``[-a, a]``, graded toward both ends and toward every line.

    ``m`` panels of power-``g`` grading fill a layer of length ``layer`` (at
    most half the distance to the next line) on each side of an end or line;
    uniform panels of the layer's largest size fill the rest.
    """
    inner = [x for x in lines if -a + tol * a < x < a - tol * a]
    points = np.unique(np.concatenate([[-a, a], inner]))
    t = np.linspace(0, 1, m + 1)
    out = []
    for x0, x1 in itertools.pairwise(points):
        length = x1 - x0
        lay = min(layer, length / 2) * t**g
        hmax = lay[-1] - lay[-2]
        gap = length - 2 * lay[-1]
        k = int(np.ceil(gap / hmax - 1e-9)) if gap > tol * a else 0
        mid = np.linspace(x0 + lay[-1], x1 - lay[-1], k + 1) if k else [x0 + lay[-1]]
        out.extend([x0 + lay, mid, x1 - lay[::-1]])
    return np.unique(np.round(np.concatenate(out) / a, 12)) * a


class Panels:
    """Rectangular panels on the six faces of a cuboid centred at the origin.

    ``face_edges[f]`` gives face ``f``'s edges along its in-plane axes
    ``i < j``; face ``f`` is normal to axis ``k = f // 2``, on the side
    ``sign = +1`` for even ``f``. A panel spans ``[p1, p2] x [q1, q2]`` and sits
    at ``w0 = sign * half[k]``.
    """

    def __init__(self, half, face_edges):
        self.half = np.asarray(half, float)
        rows = []
        for f, (ei, ej) in enumerate(face_edges):
            k, sign = f // 2, -1.0 if f % 2 else 1.0
            i, j = (a for a in range(3) if a != k)
            P1, Q1 = np.meshgrid(ei[:-1], ej[:-1], indexing="ij")
            P2, Q2 = np.meshgrid(ei[1:], ej[1:], indexing="ij")
            rows.extend(
                (k, sign, i, j, sign * self.half[k], *bounds)
                for bounds in zip(
                    P1.ravel(), P2.ravel(), Q1.ravel(), Q2.ravel(), strict=True
                )
            )
        r = np.array(rows)
        self.k = r[:, 0].astype(int)
        self.sign = r[:, 1]
        self.i = r[:, 2].astype(int)
        self.j = r[:, 3].astype(int)
        self.w0 = r[:, 4]
        self.p1, self.p2, self.q1, self.q2 = r[:, 5], r[:, 6], r[:, 7], r[:, 8]
        self.N = len(r)
        self.area = (self.p2 - self.p1) * (self.q2 - self.q1)
        idx = np.arange(self.N)
        self.center = np.zeros((self.N, 3))
        self.center[idx, self.k] = self.w0
        self.center[idx, self.i] = 0.5 * (self.p1 + self.p2)
        self.center[idx, self.j] = 0.5 * (self.q1 + self.q2)
        self.normal = np.zeros((self.N, 3))
        self.normal[idx, self.k] = self.sign
        self.face = 2 * self.k + (self.sign < 0)  # 0..5

    def corners(self):
        """Each panel's corners ``(N, 4, 3)``, counter-clockwise in ``(s, t)``:
        ``(-1, -1), (1, -1), (1, 1), (-1, 1)``."""
        out = np.zeros((self.N, 4, 3))
        idx = np.arange(self.N)
        for r, (p, q) in enumerate(
            (
                (self.p1, self.q1),
                (self.p2, self.q1),
                (self.p2, self.q2),
                (self.p1, self.q2),
            )
        ):
            out[idx, r, self.i] = p
            out[idx, r, self.j] = q
            out[idx, r, self.k] = self.w0
        return out


class Galerkin:
    """The P1 Galerkin matrices on a set of panels.

    Each panel carries three unknowns, the coefficients of ``1, s, t``.
    """

    def __init__(self, panels, nq=4):
        p = panels
        self.panels = p
        self.nq = nq
        self.hp = 0.5 * (p.p2 - p.p1)
        self.hq = 0.5 * (p.q2 - p.q1)
        self.pc = 0.5 * (p.p1 + p.p2)
        self.qc = 0.5 * (p.q1 + p.q2)
        self.mass = (p.area[:, None] * np.array([1.0, 1 / 3, 1 / 3])).ravel()
        self.n_dofs = 3 * p.N
        self._gauss = {}

    def gauss_points(self, nq):
        """Gauss points of every panel: points ``(nq^2, N, 3)``, weights
        ``(nq^2, N)``, and the local ``s, t`` of each point ``(nq^2,)``; made
        once a rule and shared, read-only."""
        if nq not in self._gauss:
            rule = self._gauss_points(nq)
            for x in rule:
                x.setflags(write=False)
            self._gauss[nq] = rule
        return self._gauss[nq]

    def _gauss_points(self, nq):
        p = self.panels
        g, w = np.polynomial.legendre.leggauss(nq)
        S, T = np.meshgrid(g, g, indexing="ij")
        weights = np.outer(w, w).ravel() * 0.25
        S, T = S.ravel(), T.ravel()
        pts = np.repeat(p.center[None], len(S), 0)
        idx = np.arange(p.N)
        pts[:, idx, p.i] = self.pc + S[:, None] * self.hp
        pts[:, idx, p.j] = self.qc + T[:, None] * self.hq
        return pts, weights[:, None] * p.area[None, :], S, T

    def assemble(self):
        """The flux matrix ``G`` (3N x 3N) between the panels of one body."""
        p = self.panels
        G = np.zeros((p.N, 3, p.N, 3))
        pts, wts, S, T = self.gauss_points(self.nq)
        faces = [np.flatnonzero(p.face == f) for f in range(6)]

        phi = np.stack([np.ones(len(S)), S, T], 1)

        def block(job):  # test rows ``tc`` against source face fb
            fb, tc = job
            ka, ia, ja, sa = p.k[tc[0]], p.i[tc[0]], p.j[tc[0]], p.sign[tc[0]]
            sb = faces[fb]
            y = pts[:, sb]  # (gauss, sb, 3)
            w3 = rectangle_field(
                y[None, ..., ia],
                y[None, ..., ja],
                y[None, ..., ka] - p.w0[tc][:, None, None],
                p.p1[tc][:, None, None],
                p.p2[tc][:, None, None],
                p.q1[tc][:, None, None],
                p.q2[tc][:, None, None],
            )  # (3 test densities, tc, gauss, sb)
            G[np.ix_(tc, [0, 1, 2], sb, [0, 1, 2])] = -sa * contract(
                w3, wts[:, sb], phi
            )

        jobs = []
        for fa in range(6):
            for fb in range(6):
                if fb != fa:
                    step = max(1, CHUNK // (len(faces[fb]) * len(S) * THREADS))
                    ta = faces[fa]
                    jobs += [(fb, ta[c : c + step]) for c in range(0, len(ta), step)]
        run(block, jobs, work=p.N * p.N * len(S))
        return G.reshape(3 * p.N, 3 * p.N)

    def load(self, field, nq=4, fine=None, nq_fine=16):
        """The load of an applied field: each test function times ``H.n``,
        integrated over its panel. ``field`` maps points ``(n, 3)`` in the
        body's frame to H ``(n, 3)`` in A/m, in that frame. Panels marked in
        ``fine``, near a magnet that is not solved, take ``nq_fine`` points a
        side: its field turns there faster than ``nq`` points resolve."""
        F = self._load(field, nq, np.arange(self.panels.N))
        if fine is not None and np.any(fine):
            idx = np.flatnonzero(fine)
            F[idx] = self._load(field, nq_fine, idx)
        return F.ravel()

    def _load(self, field, nq, idx):
        """The load on the panels ``idx`` (len, 3), by ``nq`` points a side."""
        p = self.panels
        pts, wts, S, T = self.gauss_points(nq)
        pts, wts = pts[:, idx], wts[:, idx]
        H = field(pts.reshape(-1, 3)).reshape(len(S), len(idx), 3)
        Hn = np.einsum("gnd,nd->gn", H, p.normal[idx])
        return np.stack(
            [
                (wts * Hn).sum(0),
                (wts * Hn * S[:, None]).sum(0),
                (wts * Hn * T[:, None]).sum(0),
            ],
            1,
        )

    def load_constant(self, values):
        """The load of a value constant on each panel, such as the remanence's
        ``M0.n``."""
        F = np.zeros((self.panels.N, 3))
        F[:, 0] = values * self.panels.area
        return F.ravel()
