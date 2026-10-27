"""Cuboids placed and turned freely, coupled through their face charges.

Body a's equation, tested with its own panel functions:

    (1/chi_a + 1/2) M_a c_a - G_aa c_a - sum_b G_ab c_b = F_a + F0_a / chi_a

``G_ab`` is the flux of b's charge through a's test panels, computed like the
self block: the closed-form weighted solid angle of the test panel, in a's
frame, at b's Gauss points mapped into it. Where a face of b lies in a face
plane of a (contact), that part is exact instead: no principal value, and the
one-sided limit from a's inside gives -1/2 of the integral of the two charges
over their overlap. The overlap of a test rectangle with b's panel (turned in
the plane when b is) is clipped as a polygon and integrated exactly, so the two
meshes need not match. Each body's equation is bordered by one row and column
fixing its net charge to zero.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import magpylib as magpy
import numpy as np
import scipy.linalg
import scipy.sparse

from magpylib_material_response.face_charge._contact import contact_block
from magpylib_material_response.face_charge._contact import (
    polygon_rule as _polygon_rule,
)
from magpylib_material_response.face_charge._panels import (
    Galerkin,
    Panels,
    graded_edges,
)
from magpylib_material_response.face_charge._rectangle import rectangle_field

GRADING = 2.5  # power of the panel grading toward edges and contact lines
LAYER = 2.0  # graded layer length, in half the body's smallest side
CHUNK = 2_000_000  # kernel evaluations per call, about 0.5 GB
# faces closer than this, in a body's largest half-side, touch: their lines
# grade the mesh, and coplanar panels couple exactly. Lines closer than this
# are one line. Coordinates from float32 CAD data are good to about 1e-7.
CONTACT = 1e-6


class Part:
    """What every solved part has: susceptibility ``chi`` and remanence ``J0``
    (T, in its own frame like magpylib's polarization), at ``position``, turned
    by the rotation matrix ``R`` (its frame to global), in a uniform applied
    field ``applied`` (A/m, global), meshed at level ``m``."""

    def __init__(self, position, R, chi, J0=None, applied=None, m=3):
        self.position = np.asarray(position, float)
        self.R = np.asarray(R, float)
        self.chi = float(chi)
        self.J0 = None if J0 is None else np.asarray(J0, float)
        self.applied = np.zeros(3) if applied is None else np.asarray(applied, float)
        self.m = m

    def to_global(self, x):
        return np.asarray(x) @ self.R.T + self.position


class Body(Part):
    """A cuboid of ``dimension``; the rest as for :class:`Part`."""

    def __init__(self, dimension, position, R, chi, J0=None, applied=None, m=3):
        super().__init__(position, R, chi, J0, applied, m)
        self.dimension = np.asarray(dimension, float)
        self.half = self.dimension / 2
        # interior lines to grade toward, per face and axis: a footprint
        self.lines = [[[], [], []] for _ in range(6)]
        self.extra = [[], [], []]  # edges added as they are, per axis: bisections
        self.panels = self.galerkin = self.G = None
        self.fine = None  # panels whose load takes a finer rule

    def mesh(self):
        """Mesh the faces; the self block waits for ``assemble``, so that the
        unknowns can be counted first."""
        layer = LAYER * np.min(self.half)
        graded = {}  # faces with the same lines along an axis share its edges

        def edges(f, a):
            h = self.half[a]
            lines = tuple(_merge(np.asarray(self.lines[f][a], float), h))
            if (a, lines) not in graded:
                e = np.concatenate(
                    [graded_edges(h, lines, layer, self.m, GRADING), self.extra[a]]
                )
                graded[a, lines] = _merge(e, h)
            return graded[a, lines]

        self.panels = Panels(
            self.half,
            [tuple(edges(f, a) for a in range(3) if a != f // 2) for f in range(6)],
        )
        self.galerkin = Galerkin(self.panels)
        self.G = None

    def assemble(self):
        """The self block of the faces as last meshed."""
        self.G = self.galerkin.assemble()

    def bisect(self, panels):
        """Split these panels in two along both in-plane axes: in the tensor
        mesh, the whole strips through them. Takes effect at the next mesh."""
        p = self.panels
        for n in panels:
            self.extra[p.i[n]].append(0.5 * (p.p1[n] + p.p2[n]))
            self.extra[p.j[n]].append(0.5 * (p.q1[n] + p.q2[n]))

    @property
    def n_dofs(self):
        return self.galerkin.n_dofs

    # what solve_system asks of a body, which a TriBody answers too
    def add_mass(self, block, factor):
        """``factor`` times the mass, diagonal on rectangles, into ``block``."""
        idx = np.arange(self.n_dofs)
        block[idx, idx] += factor * self.galerkin.mass

    def border(self):
        """The net charge of each unknown: the area under its constant density."""
        e = np.zeros(self.n_dofs)
        e[0::3] = self.panels.area
        return e

    def load(self, field):
        return self.galerkin.load(field, fine=self.fine)

    def load_remanence(self):
        """The load of the remanence's ``M0.n``, constant on each panel."""
        return self.galerkin.load_constant(self.panels.normal @ (self.J0 / magpy.mu_0))

    def centres_and_sizes(self):
        """Each panel's centre, global, and its diagonal."""
        g = self.galerkin
        return self.to_global(self.panels.center), 2 * np.hypot(g.hp, g.hq)

    def sources(self):
        """Global Gauss points (P, 3), and the sparse map (P x n_dofs) from the
        unknowns to the charge each point carries."""
        pts, wts, S, T = self.galerkin.gauss_points(self.galerkin.nq)
        n_gauss, N = wts.shape
        phi = np.stack([np.ones(n_gauss), S, T], 1)
        values = wts[..., None] * phi[:, None, :]  # (gauss, panel, density)
        rows = np.repeat(np.arange(n_gauss * N), 3)
        cols = (3 * np.tile(np.arange(N), n_gauss)[:, None] + np.arange(3)).ravel()
        weights = scipy.sparse.csr_matrix(
            (values.ravel(), (rows, cols)), shape=(n_gauss * N, self.n_dofs)
        )
        return self.to_global(pts.reshape(-1, 3)), weights

    def panel_polygons(self):
        """Each panel's corners (N, 4, 3) and outward normal (N, 3), global."""
        corners = self.panels.corners()
        return (
            self.to_global(corners.reshape(-1, 3)).reshape(corners.shape),
            self.panels.normal @ self.R.T,
        )

    def basis_global(self, idx, X):
        """``(1, s, t)`` of panel ``idx`` at global points ``X`` (k, 3)."""
        p = self.panels
        x = (np.asarray(X) - self.position) @ self.R
        s = (x[:, p.i[idx]] - 0.5 * (p.p1[idx] + p.p2[idx])) / (
            0.5 * (p.p2[idx] - p.p1[idx])
        )
        t = (x[:, p.j[idx]] - 0.5 * (p.q1[idx] + p.q2[idx])) / (
            0.5 * (p.q2[idx] - p.q1[idx])
        )
        return np.stack([np.ones(len(x)), s, t], 1)

    def test_flux(self, y, tol=0.0):
        """(n_dofs, P): the flux of a unit charge at each point ``y`` (this
        body's frame) through each test function; none from points within
        ``tol`` of a panel's plane, where a contact takes over."""
        p = self.panels
        out = np.zeros((p.N, 3, len(y)))
        step = max(1, CHUNK // max(len(y), 1))
        for fa in range(6):
            ta = np.flatnonzero(p.face == fa)
            k, i, j, sign = p.k[ta[0]], p.i[ta[0]], p.j[ta[0]], p.sign[ta[0]]
            for c0 in range(0, len(ta), step):
                tc = ta[c0 : c0 + step]
                w3 = rectangle_field(
                    y[None, :, i],
                    y[None, :, j],
                    y[None, :, k] - p.w0[ta[0]],
                    p.p1[tc][:, None],
                    p.p2[tc][:, None],
                    p.q1[tc][:, None],
                    p.q2[tc][:, None],
                )  # (3 test densities, tc, P)
                w3[:, :, np.abs(y[:, k] - p.w0[ta[0]]) < tol] = 0.0
                out[tc] = -sign * w3.transpose(1, 0, 2)
        return out.reshape(self.n_dofs, len(y))


def _merge(x, h):
    """The coordinates `x` on ``[-h, h]``, sorted, those closer than
    ``CONTACT h`` to the one before dropped: two lines a hair apart would make
    panels of no area. The ends stay: a point a hair inside one gives way."""
    kept = []
    for value in np.sort(x):
        if not kept or value - kept[-1] > CONTACT * h:
            kept.append(value)
        elif value >= h * (1 - 1e-12):
            kept[-1] = value
    return np.array(kept)


def near_panels(body, boxes, reach=1.0):
    """The panels of `body` within ``reach`` of their own size of any box
    ``(centre, half, R, position)``: a magnet's box in its frame, its rotation
    and its place. Their load needs a finer rule."""
    centres, size = body.centres_and_sizes()
    near = np.zeros(len(size), bool)
    for centre, half, R, position in boxes:
        local = (centres - position) @ R - centre
        gap = np.linalg.norm(np.maximum(np.abs(local) - half, 0), axis=1)
        near |= gap < reach * size
    return near


def _relative(a, b):
    """b's frame seen from a's: the rotation from b's frame to a's, and b's
    centre in a's frame."""
    return a.R.T @ b.R, (b.position - a.position) @ a.R


class Outline(NamedTuple):
    """A cuboid that is not solved, such as a magnet without susceptibility:
    its half-sides, rotation (its frame to global) and centre."""

    half: np.ndarray
    R: np.ndarray
    position: np.ndarray


def add_contact_lines(bodies, outlines=()):
    """Grade the face of each body that another body, or a fixed cuboid
    (``outlines``), rests on toward that one's footprint: the charge jumps
    there, and so does a touching magnet's field.

    Only that face: through the others the lines would cut whole strips of
    panels where nothing happens, and the assembly grows with the square of
    the panels. Exact when the two are turned by multiples of 90 degrees
    against each other; otherwise the lines bound the footprint (its box in
    this body's frame)."""
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    bodies = [
        b for b in bodies if isinstance(b, Body)
    ]  # triangle meshes are not graded
    for a in bodies:
        tol = CONTACT * np.max(a.half)
        for b in (*bodies, *outlines):
            if b is a:
                continue
            Rab, cb = _relative(a, b)
            corners = cb + (signs * b.half) @ Rab.T
            lo, hi = corners.min(0), corners.max(0)
            gap = np.maximum(0, np.maximum(lo - a.half, -a.half - hi))
            if np.linalg.norm(gap) > tol:
                continue
            # the faces b lies against: outside a across their plane
            faces = [
                f
                for k in range(3)
                for f, against in (
                    (2 * k, lo[k] >= a.half[k] - tol),
                    (2 * k + 1, hi[k] <= -a.half[k] + tol),
                )
                if against
            ]
            for f in faces or range(6):  # overlapping: every face, as before
                for k in range(3):
                    if k != f // 2:
                        a.lines[f][k] = sorted(set(a.lines[f][k]) | {lo[k], hi[k]})


def _clip(poly, x0, x1, y0, y1):
    """A convex polygon (list of (x, y)) clipped to a box (Sutherland-Hodgman)."""
    for axis, bound, keep_above in (
        (0, x0, True),
        (0, x1, False),
        (1, y0, True),
        (1, y1, False),
    ):
        out = []
        for p, q in zip(poly, poly[1:] + poly[:1], strict=True):
            p_in = p[axis] >= bound if keep_above else p[axis] <= bound
            q_in = q[axis] >= bound if keep_above else q[axis] <= bound
            if p_in:
                out.append(p)
            if p_in != q_in:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if not poly:
            break
    return poly


def cross_block(a, b, tol=CONTACT):
    """``G_ab`` (a.n_dofs x b.n_dofs): the flux of b's charge through a's test
    functions."""
    pa, pb = a.panels, b.panels
    Rab, cb = _relative(a, b)
    pts, wts, S, T = b.galerkin.gauss_points(b.galerkin.nq)
    n_gauss = len(S)
    y = pts @ Rab.T + cb  # b's Gauss points in a's frame, (gauss, Nb, 3)
    normal_b = pb.normal @ Rab.T
    center_b = pb.center @ Rab.T + cb
    phi = np.stack([np.ones(n_gauss), S, T], 1)
    G = np.zeros((pa.N, 3, pb.N, 3))
    scale = np.max(a.half)
    for fa in range(6):
        ta = np.flatnonzero(pa.face == fa)
        ka, ia, ja, sa = pa.k[ta[0]], pa.i[ta[0]], pa.j[ta[0]], pa.sign[ta[0]]
        w0 = pa.w0[ta[0]]
        # b's panels lying in this face plane are handled exactly, below
        coplanar = (np.abs(normal_b[:, ka]) > 1 - 1e-9) & (
            np.abs(center_b[:, ka] - w0) < tol * scale
        )
        src = np.flatnonzero(~coplanar)
        if len(src):
            yy = y[:, src]
            step = max(1, CHUNK // (n_gauss * len(src)))
            for c0 in range(0, len(ta), step):
                tc = ta[c0 : c0 + step]
                w3 = rectangle_field(
                    yy[None, ..., ia],
                    yy[None, ..., ja],
                    yy[None, ..., ka] - w0,
                    pa.p1[tc][:, None, None],
                    pa.p2[tc][:, None, None],
                    pa.q1[tc][:, None, None],
                    pa.q2[tc][:, None, None],
                )  # (3 test densities, tc, gauss, src)
                block = np.einsum("atgn,gn,gb->tanb", w3, wts[:, src], phi)
                G[np.ix_(tc, [0, 1, 2], src, [0, 1, 2])] = -sa * block
        if coplanar.any():
            corners_b = pb.corners() @ Rab.T + cb  # in a's frame
            for j in np.flatnonzero(coplanar):
                _add_contact(G, pa, pb, ta, j, corners_b[j], (ia, ja, ka, w0), Rab, cb)
    return G.reshape(a.n_dofs, b.n_dofs)


def _add_contact(G, pa, pb, ta, j, corners, plane, Rab, cb):
    """Add b's panel ``j`` (its ``corners`` in a's frame), lying in a's face
    plane, to ``G``: -1/2 of the integral of each pair of test and source
    functions over their overlap."""
    ia, ja, ka, w0 = plane
    ib, jb = pb.i[j], pb.j[j]
    quad = [(c[ia], c[ja]) for c in corners]
    lo, hi = corners.min(0), corners.max(0)
    hit = np.flatnonzero(
        (pa.p2[ta] > lo[ia])
        & (pa.p1[ta] < hi[ia])
        & (pa.q2[ta] > lo[ja])
        & (pa.q1[ta] < hi[ja])
    )
    pcb, hpb = 0.5 * (pb.p1[j] + pb.p2[j]), 0.5 * (pb.p2[j] - pb.p1[j])
    qcb, hqb = 0.5 * (pb.q1[j] + pb.q2[j]), 0.5 * (pb.q2[j] - pb.q1[j])
    for i in ta[hit]:
        poly = _clip(quad, pa.p1[i], pa.p2[i], pa.q1[i], pa.q2[i])
        if len(poly) < 3:
            continue
        xy, w = _polygon_rule(poly)
        if not w.size:
            continue
        psi = np.stack(
            [
                np.ones(len(w)),
                (xy[:, 0] - 0.5 * (pa.p1[i] + pa.p2[i]))
                / (0.5 * (pa.p2[i] - pa.p1[i])),
                (xy[:, 1] - 0.5 * (pa.q1[i] + pa.q2[i]))
                / (0.5 * (pa.q2[i] - pa.q1[i])),
            ]
        )
        x3 = np.zeros((len(w), 3))
        x3[:, ia], x3[:, ja], x3[:, ka] = xy[:, 0], xy[:, 1], w0
        xb = (x3 - cb) @ Rab  # into b's frame
        phi = np.stack(
            [np.ones(len(w)), (xb[:, ib] - pcb) / hpb, (xb[:, jb] - qcb) / hqb]
        )
        G[i, :, j, :] += -0.5 * np.einsum("ap,bp,p->ab", psi, phi, w)


def coupling_block(a, b):
    """``G_ab``: the flux of b's charge through a's test functions. Between
    two cuboids by their own exact scheme; otherwise by b's quadrature, which
    holds for bodies apart by about their panels' size or more, and where
    faces of the two lie in one plane, by the exact contact term."""
    if isinstance(a, Body) and isinstance(b, Body):
        return cross_block(a, b)
    tol = CONTACT * max(np.max(a.half), np.max(b.half))
    points, weights = b.sources()
    y = (points - a.position) @ a.R  # b's points in a's frame
    G = contact_block(a, b, tol)
    step = max(1, CHUNK // max(a.n_dofs, 1) // 8)
    for p0 in range(0, len(y), step):
        flux = a.test_flux(y[p0 : p0 + step], tol)  # (a.n_dofs, chunk)
        G += (weights[p0 : p0 + step].T @ flux.T).T
    return G


def solve_system(bodies, external):
    """Solve the coupled bodies.

    ``external`` maps global points ``(n, 3)`` to the field of the fixed sources
    there, H in A/m, global; each body adds its own uniform applied field.
    Returns the charge coefficients of each body.
    """
    n_bodies = len(bodies)
    offsets = np.cumsum([0] + [b.n_dofs for b in bodies])
    n = offsets[-1]
    # column-major, which the in-place factorization below needs
    A = np.zeros((n + n_bodies, n + n_bodies), order="F")
    rhs = np.zeros(n + n_bodies)
    for ia, a in enumerate(bodies):
        rows = slice(offsets[ia], offsets[ia + 1])
        # each self block is assembled here and freed once copied, so that only
        # one lives beside the system
        a.assemble()
        np.negative(a.G, out=A[rows, rows])
        a.G = None
        a.add_mass(A[rows, rows], 1 / a.chi + 0.5)
        border = a.border()
        A[rows, n + ia] = border
        A[n + ia, rows] = border / border.sum()

        def field(x, a=a):
            return (external(a.to_global(x)) + a.applied) @ a.R

        rhs[rows] += a.load(field)
        if a.J0 is not None:
            rhs[rows] += a.load_remanence() / a.chi
        for ib, b in enumerate(bodies):
            if ib != ia:
                np.negative(
                    coupling_block(a, b), out=A[rows, offsets[ib] : offsets[ib + 1]]
                )
    # in place: factorizing a copy of this dense system peaked at three times
    # its size
    x = scipy.linalg.solve(A, rhs, overwrite_a=True, check_finite=False)
    return [x[offsets[k] : offsets[k + 1]] for k in range(n_bodies)]
