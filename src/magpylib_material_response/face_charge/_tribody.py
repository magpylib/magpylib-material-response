"""A closed triangle mesh solved by face charge: a CAD part, linear across each
triangle.

The same scheme as the cuboid panels: on each triangle the charge is ``c0 +
c1 u + c2 v``, ``u, v`` its in-plane coordinates from the centroid over the
triangle's scale. The Galerkin flux of a source triangle through a test
triangle is the integral, over the source, of the test triangle's weighted
solid angle, ``-n . E[psi](y) / 4 pi``, by a 7-point rule, refined on 4 x 7
points for pairs about as close as they are long. Triangles in one plane give no
flux to each other, as the closed form gives exactly.

From the study on ``explore/face-charge`` (``triangles/tri_solver.py``,
``TriBody``).
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import (  # pylint: disable=no-name-in-module
    ConvexHull,
    QhullError,
    cKDTree,
)

from magpylib_material_response.face_charge._bodies import Body, Part, contact_tol
from magpylib_material_response.face_charge._mesh import graded_triangles, rounding
from magpylib_material_response.face_charge._parallel import CHUNK, run, threads
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import (
    frames,
    normal_field,
    normal_field_far,
)

LAYER = 0.5  # graded layer length, in each shell's thickness: half of it
GRADING = 2.5  # power of the grading toward sharp edges, as the cuboid panels'

# 7-point degree-5 rule (Dunavant) and the 3-point degree-2 rule (edge midpoints)
_A1, _B1 = 0.059715871789770, 0.470142064105115
_A2, _B2 = 0.797426985353087, 0.101286507323456
BARY7 = np.array(
    [
        [1 / 3] * 3,
        [_A1, _B1, _B1],
        [_B1, _A1, _B1],
        [_B1, _B1, _A1],
        [_A2, _B2, _B2],
        [_B2, _A2, _B2],
        [_B2, _B2, _A2],
    ]
)
W7 = np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3])
BARY3 = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
W3 = np.full(3, 1 / 3)


def rule(k=0):
    """Barycentric points and weights, summing to 1, of the 7-point rule on
    the reference triangle cut ``k`` times into four."""
    T = np.eye(3)[None]
    for _ in range(k):
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate(
            [
                np.stack(s, 1)
                for s in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            ]
        )
    points = np.einsum("qk,tkl->tql", BARY7, T).reshape(-1, 3)
    return points, np.tile(W7, len(T)) / len(T)


class TriBody(Part):
    """A closed, outward triangle mesh, ``vertices`` and ``faces`` in its own
    frame; the rest as for :class:`~._bodies.Part`."""

    def __init__(self, vertices, faces, position, R, chi, J0=None, applied=None, m=3):
        super().__init__(position, R, chi, J0, applied, m)
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, int)
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        self.centre, self.half = (lo + hi) / 2, (hi - lo) / 2  # the bounding box
        self.noise = rounding(self.vertices)  # float32's for float32 data
        self.shells = _shells(self.faces)
        inside = _nesting(self.vertices, self.faces, self.shells)
        self.faces = _orient(self.vertices, self.faces, self.shells, inside)
        self.thickness = _thickness(self.vertices, self.faces, self.shells, inside)
        self.shell = None  # each graded triangle's shell
        self.T = self.fr = self.mass = self.G = None
        self.fine = None  # triangles whose load takes a finer rule
        self.contacts = []  # lines (point, direction, normal) to grade toward
        self.split = None  # the triangles and shells of a bisection, to mesh

    def mesh(self):
        """The triangles graded toward the sharp edges, or as bisected since;
        the self block waits for ``assemble``, so that the unknowns can be
        counted first."""
        if self.split is not None:
            (self.T, self.shell), self.split = self.split, None
        else:
            T = [
                graded_triangles(
                    self.vertices,
                    self.faces[self.shells == k],
                    LAYER * self.thickness[k],
                    self.m,
                    GRADING,
                    contacts=self.contacts,
                )
                for k in range(self.shells.max() + 1)
            ]
            self.T = np.concatenate(T)
            self.shell = np.repeat(np.arange(len(T)), [len(t) for t in T])
        self.fr = frames(self.T)
        # mass blocks, exact with the degree-2 rule
        P = np.einsum("qk,fkd->fqd", BARY3, self.T)
        phi = self.basis(P)
        self.mass = np.einsum("fq,fqa,fqb->fab", self.fr[5][:, None] * W3, phi, phi)
        self.G = None

    def bisect(self, panels):
        """Cut these triangles into four at their edges' midpoints; their
        neighbours need no cut, the charge being discontinuous between
        triangles. Takes effect at the next mesh."""
        a, b, c = (self.T[panels, i] for i in range(3))
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        keep = np.ones(len(self.T), bool)
        keep[panels] = False
        children = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        self.split = (
            np.concatenate([self.T[keep], *(np.stack(t, 1) for t in children)]),
            np.concatenate([self.shell[keep], np.tile(self.shell[panels], 4)]),
        )

    @property
    def n_dofs(self):
        return 3 * len(self.T)

    def basis(self, P, idx=slice(None)):
        """``(1, u, v)`` at points ``P`` (F, q, 3) of the triangles ``idx``."""
        cen, _, e1, e2, s, _ = (a[idx] for a in self.fr)
        d = P - cen[:, None]
        return np.stack(
            [
                np.ones(P.shape[:2]),
                np.einsum("fqd,fd->fq", d, e1) / s[:, None],
                np.einsum("fqd,fd->fq", d, e2) / s[:, None],
            ],
            axis=-1,
        )

    def quad(self, k=0, idx=None):
        """Points (F, q, 3), weights (F, q) and basis (F, q, 3) of the 7-point
        rule on ``4^k`` sub-triangles, for the triangles ``idx``."""
        idx = slice(None) if idx is None else idx
        bary, w = rule(k)
        P = np.einsum("qk,fkd->fqd", bary, self.T[idx])
        return P, self.fr[5][idx][:, None] * w[None], self.basis(P, idx)

    # ── the body's own block ───────────────────────────────────────────────
    def assemble(self):
        """``G`` (3F x 3F): the flux of each triangle's densities through each
        test triangle's. All pairs with the 7-point rule on the source, then
        the pairs closer than the sum of their longest edges again on 4 x 7
        points, in batches."""
        F = len(self.T)
        G = np.zeros((F, 3, F, 3))
        P, W, B = self.quad()
        nq = P.shape[1]
        step = max(1, CHUNK // (nq * F * threads()))

        def far(j0):  # each chunk its own source columns
            js = np.arange(j0, min(j0 + step, F))
            flux = -self.far_field(P[js].reshape(-1, 3)) / FOUR_PI
            WB = W[js][..., None] * B[js]  # (js, nq, 3)
            G[:, :, js, :] = np.matmul(
                flux.reshape(3, len(js), nq, F).transpose(1, 3, 0, 2), WB[:, None]
            ).transpose(1, 2, 0, 3)

        run(far, range(0, F, step), work=F * F * nq)
        I, J = self.near_pairs()
        Pk, Wk, Bk = self.quad(1)
        Q = Pk.shape[1]
        kstep = max(1, CHUNK // (Q * threads()))

        def near(k0):  # each pair once, so each batch its own entries
            i, j = I[k0 : k0 + kstep], J[k0 : k0 + kstep]
            fr = tuple(a[i][:, None] for a in self.fr)
            En = normal_field(Pk[j], self.T[i][:, None], fr, pairs=True)  # (3, k, Q)
            WB = Wk[j][..., None] * Bk[j]  # (k, Q, 3)
            G[i, :, j, :] = np.matmul(En.transpose(1, 0, 2), WB) / -FOUR_PI

        run(near, range(0, len(I), kstep), work=len(I) * Q)
        # a triangle's own flux: the principal value is 0, the jump is the mass term
        G[np.arange(F), :, np.arange(F), :] = 0.0
        self.G = G.reshape(3 * F, 3 * F)

    def near_pairs(self):
        """Ordered pairs ``(i, j)``, ``i != j``, with centroids closer than the
        sum of their longest edges. Each triangle searches twice its own size,
        so a partner is found by the larger of the two: one radius from the
        largest triangle sorted millions of candidates on a graded mesh."""
        cen = self.fr[0]
        size = np.linalg.norm(np.roll(self.T, -1, axis=1) - self.T, axis=-1).max(axis=1)
        lists = cKDTree(cen).query_ball_point(cen, 2 * size)
        j = np.repeat(np.arange(len(cen)), [len(x) for x in lists])
        i = np.concatenate(lists).astype(int)
        # with a margin: on a regular grid, cells two apart on a diagonal lie
        # at exactly that distance, which rounding would decide
        keep = (
            (i != j)
            & (size[i] <= size[j])
            & (
                np.linalg.norm(cen[i] - cen[j], axis=1)
                < (1 + 1e-6) * (size[i] + size[j])
            )
        )
        pairs = np.unique(np.sort(np.stack([i[keep], j[keep]], 1), axis=1), axis=0)
        return (
            np.concatenate([pairs[:, 0], pairs[:, 1]]),
            np.concatenate([pairs[:, 1], pairs[:, 0]]),
        )

    def add_mass(self, block, factor):
        """``factor`` times the mass blocks into ``block`` (3F x 3F)."""
        first = 3 * np.arange(len(self.T))[:, None, None]
        rows = np.broadcast_to(first + np.arange(3)[:, None], self.mass.shape)
        cols = np.broadcast_to(first + np.arange(3), self.mass.shape)
        block[rows.ravel(), cols.ravel()] += factor * self.mass.ravel()

    def centres_and_sizes(self):
        """Each triangle's centroid, global, and its longest edge."""
        size = np.linalg.norm(np.roll(self.T, -1, axis=1) - self.T, axis=-1).max(axis=1)
        return self.to_global(self.fr[0]), size

    def border(self):
        """The net charge of each unknown on each closed shell (3F, shells):
        the area under its constant density. Each shell holds none by itself,
        a cavity's too: through the material around it ``M = chi H``, so its
        charge ``Q`` gives ``chi Q`` of M's flux, which B's zero flux makes
        ``-Q``."""
        e = np.zeros((self.n_dofs, self.shell.max() + 1))
        e[3 * np.arange(len(self.T)), self.shell] = self.fr[5]
        return e

    # ── loads ──────────────────────────────────────────────────────────────
    def load(self, field):
        """``int psi H.n`` over each triangle; ``field`` maps points in the
        body's frame to H there, in that frame. Triangles marked in ``fine``,
        near a magnet that is not solved, take 64 x 7 points: its field turns
        there faster than 7 points resolve, and 16 x 7 still left a magnet 0.1
        mm above a soft cube 1.9 % off at levels 3 and 4."""
        F = self._load(field, 0, np.arange(len(self.T)))
        if self.fine is not None and np.any(self.fine):
            idx = np.flatnonzero(self.fine)
            F[idx] = self._load(field, 3, idx)
        return F.ravel()

    def _load(self, field, k, idx):
        P, W, B = self.quad(k, idx)
        H = field(P.reshape(-1, 3)).reshape(P.shape)
        Hn = np.einsum("fqd,fd->fq", H, self.fr[1][idx])
        return np.einsum("fq,fq,fqa->fa", W, Hn, B)

    def load_remanence(self):
        """The load of the remanence's ``M0.n``, constant on each triangle."""
        m0n = self.fr[1] @ (self.J0 / magpy.mu_0)
        return np.einsum("f,fa->fa", m0n, self.mass[:, 0, :]).ravel()

    # ── coupling with other bodies ─────────────────────────────────────────
    def sources(self):
        """Global points (P, 3) of the 7-point rule, and the sparse map (P x
        3F) from the unknowns to the charge each point carries."""
        P, W, B = self.quad()
        F, nq = W.shape
        rows = np.repeat(np.arange(F * nq), 3)
        cols = (3 * np.repeat(np.arange(F), nq)[:, None] + np.arange(3)).ravel()
        S = scipy.sparse.csr_matrix(
            ((W[..., None] * B).ravel(), (rows, cols)), shape=(F * nq, 3 * F)
        )
        return self.to_global(P.reshape(-1, 3)), S

    def panel_polygons(self):
        """Each triangle's corners (F, 3, 3) and outward normal (F, 3), global."""
        return (
            self.to_global(self.T.reshape(-1, 3)).reshape(self.T.shape),
            self.fr[1] @ self.R.T,
        )

    def basis_global(self, idx, X):
        """``(1, u, v)`` of triangle ``idx`` at global points ``X`` (k, 3)."""
        cen, _, e1, e2, s, _ = (a[idx] for a in self.fr)
        d = (np.asarray(X) - self.position) @ self.R - cen
        return np.stack([np.ones(len(d)), d @ e1 / s, d @ e2 / s], 1)

    def far_field(self, X):
        """:func:`normal_field_far` at points ``X`` (its frame), in coordinates
        from the bounding box's centre: CAD exports often keep a part far from
        its own origin, where the far kernel's cancellation grows."""
        c = self.centre
        return normal_field_far(X - c, self.T - c, (self.fr[0] - c, *self.fr[1:]))

    def test_flux(self, y, tol=0.0):
        """(3F, P): the flux of a unit charge at each point ``y`` (its frame)
        through each test function; none from points within ``tol`` of a
        triangle's plane, where a contact takes over."""
        flux = -self.far_field(y) / FOUR_PI  # (3, P, F)
        n = self.fr[1]
        w = y @ n.T - np.sum(self.T[:, 0] * n, axis=1)[None]
        flux[:, np.abs(w) < tol] = 0.0
        return flux.transpose(2, 0, 1).reshape(self.n_dofs, len(y))

    def coefficients_at_corners(self, c):
        """The charge at each triangle's corners (F, 3) from its coefficients."""
        cen, _, e1, e2, s, _ = self.fr
        d = self.T - cen[:, None]
        u = np.einsum("fkd,fd->fk", d, e1) / s[:, None]
        v = np.einsum("fkd,fd->fk", d, e2) / s[:, None]
        c = np.asarray(c, float).reshape(-1, 3)
        return c[:, :1] + c[:, 1:2] * u + c[:, 2:3] * v


_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
# a box's faces as triangles, by corner index (corners as in _SIGNS)
_BOX = np.array(
    [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
)


def _shells(F):
    """Each face's closed shell, numbered from 0: the faces joined across
    shared edges, so that two solids meeting at a corner are two."""
    nf = len(F)
    pairs = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    edge = np.unique(pairs, axis=0, return_inverse=True)[1].ravel()
    links = scipy.sparse.coo_matrix(
        (np.ones(3 * nf), (np.tile(np.arange(nf), 3), nf + edge)),
        shape=(nf + edge.max() + 1,) * 2,
    )
    _, label = connected_components(links, directed=False)
    return np.unique(label[:nf], return_inverse=True)[1]


def _winding(p, T):
    """How many times the closed triangles ``T`` wind round the point ``p``:
    the sum of their solid angles over 4 pi."""
    a, b, c = (T[:, k] - p for k in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return np.arctan2(num, den).sum() / (2 * np.pi)


def _nesting(V, F, shells):
    """``inside[i, j]``: shell ``i`` lies inside shell ``j``, tested at the
    centroid of its largest triangle."""
    k = shells.max() + 1
    inside = np.zeros((k, k), bool)
    if k == 1:
        return inside
    T = V[F]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    lo = np.array([T[shells == i].min((0, 1)) for i in range(k)])
    hi = np.array([T[shells == i].max((0, 1)) for i in range(k)])
    for i in range(k):
        own = np.flatnonzero(shells == i)
        p = T[own[np.argmax(area[own])]].mean(0)
        for j in range(k):
            if j != i and np.all(lo[j] <= lo[i]) and np.all(hi[i] <= hi[j]):
                inside[i, j] = abs(_winding(p, T[shells == j])) > 0.5
    return inside


def _orient(V, F, shells, inside):
    """The faces with each shell pointing out of the material, whichever way it
    came: a shell inside an odd number of others bounds a cavity and points
    into it (magpylib's ``reorient_faces`` turns such a shell outward), any
    other points out."""
    T = V[F]
    volume = np.bincount(
        shells, np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2]))
    )
    flip = (volume > 0) != (inside.sum(1) % 2 == 0)
    F = F.copy()
    turn = flip[shells]
    F[turn] = F[turn][:, [0, 2, 1]]
    return F


def _thickness(V, F, shells, inside):
    """Each shell's thickness: the smallest width across its convex hull, a
    box's smallest side whatever the frame, or less where a shell lies inside
    it or it lies inside one: the wall between them."""
    k = shells.max() + 1
    out = np.empty(k)
    for i in range(k):
        P = V[np.unique(F[shells == i])]
        try:
            hull = ConvexHull(P)
        except QhullError:  # a shell of no volume
            out[i] = np.ptp(P, axis=0).max()
            continue
        N = np.unique(np.round(hull.equations[:, :3], 12), axis=0)
        Q = P[hull.vertices]
        out[i] = min(
            np.ptp(Q @ N[a : a + 256].T, axis=0).min() for a in range(0, len(N), 256)
        )
    for i, j in zip(*np.nonzero(inside), strict=True):
        Ti, Tj = V[F[shells == i]], V[F[shells == j]]
        wall = min(_gap(Ti.reshape(-1, 3), Tj), _gap(Tj.reshape(-1, 3), Ti))
        out[i], out[j] = min(out[i], wall), min(out[j], wall)
    return out


def _gap(P, T):
    """The smallest distance from the points ``P`` (p, 3) to the triangles ``T``
    (t, 3, 3)."""
    P = np.unique(P, axis=0)
    best = np.inf
    step = max(1, 200_000 // len(P))
    for t0 in range(0, len(T), step):
        a, b, c = (T[t0 : t0 + step, k] for k in range(3))
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n, axis=1)[:, None]
        d = P[:, None] - a  # (p, k, 3)
        w = np.einsum("pkd,kd->pk", d, n)
        foot = P[:, None] - w[..., None] * n
        within = np.ones(w.shape, bool)
        dist = np.full(w.shape, np.inf)
        for u, v in ((a, b), (b, c), (c, a)):
            e = v - u
            within &= np.einsum("pkd,kd->pk", np.cross(e, foot - u), n) >= 0
            s = np.clip(
                np.einsum("pkd,kd->pk", P[:, None] - u, e) / np.sum(e * e, 1), 0, 1
            )
            dist = np.minimum(
                dist, np.linalg.norm(P[:, None] - u - s[..., None] * e, axis=-1)
            )
        best = min(best, np.where(within, np.abs(w), dist).min())
    return best


def _triangles(part):
    """A part's surface triangles (n, 3, 3), global: a triangle mesh's own, a
    cuboid's or a fixed cuboid's (an Outline) twelve."""
    if isinstance(part, TriBody):
        return part.to_global(part.vertices[part.faces].reshape(-1, 3)).reshape(
            -1, 3, 3
        )
    corners = (_SIGNS * part.half) @ part.R.T + part.position
    return corners[_BOX]


def _in_plane(T, point, normal, tol):
    """The triangles of ``T`` lying in the plane through ``point``; never one
    of no area."""
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=1)[:, None]
    return (np.abs(n @ normal) > 1 - 1e-9) & (
        np.abs((T.mean(1) - point) @ normal) < tol
    )


def add_triangle_contact_lines(bodies, outlines=()):
    """Grade where a part touches a triangle mesh, or a triangle mesh touches
    a cuboid, as add_contact_lines does between cuboids: a triangle mesh
    toward the outline of every touching part's faces in its own face planes,
    a cuboid toward the extent of a touching mesh's triangles in its face."""
    surfaces = [_triangles(b) for b in bodies]
    others = [(_triangles(o), o) for o in outlines]
    # each part's box: parts apart are not searched, a sphere's 1,280 face
    # planes against each other part's triangles
    boxes = [
        (T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0))
        for T in [*surfaces, *(T for T, _ in others)]
    ]
    for ia, a in enumerate(bodies):
        for ib, (Tb, b) in enumerate(
            [(surfaces[k], bodies[k]) for k in range(len(bodies))] + others
        ):
            if ib == ia or not (isinstance(a, TriBody) or isinstance(b, TriBody)):
                continue
            tol = contact_tol(a, b)
            (lo_a, hi_a), (lo_b, hi_b) = boxes[ia], boxes[ib]
            if np.any(lo_a > hi_b + tol) or np.any(lo_b > hi_a + tol):
                continue
            if isinstance(a, TriBody):
                _grade_mesh(a, surfaces[ia], Tb, tol)
            elif isinstance(a, Body):
                _grade_cuboid(a, Tb, tol)


def _grade_mesh(a, Ta, Tb, tol):
    """Lines in a's face planes where b's coplanar faces end: the edges of
    b's triangles in that plane that only one of them has."""
    normals = np.cross(Ta[:, 1] - Ta[:, 0], Ta[:, 2] - Ta[:, 0])
    length = np.linalg.norm(normals, axis=1)
    has_area = length > 0  # one of no area, as STL repairs leave, has no plane
    seen = set()
    for point, normal in zip(
        Ta[has_area, 0], normals[has_area] / length[has_area, None], strict=True
    ):
        key = (tuple(np.round(normal, 6) + 0.0), round(float(point @ normal) / tol))
        if key in seen:
            continue
        seen.add(key)
        flat = Tb[_in_plane(Tb, point, normal, tol)]
        mine = Ta[_in_plane(Ta, point, normal, tol)].reshape(-1, 3)
        if len(flat):  # only where b's faces overlap a's in that plane
            lo, hi = mine.min(0) - tol, mine.max(0) + tol
            flat = flat[(flat.min(1) <= hi).all(1) & (flat.max(1) >= lo).all(1)]
        if not flat.size:
            continue
        # edges found by their rounded ends, kept with their own: rounded to a
        # large part's tolerance, a line would leave a small part's plane
        edges = {}
        for tri, key in zip(flat, np.round(flat / tol).astype(np.int64), strict=True):
            for u, v in ((0, 1), (1, 2), (2, 0)):
                k = tuple(sorted((tuple(key[u]), tuple(key[v]))))
                count, ends = edges.get(k, (0, (tri[u], tri[v])))
                edges[k] = (count + 1, ends)
        for count, (p, q) in edges.values():
            if count != 1:
                continue
            t = q - p
            if np.linalg.norm(t) == 0:
                continue
            # into a's frame
            a.contacts.append(
                ((p - a.position) @ a.R, (t / np.linalg.norm(t)) @ a.R, normal @ a.R)
            )


def _grade_cuboid(a, Tb, tol):
    """Lines on the cuboid a's faces at the extent of b's triangles lying in
    them, along each in-plane axis."""
    T = ((Tb.reshape(-1, 3) - a.position) @ a.R).reshape(Tb.shape)  # a's frame
    for k in range(3):
        for f, side in ((2 * k, 1.0), (2 * k + 1, -1.0)):
            normal = np.zeros(3)
            normal[k] = side
            flat = T[_in_plane(T, a.half[k] * normal, normal, tol)]
            inside = np.delete(np.arange(3), k)  # the face's own axes
            if len(flat):  # only where they overlap the face
                flat = flat[
                    (flat[..., inside].min(1) <= a.half[inside] + tol).all(1)
                    & (flat[..., inside].max(1) >= -a.half[inside] - tol).all(1)
                ]
            if not flat.size:
                continue
            lo, hi = flat.reshape(-1, 3).min(0), flat.reshape(-1, 3).max(0)
            for axis in range(3):
                if axis != k:
                    a.lines[f][axis] = sorted(
                        set(a.lines[f][axis]) | {lo[axis], hi[axis]}
                    )
