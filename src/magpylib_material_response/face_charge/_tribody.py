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
from scipy.spatial import cKDTree

from magpylib_material_response.face_charge._bodies import Part
from magpylib_material_response.face_charge._mesh import graded_triangles
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import (
    frames,
    normal_field,
    normal_field_far,
)

CHUNK = 2_000_000  # kernel evaluations per call, about 0.5 GB
LAYER = 0.5  # graded layer length, in the mesh's smallest extent: half its thickness
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
        self.T = self.fr = self.mass = self.G = None
        self.fine = None  # triangles whose load takes a finer rule

    def mesh(self):
        """The triangles graded toward the sharp edges; the self block waits
        for ``assemble``, so that the unknowns can be counted first."""
        ell = LAYER * np.min(2 * self.half)
        self.T = graded_triangles(self.vertices, self.faces, ell, self.m, GRADING)
        self.fr = frames(self.T)
        # mass blocks, exact with the degree-2 rule
        P = np.einsum("qk,fkd->fqd", BARY3, self.T)
        phi = self.basis(P)
        self.mass = np.einsum("fq,fqa,fqb->fab", self.fr[5][:, None] * W3, phi, phi)
        self.G = None

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
        step = max(1, CHUNK // (nq * F))
        for j0 in range(0, F, step):
            js = np.arange(j0, min(j0 + step, F))
            flux = -normal_field_far(P[js].reshape(-1, 3), self.T, self.fr) / FOUR_PI
            G[:, :, js, :] = np.einsum(
                "ajqi,jq,jqb->iajb", flux.reshape(3, len(js), nq, F), W[js], B[js]
            )
        I, J = self.near_pairs()
        Pk, Wk, Bk = self.quad(1)
        Q = Pk.shape[1]
        kstep = max(1, 200_000 // Q)
        for k0 in range(0, len(I), kstep):
            i, j = I[k0 : k0 + kstep], J[k0 : k0 + kstep]
            ii = np.repeat(i, Q)
            fr = tuple(a[ii] for a in self.fr)
            En = normal_field(Pk[j].reshape(-1, 3), self.T[ii], fr, pairs=True)
            G[i, :, j, :] = np.einsum(
                "akq,kq,kqb->kab", -En.reshape(3, len(i), Q) / FOUR_PI, Wk[j], Bk[j]
            )
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
        keep = (
            (i != j)
            & (size[i] <= size[j])
            & (np.linalg.norm(cen[i] - cen[j], axis=1) < size[i] + size[j])
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
        """The net charge of each unknown: the area under its constant density."""
        e = np.zeros(self.n_dofs)
        e[0::3] = self.fr[5]
        return e

    # ── loads ──────────────────────────────────────────────────────────────
    def load(self, field):
        """``int psi H.n`` over each triangle; ``field`` maps points in the
        body's frame to H there, in that frame. Triangles marked in ``fine``,
        near a magnet that is not solved, take 16 x 7 points: its field turns
        there faster than 7 points resolve."""
        F = self._load(field, 0, np.arange(len(self.T)))
        if self.fine is not None and np.any(self.fine):
            idx = np.flatnonzero(self.fine)
            F[idx] = self._load(field, 2, idx)
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

    def test_flux(self, y):
        """(3F, P): the flux of a unit charge at each point ``y`` (its frame)
        through each test function."""
        flux = -normal_field_far(y, self.T, self.fr) / FOUR_PI  # (3, P, F)
        return flux.transpose(2, 0, 1).reshape(self.n_dofs, len(y))

    def coefficients_at_corners(self, c):
        """The charge at each triangle's corners (F, 3) from its coefficients."""
        cen, _, e1, e2, s, _ = self.fr
        d = self.T - cen[:, None]
        u = np.einsum("fkd,fd->fk", d, e1) / s[:, None]
        v = np.einsum("fkd,fd->fk", d, e2) / s[:, None]
        c = np.asarray(c, float).reshape(-1, 3)
        return c[:, :1] + c[:, 1:2] * u + c[:, 2:3] * v
