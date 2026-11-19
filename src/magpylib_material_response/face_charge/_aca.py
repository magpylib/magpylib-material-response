"""Coupling blocks between triangle bodies apart, by adaptive cross approximation.

Between two separate parts, ``G_ab`` is numerically of low rank: on the pole
pieces of the docs, two discs of 5,724 unknowns each at level 2, rank 353
holds it to 1e-8. Partially pivoted ACA (Bebendorf 2000) builds it from that
many of its rows and columns instead of all its entries: 4.5 times faster than
the dense block there, 11 times between two cylinders their own size apart,
still 1.2 times when they are 1 mm apart (5 % of their size).

Its stopping test, two terms in a row small against the estimate of the
block's norm, ends above the asked tolerance (with one term, about ten times
above), so it is asked for ``EPS``, 1e-8: every block measured stayed within
2e-7 of the dense one, the fields of whole solves within 1e-8 of the dense
solve's. A block keeps the dense path
where ACA does not pay or does not hold: parts under ``MIN_UNKNOWNS`` (at 1,080
unknowns ACA was up to 2 times slower), parts closer than ``ADMISSIBLE`` of the
smaller one's size (where faces touch, an exact contact term takes over), and a
rank that passes ``MAX_SHARE`` of the block (the wins went to 32 %, the losses
began at 38 %).

- Bebendorf, M. (2000). Approximation of boundary element matrices.
  Numerische Mathematik, 86(4), 565-589.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from magpylib_material_response.face_charge._contact import coplanar_pairs
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import normal_field_far
from magpylib_material_response.face_charge._tribody import TriBody

#: the tolerance asked of ACA, ten times below the error it ends at
EPS = 1e-8
#: parts closer than this share of the smaller one's size keep the dense block:
#: the closest gap measured, well above the contact tolerances (1e-4, 1e-3)
ADMISSIBLE = 0.05
#: parts with fewer unknowns keep the dense block: ACA does not pay there
MIN_UNKNOWNS = 2000
#: a block whose rank passes this share of its smaller side is built dense
MAX_SHARE = 1 / 3
#: source panels whose three columns are kept, the latest used
CACHED = 16


def admissible(a, b):
    """Whether ``G_ab`` is worth approximating: two triangle bodies, both large
    enough, apart by at least ``ADMISSIBLE`` of the smaller one's size."""
    if not (isinstance(a, TriBody) and isinstance(b, TriBody)):
        return False
    if min(a.n_dofs, b.n_dofs) < MIN_UNKNOWNS:
        return False
    # between the corners of the graded triangles, which cover every face; the
    # input vertices alone can lie only on a face's rims
    corners_a = a.to_global(a.T.reshape(-1, 3))
    corners_b = b.to_global(b.T.reshape(-1, 3))
    gap = cKDTree(corners_a).query(corners_b)[0].min()
    size = 2 * min(np.max(a.half), np.max(b.half))
    return bool(gap >= ADMISSIBLE * size)


class _Entries:
    """The rows and columns of ``G_ab`` one at a time, as
    :func:`~._bodies.coupling_block` builds the whole block: b's quadrature
    points, their charge per unknown, and the flux through a's test functions."""

    def __init__(self, a, b):
        self.a = a
        points, weights = b.sources()
        self.y = (points - a.position) @ a.R  # b's points in a's frame
        self.by_point = weights.tocsc()
        self.by_unknown = weights.tocsr()
        self.per_panel = len(self.y) // (b.n_dofs // 3)
        c = a.centre  # as TriBody.far_field: about the bounding box's centre
        self.yc, self.Tc = self.y - c, a.T - c
        self.frc = (a.fr[0] - c, *a.fr[1:])
        self.panels = OrderedDict()

    def column(self, j):
        """Column ``j``: the flux of one source unknown through every test
        function. Its panel's three columns come at once, and the latest
        ``CACHED`` panels' are kept: pivots return to a panel often, though
        seldom at once, and the cache stays bounded."""
        p, k = divmod(j, 3)
        if p in self.panels:
            self.panels.move_to_end(p)
        else:
            rows = slice(p * self.per_panel, (p + 1) * self.per_panel)
            flux = self.a.test_flux(self.y[rows])
            self.panels[p] = flux @ self.by_unknown[rows, 3 * p : 3 * p + 3].toarray()
            if len(self.panels) > CACHED:
                self.panels.popitem(last=False)
        return self.panels[p][:, k]

    def row(self, i):
        """Row ``i``: the flux of every source point through one test function."""
        t, k = divmod(i, 3)
        fr = tuple(f[[t]] for f in self.frc)
        flux = -normal_field_far(self.yc, self.Tc[[t]], fr)[k, :, 0] / FOUR_PI
        return self.by_point.T @ flux


def low_rank(entries, m, n, eps=EPS, max_rank=None):
    """``U (m, r)``, ``V (r, n)`` with ``U @ V`` the block to about ``eps`` by
    partially pivoted ACA, or None if that needs more than ``max_rank`` terms.
    It stops once two terms in a row fall below ``eps`` of the estimated norm,
    a guard against one small term ending it early. The terms live in arrays,
    so that each new row, column and norm update is one product with all the
    terms so far."""
    max_rank = min(m, n) if max_rank is None else min(max_rank, m, n)
    room = min(64, max_rank)  # grown by doubling: the rank is seldom near the cap
    U, V = np.empty((m, room)), np.empty((room, n))
    norm2, scale, small = 0.0, 0.0, 0
    used = np.zeros(m, bool)
    i, k = 0, 0
    while True:
        used[i] = True
        raw = entries.row(i)
        scale = max(scale, float(np.abs(raw).max()))
        r = raw - U[i, :k] @ V[:k]
        j = int(np.argmax(np.abs(r)))
        if abs(r[j]) <= 1e-14 * scale:  # a row the terms already hold: another
            free = np.flatnonzero(~used)
            if not free.size:  # every row held: the block is exact
                return U[:, :k], V[:k]
            i = int(free[0])
            continue
        if k == max_rank:
            return None
        v = r / r[j]
        c = entries.column(j) - U[:, :k] @ V[:k, j]
        term2 = (c @ c) * (v @ v)
        norm2 += term2 + 2 * float((U[:, :k].T @ c) @ (V[:k] @ v))
        if k == room:
            room = min(2 * room, max_rank)
            U = np.concatenate([U, np.empty((m, room - k))], axis=1)
            V = np.concatenate([V, np.empty((room - k, n))])
        U[:, k], V[k] = c, v
        k += 1
        small = small + 1 if term2 <= eps**2 * norm2 else 0
        if small == 2:
            return U[:, :k], V[:k]
        largest = np.abs(c)
        largest[used] = -1.0
        if largest.max() < 0:  # every row used
            return U[:, :k], V[:k]
        i = int(np.argmax(largest))


def coupling_by_aca(a, b):
    """``G_ab`` by ACA where :func:`admissible` and no faces of the two lie in
    one plane, as a dense array; None where the dense block should be built
    instead. ACA holds the far-field part only: a pair with overlapping faces in
    one plane keeps the dense block and its exact contact term."""
    if not admissible(a, b):
        return None
    from magpylib_material_response.face_charge._bodies import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        plane_tol,
    )

    # faces of the two in one plane that overlap: the dense path takes their
    # exact contact term there
    Pa, Na = a.panel_polygons()
    Pb, Nb = b.panel_polygons()
    if coplanar_pairs(Pa, Na, Pb, Nb, plane_tol(a, b))[0].size:
        return None
    m, n = a.n_dofs, b.n_dofs
    factors = low_rank(_Entries(a, b), m, n, max_rank=int(MAX_SHARE * min(m, n)))
    if factors is None:
        return None
    U, V = factors
    return U @ V
