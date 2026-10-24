"""A magpylib source for solved face-charge bodies: a triangle mesh carrying a
magnetic surface charge that varies linearly across each face.

``ChargeSheet`` is modelled on ``magpylib.current.TriangleSheet``: vertices,
faces, and the charge at each face's corners. Its field is in closed form
(``triangles/tri.py``; ``t_source.py`` checks it against magpylib's Triangle).
It is a ``magpylib.misc.CustomSource``, so getB and getH, paths, collections and
show() work with magpylib as it is; a field function and class in magpylib core
could replace it later with the same parameters.

The charge is the magnetic surface charge density sigma = M.n in A/m. It is
given per face corner, not per vertex, because the solved charge jumps from one
face to the next. The source carries the field of that charge: H everywhere,
and B = mu0 H outside the bodies. Inside a magnetized body B also holds the
body's magnetization, which the surface charge does not carry, so B there is
not the body's B (H is).
"""

import os
import sys

import magpylib as magpy
import numpy as np
from magpylib._src.exceptions import MagpylibBadUserInput  # not exported publicly
from matplotlib.colors import to_hex, to_rgb
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "triangles"))
from tri import FOUR_PI, frames  # noqa: E402
from tri import field as _tri_field  # noqa: E402


class ChargeSheet(magpy.misc.CustomSource):
    """Magnetic surface charge on triangular faces, linear across each face.

    Parameters
    ----------
    vertices : array-like, shape (n, 3)
        Mesh points in m, in the source's own frame.
    faces : array-like, shape (m, 3)
        Triplets of vertex indices, one per triangle.
    charge_densities : array-like, shape (m, 3)
        Surface charge density M.n in A/m at each face's three corners, in the
        order of its vertex indices.
    position, orientation, style, **kwargs
        As for any magpylib source.
    """

    def __init__(self, vertices, faces, charge_densities, position=(0, 0, 0), orientation=None,
                 style=None, chunk=2_000_000, **kwargs):
        self._chunk = chunk
        self._set(vertices, faces, charge_densities)
        super().__init__(position=position, orientation=orientation, style=style, **kwargs)
        # A bound method, unlike a closure, follows copy() to the copy, and getB
        # groups sources by it, so each source keeps its own charge. magpylib after
        # 6.0.0 accepts it (#1045); 6.0.0 reads the arguments with getfullargspec,
        # which counts self, so there it is set past the setter. 6.0.0 also caches
        # its in_out check on the field function, which keeps the source alive for
        # the session; #1054 fixes that too.
        try:
            self.field_func = self._field
        except MagpylibBadUserInput:
            self._field_func = self._field

    # ── mesh and charge ────────────────────────────────────────────────────
    @property
    def vertices(self):
        """Mesh points (n, 3) in m, in the source's own frame."""
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._set(value, self._faces, self._q)

    @property
    def faces(self):
        """Vertex index triplets (m, 3)."""
        return self._faces

    @faces.setter
    def faces(self, value):
        self._set(self._vertices, value, self._q)

    @property
    def charge_densities(self):
        """Surface charge M.n at each face's corners (m, 3), in A/m."""
        return self._q

    @charge_densities.setter
    def charge_densities(self, value):
        self._set(self._vertices, self._faces, value)

    @property
    def triangles(self):
        """Corners of each face (m, 3, 3) in m, in the source's own frame."""
        return self._V

    def _set(self, vertices, faces, charge_densities):
        P = np.asarray(vertices, float).reshape(-1, 3)
        F = np.asarray(faces, int).reshape(-1, 3)
        q = np.asarray(charge_densities, float).reshape(-1, 3)
        if len(q) != len(F):
            msg = f"charge_densities needs one value per face corner, shape ({len(F)}, 3); got {np.shape(charge_densities)}"
            raise ValueError(msg)
        if F.size and (F.min() < 0 or F.max() >= len(P)):
            msg = f"faces index vertices 0 to {len(P) - 1}; got {F.min()} to {F.max()}"
            raise ValueError(msg)
        self._vertices, self._faces, self._q = P, F, q
        self._V = P[F]  # (m, 3, 3)
        self._fr = frames(self._V)
        cen, _, e1, e2, s, _ = self._fr
        d = self._V - cen[:, None]
        A = np.stack([np.ones((len(F), 3)), np.einsum("nkd,nd->nk", d, e1) / s[:, None],
                      np.einsum("nkd,nd->nk", d, e2) / s[:, None]], axis=-1)  # (m, corner, basis)
        self._c = np.linalg.solve(A, q[..., None])[..., 0]  # charge = c0 + c1 u + c2 v

    # ── field ──────────────────────────────────────────────────────────────
    def _field(self, field, observers):
        X = np.asarray(observers, float).reshape(-1, 3)
        out = np.zeros((len(X), 3))
        if field in ("M", "J"):
            return out  # the source carries no magnetization of its own
        step = max(1, self._chunk // max(len(self._V), 1))
        for a in range(0, len(X), step):
            E = _tri_field(X[a : a + step], self._V, self._fr)  # (3, k, m, 3)
            out[a : a + step] = np.einsum("kxnd,nk->xd", E, self._c) / FOUR_PI
        return out if field == "H" else magpy.mu_0 * out

    # ── display ───────────────────────────────────────────────────────────
    def get_trace(self, **kwargs):
        """The faces coloured by the charge, linear across each one, in
        magpylib's magnet colours: positive charge is a north pole (plotly and
        pyvista; matplotlib draws the faces in the object's colour). Corners are
        drawn per face, as the charge jumps from one face to the next."""
        P = self._V.reshape(-1, 3)
        i, j, k = np.arange(len(P)).reshape(-1, 3).T
        q = self._q.ravel()
        trace = {"type": "mesh3d", "x": P[:, 0], "y": P[:, 1], "z": P[:, 2], "i": i, "j": j, "k": k,
                 "color": self.style.color, "intensity": q,
                 "colorscale": _colorscale(q.min(initial=0.0), q.max(initial=0.0)), "showscale": False}
        return {**trace, **kwargs}

    # ── constructors ───────────────────────────────────────────────────────
    @classmethod
    def from_triangles(cls, triangles, charge_densities, **kwargs):
        """From corners (m, 3, 3) and their charges (m, 3): coincident corners
        become one vertex."""
        T = np.asarray(triangles, float).reshape(-1, 3, 3)
        P = T.reshape(-1, 3)
        span = np.ptp(P, axis=0).max() or 1.0
        key = np.round(P / (1e-9 * span)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return cls(P[first], inv.reshape(-1, 3), charge_densities, **kwargs)

    @classmethod
    def from_cuboid(cls, body, c, **kwargs):
        """The source of a solved cuboid Body (multibody.py): each rectangular panel,
        whose charge is c0 + c1 s + c2 t across it (s, t from -1 to 1), as two
        faces carrying the same linear charge."""
        p = body.pan
        c = np.asarray(c, float).reshape(p.N, 3)
        idx = np.arange(p.N)
        corners = np.zeros((p.N, 4, 3))
        for r, (pp, qq) in enumerate(((p.p1, p.q1), (p.p2, p.q1), (p.p2, p.q2), (p.p1, p.q2))):
            corners[idx, r, p.i] = pp
            corners[idx, r, p.j] = qq
            corners[idx, r, p.k] = p.w0
        s, t = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
        qc = c[:, :1] + c[:, 1:2] * s + c[:, 2:3] * t  # (N, 4) at the corners
        V = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
        q = np.concatenate([qc[:, [0, 1, 2]], qc[:, [0, 2, 3]]])
        n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        flip = np.einsum("fd,fd->f", n, np.concatenate([p.normal, p.normal])) < 0  # outward, for display
        V[flip], q[flip] = V[flip][:, [0, 2, 1]], q[flip][:, [0, 2, 1]]
        return cls.from_triangles(V, q, position=body.position, orientation=Rotation.from_matrix(body.R), **kwargs)


def _colorscale(lo, hi):
    """magpylib's magnet colours over the charge range [lo, hi]: the middle
    colour at zero, the north (south) colour at the largest charge of either
    sign. Built for the range, as pyvista takes no colour limits; a tuple, as
    pyvista caches it."""
    c = magpy.defaults.display.style.magnet.magnetization.color
    middle = np.array(to_rgb(c.middle))
    a = max(-lo, hi)
    if a <= 0:
        return ((0.0, to_hex(middle)), (1.0, to_hex(middle)))

    def at(v):
        end = np.array(to_rgb(c.north if v > 0 else c.south))
        return to_hex(middle + abs(v) / a * (end - middle))

    stops = [(0.0, at(lo))]
    if lo < 0 < hi:
        stops.append((-lo / (hi - lo), to_hex(middle)))
    return (*stops, (1.0, at(hi)))
