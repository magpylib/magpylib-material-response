"""A magpylib source for solved face-charge bodies: triangles whose magnetic
surface charge varies linearly across each one.

``LinearChargeTriangle`` holds one triangle or many (a whole solved body) and
computes their field in closed form (``triangles/tri.py``, checked to 1e-9). It is
a ``magpylib.misc.CustomSource``, so getB and getH, paths, collections and show()
work without any change to magpylib; a field function and class in magpylib core
could replace it later with the same parameters.

The charge is the magnetic surface charge density sigma = M.n in A/m, given at
each corner. The source carries the field of that charge: H everywhere, and
B = mu0 H outside the bodies. Inside a magnetized body B also holds the body's
magnetization, which the surface charge does not carry, so B there is not the
body's B (H is).
"""

import os
import sys

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "triangles"))
from tri import FOUR_PI, frames  # noqa: E402
from tri import field as _tri_field  # noqa: E402


class LinearChargeTriangle(magpy.misc.CustomSource):
    """Triangles with a magnetic surface charge that varies linearly across each.

    Parameters
    ----------
    vertices : array-like, shape (3, 3) or (n, 3, 3)
        Triangle corners in m, in the source's own frame.
    charge : array-like, shape (3,) or (n, 3)
        Surface charge density M.n at each corner, in A/m.
    position, orientation, style, **kwargs
        As for any magpylib source.
    """

    def __init__(self, vertices, charge, position=(0, 0, 0), orientation=None, style=None,
                 chunk=2_000_000, **kwargs):
        self._chunk = chunk
        self._set(vertices, charge)
        super().__init__(position=position, orientation=orientation, style=style, **kwargs)
        # The bound method, set past the field_func setter: magpylib's check reads
        # the arguments with getfullargspec, which counts self. A bound method, unlike
        # a closure, is rebound to the copy by copy(), and getB groups sources by it,
        # so each source is evaluated with its own charge.
        self._field_func = self._field
        self._draw()

    # ── geometry and charge ────────────────────────────────────────────────
    @property
    def vertices(self):
        """Corners (n, 3, 3) in m, in the source's own frame."""
        return self._V

    @vertices.setter
    def vertices(self, value):
        self._set(value, self._q)
        self._draw()

    @property
    def charge(self):
        """Surface charge M.n at the corners (n, 3), in A/m."""
        return self._q

    @charge.setter
    def charge(self, value):
        self._set(self._V, value)

    def _set(self, vertices, charge):
        V = np.asarray(vertices, float).reshape(-1, 3, 3)
        q = np.asarray(charge, float).reshape(-1, 3)
        if len(q) != len(V):
            msg = f"charge needs one value per corner, shape ({len(V)}, 3); got {np.shape(charge)}"
            raise ValueError(msg)
        self._V, self._q = V, q
        self._fr = frames(V)
        cen, _, e1, e2, s, _ = self._fr
        d = V - cen[:, None]
        A = np.stack([np.ones((len(V), 3)), np.einsum("nkd,nd->nk", d, e1) / s[:, None],
                      np.einsum("nkd,nd->nk", d, e2) / s[:, None]], axis=-1)  # (n, corner, basis)
        self._c = np.linalg.solve(A, q[..., None])[..., 0]  # charge = c0 + c1 u + c2 v

    # ── field ──────────────────────────────────────────────────────────────
    def _field(self, field, observers):
        X = np.asarray(observers, float).reshape(-1, 3)
        out = np.zeros((len(X), 3))
        if field in ("M", "J"):
            return out  # the source carries no magnetization of its own
        step = max(1, self._chunk // max(len(self._V), 1))
        for a in range(0, len(X), step):
            E = _tri_field(X[a : a + step], self._V, self._fr)  # (3, m, n, 3)
            out[a : a + step] = np.einsum("kmnd,nk->md", E, self._c) / FOUR_PI
        return out if field == "H" else magpy.mu_0 * out

    # ── display ────────────────────────────────────────────────────────────
    def _draw(self):
        V = self._V.reshape(-1, 3)
        trace = magpy.graphics.model3d.make_TriangularMesh(vertices=V, faces=np.arange(len(V)).reshape(-1, 3),
                                                            opacity=0.6)
        self.style.model3d.data = []
        self.style.model3d.showdefault = False
        self.style.model3d.add_trace(trace)

    # ── from the solvers ───────────────────────────────────────────────────
    @classmethod
    def from_triangles(cls, body, c, **kwargs):
        """The source of a solved TriBody, MacroBody or HBody (triangles/tri_solver.py)."""
        c = np.asarray(c, float)
        if hasattr(body, "fine") and c.size == body.Nm:
            c = body.fine(c)  # group coefficients to triangle ones
        c = c.reshape(body.F, 3)
        cen, _, e1, e2, s, _ = body.fr
        d = body.V - cen[:, None]
        u = np.einsum("nkd,nd->nk", d, e1) / s[:, None]
        v = np.einsum("nkd,nd->nk", d, e2) / s[:, None]
        q = c[:, :1] + c[:, 1:2] * u + c[:, 2:3] * v
        return cls(body.V, q, position=body.position, orientation=Rotation.from_matrix(body.R), **kwargs)

    @classmethod
    def from_cuboid(cls, body, c, **kwargs):
        """The source of a solved cuboid Body (multibody.py): each rectangular panel,
        whose charge is c0 + c1 s + c2 t across it (s, t from -1 to 1), as two
        triangles carrying the same linear charge."""
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
        return cls(V, q, position=body.position, orientation=Rotation.from_matrix(body.R), **kwargs)
