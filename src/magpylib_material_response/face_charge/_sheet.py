"""A magpylib source for a solved body: a triangle mesh carrying a magnetic
surface charge that varies linearly across each face."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.style import TriangularMeshStyle
from scipy.spatial.transform import Rotation

from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import field as triangle_field
from magpylib_material_response.face_charge._triangle import frames

CHUNK = 2_000_000  # kernel evaluations per call, about 0.5 GB


class ChargeSheet(magpy.misc.CustomSource):
    """Magnetic surface charge on triangular faces, linear across each face.

    Modelled on ``magpylib.current.TriangleSheet``: vertices, faces, and the
    charge at each face's corners, given per corner rather than per vertex
    because a solved charge jumps from one face to the next. The charge is the
    magnetic surface charge density ``sigma = M.n`` in A/m. Its field is in
    closed form. The source gives H everywhere and ``B = mu0 H`` outside the
    body; inside a magnetized body B also holds the magnetization, which the
    surface charge does not carry.

    Parameters
    ----------
    vertices : array-like, shape (n, 3)
        Mesh points in m, in the source's own frame.
    faces : array-like, shape (m, 3)
        Triplets of vertex indices, one per triangle.
    charge_densities : array-like, shape (m, 3)
        Surface charge density in A/m at each face's three corners, in the
        order of its vertex indices.
    panels : array-like, shape (m,), optional
        The surface element each face belongs to, for drawing only:
        ``style.mesh.grid`` draws the edges between elements. A solved cuboid's
        rectangular panels are two faces each. By default each face is one.
    position, orientation, style, **kwargs
        As for any magpylib source. ``style`` is a TriangularMesh's:
        ``style.magnetization.color`` colours the charge and ``style.mesh.grid``
        draws the edges.
    """

    _style_class = TriangularMeshStyle
    _style_family = ("magnet", "triangularmesh")

    def __init__(
        self,
        vertices,
        faces,
        charge_densities,
        panels=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        self._set(vertices, faces, charge_densities, panels)
        super().__init__(
            position=position, orientation=orientation, style=style, **kwargs
        )
        # A bound method, unlike a closure, follows copy() to the copy, and getB
        # groups sources by it, so each source keeps its own charge. magpylib
        # 6.0.0 reads the arguments with getfullargspec, which counts self, so
        # there it is set past the setter (fixed after 6.0.0, magpylib#1045).
        try:
            self.field_func = self._field
        except MagpylibBadUserInput:
            self._field_func = self._field

    # ── mesh and charge ───────────────────────────────────────────────────
    @property
    def vertices(self):
        """Mesh points (n, 3) in m, in the source's own frame."""
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._set(value, self._faces, self._q, self._panels)

    @property
    def faces(self):
        """Vertex index triplets (m, 3)."""
        return self._faces

    @faces.setter
    def faces(self, value):
        self._set(self._vertices, value, self._q)  # new faces, so no panels

    @property
    def charge_densities(self):
        """Surface charge density at each face's corners (m, 3), in A/m."""
        return self._q

    @charge_densities.setter
    def charge_densities(self, value):
        self._set(self._vertices, self._faces, value, self._panels)

    @property
    def triangles(self):
        """Corners of each face (m, 3, 3) in m, in the source's own frame."""
        return self._V

    def _set(self, vertices, faces, charge_densities, panels=None):
        P = np.asarray(vertices, float).reshape(-1, 3)
        F = np.asarray(faces, int).reshape(-1, 3)
        q = np.asarray(charge_densities, float).reshape(-1, 3)
        if len(q) != len(F):
            msg = (
                f"charge_densities needs one value per face corner, shape ({len(F)}, 3); "
                f"got {np.shape(charge_densities)}."
            )
            raise ValueError(msg)
        if F.size and (F.min() < 0 or F.max() >= len(P)):
            msg = f"faces index vertices 0 to {len(P) - 1}; got {F.min()} to {F.max()}."
            raise ValueError(msg)
        if panels is not None:
            panels = np.asarray(panels, int).reshape(-1)
            if len(panels) != len(F):
                msg = f"panels needs one element per face, shape ({len(F)},); got {np.shape(panels)}."
                raise ValueError(msg)
        self._vertices, self._faces, self._q, self._panels = P, F, q, panels
        self._V = P[F]
        self._frames = frames(self._V)
        cen, _, e1, e2, s, _ = self._frames
        d = self._V - cen[:, None]
        A = np.stack(
            [
                np.ones((len(F), 3)),
                np.einsum("nkd,nd->nk", d, e1) / s[:, None],
                np.einsum("nkd,nd->nk", d, e2) / s[:, None],
            ],
            axis=-1,
        )  # (face, corner, basis)
        self._c = np.linalg.solve(A, q[..., None])[..., 0]  # charge = c0 + c1 u + c2 v

    # ── what a solve tells ────────────────────────────────────────────────
    @property
    def mean_polarization(self):
        """Average polarization (3,) in T of the body a closed sheet encloses,
        in the source's own frame as magpylib gives a magnet's: ``mu0`` times
        the moment of its charge (the integral of ``sigma r``) over its volume.

        A first pass for saturation: the linear solve holds while this stays
        below the knee of the material's B-H curve.
        """
        T, q = self._V, self._q
        area = (
            np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
        )
        # a linear charge times the linear r, integrated exactly over each face
        moment = np.einsum("f,fk,fkd->d", area / 12, q, T) + np.einsum(
            "f,f,fd->d", area / 12, q.sum(1), T.sum(1)
        )
        volume = np.einsum("fd,fd->f", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6
        return magpy.mu_0 * moment / volume

    @property
    def jump_indicator(self):
        """The largest jump of the charge between panels meeting at a corner of
        a flat face, as a share of the peak charge, weighted by the panel's
        size: times ``sqrt(panel area / sheet area)``.

        On a flat face of a uniform body the charge is continuous, so the jump
        is discretization error, free to read off a solved sheet. Unweighted it
        stays large in the first panels at a soft body's edges, where the
        charge is singular and the graded panels tiny; weighted, it points
        where the panels are too coarse for the field around the body, and
        falls as the mesh is refined.
        """
        return float(self.panel_indicator().max(initial=0.0))

    def panel_indicator(self):
        """:attr:`jump_indicator` per panel, in the order of the panel ids (per
        face when the sheet has no panels)."""
        weighted, peak, total = self._panel_jumps()
        if not peak:
            return weighted
        return weighted / (peak * np.sqrt(total))

    def _panel_jumps(self):
        """Per panel, the largest jump of the charge at its corners times the
        square root of its area, in A: comparable between sheets. Also the
        peak charge and the sheet's area."""
        inv, area = self._corner_groups()
        q = self._q.ravel()
        panel = (
            np.arange(len(self._faces))
            if self._panels is None
            else np.unique(self._panels, return_inverse=True)[1]
        )
        peak = np.abs(q).max(initial=0.0)
        if not peak:
            return np.zeros(panel.max(initial=-1) + 1), 0.0, float(area.sum() / 3)
        hi = np.full(inv.max() + 1, -np.inf)
        lo = np.full(inv.max() + 1, np.inf)
        np.maximum.at(hi, inv, q)
        np.minimum.at(lo, inv, q)
        per_face = (hi - lo)[inv].reshape(-1, 3).max(1)
        jump = np.zeros(panel.max() + 1)
        size = np.zeros(panel.max() + 1)
        np.maximum.at(jump, panel, per_face)
        np.add.at(size, panel, area[::3])
        return jump * np.sqrt(size), peak, float(size.sum())

    def _corner_groups(self):
        """The group of each face corner (3m,), corners sharing a vertex and a
        face plane, and each corner's face area (3m,)."""
        n = np.cross(self._V[:, 1] - self._V[:, 0], self._V[:, 2] - self._V[:, 0])
        area = np.linalg.norm(n, axis=1) / 2
        n = np.round(n / (2 * area[:, None]), 6)
        key = np.concatenate(
            [self._faces.reshape(-1, 1), np.repeat(n, 3, axis=0)], axis=1
        )
        return np.unique(key, axis=0, return_inverse=True)[1].ravel(), np.repeat(
            area, 3
        )

    # ── field ─────────────────────────────────────────────────────────────
    def _field(self, field, observers):
        X = np.asarray(observers, float).reshape(-1, 3)
        out = np.zeros((len(X), 3))
        if field in ("M", "J"):
            return out  # the source carries no magnetization of its own
        step = max(1, CHUNK // max(len(self._V), 1))
        for a in range(0, len(X), step):
            E = triangle_field(X[a : a + step], self._V, self._frames)  # (3, k, m, 3)
            out[a : a + step] = np.einsum("kxnd,nk->xd", E, self._c) / FOUR_PI
        return out if field == "H" else magpy.mu_0 * out

    # ── display ───────────────────────────────────────────────────────────
    def get_trace(self, **kwargs):
        """The faces coloured by the charge in the object's magnet colours:
        positive charge is a north pole (plotly and pyvista; matplotlib draws
        the faces in the object's colour).

        The charge is drawn smoothed and on a square-root scale. On a flat face
        of a uniform body the charge is continuous; its jumps between panels are
        discretization error, so each corner shows the area-weighted mean over
        the corners that share its vertex and its face's plane. It still jumps
        across the body's edges, where it truly does. The square root shows the
        weak charge where flux returns into a soft part next to the strong
        charge where it leaves. Fields use the charge as solved.
        """
        from magpylib._src.display.traces_utility import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            line_dash,
        )

        P = self._V.reshape(-1, 3)
        i, j, k = np.arange(len(P)).reshape(-1, 3).T
        q = self._smoothed()
        v = np.sign(q) * np.sqrt(np.abs(q))
        style = self.style
        mesh = {
            "type": "mesh3d",
            "x": P[:, 0],
            "y": P[:, 1],
            "z": P[:, 2],
            "i": i,
            "j": j,
            "k": k,
            "color": style.color,
        }
        if style.magnetization.show:
            # the range is pinned to include zero, the colour scale's middle
            lo, hi = v.min(initial=0.0), v.max(initial=0.0)
            mesh |= {
                "intensity": v,
                "colorscale": _colorscale(lo, hi, style.magnetization.color),
                "cmin": lo,
                "cmax": hi,
                "showscale": False,
            }
        traces = [{**mesh, **kwargs}]
        grid = style.mesh.grid
        if grid.show:
            x, y, z = (
                np.insert(self._vertices[self._edges()], 2, np.nan, axis=1)
                .reshape(-1, 3)
                .T
            )
            traces.append(
                {
                    **{key: val for key, val in kwargs.items() if key != "color"},
                    "type": "scatter3d",
                    "x": x,
                    "y": y,
                    "z": z,
                    "mode": "markers+lines",
                    "marker_color": grid.marker.color,
                    "marker_size": grid.marker.size,
                    "marker_symbol": grid.marker.symbol,
                    "line_color": grid.line.color,
                    "line_width": grid.line.width,
                    "line_dash": line_dash(grid.line.style),
                    "showlegend": False,
                }
            )
        return traces

    def _smoothed(self):
        """Corner charges (3m,) averaged, by face area, over the corners that
        share a vertex and a face plane."""
        inv, w = self._corner_groups()
        return (np.bincount(inv, w * self._q.ravel()) / np.bincount(inv, w))[inv]

    def _edges(self):
        """Vertex pairs (k, 2) of the edges between surface elements: an edge
        all of whose faces belong to one panel is inside it, and not drawn."""
        F = self._faces
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        edges, inv = np.unique(e, axis=0, return_inverse=True)
        if self._panels is None:
            return edges
        inv, owner = inv.ravel(), np.tile(self._panels, 3)
        lo = np.full(len(edges), np.iinfo(np.int64).max)
        hi = np.full(len(edges), np.iinfo(np.int64).min)
        np.minimum.at(lo, inv, owner)
        np.maximum.at(hi, inv, owner)
        return edges[(lo != hi) | (np.bincount(inv) == 1)]

    # ── constructors ──────────────────────────────────────────────────────
    @classmethod
    def from_triangles(cls, triangles, charge_densities, **kwargs):
        """A sheet from triangle corners (m, 3, 3) and the charge at them
        (m, 3), in A/m; coincident corners become one vertex."""
        T = np.asarray(triangles, float).reshape(-1, 3, 3)
        P = T.reshape(-1, 3)
        span = np.ptp(P, axis=0).max() or 1.0
        key = np.round(P / (1e-9 * span)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return cls(P[first], inv.reshape(-1, 3), charge_densities, **kwargs)

    @classmethod
    def _from_body(cls, body, c, **kwargs):
        """The sheet of a solved body: each rectangular panel, whose charge is
        ``c0 + c1 s + c2 t`` across it, as two faces carrying it."""
        p = body.panels
        c = np.asarray(c, float).reshape(p.N, 3)
        corners = p.corners()
        s, t = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
        qc = c[:, :1] + c[:, 1:2] * s + c[:, 2:3] * t  # (N, 4) at the corners
        V = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
        q = np.concatenate([qc[:, [0, 1, 2]], qc[:, [0, 2, 3]]])
        n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        inward = np.einsum("fd,fd->f", n, np.concatenate([p.normal, p.normal])) < 0
        V[inward], q[inward] = V[inward][:, [0, 2, 1]], q[inward][:, [0, 2, 1]]
        idx = np.arange(p.N)
        return cls.from_triangles(
            V,
            q,
            panels=np.concatenate([idx, idx]),
            position=body.position,
            orientation=Rotation.from_matrix(body.R),
            **kwargs,
        )


def _colorscale(lo, hi, colors):
    """The magnet ``colors`` (north, middle, south) over the range ``[lo, hi]``:
    the middle colour at zero, the north (south) colour at the largest value of
    either sign. Built for the range, as pyvista takes no colour limits; a
    tuple, as pyvista caches it."""
    from matplotlib.colors import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        to_hex,
        to_rgb,
    )

    middle = np.array(to_rgb(colors.middle))
    a = max(-lo, hi)
    if a <= 0:
        return ((0.0, to_hex(middle)), (1.0, to_hex(middle)))

    def at(v):
        end = np.array(to_rgb(colors.north if v > 0 else colors.south))
        return to_hex(middle + abs(v) / a * (end - middle))

    stops = [(0.0, at(lo))]
    if lo < 0 < hi:
        stops.append((-lo / (hi - lo), to_hex(middle)))
    return (*stops, (1.0, at(hi)))
