"""A magpylib source for a solved body: a triangle mesh carrying a magnetic
surface charge that varies linearly across each face."""

from __future__ import annotations

import itertools

import magpylib as magpy
import numpy as np
from magpylib._src.defaults.property_tree import Nested, Number, PropertyNode
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.style import TriangularMeshStyle
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial.transform import Rotation

from magpylib_material_response.face_charge._parallel import CHUNK, run, threads
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import (
    charge_field,
    charge_field_far,
    far_points,
    frames,
)

SATURATE = 0.95  # the colour saturates at the charge this share of the area stays below
# an observer this many of a sheet's longest triangle sides past all of it takes
# `charge_field_far`. Each observer chooses for the whole sheet, so a near one costs
# nothing: choosing per pair made fields 2 mm above six bars 1.3 times as slow
FAR_SIDES = 64.0


class ChargeColor(PropertyNode):
    """How a sheet's charge is coloured.

    Parameters
    ----------
    level : float, default=None
        The charge in A/m at which the colours are full. By default the charge
        95 % of the sheet's area stays below.
    """

    level = Number(minimum=0, doc="Charge in A/m at which the colours are full.")


class ChargeSheetStyle(TriangularMeshStyle):
    """A TriangularMesh's style, and how the charge is coloured (``charge``)."""

    charge = Nested(ChargeColor, doc="Charge colour properties.")


class ChargeSheet(magpy.misc.CustomSource):
    """Magnetic surface charge on triangular faces, linear across each face.

    Modelled on ``magpylib.current.TriangleSheet``: vertices, faces, and the
    charge at each face's corners, given per corner rather than per vertex
    because a solved charge jumps from one face to the next. The charge is the
    magnetic surface charge density ``sigma = M.n`` in A/m. Its field is in
    closed form, or from six points on each face for observers far from the
    sheet. The source gives H everywhere and ``B = mu0 H`` outside the
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
    mean_polarization : array-like, shape (3,), optional
        What :attr:`mean_polarization` gives, in T, in place of the sheet's own:
        :func:`apply_face_charge` sets it on parts solved as one.
    position, orientation, style, **kwargs
        As for any magpylib source. ``style`` is a TriangularMesh's:
        ``style.magnetization.color`` colours the charge and ``style.mesh.grid``
        draws the edges. ``style.charge.level`` is the charge in A/m at which
        the colours are full, by default the charge 95 % of the sheet's area
        stays below; :func:`apply_face_charge` gives the sheets it solves one
        level, so that their colours compare.
    """

    _style_class = ChargeSheetStyle
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
        *,
        mean_polarization=None,
        **kwargs,
    ):
        self._set(vertices, faces, charge_densities, panels)
        if mean_polarization is not None:
            self._mean = np.asarray(mean_polarization, float).reshape(3)
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
        self.style.charge.level = None

    @property
    def faces(self):
        """Vertex index triplets (m, 3)."""
        return self._faces

    @faces.setter
    def faces(self, value):
        self._set(self._vertices, value, self._q)  # new faces, so no panels
        self.style.charge.level = None

    @property
    def charge_densities(self):
        """Surface charge density at each face's corners (m, 3), in A/m.
        Read-only: set the whole array to change it."""
        q = self._q.view()
        q.flags.writeable = False
        return q

    @charge_densities.setter
    def charge_densities(self, value):
        self._set(self._vertices, self._faces, value, self._panels)
        self.style.charge.level = None  # the solve's, of the charge it had

    @property
    def panels(self):
        """The surface element (m,) each face belongs to, or None where each
        face is one."""
        return self._panels

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
        self._mean = None  # set by apply_face_charge on parts solved as one
        self._V = P[F]
        self._frames = frames(self._V)
        cen, _, e1, e2, s, _ = self._frames
        self._center, self._far_from, self._far_rule = np.zeros(3), np.inf, None
        if len(F):
            self._center = cen.mean(0)
            side = np.linalg.norm(self._V - self._V[:, [1, 2, 0]], axis=2).max()
            reach = np.linalg.norm(self._V - self._center, axis=2).max()
            self._far_from = reach + FAR_SIDES * side
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
        below the knee of the material's B-H curve. Of parts of one material
        that touch, solved as one, each sheet leaves out the faces they share,
        and gives the polarization of the solid they make.
        """
        if self._mean is not None:
            return self._mean.copy()
        T, q = self._V, self._q
        area = self._frames[5]
        # a linear charge times the linear r, integrated exactly over each face
        moment = np.einsum("f,fk,fkd->d", area / 12, q, T) + np.einsum(
            "f,f,fd->d", area / 12, q.sum(1), T.sum(1)
        )
        # the moment does not depend on the winding, the signed volume does
        volume = (
            abs(np.einsum("fd,fd->f", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()) / 6
        )
        return magpy.mu_0 * moment / volume

    @property
    def jump_indicator(self):
        """The largest jump of the charge between panels meeting at a point of
        a flat face, as a share of the peak charge, weighted by the panel's
        size: times ``sqrt(panel area / sheet area)``.

        On a flat face of a uniform body the charge is continuous, so the jump
        is discretization error, free to read off a solved sheet. Unweighted it
        stays large in the first panels at a soft body's edges, where the
        charge is singular and the graded panels tiny; weighted, it points
        where the panels are too coarse for the field around the body, and
        falls as the mesh is refined. It leaves out how far a curved part's
        flat facets are from its true surface: that error falls with the
        level, which sets the facets, not with ``refine``.
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
        group, face, q = self._corner_groups()
        area = self._frames[5]
        panel = (
            np.arange(len(self._faces))
            if self._panels is None
            else np.unique(self._panels, return_inverse=True)[1]
        )
        peak = np.abs(self._q).max(initial=0.0)
        if not peak:
            return np.zeros(panel.max(initial=-1) + 1), 0.0, float(area.sum())
        hi = np.full(group.max() + 1, -np.inf)
        lo = np.full(group.max() + 1, np.inf)
        np.maximum.at(hi, group, q)
        np.minimum.at(lo, group, q)
        per_face = np.zeros(len(self._faces))
        np.maximum.at(per_face, face, (hi - lo)[group])
        jump = np.zeros(panel.max() + 1)
        size = np.zeros(panel.max() + 1)
        np.maximum.at(jump, panel, per_face)
        np.add.at(size, panel, area)
        return jump * np.sqrt(size), peak, float(size.sum())

    def _corner_groups(self):
        """Each corner's group, face and charge: first the faces' corners
        (3m), then every vertex inside another face's edge, a T-junction of
        the graded mesh or of a refinement split, at that edge's charge there.
        A group is the corners at one vertex in one face plane: across a flat
        face the charge is continuous, so what differs there is discretization.
        Across a curved part's facets it jumps by the magnetization's change of
        normal, which no refinement removes."""
        P, F, q = self._vertices, self._faces, self._q
        a, b = P[F].reshape(-1, 3), P[np.roll(F, -1, axis=1)].reshape(-1, 3)
        d = b - a  # edge k of face k // 3, from its corner k % 3 to the next
        length = np.linalg.norm(d, axis=1)
        tol = 1e-8 * ((np.ptp(P, axis=0).max() if len(P) else 0.0) or 1.0)
        near = cKDTree(P).query_ball_point((a + b) / 2, length / 2 + tol)
        edge = np.repeat(np.arange(len(near)), [len(c) for c in near])
        vertex = np.fromiter(itertools.chain.from_iterable(near), int, len(edge))
        t = np.einsum("kd,kd->k", P[vertex] - a[edge], d[edge]) / length[edge] ** 2
        off = np.linalg.norm(P[vertex] - a[edge] - t[:, None] * d[edge], axis=1)
        along = t * length[edge]
        inside = (off <= tol) & (along > tol) & (along < length[edge] - tol)
        edge, vertex, t = edge[inside], vertex[inside], t[inside]
        start, end = q.ravel(), np.roll(q, -1, axis=1).ravel()
        point = np.concatenate([F.ravel(), vertex])
        face = np.concatenate([np.repeat(np.arange(len(F)), 3), edge // 3])
        charge = np.concatenate(
            [q.ravel(), start[edge] + t * (end[edge] - start[edge])]
        )
        key = np.concatenate(
            [point[:, None], np.round(self._frames[1][face], 6)], axis=1
        )
        return np.unique(key, axis=0, return_inverse=True)[1].ravel(), face, charge

    # ── field ─────────────────────────────────────────────────────────────
    def _field(self, field, observers):
        X = np.asarray(observers, float).reshape(-1, 3)
        out = np.zeros((len(X), 3))
        if field in ("M", "J"):
            return out  # the source carries no magnetization of its own
        step = max(1, CHUNK // (max(len(self._V), 1) * threads()))

        def near(rows):
            out[rows] = charge_field(X[rows], self._V, self._frames, self._c) / FOUR_PI

        def away(rows):
            out[rows] = charge_field_far(X[rows], *self._far_rule) / FOUR_PI

        with np.errstate(over="ignore"):  # past 1e154 m: inf, and so far
            far = np.linalg.norm(X - self._center, axis=1) >= self._far_from
        if far.any():
            if self._far_rule is None:
                self._far_rule = far_points(self._V, self._frames, self._c)
            jobs = [
                (kernel, rows[a : a + step])
                for kernel, rows in (
                    (near, np.flatnonzero(~far)),
                    (away, np.flatnonzero(far)),
                )
                for a in range(0, len(rows), step)
            ]
        else:
            jobs = [(near, slice(a, a + step)) for a in range(0, len(X), step)]
        run(lambda job: job[0](job[1]), jobs, work=len(X) * len(self._V))
        return out if field == "H" else magpy.mu_0 * out

    # ── display ───────────────────────────────────────────────────────────
    def get_trace(self, **kwargs):
        """The faces coloured by the charge in the object's magnet colours:
        positive charge is a north pole (plotly and pyvista; matplotlib draws
        the faces in the object's colour).

        The charge is drawn smoothed and on a linear scale. On a flat face of a
        uniform body the charge is continuous; its jumps between panels are
        discretization error, so each corner shows the area-weighted mean over
        the corners that share its vertex and its face's plane. It still jumps
        across the body's edges, where it truly does. The colour is full at the
        charge that 95 % of the area stays below, not at the peak, which sits
        in a singular corner, or at ``style.charge.level``; the sheets of one
        :func:`apply_face_charge` share that level, taken from the faces not
        pressed against another part, so their colours compare. Fields use the
        charge as solved.
        """
        from magpylib._src.display.traces_utility import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            line_dash,
        )

        if not self._faces.size:  # a part inside others, all its faces shared
            return []
        P = self._V.reshape(-1, 3)
        i, j, k = np.arange(len(P)).reshape(-1, 3).T
        q = self._smoothed()
        style = self.style
        full = style.charge.level
        if full is None:
            full = saturation([self])
        v = np.clip(q, -full, full)
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
            # from 0 to 1 exactly, as a magnet's: backends place it on the scale
            # by its own range
            lo, hi = v.min(), v.max()
            mesh |= {
                "intensity": (v - lo) / (hi - lo or 1),
                "colorscale": _colorscale(lo, hi, style.magnetization.color, full),
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
        group, face, _ = self._corner_groups()
        inv, w = group[: self._q.size], self._frames[5][face[: self._q.size]]
        total, weight = np.bincount(inv, w * self._q.ravel()), np.bincount(inv, w)
        # a group of points on edges alone has no weight here
        return np.divide(total, weight, out=np.zeros(len(total)), where=weight > 0)[inv]

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
        # none for a part inside parts of its material, all its faces shared
        span = (np.ptp(P, axis=0).max() if len(P) else 0.0) or 1.0
        key = np.round(P / (1e-9 * span)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return cls(P[first], inv.reshape(-1, 3), charge_densities, **kwargs)

    @classmethod
    def _from_body(cls, body, c, keep=None, **kwargs):
        """The sheet of a solved body: each rectangular panel, whose charge is
        ``c0 + c1 s + c2 t`` across it, as two faces carrying it; only the
        panels ``keep`` marks, if given."""
        p = body.panels
        idx = np.arange(p.N) if keep is None else np.flatnonzero(keep)
        c = np.asarray(c, float).reshape(p.N, 3)[idx]
        corners, normal = p.corners()[idx], p.normal[idx]
        s, t = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
        qc = c[:, :1] + c[:, 1:2] * s + c[:, 2:3] * t  # (N, 4) at the corners
        V = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
        q = np.concatenate([qc[:, [0, 1, 2]], qc[:, [0, 2, 3]]])
        n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        inward = np.einsum("fd,fd->f", n, np.concatenate([normal, normal])) < 0
        V[inward], q[inward] = V[inward][:, [0, 2, 1]], q[inward][:, [0, 2, 1]]
        return cls.from_triangles(
            V,
            q,
            panels=np.concatenate([idx, idx]),
            position=body.position,
            orientation=Rotation.from_matrix(body.R),
            **kwargs,
        )


def _samples(n=4):
    """Barycentric centroids of the n x n equal triangles a triangle splits
    into: points that sample a linear charge by area."""
    out = []
    for i in range(n):
        for j in range(n - i):
            out.append((i + 1 / 3, j + 1 / 3))
            if i + j < n - 1:
                out.append((i + 2 / 3, j + 2 / 3))
    a, b = np.array(out).T / n
    return np.column_stack([a, b, 1 - a - b])


SAMPLES = _samples()


def saturation(sheets, hidden=None):
    """The charge in A/m that ``SATURATE`` of the sheets' area stays below, by
    its smoothed value: where their colour is full. ``hidden``, per sheet, the
    faces left out. Where that share carries no charge, the peak.

    Sampled inside each face rather than at its corners: a corner weighted by
    its face's area lets a coarse panel's strong corner count over the whole
    panel, and the level moved threefold from one mesh level to the next."""
    q, w = [], []
    for k, sheet in enumerate(sheets):
        keep = np.ones(len(sheet.faces), bool) if hidden is None else ~hidden[k]
        q.append(np.abs(sheet._smoothed().reshape(-1, 3)[keep] @ SAMPLES.T).ravel())
        w.append(np.repeat(sheet._frames[5][keep], len(SAMPLES)))
    q, w = np.concatenate(q), np.concatenate(w)
    if not q.size:
        return 0.0
    order = np.argsort(q)
    cum = np.cumsum(w[order])
    level = q[order][np.searchsorted(cum, SATURATE * cum[-1])]
    return float(level or q.max())


def _colorscale(lo, hi, colors, full=None):
    """The magnet ``colors`` (north, middle, south) over the range ``[lo, hi]``:
    the middle colour at zero, the north (south) colour at ``full``, by default
    the largest value of either sign. A tuple, as pyvista caches it."""
    from matplotlib.colors import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        to_hex,
        to_rgb,
    )

    middle = np.array(to_rgb(colors.middle))
    a = max(-lo, hi) if full is None else full
    if a <= 0:
        return ((0.0, to_hex(middle)), (1.0, to_hex(middle)))

    def at(v):
        end = np.array(to_rgb(colors.north if v > 0 else colors.south))
        return to_hex(middle + abs(v) / a * (end - middle))

    stops = [(0.0, at(lo))]
    if lo < 0 < hi:
        stops.append((-lo / (hi - lo), to_hex(middle)))
    return (*stops, (1.0, at(hi)))
