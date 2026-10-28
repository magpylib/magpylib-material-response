"""The magnets of a magpylib model as OpenCASCADE solids for netgen.

Solids are built in scaled coordinates, ``(x - center) / length``, so that the
mesher works on numbers near one whatever the model's size.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import magpylib as magpy
import numpy as np
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent, BaseMagnet
from netgen import occ

from magpylib_material_response.demag import (
    _cell_susceptibilities,
    _has_path,
    get_H_ext,
    get_susceptibilities,
)
from magpylib_material_response.meshing import _thinnest as _thinnest_part
from magpylib_material_response.meshing_utils import SOLIDS, mask_inside

#: More sharp edges than this meeting at a corner break netgen's geometric
#: refinement (an icosahedron, five at each corner, fails to assemble, and finer
#: triangle meshes crash); such corners and their edges get no edge layers.
MAX_EDGES_AT_CORNER = 4


@dataclass(frozen=True)
class Part:
    """One magnet of the model as the finite elements see it."""

    source: BaseMagnet
    #: Susceptibility tensor (3, 3) in the global frame.
    chi: np.ndarray
    #: Remanent polarization (3,) in T, in the global frame.
    polarization: np.ndarray
    #: Bounding box corners (2, 3) in m, in the global frame.
    bounds: np.ndarray
    #: Smallest extent in m.
    thinnest: float


def parts_of(collection):
    """The magnets of `collection`, the uniform applied field and the currents.

    Returns
    -------
    parts : list of Part
    B_ext : ndarray, shape (3,)
        The applied field mu0 H in T that all magnets that respond share.
    currents : list
        The current sources, whose field drives the magnets as given.
    """
    sources = (
        list(collection.sources_all)
        if hasattr(collection, "sources_all")
        else [collection]
    )
    with_paths = [s for s in sources if _has_path(s)]
    if with_paths:
        msg = (
            f"{len(with_paths)} objects with paths found; fem solves one position "
            "of the model, like apply_demag."
        )
        raise ValueError(msg)
    sheets = [s for s in sources if isinstance(s, magpy.current.TriangleSheet)]
    if sheets:
        msg = (
            "TriangleSheet currents are not supported by fem; model them as "
            "closed Polylines or TriangleStrips."
        )
        raise TypeError(msg)
    currents = [s for s in sources if isinstance(s, BaseCurrent)]
    open_paths = sorted({type(c).__name__ for c in currents if not _closed(c)})
    if open_paths:
        msg = (
            f"Open current paths ({', '.join(open_paths)}): fem needs every "
            "current to flow round a closed path, whose field has a potential "
            "in the magnets; close each Polyline (last vertex on the first) or "
            "TriangleStrip (last two vertices on the first two)."
        )
        raise ValueError(msg)
    others = [
        s for s in sources if not isinstance(s, BaseMagnet | BaseCurrent | magpy.Sensor)
    ]
    if others:
        names = sorted({type(s).__name__ for s in others})
        msg = (
            "Only magnets, currents and a uniform applied field are supported; "
            f"got {names}."
        )
        raise TypeError(msg)
    magnets = [s for s in sources if isinstance(s, BaseMagnet)]
    if not magnets:
        msg = "The model holds no magnets."
        raise ValueError(msg)
    flat = sorted({type(m).__name__ for m in magnets if not isinstance(m, SOLIDS)})
    if flat:
        msg = f"fem meshes magnets with a volume; got {flat}."
        raise TypeError(msg)

    sus = get_susceptibilities(magnets)
    chis = _cell_susceptibilities(sus, [m.orientation for m in magnets])
    parts = []
    for magnet, chi in zip(magnets, chis, strict=True):
        pol = magnet.polarization
        pol = np.zeros(3) if pol is None else magnet.orientation.apply(pol).ravel()
        corners = _corners(magnet)
        parts.append(
            Part(
                source=magnet,
                chi=chi,
                polarization=pol,
                bounds=np.array([corners.min(axis=0), corners.max(axis=0)]),
                thinnest=_thinnest(magnet),
            )
        )

    applied = np.array(get_H_ext(*magnets), dtype=float)
    responding = np.array([np.any(p.chi != 0) for p in parts])
    applied = applied[responding] if responding.any() else applied[:1]
    if np.ptp(applied, axis=0).max() > 1e-12 * max(np.abs(applied).max(), 1e-300):
        msg = (
            "The magnets that respond see different applied fields (H_ext); a "
            "finite-element model has one field in all space."
        )
        raise ValueError(msg)
    return parts, applied[0], currents


def _closed(current):
    """Whether a current flows round a closed path."""
    if isinstance(current, magpy.current.Circle):
        return True
    vertices = np.asarray(current.vertices, dtype=float)
    if isinstance(current, magpy.current.Polyline):
        return bool(np.allclose(vertices[0], vertices[-1]))
    return len(vertices) > 3 and bool(np.allclose(vertices[:2], vertices[-2:]))


def current_points(current, spacing):
    """Points (k, 3) in m, in the global frame, at most `spacing` apart along a
    current source: its wire, circle or strip, in order along a wire."""
    if isinstance(current, magpy.current.Circle):
        r = float(current.diameter) / 2
        n = max(64, int(np.ceil(2 * np.pi * r / spacing)))
        phi = np.linspace(0, 2 * np.pi, n + 1)
        local = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros(n + 1)])
    elif isinstance(current, magpy.current.Polyline):
        local = _along(np.asarray(current.vertices, dtype=float), spacing)
    else:  # TriangleStrip: a grid on each triangle
        local = np.concatenate(
            [_triangle_grid(t, spacing) for t in _triangles(current)]
        )
    return current.orientation.apply(local).reshape(-1, 3) + np.ravel(current.position)


def _along(vertices, spacing):
    """Points at most `spacing` apart along a polyline, its vertices included."""
    pieces = [vertices[:1]]
    for a, b in pairwise(vertices):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        pieces.append(a + np.outer(np.arange(1, n + 1) / n, b - a))
    return np.concatenate(pieces)


def wire_points(current, parts, spacing):
    """Points along a current source `spacing` apart, and 50 times closer
    within two spacings of a magnet, where a wire can cross a corner between
    the coarse points."""
    coarse = current_points(current, spacing)
    near = np.zeros(len(coarse), dtype=bool)
    for part in parts:
        near |= distance(part.source, coarse) < 2 * spacing
    if not near.any():
        return coarse
    if not isinstance(current, magpy.current.Polyline):
        # a circle's chords would cut inside it, a strip has no order
        return np.concatenate([coarse, current_points(current, spacing / 50)])
    # a wire: its coarse points are in order, refine the stretches that come
    # near a magnet
    close = near[:-1] | near[1:]
    fine = [_along(coarse[i : i + 2], spacing / 50) for i in np.flatnonzero(close)]
    return np.concatenate([coarse, *fine])


def _triangles(current):
    """Corners (k, 3, 3) of the triangles of a strip, in its own frame."""
    vertices = np.asarray(current.vertices, dtype=float)
    return np.stack([vertices[:-2], vertices[1:-1], vertices[2:]], axis=1)


def _triangle_grid(corners, spacing):
    """Points of a triangle at most `spacing` apart, its corners included."""
    a, b, c = corners
    longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
    n = max(1, int(np.ceil(longest / spacing)))
    i, j = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)]).T
    return a + np.outer(i / n, b - a) + np.outer(j / n, c - a)


def current_segments(current):
    """How many pieces magpylib evaluates for a current source at once."""
    if isinstance(current, magpy.current.Circle):
        return 1
    return max(
        1,
        len(current.vertices)
        - (1 if isinstance(current, magpy.current.Polyline) else 2),
    )


def contains(src, points):
    """Whether each of the global `points` (n, 3) lies in the solid `src`, its
    surface included: exact for every shape."""
    local = src.orientation.inv().apply(np.asarray(points) - np.ravel(src.position))
    local = local.reshape(-1, 3)
    if isinstance(src, magpy.magnet.Tetrahedron):
        v = np.asarray(src.vertices, dtype=float)
        weights = np.linalg.solve((v[1:] - v[0]).T, (local - v[0]).T).T
        return np.all(weights >= -1e-12, axis=1) & (weights.sum(axis=1) <= 1 + 1e-12)
    return mask_inside(src, local)


def _thinnest(src):
    """The smallest extent of a magnet in m: its default element size."""
    if isinstance(src, magpy.magnet.Sphere):
        return float(src.diameter)
    if isinstance(src, magpy.magnet.Tetrahedron):
        v = np.asarray(src.vertices, dtype=float)
        volume = abs(np.linalg.det(v[1:] - v[0])) / 6
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        area = max(
            np.linalg.norm(np.cross(v[j] - v[i], v[k] - v[i])) / 2 for i, j, k in faces
        )
        return float(3 * volume / area)  # the lowest height
    if isinstance(src, magpy.magnet.TriangularMesh):
        # the narrower of the box along the axes and along the vertices' own
        # principal axes, which a tilted plate's thickness follows
        v = np.asarray(src.vertices, dtype=float)
        _, _, axes = np.linalg.svd(v - v.mean(axis=0), full_matrices=False)
        return float(min(np.ptp(v, axis=0).min(), np.ptp(v @ axes.T, axis=0).min()))
    return _thinnest_part(src)  # Cuboid, Cylinder, CylinderSegment


def _ring(src):
    """(r1, r2, h) of a Cylinder (r1 = 0) or a CylinderSegment, in m."""
    if isinstance(src, magpy.magnet.Cylinder):
        return 0.0, src.dimension[0] / 2, src.dimension[1]
    r1, r2, h = src.dimension[:3]
    return r1, r2, h


def distance(src, points):
    """Distance in m from global `points` (n, 3) to the solid `src`, zero inside:
    exact for cuboids, spheres and cylinders, from the full ring for cylinder
    segments and from the box around the vertices, in the magnet's own frame,
    for tetrahedra and triangle meshes; never more than the true distance."""
    local = src.orientation.inv().apply(np.asarray(points) - np.ravel(src.position))
    local = local.reshape(-1, 3)
    if isinstance(src, magpy.magnet.Sphere):
        return np.maximum(np.linalg.norm(local, axis=1) - src.diameter / 2, 0)
    if isinstance(src, magpy.magnet.Cuboid):
        excess = np.abs(local) - np.asarray(src.dimension, dtype=float) / 2
        return np.linalg.norm(np.maximum(excess, 0), axis=1)
    if isinstance(src, magpy.magnet.Cylinder | magpy.magnet.CylinderSegment):
        r1, r2, h = _ring(src)
        rho = np.linalg.norm(local[:, :2], axis=1)
        radial = np.maximum(np.maximum(r1 - rho, rho - r2), 0)
        axial = np.maximum(np.abs(local[:, 2]) - h / 2, 0)
        return np.hypot(radial, axial)
    vertices = np.asarray(src.vertices, dtype=float)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0), axis=1)


def _corners(src):
    """Points (k, 3) in the global frame whose bounding box holds `src`."""
    if isinstance(src, magpy.magnet.Cuboid):
        local = 0.5 * np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        local = local * np.asarray(src.dimension, dtype=float)
    elif isinstance(src, magpy.magnet.Cylinder | magpy.magnet.CylinderSegment):
        _, r, h = _ring(src)
        local = np.array(np.meshgrid([-r, r], [-r, r], [-h / 2, h / 2])).T.reshape(
            -1, 3
        )
    elif isinstance(src, magpy.magnet.Sphere):
        r = src.diameter / 2
        local = np.array(np.meshgrid([-r, r], [-r, r], [-r, r])).T.reshape(-1, 3)
    else:
        local = np.asarray(src.vertices, dtype=float)
    return src.orientation.apply(local).reshape(-1, 3) + np.ravel(src.position)


def solid(src, center, length):
    """`src` as an OCC solid in scaled coordinates, and its sharp edges and
    corners marked for geometric refinement."""
    if isinstance(src, magpy.magnet.Cuboid):
        a, b, c = np.asarray(src.dimension, dtype=float) / (2 * length)
        shape = occ.Box(occ.Pnt(-a, -b, -c), occ.Pnt(a, b, c))
    elif isinstance(src, magpy.magnet.Cylinder):
        d, h = np.asarray(src.dimension, dtype=float) / length
        shape = occ.Cylinder(occ.Pnt(0, 0, -h / 2), occ.Z, r=d / 2, h=h)
    elif isinstance(src, magpy.magnet.CylinderSegment):
        r1, r2, h, phi1, phi2 = np.asarray(src.dimension, dtype=float)
        r1, r2, h = r1 / length, r2 / length, h / length
        section = occ.Face(
            occ.Wire(
                [
                    occ.Segment(occ.Pnt(*p), occ.Pnt(*q))
                    for p, q in zip(
                        [
                            (r1, 0, -h / 2),
                            (r2, 0, -h / 2),
                            (r2, 0, h / 2),
                            (r1, 0, h / 2),
                        ],
                        [
                            (r2, 0, -h / 2),
                            (r2, 0, h / 2),
                            (r1, 0, h / 2),
                            (r1, 0, -h / 2),
                        ],
                        strict=True,
                    )
                ]
            )
        )
        axis = occ.Axis(occ.Pnt(0, 0, 0), occ.Z)
        shape = section.Revolve(axis, phi2 - phi1).Rotate(axis, phi1)
    elif isinstance(src, magpy.magnet.Sphere):
        shape = occ.Sphere(occ.Pnt(0, 0, 0), src.diameter / (2 * length))
    elif isinstance(src, magpy.magnet.Tetrahedron | magpy.magnet.TriangularMesh):
        vertices = np.asarray(src.vertices, dtype=float) / length
        faces = (
            [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
            if isinstance(src, magpy.magnet.Tetrahedron)
            else np.asarray(src.faces)
        )
        triangles = [
            occ.Face(
                occ.Wire(
                    [
                        occ.Segment(occ.Pnt(*vertices[i]), occ.Pnt(*vertices[j]))
                        for i, j in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
                    ]
                )
            )
            for f in faces
        ]
        # coplanar triangles become one face, so that only real edges are sharp
        shape = occ.Solid(occ.Sew(triangles)).UnifySameDomain()
        if shape.mass < 0:  # faces wound inward: the solid would be the outside
            shape = shape.Reversed()
    else:
        msg = f"fem does not mesh {type(src).__name__} objects."
        raise TypeError(msg)

    rotvec = src.orientation.as_rotvec().ravel()
    angle = float(np.linalg.norm(rotvec))
    if angle > 0:
        shape = shape.Rotate(
            occ.Axis(occ.Pnt(0, 0, 0), occ.Dir(*(rotvec / angle))), np.degrees(angle)
        )
    offset = (np.ravel(src.position) - center) / length
    shape = shape.Move(occ.Vec(*offset))
    _mark_sharp(shape)
    return shape


def _mark_sharp(shape):
    """Mark for geometric refinement the edges where two faces meet, not the
    seams of a curved face, and the corners where two or more of them meet;
    not corners where more than ``MAX_EDGES_AT_CORNER`` meet, nor their edges."""
    faces_of = {}
    for face in shape.faces:
        for edge in face.edges:
            faces_of.setdefault(edge, set()).add(face)
    sharp = {edge for edge in shape.edges if len(faces_of.get(edge, ())) >= 2}
    edges_at = {}
    for edge in sharp:
        for vertex in edge.vertices:
            edges_at.setdefault(vertex, set()).add(edge)
    crowded = {v for v, edges in edges_at.items() if len(edges) > MAX_EDGES_AT_CORNER}
    marked = {e for e in sharp if not crowded.intersection(e.vertices)}
    for edge in marked:
        edge.hpref = 1
    for vertex, edges in edges_at.items():
        if vertex not in crowded and len(edges & marked) >= 2:
            vertex.hpref = 1
