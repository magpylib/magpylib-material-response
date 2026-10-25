"""The magnets of a magpylib model as OpenCASCADE solids for netgen.

Solids are built in scaled coordinates, ``(x - center) / length``, so that the
mesher works on numbers near one whatever the model's size.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from magpylib_material_response.meshing import _thinnest

#: The magnets fem meshes: those with a volume.
SOLIDS = (
    magpy.magnet.Cuboid,
    magpy.magnet.Cylinder,
    magpy.magnet.CylinderSegment,
    magpy.magnet.Sphere,
    magpy.magnet.Tetrahedron,
    magpy.magnet.TriangularMesh,
)
#: More sharp edges than this meeting at a corner break netgen's geometric
#: refinement (an icosahedron, five at each corner, fails to assemble, and finer
#: triangle meshes crash); such a shape gets no edge layers.
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
    """The magnets of `collection` and the uniform applied field.

    Returns
    -------
    parts : list of Part
    B_ext : ndarray, shape (3,)
        The applied field mu0 H in T that all magnets that respond share.
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
    currents = [s for s in sources if isinstance(s, BaseCurrent)]
    if currents:
        msg = (
            "Current sources are not supported by the fem module yet (issue #82, "
            "milestone 2); a uniform applied field (H_ext) and magnets are."
        )
        raise NotImplementedError(msg)
    others = [s for s in sources if not isinstance(s, BaseMagnet | magpy.Sensor)]
    if others:
        names = sorted({type(s).__name__ for s in others})
        msg = f"Only magnets and a uniform applied field are supported; got {names}."
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
    return parts, applied[0]


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
        if isinstance(src, magpy.magnet.Cylinder):
            r1, (r2, h) = 0.0, (src.dimension[0] / 2, src.dimension[1])
        else:
            r1, r2, h = src.dimension[:3]
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
        if isinstance(src, magpy.magnet.Cylinder):
            r, h = src.dimension[0] / 2, src.dimension[1]
        else:
            r, h = src.dimension[1], src.dimension[2]
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
    nothing when more than ``MAX_EDGES_AT_CORNER`` meet at a corner."""
    faces_of = {}
    for face in shape.faces:
        for edge in face.edges:
            faces_of.setdefault(hash(edge), set()).add(hash(face))
    sharp = {}
    for edge in shape.edges:
        if len(faces_of.get(hash(edge), ())) >= 2:
            sharp[hash(edge)] = edge
    edges_at = {}
    for key, edge in sharp.items():
        for vertex in edge.vertices:
            edges_at.setdefault(hash(vertex), (vertex, set()))[1].add(key)
    if any(len(keys) > MAX_EDGES_AT_CORNER for _, keys in edges_at.values()):
        return
    for edge in sharp.values():
        edge.hpref = 1
    for vertex, keys in edges_at.values():
        if len(keys) >= 2:
            vertex.hpref = 1
