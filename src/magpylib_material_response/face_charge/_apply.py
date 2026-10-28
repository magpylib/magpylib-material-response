"""Solve a magpylib collection's cuboids and triangle meshes by their face charges."""

from __future__ import annotations

import operator
import time
import warnings

import magpylib as magpy
import numpy as np
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
from scipy.spatial.transform import Rotation

from magpylib_material_response.demag import (
    _get_susceptibility_from_hierarchy,
    _has_path,
    get_H_ext,
)
from magpylib_material_response.face_charge._bodies import (
    CONTACT,
    Body,
    Outline,
    add_contact_lines,
    contact_tol,
    near_panels,
    solve_system,
)
from magpylib_material_response.face_charge._sheet import ChargeSheet, saturation
from magpylib_material_response.face_charge._triangle import frames
from magpylib_material_response.face_charge._tribody import (
    TriBody,
    add_triangle_contact_lines,
)
from magpylib_material_response.utils import timelog

MARK = 0.5  # a refinement round splits panels above this share of the largest jump
SOLVABLE = (
    magpy.magnet.Cuboid,
    magpy.magnet.Cylinder,
    magpy.magnet.CylinderSegment,
    magpy.magnet.Sphere,
    magpy.magnet.Tetrahedron,
    magpy.magnet.TriangularMesh,
)
# straight facets per turn of a curved face, per level plus one: 48 at level 3,
# 15 to 5 degrees apart at levels 1 to 5, below the 20 of a sharp edge. They
# are widened to keep the true volume, which matters more than more facets: on
# a soft cylinder at level 3, 24 facets widened are 0.02 % off 50 mm away, 96
# not widened 0.16 %. A sphere, with no edge to grade toward, refines by them
# alone: 0.26, 0.06 and 0.016 % off 1 mm away with 320, 1280 and 5120 facets
FACETS = 12
POINTS_PER_CALL = (
    5_000  # external-field points per magpylib call: a coil at many points takes GBs
)


def apply_face_charge(
    collection, *, level=None, refine=0, max_unknowns=20_000, min_log_time=None
):
    """Solve the collection's linear cuboids and triangle meshes on their surfaces.

    A part with a uniform, isotropic susceptibility (on itself or a parent
    collection) and a uniform remanence has no magnetic charge inside: its whole
    response is a charge on its faces. Those faces are solved together, coupled
    to each other and driven by every other source in the collection and by the
    applied field ``H_ext``, as :func:`~magpylib_material_response.demag.apply_demag`
    reads it. No volume mesh is needed.

    Parameters
    ----------
    collection : magpylib.Collection
        The model. Magnets with a nonzero susceptibility are solved: cuboids,
        cylinders, cylinder segments, spheres, tetrahedra and triangle meshes
        (``TriangularMesh``, a CAD export); every other source acts on them as
        it is. Curved faces are cut into flat facets, ``12 (level + 1)`` per
        turn, widened so that the body keeps its volume. A triangle mesh must
        be closed.
    level : int, optional
        Mesh level: the panels in each graded layer toward a face's edges. A
        triangle mesh is graded toward its sharp edges, over half the
        thickness of each closed shell: its smallest width, or the wall to a
        cavity. Fields far from the bodies are converged at low levels; fields
        within a panel's size of a surface need higher ones. By default 3, or
        the highest level whose unknowns fit `max_unknowns`, with a warning:
        slender parts are converged well below level 3, a soft rod 1 mm
        across and 20 mm long within 0.07 % at level 2.
    refine : int, default 0
        Rounds of refinement. Each splits the panels whose charge jumps most
        (see :attr:`ChargeSheet.jump_indicator`), over all solved parts, and
        solves again: a cuboid's panels in two along both axes, a triangle into
        four. A round roughly doubles the unknowns. Two rounds from level 3
        resolve the field under a magnet close to a soft face better than
        uniform panels with four times the unknowns; one round from level 2
        puts the field outside a thin tube shielding a magnet within 1.5 %,
        where level 3 alone is 9 % off.
    max_unknowns : int, default 20_000
        Refuse a solve with more unknowns than this. All bodies form one dense
        system, of 8 n² bytes for n unknowns: 20,000 take 3.2 GB, half as
        much again while it is solved (in single precision, refined to double's
        accuracy), and up to twice that while a single large body is
        assembled. Unknowns grow with the square of `level` and with every
        refinement round, and with the number of bodies: 100 small cubes take
        7,200 at level 1.
    min_log_time : float, optional
        Log steps that take longer than this, in seconds; by default the value
        of :func:`~magpylib_material_response.configure_logging`.

    Returns
    -------
    magpylib.Collection
        A copy of the collection in which each solved part is replaced by a
        :class:`ChargeSheet` carrying its solved surface charge, at the same
        place in the hierarchy and with its label; the other sources are
        copied as they are. Like a collection solved by ``apply_demag``, its
        field excludes the applied field ``H_ext``. Fields hold outside the
        solved parts: inside one, ``getB`` gives mu0 H, without the part's own
        magnetization, which depends on the field of every source at once; for
        B inside a part, mesh it and use ``apply_demag``. Its
        ``face_charge_info`` dict holds the last solve's ``unknowns``, the
        ``level``, the refinement ``rounds`` done and the wall time in
        ``seconds``; ``to_json`` keeps it.

    Raises
    ------
    ValueError
        If the first solve would need more than `max_unknowns` unknowns. A
        refinement round that would is skipped with a warning, and the round
        before it is returned.
    NotImplementedError
        For an anisotropic susceptibility (it brings back charge inside the
        body: mesh it and use ``apply_demag``), and objects moving along paths.
    """
    if not isinstance(collection, magpy.Collection):
        msg = f"apply_face_charge takes a magpylib Collection; got {type(collection).__name__}."
        raise TypeError(msg)
    chosen = level is not None
    try:
        level, refine = operator.index(level if chosen else 3), operator.index(refine)
    except TypeError:
        msg = f"level and refine must be integers; got level={level!r}, refine={refine!r}."
        raise TypeError(msg) from None
    if level < 1 or refine < 0:
        msg = f"level must be 1 or more and refine 0 or more; got level={level}, refine={refine}."
        raise ValueError(msg)
    start = time.perf_counter()
    solved, fixed = _split(collection)
    out = collection.copy()
    info = {"unknowns": 0, "level": level, "rounds": 0}
    if not solved:
        logger.info("Face charge: no part with a susceptibility to solve")
        out.face_charge_info = {**info, "seconds": time.perf_counter() - start}
        return out

    outlines = [
        Outline(
            np.asarray(src.dimension, float) / 2,
            src.orientation.as_matrix(),
            src.position,
        )
        for src in fixed
        if isinstance(src, magpy.magnet.Cuboid)
    ]
    boxes = [_box(src) for src in fixed if isinstance(src, BaseMagnet)]
    applied = get_H_ext(*(obj for obj, _ in solved))

    def meshed(level):
        bodies = [
            _part(obj, chi, np.asarray(h_ext, float) / magpy.mu_0, level)
            for (obj, chi), h_ext in zip(solved, applied, strict=True)
        ]
        add_contact_lines(bodies, outlines)
        add_triangle_contact_lines(bodies, outlines)
        for body in bodies:
            body.mesh()
            body.fine = near_panels(body, boxes)
        return bodies

    with timelog("Face-charge meshing", min_log_time=min_log_time):
        bodies = meshed(level)
        first = sum(b.n_dofs for b in bodies)
        # by default, the highest level that fits
        while not chosen and level > 1 and sum(b.n_dofs for b in bodies) > max_unknowns:
            level -= 1
            bodies = meshed(level)
        if level < 3 and not chosen and sum(b.n_dofs for b in bodies) <= max_unknowns:
            warnings.warn(
                f"Level 3 would need {first:,} unknowns, more than max_unknowns="
                f"{max_unknowns:,}; solved at level {level} "
                f"({sum(b.n_dofs for b in bodies):,}). Pass level= to choose it.",
                stacklevel=2,
            )
        info["level"] = level
        _warn_unaligned(bodies, solved, boxes, fixed)

    # oblique steps of 1e-9 to 1e-5 of the parts' size, off any edge line
    size = max(np.max(b.half) for b in bodies)
    nudges = [k * size * np.array([0.36, 0.48, 0.8]) for k in (1e-9, 1e-7, 1e-5)]

    def external(points):
        # B / mu0, not H: only the normal part enters the load, and B's is
        # continuous across a magnet's face, where a magnet touching a solved
        # part has its Gauss points; magpylib gives H there from inside the
        # magnet, which is (B - J) / mu0
        H = np.zeros((len(points), 3))
        if fixed:
            for a in range(0, len(points), POINTS_PER_CALL):
                chunk = points[a : a + POINTS_PER_CALL]
                with np.errstate(invalid="ignore", divide="ignore"):
                    B = magpy.getB(fixed, chunk, sumup=True)
                    # close to a magnet's edge magpylib can give NaN; the field is
                    # singular there but integrable: take it a little off, a step
                    # 100 times longer each time it is still NaN
                    for step in nudges:
                        bad = ~np.isfinite(B).all(axis=1)
                        if not bad.any():
                            break
                        B[bad] = magpy.getB(fixed, chunk[bad] + step, sumup=True)
                H[a : a + POINTS_PER_CALL] = B / magpy.mu_0
        return H

    for rnd in range(refine + 1):
        unknowns = sum(b.n_dofs for b in bodies)
        if unknowns > max_unknowns and rnd:
            # the rounds solved so far are kept, not thrown away
            warnings.warn(
                f"Refinement round {rnd} would need {unknowns:,} unknowns, more than "
                f"max_unknowns={max_unknowns:,}; the result is round {rnd - 1}'s.",
                stacklevel=2,
            )
            break
        if unknowns > max_unknowns:
            msg = (
                f"level={level} needs {unknowns:,} unknowns, more than "
                f"max_unknowns={max_unknowns:,}: the dense system would take about "
                f"{8 * unknowns**2 / 1e9:.1f} GB. Lower the level, raise max_unknowns, "
                "or mesh the parts for apply_demag."
            )
            raise ValueError(msg)
        # before the solve: the unknowns decide its time and memory
        note = f" (refinement round {rnd})" if rnd else ""
        logger.info(
            "Face charge: {n} bodies, {unknowns:,} unknowns{note}",
            n=len(bodies),
            unknowns=unknowns,
            note=note,
        )
        with timelog("Face-charge assembly and solve", min_log_time=min_log_time):
            coefs = solve_system(bodies, external)
        info.update(unknowns=unknowns, rounds=rnd)
        sheets = [_sheet_of(b, c) for b, c in zip(bodies, coefs, strict=True)]
        jumps = [s._panel_jumps() for s in sheets]
        top = max(
            w.max(initial=0.0) / (peak * np.sqrt(total)) if peak else 0.0
            for w, peak, total in jumps
        )
        logger.info(
            "Face charge: largest jump indicator {jump:.2%}{note}", jump=top, note=note
        )
        if rnd == refine or not top:
            break
        # marked by the jump itself, not its share of each sheet's peak, so that
        # a weakly driven part is not split as much as the one that matters
        largest = max(jump.max(initial=0.0) for jump, _, _ in jumps)
        for body, (jump, _, _) in zip(bodies, jumps, strict=True):
            marked = np.flatnonzero(jump > MARK * largest)
            if marked.size:
                body.bisect(marked)
                body.mesh()
                body.fine = near_panels(body, boxes)
    logger.info(
        "Face charge: average polarization up to {J:.3g} T; above the material's knee, saturation matters",
        J=max(np.linalg.norm(s.mean_polarization) for s in sheets),
    )
    # one colour scale for all, so that colours compare between parts
    full = saturation(
        sheets,
        [_hidden(s, b, bodies, outlines) for s, b in zip(sheets, bodies, strict=True)],
    )
    for sheet in sheets:
        sheet._saturation = full

    # each solved cuboid's copy gives way to its sheet, in the same place among
    # its parent's children
    twin = dict(
        zip((id(s) for s in collection.sources_all), out.sources_all, strict=True)
    )
    replace, parents = {}, {}
    for (obj, _), sheet in zip(solved, sheets, strict=True):
        # the sheet looks like the part it replaces; ``obj.style`` creates a
        # style on the caller's object, so it is read only if the user gave one
        styled = getattr(obj, "_style", None) is not None or obj._style_kwargs
        style = obj.style if styled else None
        label = style.label if style is not None else None
        sheet.style.label = label or f"{type(obj).__name__}, face charge"
        if style is not None:
            sheet.style.color, sheet.style.opacity = style.color, style.opacity
            sheet.style.magnetization.show = style.magnetization.show
            sheet.style.magnetization.color = style.magnetization.color.as_dict()
        old = twin[id(obj)]
        replace[id(old)] = sheet
        parents[id(old.parent)] = old.parent
    for parent in parents.values():
        children = [replace.get(id(child), child) for child in parent.children]
        parent.remove(*parent.children)
        parent.add(*children)
    out.face_charge_info = {**info, "seconds": time.perf_counter() - start}
    return out


def _split(collection):
    """The parts to solve, as (object, susceptibility), and the fixed sources."""
    if any(_has_path(obj) for obj in (collection, *collection.sources_all)):
        msg = "apply_face_charge solves one configuration; objects moving along paths are not supported."
        raise NotImplementedError(msg)
    solved, fixed = [], []
    for src in collection.sources_all:
        if not isinstance(src, BaseMagnet):
            # currents take no susceptibility, even one set on a parent
            fixed.append(src)
            continue
        try:
            chi = _get_susceptibility_from_hierarchy(src)
        except ValueError:  # none on the source or any parent
            chi = None
        if chi is None or not np.any(chi):
            fixed.append(src)
            continue
        if not isinstance(src, SOLVABLE):
            msg = (
                f"{src} has a susceptibility but no volume: apply_face_charge solves "
                "cuboids, cylinders, cylinder segments, spheres, tetrahedra and "
                "triangle meshes."
            )
            raise NotImplementedError(msg)
        chi = np.asarray(chi, float)
        if chi.ndim and not np.allclose(chi, chi.flat[0]):
            msg = (
                f"{src} has an anisotropic susceptibility, which brings back charge inside "
                "the body. Mesh it and use apply_demag instead."
            )
            raise NotImplementedError(msg)
        solved.append((src, float(chi.flat[0])))
    return solved, fixed


def _part(obj, chi, applied, level):
    """The body that solves ``obj``: a cuboid's panels, or a triangle mesh."""
    J0 = (
        None
        if obj.polarization is None or not np.any(obj.polarization)
        else obj.polarization
    )
    R = obj.orientation.as_matrix()
    if isinstance(obj, magpy.magnet.Cuboid):
        return Body(
            obj.dimension, obj.position, R, chi, J0=J0, applied=applied, m=level
        )
    vertices, faces = _closed_surface(obj, level)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    shared = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)[1]
    if np.any(shared != 2):
        msg = (
            f"{obj!r} has an edge shared by {shared[shared != 2][0]} triangles: face "
            "charge needs closed surfaces, each edge shared by two triangles. "
            "Give solids that meet along an edge a mesh each."
        )
        raise ValueError(msg)
    if len(np.unique(directed, axis=0)) < len(directed):
        msg = (
            f"The faces of {obj!r} are not wound consistently: an edge runs the "
            "same way in two of them. Create the TriangularMesh with "
            "reorient_faces=True."
        )
        raise ValueError(msg)
    return TriBody(
        vertices, faces, obj.position, R, chi, J0=J0, applied=applied, m=level
    )


def _closed_surface(src, level=3):
    """A magnet's surface, in its own frame, as vertices and outward faces;
    curved faces cut into the facets of ``level``."""
    # curved: wound outward as they are made
    n = FACETS * (level + 1)
    if isinstance(src, magpy.magnet.Cylinder):
        d, h = src.dimension
        return _turned(0.0, d / 2, h, 0.0, 360.0, n)
    if isinstance(src, magpy.magnet.CylinderSegment):
        return _turned(*src.dimension, n)
    if isinstance(src, magpy.magnet.Sphere):
        return _sphere(src.diameter, n)
    if isinstance(src, magpy.magnet.Cuboid):
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        V = signs * np.asarray(src.dimension, float) / 2
        F = ConvexHull(V).simplices
    elif isinstance(src, magpy.magnet.Tetrahedron):
        V = np.asarray(src.vertices, float)
        F = ConvexHull(V).simplices
    elif isinstance(src, magpy.magnet.TriangularMesh):
        V, F = np.asarray(src.vertices, float), np.array(src.faces, int)
    else:  # another shape: its bounding box
        centre, half, _, _ = _box(src)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        V = centre + signs * half
        F = ConvexHull(V).simplices
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    if isinstance(src, magpy.magnet.TriangularMesh):
        # consistently wound, as magpylib leaves a closed mesh: all turned
        # together if they face inward
        if np.einsum("fd,fd->", n, T.mean(1)) < 0:
            F = F[:, [0, 2, 1]]
    else:  # convex: each face turned away from the centre
        inward = np.einsum("fd,fd->f", n, T.mean(1) - V.mean(0)) < 0
        F = F.copy()
        F[inward] = F[inward][:, [0, 2, 1]]
    return V, F


def _turned(r1, r2, h, phi1, phi2, n):
    """A cylinder segment's surface (radii, height, angles in degrees) as
    vertices and outward faces: its arcs cut into straight facets, about ``n``
    per turn, their radii widened by ``sqrt(step / sin(step))`` so that the
    cross-section keeps its area. ``r1 = 0`` closes it on the axis; a full
    turn has no end faces."""
    full = np.isclose(phi2 - phi1, 360.0)
    k = n if full else max(1, int(np.ceil(n * (phi2 - phi1) / 360.0 - 1e-9)))
    step = np.radians(phi2 - phi1) / k
    angles = np.radians(phi1) + step * np.arange(k if full else k + 1)
    m, widen = len(angles), np.sqrt(step / np.sin(step))

    def ring(r, z):
        return np.c_[
            r * widen * np.cos(angles), r * widen * np.sin(angles), np.full(m, z)
        ]

    if r1 > 0:
        inner = [ring(r1, -h / 2), ring(r1, h / 2)]
        i0, i1 = 2 * m + np.arange(m), 3 * m + np.arange(m)
    else:  # on the axis: one point at each end
        inner = [np.array([[0.0, 0.0, -h / 2]]), np.array([[0.0, 0.0, h / 2]])]
        i0, i1 = np.full(m, 2 * m), np.full(m, 2 * m + 1)
    V = np.vstack([ring(r2, -h / 2), ring(r2, h / 2), *inner])
    o0, o1 = np.arange(m), m + np.arange(m)
    F = []
    for a in range(k):
        b = (a + 1) % m
        F += [(o0[a], o0[b], o1[b]), (o0[a], o1[b], o1[a])]  # outer side
        F += [(i1[a], o1[a], o1[b]), (i0[a], o0[b], o0[a])]  # top and bottom
        if r1 > 0:
            F += [(i1[a], o1[b], i1[b]), (i0[a], i0[b], o0[b])]
            F += [(i0[a], i1[b], i0[b]), (i0[a], i1[a], i1[b])]  # inner side
    if not full:  # the two flat ends
        F += [(i0[0], o0[0], o1[0]), (i0[0], o1[0], i1[0])]
        F += [(i0[k], o1[k], o0[k]), (i0[k], i1[k], o1[k])]
    return V, np.array(F)


def _sphere(d, n):
    """A sphere's surface as vertices and outward faces: each face of an
    icosahedron cut into ``f**2`` triangles, their edges spanning about ``360 /
    n`` degrees, projected onto the sphere and scaled to its volume."""
    p = (1 + 5**0.5) / 2
    corners = np.array([
        (-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p),
        (0, -1, -p), (0, 1, -p), (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1),
    ])  # fmt: skip
    icosahedron = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
        (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
        (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
        (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    f = max(1, round(n * 63.43 / 360))  # an icosahedron's edge spans 63.43 degrees
    V, index, F = [], {}, []

    def vertex(face, weights):
        # one key per point, whichever face reaches it: its corners and weights
        key = tuple(sorted((c, w) for c, w in zip(face, weights, strict=True) if w))
        if len(key) == 3:  # inside the face: only this face has it
            key = (face, key)
        if key not in index:
            index[key] = len(V)
            V.append(np.asarray(weights, float) @ corners[list(face)])
        return index[key]

    for face in icosahedron:
        grid = {
            (i, j): vertex(face, (f - i - j, i, j))
            for i in range(f + 1)
            for j in range(f + 1 - i)
        }
        for i in range(f):
            for j in range(f - i):
                F.append((grid[i, j], grid[i + 1, j], grid[i, j + 1]))
                if i + j < f - 1:
                    F.append((grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]))
    V = np.array(V)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = np.array(F)
    volume = np.einsum("fd,fd->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6
    return V * (d / 2) * (4 * np.pi / 3 / volume) ** (1 / 3), F


def _global_triangles(src_or_body):
    """A solved body's or a magnet's surface triangles (n, 3, 3), global."""
    if isinstance(src_or_body, TriBody):
        b = src_or_body
        return b.to_global(b.vertices[b.faces].reshape(-1, 3)).reshape(-1, 3, 3)
    if isinstance(src_or_body, Body):
        b = src_or_body
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        V = signs * b.half
        return b.to_global(V[ConvexHull(V).simplices].reshape(-1, 3)).reshape(-1, 3, 3)
    V, F = _closed_surface(src_or_body)
    R, position = (
        src_or_body.orientation.as_matrix(),
        np.asarray(src_or_body.position, float),
    )
    return (V[F].reshape(-1, 3) @ R.T + position).reshape(-1, 3, 3)


def _vertices_on(corners, triangles, tol):
    """True if a corner of ``corners`` (n, 3, 3) lies on one of ``triangles``;
    one of no area has no plane to lie in."""
    P = np.unique(corners.reshape(-1, 3), axis=0)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    triangles = triangles[np.linalg.norm(cross, axis=1) > 0]
    _, n, _, _, _, _ = frames(triangles)
    a = triangles[:, 0]
    v0, v1 = triangles[:, 1] - a, triangles[:, 2] - a
    d00, d01, d11 = (
        np.einsum("fd,fd->f", x, y) for x, y in ((v0, v0), (v0, v1), (v1, v1))
    )
    denom = d00 * d11 - d01**2
    step = max(1, 2_000_000 // max(len(triangles), 1))
    for k in range(0, len(P), step):
        v2 = P[k : k + step, None] - a[None]  # (k, f, 3)
        near = np.abs(np.einsum("kfd,fd->kf", v2, n)) <= tol
        d20, d21 = np.einsum("kfd,fd->kf", v2, v0), np.einsum("kfd,fd->kf", v2, v1)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        inside = (v >= -1e-9) & (w >= -1e-9) & (v + w <= 1 + 1e-9)
        if np.any(near & inside):
            return True
    return False


def _touch(first, second, tol):
    """True if two surfaces touch: a corner of either on the other."""
    return _vertices_on(first, second, tol) or _vertices_on(second, first, tol)


def _sheet_of(body, c):
    """The ChargeSheet of a solved body."""
    if isinstance(body, Body):
        return ChargeSheet._from_body(body, c)
    return ChargeSheet.from_triangles(
        body.T,
        body.coefficients_at_corners(c),
        position=body.position,
        orientation=Rotation.from_matrix(body.R),
    )


def _box(src):
    """A magnet's bounding box in its own frame, its rotation and its place:
    ``(centre, half, R, position)``."""
    if isinstance(src, magpy.magnet.Cuboid):
        centre, half = np.zeros(3), np.asarray(src.dimension, float) / 2
    elif isinstance(src, magpy.magnet.Cylinder):
        d, h = src.dimension
        centre, half = np.zeros(3), np.array([d / 2, d / 2, h / 2])
    elif isinstance(src, magpy.magnet.CylinderSegment):
        r2, h = src.dimension[1], src.dimension[2]
        centre, half = np.zeros(3), np.array([r2, r2, h / 2])
    elif isinstance(src, magpy.magnet.Sphere):
        centre, half = np.zeros(3), np.full(3, src.diameter / 2)
    else:  # vertices: tetrahedra and triangular meshes
        v = np.asarray(src.vertices, float).reshape(-1, 3)
        centre, half = (v.min(0) + v.max(0)) / 2, (v.max(0) - v.min(0)) / 2
    return centre, half, src.orientation.as_matrix(), np.asarray(src.position, float)


def _hidden(sheet, body, bodies, outlines):
    """Per face of the body's sheet: True where it lies against another solved
    body or a fixed cuboid, out of sight. Its charge, large and nearly
    cancelled by the other side's, would set the colour scale."""
    C = sheet.triangles.mean(axis=1) @ body.R.T + body.position
    hidden = np.zeros(len(C), bool)
    # against exact boxes only: a triangle mesh's bounding box is not its shape
    for other in (*(b for b in bodies if isinstance(b, Body)), *outlines):
        if other is not body:
            local = np.abs((C - other.position) @ other.R)
            hidden |= np.all(local <= other.half + CONTACT * np.max(other.half), axis=1)
    return hidden


def _warn_unaligned(bodies, solved, boxes, fixed):
    """Warn when a magnet that is not solved touches a solved part and is not
    a cuboid turned by multiples of 90 degrees against it: the panels cannot
    follow its footprint, where its field jumps."""
    magnets = [src for src in fixed if isinstance(src, BaseMagnet)]
    for body, (obj, _) in zip(bodies, solved, strict=True):
        if isinstance(body, TriBody):
            # a cuboid's footprint is graded toward; another shape's is not
            surface = _global_triangles(body)
            for src in magnets:
                if not isinstance(src, magpy.magnet.Cuboid) and _touch(
                    surface, _global_triangles(src), contact_tol(body)
                ):
                    warnings.warn(
                        f"{src} touches {obj}, a triangle mesh face charge solves, "
                        "and is not a cuboid: its triangles are not graded toward "
                        "the magnet's footprint, where its field jumps. Expect "
                        "errors of a few percent near it, falling slowly with "
                        "level; fem_solve checks it.",
                        stacklevel=3,
                    )
            continue
        for src, (centre, half, R, position) in zip(magnets, boxes, strict=True):
            local = (position + centre @ R.T - body.position) @ body.R
            Rab = body.R.T @ R
            extent = np.abs(Rab) @ half
            gap = np.maximum(
                0, np.maximum(local - extent - body.half, -body.half - local - extent)
            )
            if np.linalg.norm(gap) > CONTACT * np.max(body.half):
                continue
            aligned = np.allclose(np.abs(Rab), np.round(np.abs(Rab)), atol=1e-9)
            if isinstance(src, magpy.magnet.Cuboid) and aligned:
                continue
            warnings.warn(
                f"{src} touches {obj}, which face charge solves, and is not a "
                "cuboid aligned with it: the panels cannot follow its footprint, "
                "where its field jumps. Expect errors of a few percent near it, "
                "falling slowly with level; refine= helps, and fem_solve checks it.",
                stacklevel=3,
            )
