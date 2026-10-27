"""Solve a magpylib collection's cuboids by their face charges."""

from __future__ import annotations

import operator
import warnings

import magpylib as magpy
import numpy as np
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet

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
    near_panels,
    solve_system,
)
from magpylib_material_response.face_charge._sheet import ChargeSheet, saturation
from magpylib_material_response.utils import timelog

MARK = 0.5  # a refinement round splits panels above this share of the largest jump
POINTS_PER_CALL = (
    5_000  # external-field points per magpylib call: a coil at many points takes GBs
)


def apply_face_charge(
    collection, *, level=3, refine=0, max_unknowns=20_000, min_log_time=None
):
    """Solve the collection's linear cuboids on their surfaces.

    A cuboid with a uniform, isotropic susceptibility (on itself or a parent
    collection) and a uniform remanence has no magnetic charge inside: its whole
    response is a charge on its faces. Those faces are solved together, coupled
    to each other and driven by every other source in the collection and by the
    applied field ``H_ext``, as :func:`~magpylib_material_response.demag.apply_demag`
    reads it. No volume mesh is needed.

    Parameters
    ----------
    collection : magpylib.Collection
        The model. Cuboids with a nonzero susceptibility are solved; every
        other source acts on them as it is.
    level : int, default 3
        Mesh level: the panels in each graded layer toward a face's edges.
        Fields far from the bodies are converged at low levels; fields within a
        panel's size of a surface need higher ones.
    refine : int, default 0
        Rounds of refinement. Each splits the panels whose charge jumps most
        (see :attr:`ChargeSheet.jump_indicator`), over all bodies, and solves
        again; a round roughly doubles the unknowns. Two rounds from level 3
        resolve the field under a magnet close to a soft face better than
        uniform panels with four times the unknowns.
    max_unknowns : int, default 20_000
        Refuse a solve with more unknowns than this. All bodies form one dense
        system, of 8 n² bytes for n unknowns: 20,000 take 3.2 GB, and up to
        twice that while a single large body is assembled. Unknowns grow
        with the square of `level` and with every refinement round, and with
        the number of bodies: 100 small cubes take 7,200 at level 1.
    min_log_time : float, optional
        Log steps that take longer than this, in seconds; by default the value
        of :func:`~magpylib_material_response.configure_logging`.

    Returns
    -------
    magpylib.Collection
        A copy of the collection in which each solved cuboid is replaced by a
        :class:`ChargeSheet` carrying its solved surface charge, at the same
        place in the hierarchy and with its label; the other sources are
        copied as they are. Like a collection solved by ``apply_demag``, its
        field excludes the applied field ``H_ext``. Fields hold outside the
        solved parts: inside one, ``getB`` gives mu0 H, without the part's own
        magnetization, which depends on the field of every source at once; for
        B inside a part, mesh it and use ``apply_demag``.

    Raises
    ------
    ValueError
        If the first solve would need more than `max_unknowns` unknowns. A
        refinement round that would is skipped with a warning, and the round
        before it is returned.
    NotImplementedError
        For a non-cuboid source or an anisotropic susceptibility (both bring
        back charge inside the body: mesh it and use ``apply_demag``), and for
        objects moving along paths.
    """
    if not isinstance(collection, magpy.Collection):
        msg = f"apply_face_charge takes a magpylib Collection; got {type(collection).__name__}."
        raise TypeError(msg)
    try:
        level, refine = operator.index(level), operator.index(refine)
    except TypeError:
        msg = f"level and refine must be integers; got level={level!r}, refine={refine!r}."
        raise TypeError(msg) from None
    if level < 1 or refine < 0:
        msg = f"level must be 1 or more and refine 0 or more; got level={level}, refine={refine}."
        raise ValueError(msg)
    solved, fixed = _split(collection)
    out = collection.copy()
    if not solved:
        logger.info("Face charge: no cuboid with a susceptibility to solve")
        return out

    with timelog("Face-charge meshing", min_log_time=min_log_time):
        bodies = [
            Body(
                obj.dimension,
                obj.position,
                obj.orientation.as_matrix(),
                chi,
                J0=None
                if obj.polarization is None or not np.any(obj.polarization)
                else obj.polarization,
                applied=np.asarray(h_ext, float) / magpy.mu_0,
                m=level,
            )
            for (obj, chi), h_ext in zip(
                solved, get_H_ext(*(obj for obj, _ in solved)), strict=True
            )
        ]
        outlines = [
            Outline(
                np.asarray(src.dimension, float) / 2,
                src.orientation.as_matrix(),
                src.position,
            )
            for src in fixed
            if isinstance(src, magpy.magnet.Cuboid)
        ]
        add_contact_lines(bodies, outlines)
        boxes = [_box(src) for src in fixed if isinstance(src, BaseMagnet)]
        _warn_unaligned(bodies, solved, boxes, fixed)
        for body in bodies:
            body.mesh()
            body.fine = near_panels(body, boxes)

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
        with timelog("Face-charge assembly and solve", min_log_time=min_log_time):
            coefs = solve_system(bodies, external)
        sheets = [
            ChargeSheet._from_body(b, c) for b, c in zip(bodies, coefs, strict=True)
        ]
        jumps = [s._panel_jumps() for s in sheets]
        top = max(
            w.max(initial=0.0) / (peak * np.sqrt(total)) if peak else 0.0
            for w, peak, total in jumps
        )
        logger.info(
            "Face charge: {n} bodies, {unknowns} unknowns, largest jump indicator {jump:.2%}{round}",
            n=len(bodies),
            unknowns=sum(b.n_dofs for b in bodies),
            jump=top,
            round=f" (refinement round {rnd})" if rnd else "",
        )
        if rnd == refine or not top:
            break
        # marked by the jump itself, not its share of each sheet's peak, so that
        # a weakly driven part is not split as much as the one that matters
        largest = max(w.max(initial=0.0) for w, _, _ in jumps)
        for body, (w, _, _) in zip(bodies, jumps, strict=True):
            marked = np.flatnonzero(w > MARK * largest)
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
    return out


def _split(collection):
    """The cuboids to solve, as (object, susceptibility), and the fixed sources."""
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
        if not isinstance(src, magpy.magnet.Cuboid):
            msg = (
                f"apply_face_charge solves cuboids; {src} has a susceptibility. "
                "Mesh it and use apply_demag instead."
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
    for other in (*bodies, *outlines):
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
