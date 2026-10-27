"""Solve a magpylib collection's cuboids by their face charges."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
from loguru import logger

from magpylib_material_response.demag import get_H_ext
from magpylib_material_response.face_charge._bodies import (
    Body,
    add_contact_lines,
    solve_system,
)
from magpylib_material_response.face_charge._sheet import ChargeSheet
from magpylib_material_response.utils import timelog

MARK = 0.5  # a refinement round splits panels above this share of the largest indicator
POINTS_PER_CALL = (
    5_000  # external-field points per magpylib call: a coil at many points takes GBs
)


def apply_face_charge(
    collection, *, level=3, refine=0, max_unknowns=30_000, min_log_time=None
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
    max_unknowns : int, default 30_000
        Refuse a solve with more unknowns than this. All bodies form one dense
        system, of 8 n² bytes for n unknowns: 30,000 take 7 GB. Unknowns grow
        with the square of `level` and with every refinement round, and with
        the number of bodies: 100 small cubes take 14,400 at level 1.
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
        field excludes the applied field ``H_ext``.

    Raises
    ------
    ValueError
        If a solve would need more than `max_unknowns` unknowns.
    NotImplementedError
        For a non-cuboid source or an anisotropic susceptibility (both bring
        back charge inside the body: mesh it and use ``apply_demag``), and for
        objects moving along paths.
    """
    if not isinstance(collection, magpy.Collection):
        msg = f"apply_face_charge takes a magpylib Collection; got {type(collection).__name__}."
        raise TypeError(msg)
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
        add_contact_lines(bodies)
        for body in bodies:
            body.mesh()

    def external(points):
        H = np.zeros((len(points), 3))
        if fixed:
            for a in range(0, len(points), POINTS_PER_CALL):
                H[a : a + POINTS_PER_CALL] = magpy.getH(
                    fixed, points[a : a + POINTS_PER_CALL], sumup=True
                )
        return H

    for rnd in range(refine + 1):
        unknowns = sum(b.n_dofs for b in bodies)
        if unknowns > max_unknowns:
            after = f" after {rnd} refinement rounds" if rnd else ""
            msg = (
                f"level={level}{after} needs {unknowns:,} unknowns, more than "
                f"max_unknowns={max_unknowns:,}: the dense system would take about "
                f"{8 * unknowns**2 / 1e9:.0f} GB. Lower the level or the refinement, "
                "raise max_unknowns, or mesh the parts for apply_demag."
            )
            raise ValueError(msg)
        with timelog("Face-charge assembly and solve", min_log_time=min_log_time):
            for body in bodies:
                if body.G is None:
                    body.assemble()
            coefs = solve_system(bodies, external)
        sheets = [
            ChargeSheet._from_body(b, c) for b, c in zip(bodies, coefs, strict=True)
        ]
        indicators = [s.panel_indicator() for s in sheets]
        top = max(e.max(initial=0.0) for e in indicators)
        logger.info(
            "Face charge: {n} bodies, {unknowns} unknowns, largest jump indicator {jump:.2%}{round}",
            n=len(bodies),
            unknowns=sum(b.n_dofs for b in bodies),
            jump=top,
            round=f" (refinement round {rnd})" if rnd else "",
        )
        if rnd == refine or not top:
            break
        for body, e in zip(bodies, indicators, strict=True):
            marked = np.flatnonzero(e > MARK * top)
            if marked.size:
                body.bisect(marked)
                body.mesh()
    logger.info(
        "Face charge: average polarization up to {J:.3g} T; above the material's knee, saturation matters",
        J=max(np.linalg.norm(s.mean_polarization) for s in sheets),
    )

    # each solved cuboid's copy gives way to its sheet, in the same place among
    # its parent's children
    twin = dict(
        zip((id(s) for s in collection.sources_all), out.sources_all, strict=True)
    )
    replace, parents = {}, {}
    for (obj, _), sheet in zip(solved, sheets, strict=True):
        sheet.style.label = obj.style.label or f"{type(obj).__name__}, face charge"
        old = twin[id(obj)]
        replace[id(old)] = sheet
        parents[id(old.parent)] = old.parent
    for parent in parents.values():
        children = [replace.get(id(child), child) for child in parent.children]
        parent.remove(*parent.children)
        parent.add(*children)
    return out


def _inherited(obj, name):
    """An object's attribute, else its nearest parent's, else None."""
    while obj is not None:
        value = getattr(obj, name, None)
        if value is not None:
            return value
        obj = obj.parent
    return None


def _split(collection):
    """The cuboids to solve, as (object, susceptibility), and the fixed sources."""
    objects = [collection, *collection.sources_all]
    if any(
        np.asarray(obj.position).ndim > 1 and len(obj.position) > 1 for obj in objects
    ):
        msg = "apply_face_charge solves one configuration; objects moving along paths are not supported."
        raise NotImplementedError(msg)
    solved, fixed = [], []
    for src in collection.sources_all:
        chi = _inherited(src, "susceptibility")
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
