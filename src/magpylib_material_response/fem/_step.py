"""A part read from a STEP file: a TriangularMesh that keeps its exact solid.

A STEP file holds a part's exact geometry, its faces on planes, cylinders,
spheres and free-form surfaces (a B-rep). Magpylib computes the field of a
triangle mesh, so the part is one, cut from the solid; finite elements mesh
the solid itself, its elements curved onto the real fillets and bores, which
no faceting gives (issue #195).
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
from netgen import meshing, occ

from magpylib_material_response.cells import _outside_this_package

#: netgen's curvature safety of 4 cuts a curve at about 15 degrees a facet, 8
#: at about 7.5: the angle is about 60 degrees over it.
_DEGREES_TIMES_SAFETY = 60.0


def from_step(path, *, scale=1e-3, facet_angle=15.0, **kwargs):
    """A part read from a STEP file, as a ``TriangularMesh`` that keeps its
    exact solid for :func:`solve_fem`.

    The triangles, cut from the solid by netgen, are what Magpylib computes
    the field of and draws, and what :func:`~magpylib_material_response.solve_face_charge`
    solves. :func:`solve_fem` meshes the exact solid instead, its elements
    curved onto the real fillets and bores, and tells inside from outside by
    it. ``to_json`` keeps the STEP file's text, and copies keep the solid; if
    the part's vertices are changed, it is no longer the solid's, and
    :func:`solve_fem` meshes the triangles, with a warning.

    Parameters
    ----------
    path : str or path-like
        A STEP file (``.step``, ``.stp``) of one solid.
    scale : float, default 1e-3
        Metres per unit of the file as netgen reads it, which is millimetres,
        whatever unit the file was written in: the default turns millimetres
        into metres.
    facet_angle : float, default 15
        Degrees by which the triangles turn along a curved face, about: the
        smaller, the more triangles. Face charge solves netgen's triangles
        less cheaply than a CAD tool's own export, which cuts a bore or a
        fillet into strips: for face charge, an STL from the CAD tool with
        ``widen_curves=True`` takes up to half the unknowns.
    **kwargs
        Passed to ``magpylib.magnet.TriangularMesh``: ``polarization``,
        ``position``, ``orientation``, ``style_label``, ... ``check_open``
        and ``reorient_faces`` default to ``"raise"`` and ``True``.

    Returns
    -------
    magpylib.magnet.TriangularMesh

    Raises
    ------
    ValueError
        If the file holds no solid or more than one, ``scale`` is not a
        positive length, ``facet_angle`` is not between 1 and 30 degrees, or
        ``vertices`` or ``faces`` are given.

    Examples
    --------
    >>> from magpylib_material_response.fem import from_step, solve_fem  # doctest: +SKIP
    >>> part = from_step("pole.step", polarization=(0, 0, 0))  # doctest: +SKIP
    >>> part.susceptibility = 1000  # doctest: +SKIP
    """
    if not 1 <= facet_angle <= 30:
        msg = f"facet_angle is in degrees, between 1 and 30; got {facet_angle!r}."
        raise ValueError(msg)
    if not (np.isfinite(scale) and scale > 0):
        msg = f"scale is metres per millimetre of the file, a positive number; got {scale!r}."
        raise ValueError(msg)
    if {"vertices", "faces"} & set(kwargs):
        msg = "from_step makes the vertices and faces from the solid: do not pass them."
        raise ValueError(msg)
    text = Path(path).read_bytes().decode("latin-1")
    try:
        solid = _parsed(text)
    except ValueError as error:
        msg = f"{path}: {error}"
        raise ValueError(msg) from None
    mesh = occ.OCCGeometry(solid).GenerateMesh(
        maxh=1e6,
        curvaturesafety=_DEGREES_TIMES_SAFETY / facet_angle,
        perfstepsend=meshing.MeshingStep.MESHSURFACE,
    )
    vertices = np.array([p.p for p in mesh.Points()]) * scale
    faces = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.Elements2D()])
    part = magpy.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        **{"reorient_faces": True, "check_open": "raise", **kwargs},
    )
    part._step = {"text": text, "scale": float(scale), "vertices": _fingerprint(part)}
    return part


def exact_solid(src):
    """The exact solid of a part made by :func:`from_step`, in its own frame
    and in metres; None for any other object, or one whose vertices changed
    since, with a warning."""
    step = getattr(src, "_step", None)
    if step is None:
        return None
    if step.get("vertices") != _fingerprint(src):
        warnings.warn(
            f"{src!r}: its vertices changed since from_step read it, so it is no "
            "longer its STEP solid: finite elements mesh its triangles.",
            stacklevel=_outside_this_package(),
        )
        return None
    return _parsed(step["text"]).Scale(occ.Pnt(0, 0, 0), step["scale"])


def exact_box(src):
    """The bounding box (lo, hi) of a from_step part's exact solid, in its
    own frame and in metres, or None: the solid bulges past its triangles."""
    solid = exact_solid(src)
    if solid is None:
        return None
    lo, hi = solid.bounding_box
    return np.array([lo.x, lo.y, lo.z]), np.array([hi.x, hi.y, hi.z])


def exact_inside(src, local, near):
    """Whether each of the `near` points of `local` (n, 3), in the part's own
    frame and in metres, lies in a from_step part's exact solid: OpenCASCADE's
    distance, zero inside. Asked in the file's millimetres, since its
    tolerances are absolute: scaled to metres, a point 17 µm inside a 5 mm
    cylinder reads as outside."""
    step = src._step
    solid = _parsed(step["text"])
    out = np.zeros(len(local), bool)
    for k in np.flatnonzero(near):
        point = occ.Pnt(*(local[k] / step["scale"]))
        out[k] = solid.Distance(occ.Vertex(point)) <= 0
    return out


def _fingerprint(src):
    """A digest of the part's vertices: the triangles from_step cut."""
    vertices = np.ascontiguousarray(src.vertices, dtype=float)
    return hashlib.sha256(vertices.tobytes()).hexdigest()


@functools.lru_cache(maxsize=16)
def _parsed(text):
    """The one solid of a STEP file's text, in the file's units as netgen
    reads them; parsed once per text."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "part.step"
        path.write_bytes(text.encode("latin-1"))
        solids = occ.OCCGeometry(str(path)).shape.solids
    if len(solids) != 1:
        msg = (
            f"it holds {len(solids)} solids: from_step reads one part. "
            "Export each part to a file of its own."
        )
        raise ValueError(msg)
    return solids[0]
