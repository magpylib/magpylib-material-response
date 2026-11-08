"""A part read from a STEP file: a TriangularMesh that keeps its exact solid.

A STEP file holds a part's exact geometry, its faces on planes, cylinders,
spheres and free-form surfaces (a B-rep). Magpylib computes the field of a
triangle mesh, so the part is one, cut from the solid; finite elements mesh
the solid itself, its elements curved onto the real fillets and bores, which
no faceting gives (issue #195).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import magpylib as magpy
import numpy as np
from netgen import meshing, occ

#: netgen's curvature safety of 4 cuts a curve at about 15 degrees a facet, 8
#: at about 7.5: the angle is about 60 degrees over it.
_DEGREES_TIMES_SAFETY = 60.0


def from_step(path, *, scale=1e-3, facet_angle=15.0, **kwargs):
    """A part read from a STEP file, as a ``TriangularMesh`` that keeps its
    exact solid for :func:`solve_fem`.

    The triangles, cut from the solid by netgen, are what Magpylib computes
    the field of and draws, and what :func:`~magpylib_material_response.solve_face_charge`
    solves. :func:`solve_fem` meshes the exact solid instead, its elements
    curved onto the real fillets and bores. ``to_json`` keeps the STEP file's
    text, and copies keep the solid.

    Parameters
    ----------
    path : str or path-like
        A STEP file (``.step``, ``.stp``) of one solid.
    scale : float, default 1e-3
        Metres per unit of the file as netgen reads it, which is millimetres:
        the default turns millimetres into metres.
    facet_angle : float, default 15
        Degrees by which the triangles turn along a curved face, about: the
        smaller, the more triangles. Face charge solves netgen's triangles
        less cheaply than a CAD tool's own export, which cuts a bore or a
        fillet into strips: for face charge, an STL from the CAD tool with
        ``widen_curves=True`` takes up to half the unknowns.
    **kwargs
        Passed to ``magpylib.magnet.TriangularMesh``: ``polarization``,
        ``position``, ``orientation``, ``style_label``, ...

    Returns
    -------
    magpylib.magnet.TriangularMesh

    Raises
    ------
    ValueError
        If the file holds no solid or more than one, or ``facet_angle`` is not
        between 1 and 30 degrees.

    Examples
    --------
    >>> from magpylib_material_response.fem import from_step, solve_fem  # doctest: +SKIP
    >>> part = from_step("pole.step", polarization=(0, 0, 0))  # doctest: +SKIP
    >>> part.susceptibility = 1000  # doctest: +SKIP
    """
    if not 1 <= facet_angle <= 30:
        msg = f"facet_angle is in degrees, between 1 and 30; got {facet_angle!r}."
        raise ValueError(msg)
    text = Path(path).read_bytes().decode("latin-1")
    solid = _solid(text, str(path))
    mesh = occ.OCCGeometry(solid).GenerateMesh(
        maxh=1e6,
        curvaturesafety=_DEGREES_TIMES_SAFETY / facet_angle,
        perfstepsend=meshing.MeshingStep.MESHSURFACE,
    )
    vertices = np.array([p.p for p in mesh.Points()]) * scale
    faces = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.Elements2D()])
    part = magpy.magnet.TriangularMesh(
        vertices=vertices, faces=faces, reorient_faces=True, check_open="raise", **kwargs
    )
    part._step = {"text": text, "scale": float(scale)}  # noqa: SLF001
    return part


def exact_solid(src):
    """The exact solid of a part made by :func:`from_step`, in its own frame
    and in metres, or None for any other object."""
    step = getattr(src, "_step", None)
    if step is None:
        return None
    return _solid(step["text"]).Scale(occ.Pnt(0, 0, 0), step["scale"])


def _solid(text, name="the saved STEP text"):
    """The one solid of a STEP file's text, in the file's units as netgen
    reads them."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "part.step"
        path.write_bytes(text.encode("latin-1"))
        solids = occ.OCCGeometry(str(path)).shape.solids
    if len(solids) != 1:
        msg = (
            f"{name} holds {len(solids)} solids: from_step reads one part. "
            "Export each part to a file of its own."
        )
        raise ValueError(msg)
    return solids[0]
