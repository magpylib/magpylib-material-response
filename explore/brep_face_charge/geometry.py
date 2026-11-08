"""The study's two parts, exact and faceted.

Exact: an OpenCASCADE solid (netgen.occ), the B-rep a STEP file holds.
Faceted: a closed triangle mesh the way a CAD export cuts it, flat faces whole
and each curved face in strips of one facet per ``step`` degrees, with its
vertices on the curve ("export"); or the way the solver cuts a primitive today,
``12 (level + 1)`` facets per turn ("brep"), with its vertices moved out so
that the faceted body keeps the exact volume ("widened") or not.

Lengths in millimetres here; ``as_magnet`` converts to metres.
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np
from netgen import meshing, occ
from scipy.optimize import brentq

MM = 1e-3

# ── the plate: 20 x 12 x 4 mm, 3 mm fillets on its vertical edges, a 3 mm hole
# 3 mm off centre (the validation's CAD part, exact)
A, B, T, RF = 10.0, 6.0, 4.0, 3.0
HOLE_C, HOLE_R = (3.0, 0.0), 1.5

# ── the pole piece: a 20 mm round bar, 10 mm long, its top rim filleted 3 mm
R_POLE, H_POLE, RF_POLE = 10.0, 10.0, 3.0


def plate_exact():
    box = occ.Box(occ.Pnt(-A, -B, -T / 2), occ.Pnt(A, B, T / 2))
    vertical = [e for e in box.edges if abs(e.start.z - e.end.z) > 1]
    hole = occ.Cylinder(occ.Pnt(*HOLE_C, -T), occ.Z, r=HOLE_R, h=2 * T)
    return box.MakeFillet(vertical, RF) - hole


def pole_exact():
    bar = occ.Cylinder(occ.Pnt(0, 0, -H_POLE / 2), occ.Z, r=R_POLE, h=H_POLE)
    rim = [
        e for e in bar.edges if abs(e.start.z - H_POLE / 2) < 1e-9 and abs(e.end.z - H_POLE / 2) < 1e-9
    ]
    return bar.MakeFillet(rim, RF_POLE)


def _arc(centre, r, a0, a1, n, widen):
    """n + 1 points on an arc from angle a0 to a1 (radians); with `widen`,
    the inner points moved out so that the polygon from the centre has the
    sector's area (the end points stay on the tangent lines)."""
    a = np.linspace(a0, a1, n + 1)
    s = (a1 - a0) / n
    rr = np.full(n + 1, r)
    if widen and n > 1:
        sector = 0.5 * r * r * (a1 - a0)

        def area(x):  # fan: 2 end triangles r-x, n-2 inner x-x
            return 0.5 * np.sin(s) * (2 * r * x + (n - 2) * x * x)

        rr[1:-1] = brentq(lambda x: area(x) - sector, r, 2 * r)
    elif widen:  # one facet: nothing inside to move
        pass
    return np.column_stack([centre[0] + rr * np.cos(a), centre[1] + rr * np.sin(a)])


def _circle(centre, r, n, widen):
    """n points on a circle, all moved out by sqrt(step / sin(step)) with
    `widen`, which keeps the polygon's area that of the circle."""
    s = 2 * np.pi / n
    rr = r * (np.sqrt(s / np.sin(s)) if widen else 1.0)
    a = s * np.arange(n)
    return np.column_stack([centre[0] + rr * np.cos(a), centre[1] + rr * np.sin(a)])


def plate_faceted(n_quarter, n_hole, widen=False):
    """The plate with n_quarter facets on each fillet and n_hole on the hole,
    as a closed triangle mesh (vertices mm, faces)."""
    centres = [(A - RF, B - RF), (-A + RF, B - RF), (-A + RF, -B + RF), (A - RF, -B + RF)]
    outline = np.concatenate(
        [
            _arc(c, RF, np.pi / 2 * k, np.pi / 2 * (k + 1), n_quarter, widen)
            for k, c in enumerate(centres)
        ]
    )
    hole = _circle(HOLE_C, HOLE_R, n_hole, widen)

    def wire(points):
        pts = [occ.Pnt(x, y, -T / 2) for x, y in points]
        return occ.Wire([occ.Segment(p, q) for p, q in zip(pts, pts[1:] + pts[:1])])

    section = occ.Face(wire(outline)) - occ.Face(wire(hole))
    solid = section.Extrude(T * occ.Z)
    return _surface(solid)


def pole_faceted(n_turn, n_quarter, widen=False):
    """The pole piece with n_turn facets round its axis and n_quarter on the
    fillet's profile, as a closed triangle mesh (vertices mm, faces)."""
    zb, zt = -H_POLE / 2, H_POLE / 2
    arc = _arc((R_POLE - RF_POLE, zt - RF_POLE), RF_POLE, 0, np.pi / 2, n_quarter, widen)
    profile = np.concatenate([[(R_POLE, zb)], arc])  # (r, z), bottom rim to top edge
    s = 2 * np.pi / n_turn
    w = np.sqrt(s / np.sin(s)) if widen else 1.0
    phi = s * np.arange(n_turn)
    rings = [
        np.column_stack([w * r * np.cos(phi), w * r * np.sin(phi), np.full(n_turn, z)])
        for r, z in profile
    ]
    V = np.concatenate([*rings, [(0, 0, zb), (0, 0, zt)]])
    bottom, top = len(V) - 2, len(V) - 1
    F = []
    for k in range(n_turn):
        k1 = (k + 1) % n_turn
        F.append((bottom, k1, k))
        for j in range(len(rings) - 1):
            a, b = j * n_turn, (j + 1) * n_turn
            F += [(a + k, a + k1, b + k1), (a + k, b + k1, b + k)]
        last = (len(rings) - 1) * n_turn
        F.append((top, last + k, last + k1))
    return V, np.array(F)


def _surface(solid):
    """A solid with flat faces as a closed triangle mesh: netgen triangulates
    the faces, coarsely, since face charge re-meshes flat faces anyway."""
    mesh = occ.OCCGeometry(solid).GenerateMesh(
        maxh=1000, perfstepsend=meshing.MeshingStep.MESHSURFACE
    )
    V = np.array([p.p for p in mesh.Points()])
    F = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.Elements2D()])
    return V, F


def as_magnet(V, F, exact=None, chi=1000.0, label=""):
    """A TriangularMesh in metres; `exact`, the OCC solid in mm, rides along
    for the finite-element reference (see reference.py)."""
    part = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=np.asarray(V) * MM, faces=F,
        reorient_faces=True, check_open="raise", style_label=label,
    )
    part.susceptibility = chi
    part._exact_mm = exact  # noqa: SLF001
    return part


# ── what each part sits in and where it is read

FIELD = 0.01 * np.array([0.8, 0.36, 0.48])  # T, a direction no symmetry helps

SENSORS = {
    "plate": {
        "near the curves": np.array(
            [(3, 0, 0), (8, 0, 3), (7 + 4 / 2**0.5, 3 + 4 / 2**0.5, 1), (3, 0.9, 0)]
        )
        * MM,
        "60 mm away": np.array([(60, 0, 0), (0, 0, 60)]) * MM,
    },
    "pole": {
        "near the curves": np.array(
            [
                (r * np.cos(a), r * np.sin(a), z)
                for a in np.radians([0, 120, 240])
                for r, z in [(7 + 4 / 2**0.5, 2 + 4 / 2**0.5)]
            ]
            + [(0, 0, 6), (11 * np.cos(np.pi / 3), 11 * np.sin(np.pi / 3), -1.5)]
        )
        * MM,
        "60 mm away": np.array([(60, 0, 0), (0, 0, 60)]) * MM,
    },
}

EXACT = {"plate": plate_exact, "pole": pole_exact}


def fine(part):
    """A fine faceting for the reference object's own vertices (the FEM reads
    the exact solid; these serve only its inside tests and sizes)."""
    if part == "plate":
        return plate_faceted(24, 96)
    return pole_faceted(192, 24)


def faceted(part, n_turn, widen):
    """`part` faceted with n_turn facets per full turn of every curve."""
    if part == "plate":
        return plate_faceted(n_turn // 4, n_turn, widen)
    return pole_faceted(n_turn, n_turn // 4, widen)


# ── widening from the mesh alone: what a solver could do with an export

SHARP = 20.0  # degrees: the face-charge solver's own sharp-edge threshold


def mesh_widen(V, F, sharp=SHARP):
    """Move an export's curved-face vertices out as the solver widens its own
    facets, from the mesh alone: each smooth edge (its faces turn by less than
    `sharp`) is a polygon corner of a curve across it, of turning angle theta
    and local radius r = width / (2 sin(theta / 2)), the width being the
    narrower adjacent facet's extent across the edge; it moves out by
    r (sqrt(theta / sin theta) - 1), away from the curve's centre. A vertex
    takes the mean of its edges along one direction, summed over directions.
    `F` must be wound consistently outward."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    edges = {}
    for f, tri in enumerate(F):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            edges.setdefault((min(a, b), max(a, b)), []).append((f, tri[(k + 2) % 3]))
    moves = {}  # vertex -> list of (edge direction, displacement)
    for (a, b), side in edges.items():
        if len(side) != 2:
            continue
        (f1, c1), (f2, c2) = side
        cos = np.clip(n[f1] @ n[f2], -1, 1)
        theta = np.arccos(cos)
        if theta < 1e-6 or np.degrees(theta) >= sharp:
            continue
        e = V[b] - V[a]
        e /= np.linalg.norm(e)

        def width(c):
            d = V[c] - V[a]
            return np.linalg.norm(d - (d @ e) * e)

        r = min(width(c1), width(c2)) / (2 * np.sin(theta / 2))
        delta = r * (np.sqrt(theta / np.sin(theta)) - 1)
        m = n[f1] + n[f2]
        m /= np.linalg.norm(m)
        convex = n[f1] @ (V[c2] - V[a]) < 0
        d = (delta if convex else -delta) * m
        for v in (a, b):
            moves.setdefault(v, []).append((e, d))
    W = V.copy()
    for v, items in moves.items():
        groups = []  # edges along one direction
        for e, d in items:
            for g in groups:
                if abs(g[0] @ e) > np.cos(np.radians(15)):
                    g[1].append(d)
                    break
            else:
                groups.append((e, [d]))
        W[v] += sum(np.mean(ds, axis=0) for _, ds in groups)
    return W


def export_mesh_widened(part, n_turn):
    """`part` exported at n_turn facets per turn, then widened from the mesh."""
    V, F = faceted(part, n_turn, False)
    oriented = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=V, faces=F, reorient_faces=True
    )
    F = np.asarray(oriented.faces)
    return mesh_widen(V, F), F
