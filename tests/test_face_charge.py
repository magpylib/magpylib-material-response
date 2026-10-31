"""Tests of the face-charge solver and its ChargeSheet source."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import magpylib as magpy
import numpy as np
import pytest
from magpylib.graphics.backend import colors_from_colorscale
from matplotlib.colors import to_rgb
from scipy.linalg import LinAlgError, LinAlgWarning
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.face_charge import (
    ChargeSheet,
    _cache,
    apply_face_charge,
    clear_cache,
    set_cache_limit,
)
from magpylib_material_response.face_charge._apply import _closed_surface
from magpylib_material_response.face_charge._bodies import _solve, solve_system
from magpylib_material_response.face_charge._mesh import (
    _convex,
    _merge_coplanar,
    graded_triangles,
    rounding,
)
from magpylib_material_response.face_charge._parallel import run
from magpylib_material_response.face_charge._sheet import saturation
from magpylib_material_response.face_charge._triangle import (
    charge_field,
    frames,
    normal_field,
    normal_field_far,
)
from magpylib_material_response.face_charge._tribody import TriBody

TRIANGLE = np.array([[0.0, 0.0, 0.0], [0.004, 0.001, 0.0], [0.001, 0.003, 0.0005]])
NORMAL = np.cross(TRIANGLE[1] - TRIANGLE[0], TRIANGLE[2] - TRIANGLE[0])
NORMAL /= np.linalg.norm(NORMAL)
CHARGE = np.array([1e5, -3e4, 6e4])  # A/m at the triangle's corners


def _observers():
    rng = np.random.default_rng(3)
    centre = TRIANGLE.mean(0)
    return np.concatenate(
        [
            centre + rng.normal(size=(30, 3)) * 0.01,
            centre + rng.normal(size=(30, 3)) * 8e-4,
        ]
    )


def _one_face(charge, **kwargs):
    return ChargeSheet(TRIANGLE, [[0, 1, 2]], [charge], **kwargs)


def _plate_under_magnet(shape="cuboid"):
    plate = (
        magpy.magnet.Cuboid(
            dimension=(0.01, 0.01, 0.004), polarization=(0, 0, 0), style_label="plate"
        )
        if shape == "cuboid"
        else magpy.magnet.Cylinder(dimension=(0.012, 0.004), polarization=(0, 0, 0))
    )
    plate.susceptibility = 1000.0
    magnet = magpy.magnet.Cuboid(
        dimension=(0.004, 0.004, 0.003),
        position=(0, 0, 0.0045),
        polarization=(0, 0, 1.2),
    )
    magnet.susceptibility = 0.0  # exactly 0: not solved
    return magpy.Collection(plate, magnet)


def _relative(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_uniform_charge_is_magpylib_triangle():
    J = np.array([0.3, -0.2, 0.9])
    sheet = _one_face([J @ NORMAL / magpy.mu_0] * 3)
    triangle = magpy.misc.Triangle(vertices=TRIANGLE, polarization=J)
    obs = _observers()
    assert _relative(sheet.getH(obs), triangle.getH(obs)) < 1e-13
    assert _relative(sheet.getB(obs), triangle.getB(obs)) < 1e-13


def test_linear_charge_is_the_limit_of_uniform_pieces():
    bary = np.eye(3)[None]
    for _ in range(4):  # 256 pieces, each charged at its centroid
        a, b, c = bary[:, 0], bary[:, 1], bary[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        bary = np.concatenate(
            [
                np.stack(t, 1)
                for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            ]
        )
    pieces = [
        magpy.misc.Triangle(
            vertices=bv @ TRIANGLE,
            polarization=magpy.mu_0 * (bv.mean(0) @ CHARGE) * NORMAL,
        )
        for bv in bary
    ]
    far = _observers()[:30]
    assert (
        _relative(magpy.getH(pieces, far, sumup=True), _one_face(CHARGE).getH(far))
        < 1e-3
    )


def test_normal_fields_are_the_fields_normal_part():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(40, 3, 3))
    fr = frames(V)
    near, far = rng.normal(size=(30, 3)) * 3, rng.normal(size=(30, 3)) * 50
    for X, kernel, tol in ((near, normal_field, 1e-13), (far, normal_field_far, 1e-9)):
        # each triangle's field of each density alone, along its normal
        En = np.array(
            [
                [
                    charge_field(X, V[[k]], tuple(f[[k]] for f in fr), np.eye(3)[[a]])
                    @ fr[1][k]
                    for k in range(len(V))
                ]
                for a in range(3)
            ]
        ).transpose(0, 2, 1)
        assert np.abs(kernel(X, V, fr) - En).max() < tol * np.abs(En).max()
    one = rng.integers(0, len(V), len(near))  # observer i with triangle one[i]
    pairs = normal_field(near, V[one], tuple(a[one] for a in fr), pairs=True)
    assert np.allclose(pairs, normal_field(near, V, fr)[:, np.arange(len(near)), one])
    assert all(np.isfinite(normal_field(V[:, k], V, fr)).all() for k in range(3))


def _cube_as_triangles(half=0.005):
    """A cube as 12 triangles, outward."""
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    F = ConvexHull(V).simplices
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    inward = np.einsum("fd,fd->f", n, V[F].mean(1)) < 0
    F[inward] = F[inward][:, [0, 2, 1]]
    return half * V, F


def _cube_cut(n, half=0.005):
    """A cube with each face cut into n x n squares of two triangles, outward."""
    g, V, F = np.linspace(-1, 1, n + 1), [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (-1, 1):
            first = len(V)
            for a in g:
                for b in g:
                    p = np.zeros(3)
                    p[axis], p[u], p[v] = side, a, b
                    V.append(p)
            for i in range(n):
                for j in range(n):
                    q = first + i * (n + 1) + j
                    F += [[q, q + n + 1, q + n + 2], [q, q + n + 2, q + 1]]
    V, weld = np.unique(np.round(V, 12), axis=0, return_inverse=True)
    F = weld.ravel()[np.array(F)]
    n_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    inward = np.einsum("fd,fd->f", n_, V[F].mean(1)) < 0
    F[inward] = F[inward][:, [0, 2, 1]]
    return half * V, F


def test_graded_triangles_cover_the_mesh_toward_its_edges():
    V, F = _cube_as_triangles()
    T = graded_triangles(V, F, ell=0.005, m=3, g=2.5)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    assert np.isclose(area.sum(), 6 * 0.01**2)
    # the first layer, 0.005 / 3**2.5 deep, lies along the cube's edges
    on_face = np.isclose(np.abs(T[..., 2]), 0.005).all(1)
    assert np.isclose(
        np.sort(np.unique(np.round(np.abs(T[on_face][..., 0]), 12)))[-2],
        0.005 * (1 - 3**-2.5),
    )
    with pytest.raises(ValueError, match="closed mesh"):
        graded_triangles(V, F[:-1], ell=0.005, m=2, g=2.5)


@pytest.mark.parametrize("irregular", [False, True], ids=["grid", "irregular"])
@pytest.mark.parametrize(
    ("left_out", "polygons"),
    [
        (lambda x, y: x >= 0.5 and y >= 0.5, 2),
        (lambda x, y: 0.3 <= x < 0.6 and 0.4 <= y < 0.7, 4),
    ],
    ids=["L-shaped", "a square hole"],
)
def test_finely_cut_faces_become_the_rectangles_of_their_outline(
    left_out, polygons, irregular
):
    """#106: a flat face that is not convex, cut into about 10,000 triangles
    as a fine export, becomes the rectangles its outline gives, in time, and
    the same ones cut irregularly, as an FEM surface mesh: merged, its
    triangles left 5 times the cells. The L took 13.7 s, merged triangle by
    triangle from the start after each merge."""
    V, F = _grid_face(80, left_out, irregular)
    start = time.perf_counter()
    merged = _merge_coplanar(V, F, 1e-12)
    assert time.perf_counter() - start < 5
    assert len(merged) == polygons
    assert all(len(poly) == 4 for poly, _ in merged)


def _grid_face(k, left_out, irregular):
    """A unit square face cut into k x k squares, those ``left_out`` left out;
    irregularly, the diagonals either way and the vertices inside moved."""
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 1, k + 1)
    P = np.array([(x, y) for x in xs for y in xs])
    at = np.arange((k + 1) ** 2).reshape(k + 1, k + 1)
    cells = [(i, j) for i in range(k) for j in range(k) if not left_out(i / k, j / k)]
    turns = rng.random(len(cells)) < 0.5 if irregular else np.zeros(len(cells), bool)
    F = np.reshape(
        [
            at[i : i + 2, j : j + 2].ravel()[
                [0, 2, 1, 2, 3, 1] if turn else [0, 2, 3, 0, 3, 1]
            ]
            for (i, j), turn in zip(cells, turns, strict=True)
        ],
        (-1, 3),
    )
    if irregular:
        edges = {(u, v) for f in F for u, v in zip(f, np.roll(f, -1), strict=True)}
        inside = np.ones(len(P), bool)
        inside[[u for u, v in edges if (v, u) not in edges]] = False
        P[inside] += rng.uniform(-0.3, 0.3, (inside.sum(), 2)) / k
    return np.c_[P, np.zeros(len(P))], F


def _face_with_round_holes(sides):
    """A 30 mm square face with 3 x 3 holes of ``sides`` sides, cut as an FEM
    surface mesh: Delaunay on its outline, each triangle split at its centre."""
    a = np.pi / sides + 2 * np.pi * np.arange(sides) / sides
    centres = [(x, y) for x in (-10, 0, 10) for y in (-10, 0, 10)]
    B = np.concatenate(
        [[(-15, -15), (15, -15), (15, 15), (-15, 15)]]
        + [np.c_[x + 3 * np.cos(a), y + 3 * np.sin(a)] for x, y in centres]
    )
    P, F = list(B), []
    for simplex in Delaunay(B).simplices:
        if min(np.hypot(*(B[simplex].mean(0) - c)) for c in centres) < 3:
            continue  # in a hole
        d1, d2 = B[simplex[1]] - B[simplex[0]], B[simplex[2]] - B[simplex[0]]
        t = simplex if d1[0] * d2[1] - d1[1] * d2[0] > 0 else simplex[[0, 2, 1]]
        F += [(t[0], t[1], len(P)), (t[1], t[2], len(P)), (t[2], t[0], len(P))]
        P.append(B[t].mean(0))
    return np.c_[np.array(P), np.zeros(len(P))] * 1e-3, np.array(F)


@pytest.mark.parametrize(
    ("face", "scale"),
    [
        (
            lambda: _grid_face(
                24, lambda x, y: not (0.33 <= x < 0.67 or 0.33 <= y < 0.67), True
            ),
            0.03,
        ),
        (lambda: _face_with_round_holes(12), 1.0),
    ],
    ids=["a plus", "nine round holes"],
)
def test_float32_faces_are_cut_whole(face, scale):
    """A face cut irregularly, in float32 as an STL export, turned out of the
    axes: rounding left the cuts from its outline without end (a plus) or
    losing 0.5 % of the face (round holes)."""
    V, F = face()
    turn = R.from_euler("xyz", (0.3, 0.7, 1.1)).as_matrix()
    V = (scale * V @ turn.T + (0.05, -0.08, 0.02)).astype(np.float32).astype(float)
    tol = max(1e-9 * np.ptp(V), 4 * rounding(V))
    merged = _merge_coplanar(V, F, tol)
    assert all(_convex(poly, normal, tol) for poly, normal in merged)
    area = np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1
    )
    covered = sum(np.cross(P, np.roll(P, -1, 0)).sum(0) @ n / 2 for P, n in merged)
    np.testing.assert_allclose(covered, area.sum() / 2, rtol=1e-6)


def test_cube_as_triangles_converges_as_the_cuboid():
    """A cube given as 12 triangles, chi -> inf: the polarizability converges
    onto Helsing & Perfekt's 3.644305 as the cuboid solver's does."""
    V, F = _cube_as_triangles(half=1.0)
    errors = []
    for m in (2, 3):
        body = TriBody(V, F, (0, 0, 0), np.eye(3), chi=1e12, applied=(0, 0, 1.0), m=m)
        body.mesh()
        (c,) = solve_system([body], np.zeros_like)
        P, W, B = body.quad()
        q = np.einsum("fqa,fa->fq", B, c.reshape(-1, 3))
        errors.append(np.einsum("fq,fq,fq->", W, q, P[..., 2]) / 8 / 3.644305 - 1)
    assert -3.5e-3 < errors[0] < errors[1] < 0
    assert abs(errors[1]) < 1.1e-3


def test_triangle_mesh_part_couples_as_a_cuboid():
    """A cube given as a triangle mesh beside a turned soft cuboid, in an
    applied field: the field around them as with two cuboids."""
    turned = R.from_euler("z", 30, degrees=True)
    sensors = np.array([[0, 0.02, 0], [0.0075, 0, 0.012], [-0.02, 0, 0]])
    fields = []
    for first in (_mesh_cube(), _cuboid()):
        coll = magpy.Collection(
            first, _cuboid(position=(0.015, 0, 0), orientation=turned)
        )
        coll.susceptibility = 100.0
        coll.H_ext = (8e-3, 2.4e-3, 4e-3)
        solved = apply_face_charge(coll, level=3)
        assert all(isinstance(s, ChargeSheet) for s in solved.sources_all)
        fields.append(solved.getB(sensors))
    assert _relative(*fields) < 5e-4


@pytest.mark.parametrize(
    ("turn", "offset"),
    [(0, 0.0), (-45, 0.0), (30, 1.0)],
    ids=["aligned", "turned, float32", "turned, far away, float32, finely cut"],
)
def test_triangle_mesh_touching_a_cuboid_solves_as_two_cuboids(turn, offset):
    """A smaller soft cuboid against a face of a cube given as a triangle
    mesh, offset on it: their faces in contact couple exactly, and the mesh
    is graded toward the cuboid's footprint. Turned, with the turn in the
    mesh's float32 vertices, as a CAD export gives: the contact planes found
    whichever sign their normals round to, and no slivers where the layers
    meet. Far from the origin in float32, where rounding grows with the
    distance, its faces cut into strips: the same."""
    rot = R.from_euler("z", turn, degrees=True)
    at = offset * np.array([1.0, -0.6, 0.4])
    V, F = _cube_cut(4) if offset else _cube_as_triangles()
    mesh = magpy.magnet.TriangularMesh(
        vertices=(rot.apply(V) + at).astype(np.float32 if turn or offset else float),
        faces=F,
        polarization=(0, 0, 0),
    )
    sensors = at + np.array([[0, 0.02, 0], [0.005, 0, 0.012], [0.0125, 0.0061, 0]])
    fields = []
    for first in (mesh, _cuboid(position=at, orientation=rot)):
        side = magpy.magnet.Cuboid(
            dimension=(0.006, 0.004, 0.004),
            position=at + rot.apply((0.008, 0.001, 0)),
            orientation=rot,
            polarization=(0, 0, 0),
        )
        coll = magpy.Collection(first, side)
        coll.susceptibility = 100.0
        coll.H_ext = (8e-3, 2.4e-3, 4e-3)
        fields.append(apply_face_charge(coll, level=3).getB(sensors))
    assert _relative(*fields) < 3e-4
    part = _mesh_cube()
    part.susceptibility = 100.0
    disc = magpy.magnet.Cylinder(
        dimension=(0.004, 0.002), position=(0, 0, 0.006), polarization=(0, 0, 1)
    )
    with pytest.warns(UserWarning, match="is not a cuboid"):
        apply_face_charge(magpy.Collection(part, disc), level=1)


def _bars(as_mesh):
    """#112's soft bars of one material: wide, a short narrow bridge touching
    each on part of its face, wide again; as three cuboids or one closed mesh."""
    t, slabs = 0.008, [(-0.04, 0.0, 0.02), (0.0, 0.005, 0.01), (0.005, 0.045, 0.02)]
    if not as_mesh:
        return [
            magpy.magnet.Cuboid(
                dimension=(x1 - x0, 2 * h, t),
                position=((x0 + x1) / 2, 0, -t / 2),
                polarization=(0, 0, 0),
            )
            for x0, x1, h in slabs
        ]
    outline = [(-0.04, -0.02), (0, -0.02), (0, -0.01), (0.005, -0.01), (0.005, -0.02)]
    outline += [(0.045, -0.02), (0.045, 0.02), (0.005, 0.02), (0.005, 0.01)]
    outline += [(0, 0.01), (0, 0.02), (-0.04, 0.02)]
    caps = [[0, 1, 2, 9, 10, 11], [2, 3, 8, 9], [5, 6, 7, 8, 3, 4]]  # convex, fanned
    n = len(outline)
    V = [(x, y, z) for z in (-t, 0.0) for x, y in outline]
    F = [(k, (k + 1) % n, n + (k + 1) % n) for k in range(n)]
    F += [(k, n + (k + 1) % n, n + k) for k in range(n)]
    for cap in caps:
        F += [(cap[0], cap[q + 1], cap[q]) for q in range(1, len(cap) - 1)]
        F += [(n + cap[0], n + cap[q], n + cap[q + 1]) for q in range(1, len(cap) - 1)]
    mesh = magpy.magnet.TriangularMesh(
        vertices=V, faces=F, polarization=(0, 0, 0), reorient_faces=True
    )
    return [mesh]


def test_touching_parts_of_one_material_solve_as_one():
    """#112: across a face where parts of one material touch, nothing changes,
    so it carries no charge. Both parts' charges there, which had to cancel,
    leaked on a partial contact: 16 % at level 2 against the same solid as one
    mesh."""
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 1.0, 0),
        dimension=(0.005,) * 3,
        position=(-0.001, 0.01, 0.0035),
    )
    line = [(x, 0.008, 0.0025) for x in np.linspace(-0.02, 0.02, 21)]
    fields = []
    for as_mesh in (False, True):
        coll = magpy.Collection(*_bars(as_mesh), magnet.copy())
        for part in coll.sources_all[:-1]:
            part.susceptibility = 999.0
        coll.sources_all[-1].susceptibility = 0.0
        fields.append(apply_face_charge(coll, level=2).getB(line))
    assert _relative(*fields) < 0.015


def test_touching_shells_of_one_mesh_solve_as_two_meshes():
    """A multi-body export: two touching boxes in one mesh. Graded and joined
    as two meshes are; 28 % apart before."""

    def box(a, z):
        V = np.array([[x, y, w] for x in (-a, a) for y in (-a, a) for w in (-a, a)])
        V = V / 2 + (0, 0, z)
        return V, ConvexHull(V).simplices

    (V1, F1), (V2, F2) = box(0.01, 0), box(0.006, 0.008)
    one = magpy.magnet.TriangularMesh(
        vertices=np.vstack([V1, V2]),
        faces=np.vstack([F1, F2 + len(V1)]),
        polarization=(0, 0, 0),
        check_disconnected=False,
    )
    two = [
        magpy.magnet.TriangularMesh(vertices=V, faces=F, polarization=(0, 0, 0))
        for V, F in ((V1, F1), (V2, F2))
    ]
    fields = []
    for parts in ([one], two):
        coll = magpy.Collection(*parts)
        coll.susceptibility, coll.H_ext = 100.0, (0.004, 0, 0.008)
        fields.append(
            apply_face_charge(coll, level=2).getB([(0, 0, 0.012), (0.009, 0, 0.004)])
        )
    assert _relative(*fields) < 1e-4


def test_a_part_inside_parts_of_its_material_keeps_no_faces():
    """A block of 3 x 3 x 3 touching cubes: the centre one shares all its
    faces, so its sheet has none, draws nothing and reports the block's
    polarization. Building that empty sheet crashed."""

    def cube(a, at=(0, 0, 0)):
        return magpy.magnet.Cuboid(
            dimension=(a,) * 3, polarization=(0, 0, 0), position=at
        )

    a, steps = 0.002, (-1, 0, 1)
    block = [
        cube(a, (a * i, a * j, a * k)) for i in steps for j in steps for k in steps
    ]
    models = [magpy.Collection(*block), magpy.Collection(cube(3 * a))]
    solved = []
    for model, level in zip(models, (1, 4), strict=True):
        model.susceptibility, model.H_ext = 1000, (0.002, 0, 0.01)
        solved.append(apply_face_charge(model, level=level))
    centre, whole = solved[0].children[13], solved[1].children[0]
    assert len(centre.faces) == 0
    assert centre.get_trace() == []
    assert _relative(centre.mean_polarization, whole.mean_polarization) < 0.01
    sensors = [(0.006, 0.002, 0.001), (0, 0, 0.007), (-0.005, 0.004, -0.003)]
    assert _relative(*(s.getB(sensors) for s in solved)) < 0.01


@pytest.mark.parametrize(
    ("mesh", "gap", "rtol"),
    [(False, 1e-8, 1e-4), (True, 1e-7, 2e-3)],
    ids=["cuboids", "meshes"],
)
def test_a_gap_below_the_panels_moves_the_field_as_little(mesh, gap, rtol):
    """#105, #112: across a gap far below the panels the two faces' charges
    nearly cancel. Graded toward each other's outline and coupled by the
    exact contact term, they move off touching as little as the gap does:
    6.5e-3 for the cuboids and 2.4e-2 for the meshes, before."""

    def cube(a, at):
        if not mesh:
            return magpy.magnet.Cuboid(
                dimension=(a,) * 3, polarization=(0, 0, 0), position=at
            )
        V = np.array([[x, y, z] for x in (-a, a) for y in (-a, a) for z in (-a, a)]) / 2
        part = magpy.magnet.TriangularMesh(
            vertices=V, faces=ConvexHull(V).simplices, polarization=(0, 0, 0)
        )
        part.position = at
        return part

    sensors = [(0, 0.02, 0), (0.0075, 0, 0.012), (0, 0, -0.0105), (0.006, 0.006, 0.006)]
    fields = []
    for z in (0.008, 0.008 + gap):
        coll = magpy.Collection(cube(0.01, (0, 0, 0)), cube(0.006, (0.001, 0.0015, z)))
        coll.susceptibility = 100.0
        coll.H_ext = (8e-3, 2.4e-3, 4e-3)
        fields.append(apply_face_charge(coll, level=2).getB(sensors))
    assert _relative(*fields) < rtol


def _cube_with_a_sliver(edge, turn=None, dtype=float):
    """The cube mesh with a triangle split at the midpoint of one of its edges,
    a cube edge or a face's diagonal, and the sliver an STL repair leaves in
    the T-junction: of no area, or of rounding's once turned."""
    V, F = _cube_as_triangles()
    k, j = next(
        (k, j)
        for k in range(len(F))
        for j in range(3)
        if (np.count_nonzero(V[F[k, j]] != V[F[k, (j + 1) % 3]]) == 1) == edge
    )
    a, b, c = F[k, j], F[k, (j + 1) % 3], F[k, (j + 2) % 3]
    m = len(V)
    V = np.vstack([V, (V[a] + V[b]) / 2])
    F = np.vstack([np.delete(F, k, axis=0), [[a, m, c], [m, b, c], [a, b, m]]])
    # magpylib's own checks divide by its zero area
    return magpy.magnet.TriangularMesh(
        vertices=(V if turn is None else turn.apply(V)).astype(dtype),
        faces=F,
        polarization=(0, 0, 0),
        reorient_faces=False,
        check_selfintersecting="skip",
    )


def _awkward(case):
    """An awkward mesh, as parts, its clean twin, as parts, and the
    susceptibility they are compared at."""
    V, F = _cube_as_triangles(half=0.004)
    shift = np.array([0.004, 0.004, 0.004])
    if case == "two shells, one inside out":
        far = np.array([0.006, 0, 0])
        parts = [
            magpy.magnet.TriangularMesh(
                vertices=np.vstack([V - far, V + far]),
                faces=np.vstack([F, F[:, [0, 2, 1]] + len(V)]),
                polarization=(0, 0, 0),
                reorient_faces=False,
                check_disconnected="ignore",
            )
        ]
        twins = [_mesh_at(V + x, F) for x in (-far, far)]
        return parts, twins, 1e4
    if case == "two cubes welded at a corner":
        # the second cube's corner at the origin is the first's
        own, theirs = np.argmin(V.sum(1)), np.argmax(V.sum(1))
        index = np.r_[np.arange(own), -1, np.arange(own, len(V) - 1)] + len(V)
        index[own] = theirs
        parts = [
            magpy.magnet.TriangularMesh(
                vertices=np.vstack([V - shift, np.delete(V + shift, own, axis=0)]),
                faces=np.vstack([F, index[F]]),
                polarization=(0, 0, 0),
            )
        ]
        return parts, [_mesh_at(V - shift, F), _mesh_at(V + shift, F)], 1e6
    if case == "a cavity magpylib turned outward":
        Vi, Fi = _cube_as_triangles(half=0.002)
        V, F = 1.25 * V, np.vstack([F, Fi[:, [0, 2, 1]] + len(V)])
        V = np.vstack([V, Vi])
        parts, twins = (
            [
                magpy.magnet.TriangularMesh(
                    vertices=V,
                    faces=F,
                    polarization=(0, 0, 0),
                    check_disconnected="ignore",
                    **kwargs,
                )
            ]
            for kwargs in ({}, {"reorient_faces": False})
        )
        return parts, twins, 1e4
    if case == "a sliver on a sharp edge, turned, float32":
        turn = R.from_euler("xyz", (17, 33, 61), degrees=True)
        V, F = _cube_as_triangles()
        twin = _mesh_at(turn.apply(V).astype(np.float32), F)
        return [_cube_with_a_sliver(True, turn, np.float32)], [twin], 100.0
    return [_cube_with_a_sliver(False)], [_mesh_cube()], 100.0


def _mesh_at(V, F):
    return magpy.magnet.TriangularMesh(vertices=V, faces=F, polarization=(0, 0, 0))


@pytest.mark.parametrize(
    "case",
    [
        "two shells, one inside out",
        "two cubes welded at a corner",
        "a cavity magpylib turned outward",
        "a sliver on a sharp edge, turned, float32",
        "a flat triangle on a diagonal",
    ],
)
def test_awkward_meshes_solve_as_their_clean_twins(case):
    """Each closed shell of a mesh keeps its own net charge zero, whichever way
    it is wound, and two solids sharing only a corner are two; a shell inside
    another bounds a cavity, whichever way magpylib turned it; a sliver
    changes nothing, on a sharp edge too, beside another part too."""
    parts, twins, chi = _awkward(case)
    sensors = np.array(
        [[0, 0.02, 0], [0.013, 0, 0.004], [0, -0.004, -0.012], [0.001, 0.0005, 0]]
    )
    fields = []
    for soft in (parts, twins):
        for part in soft:
            part.susceptibility = chi
        side = _cuboid(position=(0.004, -0.002, 0.016))
        side.susceptibility = 50.0
        magnet = magpy.magnet.Cuboid(
            dimension=(0.004,) * 3, position=(0, 0.012, 0), polarization=(0, 0.5, 1)
        )
        coll = magpy.Collection(*soft, side, magnet)
        fields.append(apply_face_charge(coll, level=2).getB(sensors))
    # the welded cubes' triangles near their corner couple as close pairs of
    # one mesh, more finely than two meshes do: 7e-5 apart
    assert _relative(*fields) < 1e-4


def test_copies_keep_their_own_charge():
    a = _one_face(CHARGE)
    b = a.copy()
    b.charge_densities = [2 * CHARGE]
    obs = _observers()
    Ha = a.getH(obs)
    assert _relative(b.getH(obs), 2 * Ha) < 1e-14
    assert _relative(magpy.getH([a, b], obs, sumup=True), 3 * Ha) < 1e-14


@pytest.mark.parametrize("winding", ["inward", "outward"])
def test_mean_polarization_of_a_uniformly_magnetized_cube(winding):
    """The charge of a cube magnetized J, sigma = J.n on its outward normals,
    gives J whichever way the faces are wound: the field depends on the charge
    only."""
    corners = (
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        * 0.002
    )
    faces = np.array([[0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4],
                      [2, 6, 7], [2, 7, 3], [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6]])  # fmt: skip
    if winding == "outward":
        faces = faces[:, ::-1]
    J = np.array([0.2, -0.5, 1.1])
    T = corners[faces]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    n *= np.sign(np.einsum("fd,fd->f", n, T.mean(1)))[:, None]  # outward
    sigma = n @ J / magpy.mu_0
    sheet = ChargeSheet(corners, faces, np.repeat(sigma[:, None], 3, axis=1))
    np.testing.assert_allclose(sheet.mean_polarization, J, atol=1e-14)
    cube = magpy.magnet.Cuboid(dimension=(0.004,) * 3, polarization=J)
    np.testing.assert_allclose(
        sheet.getB((0.01, 0.007, 0.012)), cube.getB((0.01, 0.007, 0.012)), rtol=1e-9
    )


def test_cube_polarizability_converges_from_below():
    """Helsing and Perfekt (2013): 3.644305 for a cube of infinite susceptibility."""
    alphas = []
    for level in (3, 4):
        cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
        cube.susceptibility = 1e6
        cube.H_ext = (0.0, 0.0, 1e-3)
        sheet = apply_face_charge(magpy.Collection(cube), level=level).sources_all[0]
        alphas.append(sheet.mean_polarization[2] / 1e-3)
    assert alphas[0] < alphas[1] < 3.644305
    assert abs(alphas[1] / 3.644305 - 1) < 6e-4


def test_turned_collection_turns_the_field():
    def model():
        plate = magpy.magnet.Cuboid(
            dimension=(0.01, 0.006, 0.004), polarization=(0, 0, 0)
        )
        plate.susceptibility = 500.0
        block = magpy.magnet.Cuboid(
            dimension=(0.004, 0.004, 0.006),
            position=(0.007, 0, 0.001),
            polarization=(0, 0, 0),
        )
        block.susceptibility = 200.0  # touches the plate's +x face
        magnet = magpy.magnet.Cuboid(
            dimension=(0.003,) * 3, position=(0, 0, 0.005), polarization=(0.2, 0, 1.0)
        )
        return magpy.Collection(plate, block, magnet)

    turn = R.from_euler("xyz", (20, -35, 50), degrees=True)
    obs = np.array([[0.0, 0.0, 0.012], [0.015, 0.004, 0.0], [-0.01, -0.008, -0.006]])
    B = apply_face_charge(model(), level=2).getB(obs)
    B_turned = apply_face_charge(model().rotate(turn, anchor=0), level=2).getB(
        turn.apply(obs)
    )
    assert _relative(B_turned, turn.apply(B)) < 1e-10


def test_returns_a_solved_copy():
    coll = _plate_under_magnet()
    plate, _ = coll.sources_all
    plate.H_ext = (0.0, 0.0, 0.005)
    solved = apply_face_charge(coll, level=2)
    sheet, kept = solved.sources_all
    assert isinstance(sheet, ChargeSheet)
    assert sheet.style.label == "plate"
    assert isinstance(
        kept, magpy.magnet.Cuboid
    )  # susceptibility 0: magpylib's own magnet
    assert coll.sources_all[0] is plate  # the input is untouched
    far = [0.0, 0.0, -0.1]  # the applied field (5 mT) is left out, as by apply_demag
    assert np.linalg.norm(solved.getB(far)) < 1e-4


@pytest.mark.parametrize(
    ("shape", "factor"), [("cuboid", 5), ("disc", 2)], ids=["panels", "triangles"]
)
def test_refine_lowers_the_jump_indicator(shape, factor):
    solved = [
        apply_face_charge(_plate_under_magnet(shape), level=3, refine=refine)
        for refine in (0, 2)
    ]
    jumps = [s.sources_all[0].jump_indicator for s in solved]
    assert jumps[1] < jumps[0] / factor
    assert [s.face_charge_info["rounds"] for s in solved] == [0, 2]


@pytest.mark.parametrize(
    "src",
    [
        magpy.magnet.Cylinder(dimension=(2, 2)),
        magpy.magnet.CylinderSegment(dimension=(0.5, 1, 2, 0, 360)),
        magpy.magnet.CylinderSegment(dimension=(0.5, 1, 2, -30, 250)),
        magpy.magnet.CylinderSegment(dimension=(0, 1, 2, 10, 200)),
        magpy.magnet.Sphere(diameter=2),
    ],
    ids=["cylinder", "tube", "segment", "slice", "sphere"],
)
def test_curved_parts_are_faceted_closed_outward_at_their_volume(src):
    V, F = _closed_surface(src, level=2)
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    assert len(np.unique(directed, axis=0)) == len(directed)  # wound one way
    pairs = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)[1]
    assert np.all(pairs == 2)  # closed
    volume = np.einsum("fd,fd->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6
    exact = (
        np.pi * src.diameter**3 / 6
        if isinstance(src, magpy.magnet.Sphere)
        else np.pi * src.dimension[0] ** 2 / 4 * src.dimension[1]
        if isinstance(src, magpy.magnet.Cylinder)
        else np.radians(src.dimension[4] - src.dimension[3])
        * (src.dimension[1] ** 2 - src.dimension[0] ** 2)
        / 2
        * src.dimension[2]
    )
    assert volume == pytest.approx(exact, rel=1e-12)  # positive: outward


def test_coaxial_parts_share_their_facets():
    """A cylinder in a tube that touches it: the tube's inner corners are the
    cylinder's, wherever the tube's angles start, so the two do not overlap."""
    core, _ = _closed_surface(magpy.magnet.Cylinder(dimension=(2, 2)), level=2)
    tube, _ = _closed_surface(
        magpy.magnet.CylinderSegment(dimension=(1, 1.5, 2, -45, 315)), level=2
    )
    rim = core[np.abs(np.linalg.norm(core[:, :2], axis=1)) > 0.5]
    assert {tuple(v) for v in np.round(rim, 9)} <= {tuple(v) for v in np.round(tube, 9)}


def test_soft_sphere_gives_the_exact_dipole():
    a, chi, B0 = 0.005, 1000.0, 0.01
    sphere = magpy.magnet.Sphere(diameter=2 * a, polarization=(0, 0, 0))
    sphere.susceptibility = chi
    model = magpy.Collection(sphere)
    model.H_ext = (0, 0, B0)
    m = np.array([0, 0, 4 * np.pi * a**3 * chi / (chi + 3) * B0 / magpy.mu_0])
    dipole = magpy.misc.Dipole(moment=m)
    solved = apply_face_charge(model, level=1)
    for points, rtol in (
        ([(0, 0, 0.006), (0.006, 0, 0)], 5e-3),
        ([(0, 0, 0.05)], 1e-3),
    ):
        assert _relative(solved.getB(points), dipole.getB(points)) < rtol


def test_show_colours_the_charge_and_draws_panel_edges():
    sheet = apply_face_charge(_plate_under_magnet(), level=2).sources_all[0]
    sheet.style.mesh.grid.show = True
    fig = magpy.show(sheet, backend="plotly", return_fig=True)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    edges = next(t for t in fig.data if t.type == "scatter3d")
    q = sheet._smoothed()
    full = sheet.style.charge.level
    assert full < np.abs(q).max()  # not the singular corner's
    v = np.clip(q, -full, full)
    np.testing.assert_allclose(mesh.intensity, (v - v.min()) / np.ptp(v))
    F = sheet.faces
    triangle_edges = len(
        np.unique(
            np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1),
            axis=0,
        )
    )
    assert len(edges.x) // 3 == triangle_edges - len(F) // 2  # no panel's diagonal


@pytest.mark.parametrize("name", ["vertices", "faces", "charge_densities"])
def test_new_data_drops_the_colour_level_of_its_solve(name):
    triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    sheet = ChargeSheet(triangle, [(0, 1, 2)], [(1, 2, 4)], style_charge_level=4)
    setattr(sheet, name, getattr(sheet, name))
    assert sheet.style.charge.level is None


def test_parts_solved_together_share_one_colour_scale():
    # a magnet with a susceptibility, solved too, glued on an iron plate
    plate = magpy.magnet.Cuboid(dimension=(0.02, 0.02, 0.005), polarization=(0, 0, 0))
    plate.susceptibility = 1000.0
    magnet = magpy.magnet.Cuboid(
        dimension=(0.006, 0.006, 0.004),
        position=(0, 0, 0.0045),
        polarization=(0, 0, 1.2),
    )
    magnet.susceptibility = 0.05
    solved = apply_face_charge(magpy.Collection(plate, magnet), level=2)
    near, far = solved.sources_all
    assert near.style.charge.level == far.style.charge.level
    # where parts touch, the faces carry large charges, out of sight and
    # nearly cancelled by the other side's: they do not set the scale
    assert near.style.charge.level < saturation([near, far]) / 2


@pytest.mark.parametrize(
    "charge",
    [(-4, 1, 2), (1, 2, 4), (-4, -2, -1), (-8, 0, 8), (2, 2, 2), (0, 0, 0)],
    ids=["both signs", "north", "south", "clipped", "constant", "none"],
)
def test_the_colour_at_each_corner_follows_the_charge_and_the_level(charge):
    triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    sheet = ChargeSheet(
        triangle,
        [(0, 1, 2)],
        [charge],
        style_charge_level=4,
        style_magnetization_color_north="#ff0000",
        style_magnetization_color_middle="#808080",
        style_magnetization_color_south="#00ff00",
    )
    fig = magpy.show(sheet, backend="plotly", return_fig=True)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    middle = np.array(to_rgb("#808080"))
    # plotly, pyvista and studio read it by its own range, a backend may as it is
    for limits in ({}, {"cmin": 0, "cmax": 1}):
        drawn = colors_from_colorscale(mesh.intensity, mesh.colorscale, **limits)
        for q, color in zip(charge, drawn, strict=True):
            end = np.array(to_rgb("#ff0000" if q > 0 else "#00ff00"))
            expected = middle + min(abs(q) / 4, 1) * (end - middle)
            np.testing.assert_allclose(to_rgb(color), expected, atol=2 / 255)


def _cuboid(**kwargs):
    return magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0), **kwargs)


def _mesh_cube(position=(0, 0, 0), **kwargs):
    """A 10 mm cube as a TriangularMesh of 12 triangles."""
    V, F = _cube_as_triangles()
    return magpy.magnet.TriangularMesh(
        vertices=V, faces=F, position=position, polarization=(0, 0, 0), **kwargs
    )


def _cubes_sharing_an_edge():
    """Two cubes welded along an edge: four triangles on it."""
    V, F = _cube_as_triangles()
    shift = np.array([0.01, 0.01, 0])
    V2 = V + shift
    # the second cube's vertices on the first's edge are the first's
    same = [int(np.flatnonzero(np.all(np.isclose(V, v), 1))[0]) for v in V2[:2]]
    index = np.arange(len(V), 2 * len(V))
    index[:2] = same
    return magpy.magnet.TriangularMesh(
        vertices=np.vstack([V, V2]),
        faces=np.vstack([F, index[F]]),
        polarization=(0, 0, 0),
        reorient_faces=False,
        check_open="skip",
    )


def _flipped_cube():
    """The cube mesh with one triangle turned over, left so."""
    V, F = _cube_as_triangles()
    F[0] = F[0, [0, 2, 1]]
    return magpy.magnet.TriangularMesh(
        vertices=V, faces=F, polarization=(0, 0, 0), reorient_faces=False
    )


@pytest.mark.parametrize(
    ("model", "chi", "error"),
    [
        (
            lambda: magpy.Collection(
                magpy.misc.Triangle(
                    vertices=[(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)],
                    polarization=(0, 0, 0),
                )
            ),
            10.0,
            NotImplementedError,
        ),
        (lambda: magpy.Collection(_cuboid()), (1.0, 2.0, 3.0), NotImplementedError),
        (
            lambda: magpy.Collection(_cuboid(position=[(0, 0, 0), (0, 0, 0.01)])),
            10.0,
            NotImplementedError,
        ),
        (
            lambda: magpy.Collection(
                _cuboid(),
                magpy.current.Circle(current=[1, 2, 3], diameter=0.02),
            ),
            10.0,
            NotImplementedError,
        ),
        (
            lambda: magpy.Collection(
                magpy.magnet.Cuboid(
                    dimension=(0.01,) * 3, polarization=[(0, 0, 1), (0, 0, 2)]
                )
            ),
            10.0,
            NotImplementedError,
        ),
        (_cuboid, 10.0, TypeError),
        (lambda: magpy.Collection(_flipped_cube()), 10.0, ValueError),
        (lambda: magpy.Collection(_cubes_sharing_an_edge()), 10.0, ValueError),
    ],
    ids=[
        "no volume",
        "anisotropic",
        "path",
        "current path",
        "polarization path",
        "not a collection",
        "faces wound both ways",
        "an edge of four triangles",
    ],
)
def test_unsupported_input_raises(model, chi, error):
    model = model()
    for src in getattr(model, "sources_all", [model]):
        src.susceptibility = chi
    with pytest.raises(error):
        apply_face_charge(model)


def test_too_many_unknowns_raise_before_assembly():
    cube = _cuboid()
    cube.susceptibility = 10
    with pytest.raises(ValueError, match="max_unknowns"):
        apply_face_charge(magpy.Collection(cube), max_unknowns=10)
    cube = _cuboid()
    cube.susceptibility, cube.H_ext = 10, (0, 0, 0.01)
    # by default the highest level that fits; a level asked for raises
    with pytest.warns(UserWarning, match="solved at level 2"):
        solved = apply_face_charge(magpy.Collection(cube.copy()), max_unknowns=300)
    assert solved.face_charge_info["level"] == 2
    with pytest.raises(ValueError, match="max_unknowns"):
        apply_face_charge(magpy.Collection(cube.copy()), level=3, max_unknowns=300)
    # a refinement round over the limit is skipped: the round before is kept
    with pytest.warns(UserWarning, match="round 0's"):
        solved = apply_face_charge(
            magpy.Collection(cube), level=2, refine=1, max_unknowns=300
        )
    assert np.isfinite(solved.getB((0, 0, 0.01))).all()
    info = solved.face_charge_info  # the solve that was kept, not the skipped one
    assert (info["level"], info["rounds"]) == (2, 0)
    assert 0 < info["unknowns"] <= 300
    assert info["seconds"] > 0
    # a thin plate as a triangle mesh is refused from a count of its cells,
    # before cutting them, which did not end (#106)
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    V = V * (0.005, 0.005, 2.5e-5)
    plate = magpy.magnet.TriangularMesh(
        vertices=V, faces=ConvexHull(V).simplices, polarization=(0, 0, 0)
    )
    plate.susceptibility = 100
    with pytest.raises(ValueError, match="Even level 1 needs at least"):
        apply_face_charge(magpy.Collection(plate))


def test_threads_take_the_callers_numpy_error_settings():
    def divide(_):
        return np.ones(1) / np.zeros(1)

    with np.errstate(divide="raise"), pytest.raises(FloatingPointError):
        run(divide, range(8))


#: a parent fills its pool, forks, and its child solves: in a process of its
#: own, as forking beside this suite's other threads hung the fork itself on
#: Linux with Python 3.11
FORK_AFTER_A_SOLVE = """
import os, sys, time
from magpylib_material_response.face_charge import _parallel
_parallel.threads = lambda: 4
_parallel.run(time.sleep, [1e-3] * 8, work=10**7)
pid = os.fork()
if pid == 0:
    try:
        _parallel.run(time.sleep, [1e-3] * 8, work=10**7)
    finally:
        os._exit(0)
deadline = time.monotonic() + 20
while not os.waitpid(pid, os.WNOHANG)[0]:
    if time.monotonic() > deadline:
        os.kill(pid, 9)
        sys.exit("the forked child hung")
    time.sleep(0.05)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork here")
def test_a_forked_child_runs_on_threads_of_its_own():
    """A child forked after a solve, as a multiprocessing.Pool on Linux does,
    inherited the pool but none of its threads, and waited forever."""
    done = subprocess.run(
        [sys.executable, "-c", FORK_AFTER_A_SOLVE],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    "condition", [1e3, 1e10], ids=["refined", "past single precision"]
)
def test_mixed_precision_solves_to_double_accuracy(monkeypatch, condition):
    monkeypatch.setattr("magpylib_material_response.face_charge._bodies.MIXED", 0)
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.normal(size=(200, 200)))[0]
    A = np.asfortranarray(q * np.geomspace(1, 1 / condition, 200) @ q.T)
    b = rng.normal(size=200)
    x = _solve(A.copy(order="F"), b)
    residual = np.abs(b - A @ x).max() / (np.abs(A).sum(1).max() * np.abs(x).max())
    assert residual < 1e-14


@pytest.mark.parametrize(
    ("gap", "fem", "rtol"),
    [
        (0.0, (296.762, 7.806, 3.882), 0.01),
        (1e-9, (296.762, 7.806, 3.882), 0.01),
        (1e-4, (308.073, 7.721, 3.858), 0.02),
    ],
)
@pytest.mark.parametrize("soft", [_cuboid, _mesh_cube], ids=["cuboid", "mesh"])
def test_magnet_on_a_soft_part_matches_fem(gap, fem, rtol, soft):
    """A magnet without susceptibility on soft iron, given as a cuboid or as a
    triangle mesh, touching or 0.1 mm above: its field jumps at the edges of
    its footprint, which the mesh and the load must follow. The references are
    fem_study at orders 3 to 5, good to 5e-5 at each sensor."""
    soft = soft()
    soft.susceptibility = 1000
    magnet = magpy.magnet.Cuboid(
        dimension=(0.004, 0.004, 0.002),
        polarization=(0, 0, 1.2),
        position=(0, 0, 0.006 + gap),
    )
    magnet.susceptibility = 0
    sensors = np.array([(0, 0, 8), (6, 0, 2), (0, 0, -7)]) * 1e-3
    B = apply_face_charge(magpy.Collection(soft, magnet)).getB(sensors)
    np.testing.assert_allclose(
        np.linalg.norm(B, axis=1), np.array(fem) * 1e-3, rtol=rtol
    )


@pytest.mark.parametrize(
    "magnet",
    [
        lambda: magpy.magnet.Cylinder(dimension=(0.004, 0.002), polarization=(0, 0, 1)),
        lambda: magpy.magnet.Cuboid(
            dimension=(0.004, 0.004, 0.002), polarization=(0, 0, 1)
        ).rotate_from_angax(45, "z"),
    ],
    ids=["cylinder", "turned cuboid"],
)
def test_unaligned_magnet_touching_warns(magnet):
    soft = _cuboid()
    soft.susceptibility = 1000
    magnet = magnet().move((0, 0, 0.006))
    with pytest.warns(UserWarning, match="cannot follow its footprint"):
        apply_face_charge(magpy.Collection(soft, magnet), level=1)


@pytest.mark.parametrize("offset", [1e-13, 1e-10])
def test_parts_a_hair_apart_solve_as_touching(offset):
    """Two soft blocks side by side on a soft plate, their shared face off by a
    hair, as in rounded CAD data."""

    def solve(shift):
        plate = magpy.magnet.Cuboid(
            dimension=(0.01, 0.01, 0.001), polarization=(0, 0, 0)
        )
        blocks = [
            magpy.magnet.Cuboid(
                dimension=(0.005, 0.01, 0.002),
                polarization=(0, 0, 0),
                position=(x, 0, 0.0015),
            )
            for x in (-0.0025, 0.0025 + shift)
        ]
        coll = magpy.Collection(plate, *blocks)
        coll.susceptibility, coll.H_ext = 1000, (0, 0, 0.01)
        return apply_face_charge(coll, level=2).getB([(0, 0, 0.004), (0.008, 0, 0)])

    # the block moved by the offset itself: 2e-8 of its size
    np.testing.assert_allclose(solve(offset), solve(0.0), rtol=1e-6, atol=1e-9)


def test_currents_take_no_susceptibility():
    """A susceptibility set on the collection reaches its magnets, not a coil in
    it, as with apply_demag."""
    plate = _cuboid()
    coil = magpy.current.Circle(current=10, diameter=0.02, position=(0, 0, 0.01))
    coll = magpy.Collection(plate, coil)
    coll.susceptibility = 1000
    solved = apply_face_charge(coll, level=1)
    assert isinstance(solved.sources_all[0], ChargeSheet)
    assert np.isfinite(solved.getB((0, 0, 0.02))).all()


def test_a_flat_magnet_beside_a_meshed_part_drives_it():
    """A misc.Triangle has no hull for the contact checks: its triangle stands
    for it."""
    disc = magpy.magnet.Cylinder(dimension=(0.01, 0.01), polarization=(0, 0, 0))
    disc.susceptibility = 100
    flat = magpy.misc.Triangle(
        polarization=(0, 0, 1), vertices=[(0.02, 0, 0), (0.03, 0, 0), (0.02, 0.01, 0)]
    )
    solved = apply_face_charge(magpy.Collection(disc, flat), level=1)
    assert np.isfinite(solved.getB((0, 0, 0.02))).all()


@pytest.mark.parametrize(
    ("level", "refine", "error"),
    [(0, 0, ValueError), (2, -1, ValueError), (2.5, 0, TypeError), (2, 1.0, TypeError)],
)
def test_bad_level_or_refine_raises(level, refine, error):
    cube = _cuboid()
    cube.susceptibility = 10
    with pytest.raises(error):
        apply_face_charge(magpy.Collection(cube), level=level, refine=refine)


def test_sheet_is_closed_finite_at_its_corners_and_read_only():
    """A block flush with the plate's side up to CAD rounding: the plate's
    panels still reach its edges, its field is finite at every panel corner,
    and its charge cannot be edited behind the field's back."""
    plate = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.001), polarization=(0, 0, 0))
    block = magpy.magnet.Cuboid(
        dimension=(0.004, 0.004, 0.002),
        polarization=(0, 0, 0),
        position=(0.003 - 3e-9, 0, 0.0015),
    )
    coll = magpy.Collection(plate, block)
    coll.susceptibility, coll.H_ext = 1000, (0, 0, 0.01)
    sheet = apply_face_charge(coll, level=2).sources_all[0]
    np.testing.assert_allclose(np.ptp(sheet.vertices, axis=0), plate.dimension)
    assert np.isfinite(sheet.getB(sheet.vertices + sheet.position)).all()
    with pytest.raises(ValueError, match="read-only"):
        sheet.charge_densities[:] *= 2


def _soft(dimension=(0.01, 0.01, 0.002), position=(0, 0, 0), chi=1000.0, J=(0, 0, 0)):
    part = magpy.magnet.Cuboid(polarization=J, dimension=dimension, position=position)
    part.susceptibility = chi
    return part


def _drive(x=0.0, z=0.0045, tilt=0.0):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(0.005, 0.005, 0.005),
        position=(x, 0, z),
        orientation=R.from_euler("x", tilt, degrees=True),
    )


def _scaled(k):
    return magpy.Collection(
        magpy.magnet.Cuboid(
            polarization=(0, 0, 1),
            dimension=(0.005 * k, 0.005 * k, 0.005 * k),
            position=(0, 0, 0.0045 * k),
        ),
        _soft((0.01 * k, 0.01 * k, 0.002 * k)),
    )


def _with_field(field):
    model = magpy.Collection(_drive(), _soft())
    model.H_ext = (field, -field, 0)
    return model


def _turned(k):
    """Two soft parts turned and moved together, a magnet beside them."""
    parts = magpy.Collection(_soft(), _soft((0.004, 0.004, 0.004), (0, 0, 0.006)))
    parts.rotate(R.from_euler("xyz", (20 * k, 13 * k, 7 * k), degrees=True), anchor=0)
    parts.move((0.001 * k, -0.002 * k, 0))
    return magpy.Collection(_drive(0.015, 0), parts)


def _shaft(angle):
    shaft = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(0.02, 0.04), position=(0, 0, -0.02)
    )
    shaft.susceptibility = 3999
    magnet = magpy.magnet.Cylinder(
        polarization=(0.5, 0, 0), dimension=(0.006, 0.003), position=(0, 0, 0.0025)
    )
    return magpy.Collection(
        magpy.Collection(shaft, magnet).rotate_from_angax(angle, "z", anchor=0)
    )


def _models(make, *values):
    return [lambda v=v: make(v) for v in values]


HITS = (False, False, True, True)  # kept on its second call, reused from its third
KEPT = {
    "a magnet moves": (
        _models(
            lambda x: magpy.Collection(_drive(x, tilt=x * 1e3), _soft()),
            0,
            1e-3,
            2e-3,
            3e-3,
        ),
        {},
        HITS,
    ),
    "above the mixed-precision size": (
        _models(lambda x: magpy.Collection(_drive(x), _soft()), 0, 1e-3, 2e-3, 3e-3),
        {"level": 4},
        HITS,
    ),
    "H_ext changes": (_models(_with_field, 0.0, 5e-3, -0.2, 1.0), {}, HITS),
    "the part's remanence changes": (
        _models(
            lambda j: magpy.Collection(_drive(), _soft(chi=0.05, J=(0, 0, j))),
            0.2,
            0.5,
            -1.0,
            0.1,
        ),
        {},
        HITS,
    ),
    "parts move as one": (_models(_turned, 0, 1, 2, 3), {}, HITS),
    "a mesh turns": (_models(_shaft, 0, 30, 77, 190), {}, HITS),
    "parts of one material touch": (
        _models(
            lambda x: magpy.Collection(
                _drive(x, 0.01),
                _soft(position=(-0.005, 0, 0)),
                _soft(position=(0.005, 0, 0)),
            ),
            0,
            1e-3,
            2e-3,
            3e-3,
        ),
        {},
        HITS,
    ),
    "parts move apart": (
        _models(
            lambda z: magpy.Collection(
                _drive(0.015, 0), _soft(), _soft(position=(0, 0, z))
            ),
            0.003,
            0.004,
            0.005,
        ),
        {},
        (False, False, False),
    ),
    "chi changes": (
        _models(lambda x: magpy.Collection(_drive(), _soft(chi=x)), 1000, 999, 10),
        {},
        (False, False, False),
    ),
    "the model is scaled": (
        _models(_scaled, 1, 1, 1, 2, 1000),
        {},
        (False, False, True, False, False),
    ),
    "a length changes": (
        _models(
            lambda L: magpy.Collection(_drive(), _soft((L, 0.01, 0.002))),
            0.01,
            0.01001,
            0.0105,
        ),
        {},
        (False, False, False),
    ),
    "a magnet's footprint moves on a part": (
        _models(
            lambda x: magpy.Collection(_drive(x, 0.0035), _soft()), 0, 7e-4, 1.4e-3
        ),
        {},
        (False, False, False),
    ),
    "parts swap places": (
        [
            lambda: magpy.Collection(
                _drive(0.015, 0), _soft(), _soft((0.004, 0.004, 0.004), (0, 0, 0.006))
            ),
            lambda: magpy.Collection(
                _drive(0.015, 0), _soft((0.004, 0.004, 0.004), (0, 0, 0.006)), _soft()
            ),
        ],
        {},
        (False, False),
    ),
    "refinement": (
        _models(lambda x: magpy.Collection(_drive(x), _soft()), 0, 1e-3, 2e-3),
        {"refine": 1},
        None,  # the layout follows the solved charge
    ),
}


@pytest.fixture
def _no_cache_left():
    clear_cache()
    yield
    set_cache_limit(1)
    clear_cache()


@pytest.mark.usefixtures("_no_cache_left")
@pytest.mark.parametrize("case", KEPT)
def test_a_kept_factorization_serves_only_its_own_matrix(case):
    """A key that misses what makes the matrix would give wrong fields silently,
    so every call is checked against a solve with no cache."""
    models, kwargs, reused = KEPT[case]
    observers = np.array([[0.012, 0, 0], [0, 0.012, 0.001], [0.007, 0.007, 0.007]])
    kept = []
    for model in models:
        solved = apply_face_charge(
            model(), level=kwargs.get("level", 2), refine=kwargs.get("refine", 0)
        )
        kept.append((magpy.getB(solved, observers), solved.face_charge_info["reused"]))
    set_cache_limit(0)
    for model, (B, _) in zip(models, kept, strict=True):
        fresh = magpy.getB(
            apply_face_charge(
                model(), level=kwargs.get("level", 2), refine=kwargs.get("refine", 0)
            ),
            observers,
        )
        np.testing.assert_allclose(B, fresh, rtol=0, atol=1e-11 * np.abs(fresh).max())
    if reused is not None:
        assert tuple(r for _, r in kept) == reused


@pytest.mark.usefixtures("_no_cache_left")
def test_a_kept_factorization_is_safe_from_threads():
    """Threads share one kept factorization, whose pivots scipy's lu_solve shifts
    in place (#131): shared as they are, nearly every answer here is wrong."""
    observers = np.array([[0.012, 0, 0], [0, 0, 0.012]])
    positions = np.linspace(-1e-3, 1e-3, 40)

    def read(x):
        return magpy.getB(
            apply_face_charge(magpy.Collection(_drive(x), _soft()), level=2), observers
        )

    fresh = [read(x) for x in positions]  # keeps it, for the threads to share
    with ThreadPoolExecutor(8) as pool:
        for _ in range(3):
            threaded = list(pool.map(read, positions))
            np.testing.assert_allclose(threaded, fresh, rtol=1e-10)
    # and the factorization is still good afterwards
    np.testing.assert_allclose(read(positions[0]), fresh[0], rtol=1e-10)


@pytest.mark.usefixtures("_no_cache_left")
def test_the_cache_stays_within_its_limit():
    def reused(chi=1000.0):
        model = magpy.Collection(_drive(), _soft(chi=chi))
        return apply_face_charge(model, level=2).face_charge_info["reused"]

    assert [reused() for _ in range(3)] == [False, False, True]
    clear_cache()
    assert [reused() for _ in range(3)] == [False, False, True]
    # a limit that takes one: the older goes
    one = sum(lu.nbytes + piv.nbytes for lu, piv in _cache._kept.values())
    set_cache_limit(1.5 * one / 1e9)
    assert [reused(500.0) for _ in range(3)] == [False, False, True]
    assert len(_cache._kept) == 1
    # too small for any: never factored for keeping, nor reused
    set_cache_limit(one / 2e9)
    assert not _cache._kept
    assert [reused() for _ in range(3)] == [False, False, False]
    assert not _cache._kept
    with pytest.raises(ValueError, match="0 or more"):
        set_cache_limit(-1)
    # more than fit, in a cycle: some still reused, not factored on every call
    set_cache_limit(2.5 * one / 1e9)
    cycle = [reused(chi) for chi in (100.0, 200.0, 300.0) * 5]
    assert sum(cycle) >= 4
    set_cache_limit(float("inf"))
    clear_cache()
    assert [reused() for _ in range(3)] == [False, False, True]


@pytest.mark.parametrize(
    ("matrix", "kept"), [(np.eye(3), True), (np.diag([1, 1, 1e-20]), False)]
)
def test_a_kept_factorization_needs_a_sound_matrix(matrix, kept):
    """Later calls only back-substitute: they would lose scipy.linalg.solve's
    error on a singular matrix and warning on an ill-conditioned one."""
    if kept:
        assert _cache._factor(matrix.copy())[2]
    else:
        with pytest.warns(LinAlgWarning, match="Ill-conditioned"):
            assert not _cache._factor(matrix.copy())[2]
        with pytest.raises(LinAlgError, match="singular"):
            _cache._factor(np.zeros((3, 3)))
