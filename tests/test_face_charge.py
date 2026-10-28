"""Tests of the face-charge solver and its ChargeSheet source."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.face_charge import ChargeSheet, apply_face_charge
from magpylib_material_response.face_charge._apply import _closed_surface
from magpylib_material_response.face_charge._bodies import _solve, solve_system
from magpylib_material_response.face_charge._mesh import graded_triangles
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
    full = sheet._saturation
    assert full < np.abs(q).max()  # not the singular corner's
    np.testing.assert_allclose(mesh.intensity, np.clip(q, -full, full))
    F = sheet.faces
    triangle_edges = len(
        np.unique(
            np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1),
            axis=0,
        )
    )
    assert len(edges.x) // 3 == triangle_edges - len(F) // 2  # no panel's diagonal


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
    assert near._saturation == far._saturation
    # where parts touch, the faces carry large charges, out of sight and
    # nearly cancelled by the other side's: they do not set the scale
    assert near._saturation < saturation([near, far]) / 2
    fig = magpy.show(solved, backend="plotly", return_fig=True)
    for mesh in (t for t in fig.data if t.type == "mesh3d"):
        assert np.abs(mesh.intensity).max() <= near._saturation


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


def test_threads_take_the_callers_numpy_error_settings():
    def divide(_):
        return np.ones(1) / np.zeros(1)

    with np.errstate(divide="raise"), pytest.raises(FloatingPointError):
        run(divide, range(8))


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
