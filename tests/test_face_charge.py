"""Tests of the face-charge solver and its ChargeSheet source."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.face_charge import ChargeSheet, apply_face_charge

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


def _plate_under_magnet():
    plate = magpy.magnet.Cuboid(
        dimension=(0.01, 0.01, 0.004), polarization=(0, 0, 0), style_label="plate"
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


def test_refine_lowers_the_jump_indicator():
    jumps = [
        apply_face_charge(_plate_under_magnet(), level=3, refine=refine)
        .sources_all[0]
        .jump_indicator
        for refine in (0, 2)
    ]
    assert jumps[1] < jumps[0] / 5


def test_show_colours_the_charge_and_draws_panel_edges():
    sheet = apply_face_charge(_plate_under_magnet(), level=2).sources_all[0]
    sheet.style.mesh.grid.show = True
    fig = magpy.show(sheet, backend="plotly", return_fig=True)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    edges = next(t for t in fig.data if t.type == "scatter3d")
    q = sheet._smoothed()
    np.testing.assert_allclose(mesh.intensity, np.sign(q) * np.sqrt(np.abs(q)))
    F = sheet.faces
    triangle_edges = len(
        np.unique(
            np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1),
            axis=0,
        )
    )
    assert len(edges.x) // 3 == triangle_edges - len(F) // 2  # no panel's diagonal


def _cuboid(**kwargs):
    return magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0), **kwargs)


@pytest.mark.parametrize(
    ("model", "chi", "error"),
    [
        (
            lambda: magpy.Collection(
                magpy.magnet.Cylinder(dimension=(0.01, 0.01), polarization=(0, 0, 0))
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
    ],
    ids=[
        "not a cuboid",
        "anisotropic",
        "path",
        "current path",
        "polarization path",
        "not a collection",
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
    # a refinement round over the limit is skipped: the round before is kept
    cube = _cuboid()
    cube.susceptibility, cube.H_ext = 10, (0, 0, 0.01)
    with pytest.warns(UserWarning, match="round 0's"):
        solved = apply_face_charge(
            magpy.Collection(cube), level=2, refine=1, max_unknowns=300
        )
    assert np.isfinite(solved.getB((0, 0, 0.01))).all()


@pytest.mark.parametrize(
    ("gap", "fem", "rtol"),
    [
        (0.0, (296.762, 7.806, 3.882), 0.01),
        (1e-9, (296.762, 7.806, 3.882), 0.01),
        (1e-4, (308.073, 7.721, 3.858), 0.02),
    ],
)
def test_magnet_on_a_soft_part_matches_fem(gap, fem, rtol):
    """A magnet without susceptibility on soft iron, touching or 0.1 mm above:
    its field jumps at the edges of its footprint, which the mesh and the load
    must follow. The references are fem_study at orders 3 to 5, good to 5e-5
    at each sensor."""
    soft = _cuboid()
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
