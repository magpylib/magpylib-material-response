"""Saturation: cells on a B-H curve (#81)."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.optimize import brentq

from magpylib_material_response import from_json, to_json
from magpylib_material_response._saturation import MU0, Curve, curve_of
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.face_charge import apply_face_charge
from magpylib_material_response.meshing import mesh_Cuboid, mesh_TriangularMesh

CHI0, JS = 1000.0, 1.6  # the test curve: J = chi0 mu0 H / (1 + chi0 mu0 H / Js)


def polarization(H):
    mu0H = MU0 * np.asarray(H, float)
    return CHI0 * mu0H / (1 + CHI0 * mu0H / JS)


def table(H_max=1e6, n=400):
    H = np.geomspace(1e-2, H_max, n)
    H = np.r_[0.0, H]
    return np.c_[H, polarization(H) + MU0 * H]


def icosphere(diameter, splits=3):
    """A triangle mesh of a sphere, from an icosahedron split `splits` times."""
    t = (1 + 5**0.5) / 2
    V = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        float,
    )  # fmt: skip
    F = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    V = list(V / np.linalg.norm(V, axis=1)[:, None])
    for _ in range(splits):
        mid, new = {}, []
        for a, b, c in F:
            m = []
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                if key not in mid:
                    p = V[i] + V[j]
                    V.append(p / np.linalg.norm(p))
                    mid[key] = len(V) - 1
                m.append(mid[key])
            new += [
                (a, m[0], m[2]),
                (b, m[1], m[0]),
                (c, m[2], m[1]),
                (m[0], m[1], m[2]),
            ]
        F = new
    return magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=np.array(V) * diameter / 2, faces=np.array(F)
    )


def exact_sphere(B0):
    """A sphere's uniform polarization in the applied field `B0` (T), on the
    curve: ``mu0 H_in = B0 - J / 3``, ``J = J(H_in)``."""
    return brentq(lambda J: polarization((B0 - J / 3) / MU0) - J, 0.0, 3 * B0)


@pytest.mark.parametrize("B0", [0.3, 1.0, 2.0])
def test_a_sphere_follows_its_curve(B0):
    pytest.importorskip("tetgen")
    sphere = icosphere(0.01, splits=2)
    sphere.bh_curve = table()
    model = mesh_TriangularMesh(sphere, 300)
    model.H_ext = (0, 0, B0)
    solved = apply_demag(model, solver="direct")
    cells = solved.sources_all
    volumes = np.array([c.volume for c in cells])
    J = np.array([c.polarization for c in cells])
    mean = (J * volumes[:, None]).sum(0) / volumes.sum()
    exact = exact_sphere(B0)
    assert mean[2] == pytest.approx(exact, rel=0.01)
    assert mean[2] < JS  # saturating, where chi0 alone would give 3 B0 / (1 + 3 / chi0)
    assert abs(mean[:2]).max() < 0.01 * exact


def _plate(curve=None, chi=None):
    plate = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.002), polarization=(0, 0, 0))
    if curve is not None:
        plate.bh_curve = curve
    if chi is not None:
        plate.susceptibility = chi
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(0.005,) * 3, position=(0, 0, 0.004)
    )
    return magpy.Collection(magnet, mesh_Cuboid(plate, (5, 5, 2)))


@pytest.mark.parametrize("solver", ["direct", "iterative"])
def test_a_linear_curve_is_a_susceptibility(solver):
    """A straight curve of slope 1 + chi solves as chi does, in one more solve."""
    H = np.array([0.0, 1e7])
    straight = np.c_[H, (1 + CHI0) * MU0 * H]
    on_curve = apply_demag(_plate(straight), solver=solver)
    linear = apply_demag(_plate(chi=CHI0), solver=solver)
    points = np.array([[0.003, 0, 0.0015], [0, 0.006, 0], [0, 0, -0.003]])
    np.testing.assert_allclose(
        magpy.getB(on_curve, points), magpy.getB(linear, points), rtol=1e-5, atol=1e-9
    )


def test_saturation_lowers_the_field_and_the_rest_stays_linear():
    """In a field along the plate, the saturating plate carries less than chi0
    would give, and the fixed magnet stays as given."""
    along = _plate(table())
    along.H_ext = (0.5, 0, 0)
    saturating = apply_demag(along)
    along = _plate(chi=CHI0)
    along.H_ext = (0.5, 0, 0)
    linear = apply_demag(along)
    J_sat = max(
        np.linalg.norm(c.polarization)
        for c in saturating.sources_all
        if c.dimension[0] < 0.004
    )
    J_lin = max(
        np.linalg.norm(c.polarization)
        for c in linear.sources_all
        if c.dimension[0] < 0.004
    )
    assert J_sat < JS < J_lin
    # the drive magnet itself is fixed: its polarization is as given
    magnet = next(c for c in saturating.sources_all if c.dimension[0] == 0.005)
    np.testing.assert_allclose(magnet.polarization, (0, 0, 1))


def test_a_curve_set_on_a_parent_reaches_the_cells_and_survives_json():
    plate = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.002), polarization=(0, 0, 0))
    model = magpy.Collection(mesh_Cuboid(plate, (3, 3, 1)))
    model.bh_curve = table(n=30)
    model.H_ext = (0, 0, 0.5)
    for cell in model.sources_all:
        assert curve_of(cell) is not None
    solved = apply_demag(model)
    assert all(np.linalg.norm(c.polarization) < JS for c in solved.sources_all)
    back = from_json(to_json(model))[0]
    np.testing.assert_array_equal(back.bh_curve, model.bh_curve)
    np.testing.assert_allclose(
        apply_demag(back).sources_all[0].polarization,
        solved.sources_all[0].polarization,
    )


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ([[0, 0]], "at least two"),
        ([[1, 0], [2, 1]], r"start at \(0, 0\)"),
        ([[0, 0], [2, 1], [1, 2]], "H increasing"),
        ([[0, 0], [1, 1], [2, 0.5]], "B not decreasing"),
        ([[0, 0], [1e6, 0.1]], "below vacuum"),
        ([[0, 0], [1e6, MU0 * 1e6]], "vacuum everywhere"),
    ],
)
def test_a_bad_curve_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        Curve(bad)


def test_what_a_curve_cannot_go_with():
    model = _plate(table(n=30))
    with pytest.raises(ValueError, match="susceptibility argument"):
        apply_demag(model, susceptibility=10.0)
    part = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.002), polarization=(0, 0, 0))
    part.bh_curve = table(n=30)
    with pytest.raises(NotImplementedError, match="B-H curve"):
        apply_face_charge(magpy.Collection(part))


def test_a_curve_that_starts_flat_solves():
    """A steel's curve can rise slowly at first: the iteration starts from a
    small susceptibility, not from log(0)."""
    H = np.r_[0.0, np.geomspace(1.0, 1e6, 200)]
    rise = np.clip((H - 1e3) / 2e3, 0, 1) ** 1.5
    flat = np.c_[H, 4 * JS * rise / (1 + 3 * rise) + MU0 * H]
    curve = Curve(flat)
    assert curve.chi0 == 1e-6
    assert curve.chi_max > 100
    model = _plate(flat)
    model.H_ext = (0.5, 0, 0)
    solved = apply_demag(model)
    J_cells = [
        np.linalg.norm(c.polarization)
        for c in solved.sources_all
        if c.dimension[0] < 0.004
    ]
    assert 0.5 * JS < max(J_cells) < 1.001 * JS  # the iteration stops at 1e-4


@pytest.mark.parametrize("B0", [0.3, 1.0, 2.0])
def test_finite_elements_follow_a_curve(B0):
    """A sphere stays uniform while it saturates, so its answer is exact, inside
    the part and by its moment outside."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import fem_solve  # noqa: PLC0415

    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
    sphere.bh_curve = table()
    model = magpy.Collection(sphere)
    model.H_ext = (0, 0, B0)
    solved = fem_solve(model, order=4)
    J = exact_sphere(B0)
    # B leaves out the applied field: inside, J - J / 3
    assert solved.getB(np.array([[0, 0, 0]]))[0, 2] == pytest.approx(
        2 * J / 3, rel=5e-3
    )
    assert solved.moment[2] == pytest.approx(J * np.pi / 6 * 0.01**3 / MU0, rel=2e-3)


def test_finite_elements_and_cells_agree_on_a_coil_in_a_saturating_core():
    """A coil at a current that saturates the bar: the reduced and the total
    potential both take chi from the curve, and the two methods agree."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import fem_solve  # noqa: PLC0415
    from magpylib_material_response.meshing import mesh_uniform  # noqa: PLC0415

    def model():
        bar = magpy.magnet.Cuboid(dimension=(0.04, 0.01, 0.01), polarization=(0, 0, 0))
        bar.bh_curve = table(n=120)
        coil = magpy.current.Circle(
            current=2e4, diameter=0.03, orientation=None, position=(0, 0, 0)
        ).rotate_from_angax(90, "y")
        return magpy.Collection(bar, coil)

    points = np.array([[0.03, 0, 0], [0, 0.02, 0], [0.03, 0.01, 0]])
    by_fem = fem_solve(
        model(), order=3, observers=points, solver_tol=1e-7, max_iter=30000
    )
    by_cells = apply_demag(mesh_uniform(model(), 4))
    np.testing.assert_allclose(
        by_fem.getB(points), magpy.getB(by_cells, points), rtol=0.05, atol=1e-3
    )
