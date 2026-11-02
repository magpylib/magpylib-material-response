"""Saturation: cells on a B-H curve (#81)."""

from __future__ import annotations

import copy
import pickle

import magpylib as magpy
import numpy as np
import pytest
from scipy.optimize import brentq

from magpylib_material_response import (
    UniformField,
    from_json,
    study_cells,
    to_json,
)
from magpylib_material_response._saturation import MU0, BHCurve, curve_of
from magpylib_material_response.cells import solve_cells
from magpylib_material_response.face_charge import solve_face_charge
from magpylib_material_response.meshing import (
    mesh_Cuboid,
    mesh_TriangularMesh,
    mesh_uniform,
)

CHI0, JS = 1000.0, 1.6  # the test curve: J = chi0 mu0 H / (1 + chi0 mu0 H / Js)


def polarization(H):
    mu0H = MU0 * np.asarray(H, float)
    return CHI0 * mu0H / (1 + CHI0 * mu0H / JS)


def table(H_max=1e6, n=400):
    H = np.geomspace(1e-2, H_max, n)
    H = np.r_[0.0, H]
    return BHCurve(H=H, B=polarization(H) + MU0 * H)


def mesh_uniform_six(model):
    return mesh_uniform(model, 6)


def mesh_cuboid_two(model):
    return mesh_Cuboid(model.sources_all[0], (2, 2, 2))


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
    sphere.susceptibility = table()
    model = mesh_TriangularMesh(sphere, 300)
    model.H_ext = (0, 0, B0)
    solved = solve_cells(model, solver="direct")
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
        plate.susceptibility = curve
    if chi is not None:
        plate.susceptibility = chi
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(0.005,) * 3, position=(0, 0, 0.004)
    )
    return magpy.Collection(magnet, mesh_Cuboid(plate, (5, 5, 2)))


@pytest.mark.filterwarnings("ignore:This B-H curve reaches")
@pytest.mark.parametrize("solver", ["direct", "iterative"])
def test_a_linear_curve_is_a_susceptibility(solver):
    """A straight curve of slope 1 + chi solves as chi does, in one more solve."""
    H = np.array([0.0, 1e7])
    straight = BHCurve(H=H, B=(1 + CHI0) * MU0 * H)
    on_curve = solve_cells(_plate(straight), solver=solver)
    linear = solve_cells(_plate(chi=CHI0), solver=solver)
    points = np.array([[0.003, 0, 0.0015], [0, 0.006, 0], [0, 0, -0.003]])
    np.testing.assert_allclose(
        magpy.getB(on_curve, points), magpy.getB(linear, points), rtol=1e-5, atol=1e-9
    )


def test_saturation_lowers_the_field_and_the_rest_stays_linear():
    """In a field along the plate, the saturating plate carries less than chi0
    would give, and the fixed magnet stays as given."""
    along = _plate(table())
    along.H_ext = (0.5, 0, 0)
    saturating = solve_cells(along)
    along = _plate(chi=CHI0)
    along.H_ext = (0.5, 0, 0)
    linear = solve_cells(along)
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
    model.susceptibility = table(n=30)
    model.H_ext = (0, 0, 0.5)
    for cell in model.sources_all:
        assert curve_of(cell) is not None
    solved = solve_cells(model)
    assert all(np.linalg.norm(c.polarization) < JS for c in solved.sources_all)
    back = from_json(to_json(model))[0]
    np.testing.assert_array_equal(back.susceptibility.B, model.susceptibility.B)
    np.testing.assert_allclose(
        solve_cells(back).sources_all[0].polarization,
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
        BHCurve(H=np.asarray(bad)[:, 0], B=np.asarray(bad)[:, 1])


def test_what_a_curve_cannot_go_with():
    model = _plate(table(n=30))
    with pytest.raises(ValueError, match="susceptibility argument"):
        solve_cells(model, susceptibility=10.0)
    part = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.002), polarization=(0, 0, 0))
    part.susceptibility = table(n=30)
    with pytest.raises(NotImplementedError, match="B-H curve"):
        solve_face_charge(magpy.Collection(part))


@pytest.mark.parametrize("own", ["on the part", "on the parent"])
def test_the_nearest_susceptibility_wins_between_a_number_and_a_table(own):
    """As everywhere: a part's own value beats its parent's, either way round."""
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    cells = mesh_Cuboid(part, (2, 2, 2))
    model = magpy.Collection(cells)
    model.H_ext = (0, 0, 1.0)
    if own == "on the part":  # linear: past saturation, as a chi of 1000 goes
        model.susceptibility = table(n=30)
        cells.susceptibility = 1000.0
        assert curve_of(cells.sources_all[0]) is None
        expected_above = True
    else:
        model.susceptibility = 1000.0
        cells.susceptibility = table(n=30)
        assert curve_of(cells.sources_all[0]) is not None
        expected_above = False
    J = max(np.linalg.norm(c.polarization) for c in solve_cells(model).sources_all)
    assert bool(J > JS) is expected_above


def test_a_linear_part_under_a_table_is_solved_by_face_charge():
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    part.susceptibility = 1000.0
    model = magpy.Collection(part)
    model.susceptibility = table(n=30)
    model.H_ext = (0, 0, 0.01)
    assert solve_face_charge(model, level=1).solved_by == "face_charge"


@pytest.mark.parametrize(
    "use",
    [solve_cells, solve_face_charge, mesh_uniform_six, mesh_cuboid_two],
    ids=["solve_cells", "solve_face_charge", "mesh_uniform", "mesh_Cuboid"],
)
def test_the_old_attribute_raises_and_does_not_silently_give_a_linear_part(use):
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    part.bh_curve = table(n=30)
    with pytest.raises(ValueError, match="bh_curve"):
        use(magpy.Collection(part))


def test_a_table_as_argument_is_refused():
    for curve in (table(n=30), np.c_[[0.0, 1e6], [0.0, 2.0]]):
        with pytest.raises(ValueError, match="set it on the part"):
            solve_cells(_plate(chi=1.0), susceptibility=curve)


def test_a_steep_curve_solves():
    """Anderson's steps overshoot on it: unbounded, log chi left what a solve can
    take, and a raw LinAlgError came out."""
    pytest.importorskip("tetgen")
    sphere = icosphere(0.01, splits=1)
    sphere.susceptibility = BHCurve(H=[0, 100, 1e6], B=[0, 1, 2.5])
    model = mesh_TriangularMesh(sphere, 50)
    model.H_ext = (0, 0, 1.0)
    J = [np.linalg.norm(c.polarization) for c in solve_cells(model).sources_all]
    assert 0.9 < max(J) < 2.5


@pytest.mark.parametrize(
    "scale",
    [
        lambda c: {"H": c.B, "B": c.H},
        lambda c: {"H": c.H, "B": c.B * 1e4},
    ],
    ids=["swapped", "B in gauss"],
)
def test_a_curve_past_any_material_warns(scale):
    with pytest.warns(UserWarning, match="past any material"):
        BHCurve(**scale(table()))


def test_a_table_is_refused_and_a_curve_is_shared():
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    part.susceptibility = np.c_[[0.0, 1e6], [0.0, 2.0]]
    with pytest.raises(TypeError, match="BHCurve"):
        solve_cells(magpy.Collection(mesh_Cuboid(part, (1, 1, 1))))
    with pytest.raises(TypeError, match=r"keyword|positional"):
        BHCurve([0, 1], [0, 1])  # pylint: disable=too-many-function-args,missing-kwoa
    curve = table(n=5)
    assert copy.deepcopy(curve) is curve
    assert pickle.loads(pickle.dumps(curve)).B.tolist() == curve.B.tolist()
    part.susceptibility = curve
    cells = mesh_Cuboid(part, (2, 2, 2)).sources_all
    assert all(c.susceptibility is curve for c in cells)


S_SHAPED = {
    "S-shaped": {
        "H": [0, 100, 200, 400, 800, 1600, 5000, 2e4, 1e5],
        "B": [0, 0.05, 0.4, 1.0, 1.3, 1.5, 1.65, 1.8, 1.95],
    },
    "vacuum first": {
        "H": [0, 500, 1000, 2000, 5000, 1e4, 1e5],
        "B": [0, MU0 * 500, 0.1, 0.5, 1.1, 1.4, 1.8],
    },
}


def _big_magnet():
    return magpy.magnet.Cuboid(
        dimension=(0.05,) * 3, polarization=(0, 0, 1.3), position=(0, 0, 0.1)
    )


def _in_a_magnets_field(curve):
    """A part driven by a meshed magnet alone: its own residual is nothing beside
    the magnet's 1.3 T, so GMRES does not see a part at a small chi."""
    part = magpy.magnet.Cuboid(dimension=(0.02,) * 3, polarization=(0, 0, 0))
    part.susceptibility = curve
    return magpy.Collection(
        mesh_Cuboid(part, (6, 6, 6)), mesh_Cuboid(_big_magnet(), (10, 10, 10))
    )


@pytest.mark.parametrize("solver", ["auto", "iterative"])
@pytest.mark.parametrize("curve", S_SHAPED, ids=list(S_SHAPED))
def test_a_curve_that_starts_flat_responds_beside_a_magnet(curve, solver):
    """It stayed at J = 0 under GMRES, reported as converged, in the magnet's field."""
    curve = BHCurve(**S_SHAPED[curve])
    point = (0, 0.015, 0)
    reference = magpy.getB(
        solve_cells(_in_a_magnets_field(curve), solver="direct"), point
    )
    got = magpy.getB(solve_cells(_in_a_magnets_field(curve), solver=solver), point)
    np.testing.assert_allclose(got, reference, atol=1e-5)
    assert np.abs(got - _big_magnet().getB(point)).max() > 1e-3


def test_a_mesh_study_of_a_flat_start_sees_the_part():
    """Three meshes of a part at J = 0 agreed exactly, and the study said so."""
    part = magpy.magnet.Cuboid(dimension=(0.02,) * 3, polarization=(0, 0, 0))
    part.susceptibility = BHCurve(**S_SHAPED["vacuum first"])
    point = (0, 0.015, 0)
    magnet = mesh_Cuboid(_big_magnet(), (10, 10, 10))
    study = study_cells(magpy.Collection(part, magnet), [point], (3, 4, 6))
    assert np.abs(study.values[-1] - _big_magnet().getB(point)).max() > 1e-3


def test_remanence_on_a_curve_matches_a_sphere():
    """The field read leaves the remanence out of chi, and J adds it."""
    pytest.importorskip("tetgen")
    sphere = icosphere(0.01, splits=2)
    sphere.polarization = (0, 0, 0.5)
    sphere.susceptibility = table()
    model = magpy.Collection(
        mesh_TriangularMesh(sphere, 300), UniformField(B=(0, 0, 0.2))
    )
    cells = solve_cells(model, solver="direct").children[0].sources_all
    volumes = np.array([c.volume for c in cells])
    mean = (
        np.array([c.polarization[2] for c in cells]) * volumes
    ).sum() / volumes.sum()
    exact = brentq(lambda J: 0.5 + polarization((0.2 - J / 3) / MU0) - J, 0.5, 0.6)
    assert mean == pytest.approx(exact, rel=0.01)


@pytest.mark.parametrize("mu0", [magpy.mu_0, 4e-7 * np.pi])
def test_a_curve_on_vacuum_everywhere_is_refused_with_either_mu0(mu0):
    with pytest.raises(ValueError, match="vacuum everywhere"):
        BHCurve(H=[0.0, 1e5], B=[0.0, mu0 * 1e5])


@pytest.mark.parametrize("mu0", [magpy.mu_0, 4e-7 * np.pi])
def test_a_point_on_vacuum_is_no_error_with_either_mu0(mu0):
    curve = BHCurve(H=[0.0, 50.0, 100.0, 1000.0], B=[0, mu0 * 50, 0.05, 1.0])
    assert curve.chi_max > 100


@pytest.mark.parametrize(
    ("given", "same"),
    [
        ([1000.0, (1, 2, 3)], [(1000, 1000, 1000), (1, 2, 3)]),
        ([np.array((1, 2, 3)), 5.0], [(1, 2, 3), (5, 5, 5)]),
    ],
)
def test_a_list_may_mix_numbers_and_vectors(given, same):
    def model():
        return magpy.Collection(
            magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0)),
            magpy.magnet.Cuboid(
                dimension=(0.01,) * 3, polarization=(0, 0, 0), position=(0.02, 0, 0)
            ),
            UniformField(B=(0, 0, 0.5)),
        )

    # in the list, one entry per magnet; the field takes none
    point = (0.01, 0.02, 0.0)
    np.testing.assert_allclose(
        magpy.getB(solve_cells(model(), susceptibility=given), point),
        magpy.getB(solve_cells(model(), susceptibility=same), point),
    )


def test_a_curve_in_a_list_says_to_set_it_on_the_part():
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    with pytest.raises(ValueError, match="set it on the part"):
        solve_cells(magpy.Collection(part), susceptibility=[table(n=5)])


@pytest.mark.parametrize(
    "wrap",
    [
        lambda r: r,
        lambda r: r.copy(),
        magpy.Collection,
        lambda r: r.children[0],
    ],
    ids=["itself", "a copy", "nested", "taken out"],
)
@pytest.mark.parametrize("solve", [solve_cells, solve_face_charge])
def test_solving_a_result_warns(solve, wrap):
    part = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    part.susceptibility = 10.0
    meshed = mesh_Cuboid(part, (1, 1, 1))
    solved = solve_cells(magpy.Collection(meshed, UniformField(B=(0, 0, 0.1))))
    with pytest.warns(UserWarning, match="already solved"):
        solve(wrap(solved))


def test_a_part_with_nothing_on_it_warns():
    """A child's copy leaves its parent's susceptibility behind, and does nothing."""
    child = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    holder = magpy.Collection(child)
    holder.susceptibility = 100.0
    other = magpy.magnet.Cuboid(
        dimension=(0.01,) * 3, polarization=(0, 0, 0), position=(0.03, 0, 0)
    )
    other.susceptibility = 100.0
    model = magpy.Collection(child.copy(), other, UniformField(B=(0, 0, 0.1)))
    with pytest.warns(UserWarning, match="does nothing"):
        solve_cells(model)


def test_a_fem_solution_keeps_its_own_coil():
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    sphere = magpy.magnet.Sphere(diameter=0.02, polarization=(0, 0, 0))
    sphere.susceptibility = 10.0
    coil = magpy.current.Circle(current=8000, diameter=0.1)
    solution = solve_fem(magpy.Collection(sphere, coil), order=2)
    point = (0.03, 0, 0.01)
    before = solution.getB(point)
    coil.current = 0
    coil.move((0, 0, 0.05))
    np.testing.assert_array_equal(solution.getB(point), before)


def test_a_fem_model_with_a_coil_and_a_field_reads_both():
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    sphere = magpy.magnet.Sphere(diameter=0.02, polarization=(0, 0, 0))
    sphere.susceptibility = 10.0
    coil = magpy.current.Circle(current=2e5, diameter=2.0)
    B0 = coil.getB((0, 0, 0))[2]
    field = UniformField(B=(0, 0, B0))
    both = solve_fem(magpy.Collection(sphere, coil, field), order=3)
    J = 10 * 2 * B0 / (1 + 10 / 3)  # the coil and the field add at the sphere
    np.testing.assert_allclose(magpy.getJ(both, (0, 0, 0)), (0, 0, J), atol=0.02 * J)
    np.testing.assert_allclose(
        magpy.getB(both, (0, 0, 0)), (0, 0, 2 * B0 + 2 * J / 3), atol=0.02 * J
    )
    far = (0, 0, 0.2)
    np.testing.assert_allclose(
        magpy.getB(both, far), coil.getB(far) + field.getB(far), atol=0.02 * B0
    )


def test_a_curve_that_starts_flat_solves():
    """A steel's curve can rise slowly at first: the iteration starts from a
    small susceptibility, not from log(0)."""
    H = np.r_[0.0, np.geomspace(1.0, 1e6, 200)]
    rise = np.clip((H - 1e3) / 2e3, 0, 1) ** 1.5
    flat = BHCurve(H=H, B=4 * JS * rise / (1 + 3 * rise) + MU0 * H)
    curve = flat
    assert curve.chi0 == 1e-6
    assert curve.chi_max > 100
    model = _plate(flat)
    model.H_ext = (0.5, 0, 0)
    solved = solve_cells(model)
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
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
    sphere.susceptibility = table()
    model = magpy.Collection(sphere)
    model.H_ext = (0, 0, B0)
    solved = solve_fem(model, order=4)
    J = exact_sphere(B0)
    # B leaves out the applied field: inside, J - J / 3
    assert solved.getB((0, 0, 0))[2] == pytest.approx(2 * J / 3, rel=5e-3)
    assert solved.moment[2] == pytest.approx(J * np.pi / 6 * 0.01**3 / MU0, rel=2e-3)


@pytest.mark.filterwarnings("ignore:This B-H curve reaches")
def test_finite_elements_bring_a_tiny_part_onto_its_curve():
    """A part a billionth of the other's volume converges too: the iteration
    stops on its worst element, not on an average over the volume."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    H = np.r_[0.0, np.geomspace(1.0, 1e8, 50)]
    big = magpy.magnet.Sphere(diameter=0.02, polarization=(0, 0, 0))
    big.susceptibility = BHCurve(H=H, B=11 * MU0 * H)  # a straight curve: chi = 10
    tiny = magpy.magnet.Sphere(
        diameter=2e-5, polarization=(0, 0, 0), position=(0.03, 0, 0)
    )
    tiny.susceptibility = table()
    model = magpy.Collection(big, tiny)
    model.H_ext = (0, 0, 2.0)
    point = np.array([[0.03, 0, 0]])
    solved = solve_fem(
        model, order=3, solver_tol=1e-8, observers=point, observer_maxh=1e-5
    )
    uniform = magpy.magnet.Sphere(
        diameter=0.02, polarization=(0, 0, 10 / (1 + 10 / 3) * 2)
    )
    from_big = uniform.getB(point[0])[2]
    expected = from_big + 2 * exact_sphere(2.0 + from_big) / 3
    assert solved.getB(point)[2] == pytest.approx(expected, rel=5e-3)


@pytest.mark.filterwarnings("ignore:This B-H curve reaches")
def test_finite_elements_on_a_straight_curve_match_its_susceptibility():
    """On curved elements too: projecting chi must keep a constant."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    def solved(**attributes):
        sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
        for name, value in attributes.items():
            setattr(sphere, name, value)
        model = magpy.Collection(sphere)
        model.H_ext = (0, 0, 1.0)
        return solve_fem(model, order=3, observers=points)

    points = np.array([[0, 0, 0], [0.003, 0, 0], [0, 0, 0.012]])
    H = np.r_[0.0, np.geomspace(1.0, 1e8, 50)]
    on_curve = solved(susceptibility=BHCurve(H=H, B=11 * MU0 * H))
    linear = solved(susceptibility=10.0)
    np.testing.assert_allclose(
        on_curve.getB(points), linear.getB(points), rtol=1e-4, atol=1e-5
    )
    assert on_curve.moment[2] == pytest.approx(linear.moment[2], rel=1e-5)


def test_finite_elements_keep_a_remanent_part_on_its_curve():
    """B and the moment come from the chi of the solve, and they agree: with
    no field, ``J = J_r - J(J / 3)``."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415

    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 1.0))
    sphere.susceptibility = table()
    solved = solve_fem(magpy.Collection(sphere), order=4)
    J = brentq(lambda J: 1.0 - polarization(J / 3 / MU0) - J, 0.0, 1.0)
    assert solved.getB((0, 0, 0))[2] == pytest.approx(2 * J / 3, rel=0.01)
    assert solved.moment[2] == pytest.approx(J * np.pi / 6 * 0.01**3 / MU0, rel=0.01)


def test_finite_elements_say_what_to_do_when_a_curve_does_not_settle(monkeypatch):
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import _solve  # noqa: PLC0415

    monkeypatch.setattr(_solve, "FEM_SOLVES", 1)
    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
    sphere.susceptibility = table(n=30)
    model = magpy.Collection(sphere)
    model.H_ext = (0, 0, 1.0)
    with pytest.raises(RuntimeError, match="higher order"):
        _solve.solve_fem(model, order=1)


def test_finite_elements_and_cells_agree_on_a_coil_in_a_saturating_core():
    """A coil at a current that saturates the bar: the reduced and the total
    potential both take chi from the curve, and the two methods agree."""
    pytest.importorskip("ngsolve")
    from magpylib_material_response.fem import solve_fem  # noqa: PLC0415
    from magpylib_material_response.meshing import mesh_uniform  # noqa: PLC0415

    def model():
        bar = magpy.magnet.Cuboid(dimension=(0.04, 0.01, 0.01), polarization=(0, 0, 0))
        bar.susceptibility = table(n=120)
        coil = magpy.current.Circle(
            current=2e4, diameter=0.03, orientation=None, position=(0, 0, 0)
        ).rotate_from_angax(90, "y")
        return magpy.Collection(bar, coil)

    points = np.array([[0.03, 0, 0], [0, 0.02, 0], [0.03, 0.01, 0]])
    by_fem = solve_fem(
        model(), order=3, observers=points, solver_tol=1e-7, max_iter=30000
    )
    by_cells = solve_cells(mesh_uniform(model(), 4))
    np.testing.assert_allclose(
        by_fem.getB(points), magpy.getB(by_cells, points), rtol=0.05, atol=1e-3
    )
