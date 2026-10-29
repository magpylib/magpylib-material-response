from __future__ import annotations

import itertools

import magpylib as magpy
import numpy as np
import pytest

import magpylib_material_response
from magpylib_material_response import demag
from magpylib_material_response.demag import (
    apply_demag,
    get_susceptibilities,
)
from magpylib_material_response.face_charge import apply_face_charge
from magpylib_material_response.meshing import mesh_Cuboid, mesh_Cylinder, mesh_uniform


def test_version():
    assert isinstance(magpylib_material_response.__version__, str)


def test_apply_demag_integration():
    """Integration test: verify get_susceptibilities works correctly with apply_demag"""
    zone = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    mesh = mesh_Cuboid(zone, (2, 2, 2))

    # Test that different equivalent susceptibility inputs give same result
    dm1 = apply_demag(mesh, susceptibility=4)
    dm2 = apply_demag(mesh, susceptibility=(4, 4, 4))

    zone.susceptibility = 4
    mesh_with_attr = mesh_Cuboid(zone, (2, 2, 2))
    dm3 = apply_demag(mesh_with_attr)

    # All should give same magnetic field result
    b_ref = dm1.getB((1, 2, 3))
    np.testing.assert_allclose(dm2.getB((1, 2, 3)), b_ref)
    np.testing.assert_allclose(dm3.getB((1, 2, 3)), b_ref)


@pytest.mark.parametrize(
    ("test_case", "susceptibility_input", "expected_output"),
    [
        pytest.param(
            "source_scalar",
            [(2.5,), (3.0,)],
            np.array([2.5, 3.0, 2.5, 3.0, 2.5, 3.0]),
            id="source_scalar",
        ),
        pytest.param(
            "source_none",
            [(None,), (3.0,)],
            np.array([0.0, 3.0, 0.0, 3.0, 0.0, 3.0]),
            id="source_none_is_fixed",
        ),
        pytest.param(
            "source_vector",
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            id="source_vector",
        ),
        pytest.param(
            "function_scalar",
            1.5,
            np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
            id="function_scalar",
        ),
        pytest.param(
            "function_0d_array",
            np.array(1.5),
            np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
            id="function_0d_array",
        ),
        pytest.param(
            "function_vector",
            (2.0, 3.0, 4.0),
            np.array([2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
            id="function_vector",
        ),
        pytest.param(
            "function_list",
            [1.5, 2.5],
            np.array([1.5, 2.5, 1.5, 2.5, 1.5, 2.5]),
            id="function_list",
        ),
    ],
)
def test_get_susceptibilities_basic(test_case, susceptibility_input, expected_output):
    """Test basic get_susceptibilities functionality with source attributes and function inputs"""
    sources = []
    for _ in range(2):
        zone = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
        sources.append(zone)

    if test_case.startswith("source"):
        # Set susceptibility on sources
        for i, sus_val in enumerate(susceptibility_input):
            if len(sus_val) == 1:
                sources[i].susceptibility = sus_val[0]
            else:
                sources[i].susceptibility = sus_val
        result = get_susceptibilities(sources)
    else:
        # Use function input
        result = get_susceptibilities(sources, susceptibility=susceptibility_input)

    np.testing.assert_allclose(result, expected_output)


def test_get_susceptibilities_hierarchy():
    """Test susceptibility inheritance from parent collections and mixed scenarios"""
    # Create collection with susceptibility
    collection = magpy.Collection()
    collection.susceptibility = 2.0

    # Source with its own susceptibility
    zone_own = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    zone_own.susceptibility = 5.0

    # Source inheriting from parent
    zone_inherit = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    collection.add(zone_inherit)

    # Test mixed sources (critical edge case)
    result = get_susceptibilities([zone_own, zone_inherit])
    expected = np.array([5.0, 2.0, 5.0, 2.0, 5.0, 2.0])
    np.testing.assert_allclose(result, expected)

    # Test single inheritance
    result_single = get_susceptibilities([zone_inherit])
    expected_single = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(result_single, expected_single)


@pytest.mark.parametrize(
    ("error_case", "setup_func", "error_message"),
    [
        pytest.param(
            "invalid_format",
            lambda: [_create_zone_with_bad_susceptibility()],
            "susceptibility is not scalar or array of length 3",
            id="invalid_format",
        ),
        pytest.param(
            "wrong_length",
            lambda: [
                magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
                for _ in range(4)
            ],
            "Apply_demag input susceptibility must be scalar, 3-vector, or same length as input Collection",
            id="wrong_length",
        ),
        pytest.param(
            "ambiguous_input",
            lambda: [
                magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
                for _ in range(3)
            ],
            "Apply_demag input susceptibility is ambiguous",
            id="ambiguous_input",
        ),
    ],
)
def test_get_susceptibilities_errors(error_case, setup_func, error_message):
    """Test error cases for get_susceptibilities function"""
    sources = setup_func()

    if error_case == "wrong_length":
        with pytest.raises(ValueError, match=error_message):
            get_susceptibilities(sources, susceptibility=[1.0, 2.0, 3.0, 4.0, 5.0])
    elif error_case == "ambiguous_input":
        for value in ((1.0, 2.0, 3.0), (np.array(1.0), 2.0, 3.0)):
            with pytest.raises(ValueError, match=error_message):
                get_susceptibilities(sources, susceptibility=value)
    else:
        with pytest.raises(ValueError, match=error_message):
            get_susceptibilities(sources)


def _create_zone_with_bad_susceptibility():
    """Helper to create a zone with invalid susceptibility format"""
    zone = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    zone.susceptibility = (1, 2)  # Invalid: should be scalar or length 3
    return zone


def test_get_susceptibilities_edge_cases():
    """Test edge cases: empty list, single source"""
    # Empty sources
    result = get_susceptibilities([])
    assert len(result) == 0

    # Single source
    zone = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    zone.susceptibility = 3.0
    result = get_susceptibilities([zone])
    expected = np.array([3.0, 3.0, 3.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "make",
    [
        lambda: magpy.magnet.Cuboid(
            polarization=(0, 0, 1), dimension=[(0.01, 0.01, 0.01), (0.012, 0.01, 0.01)]
        ),
        lambda: magpy.magnet.Cuboid(
            polarization=[(0, 0, 1), (0, 0, 0.9)], dimension=(0.01, 0.01, 0.01)
        ),
        lambda: magpy.current.Circle(diameter=0.02, current=[1, 2, 3]),
    ],
    ids=["dimension", "polarization", "current"],
)
def test_apply_demag_refuses_property_paths(make):
    """Demagnetization along a path is not supported, whichever property varies."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01))
    cube.susceptibility = 1
    obj = make()
    obj.susceptibility = 1
    with pytest.raises(ValueError, match="paths"):
        apply_demag(magpy.Collection(cube, obj))


@pytest.mark.parametrize(
    "make",
    [
        lambda: magpy.magnet.Cuboid(
            dimension=(0.01, 0.02, 0.01), position=(0.03, 0, 0)
        ),
        lambda: magpy.magnet.Tetrahedron(
            vertices=[(0.03, 0, 0), (0.04, 0, 0), (0.03, 0.01, 0), (0.03, 0, 0.01)]
        ),
        lambda: magpy.magnet.CylinderSegment(dimension=(0.01, 0.02, 0.01, 0, 90)),
    ],
    ids=["cuboid", "tetrahedron", "cylinder_segment"],
)
def test_current_field_is_taken_at_the_cell_centroid(make):
    """The coil comes first in the collection, ahead of the magnet it drives."""
    coil = magpy.current.Circle(diameter=0.05, current=1000, position=(0.02, 0, 0.01))
    cell = make()
    cell.polarization = (0, 0, 0)
    J_coil = apply_demag(magpy.Collection(coil, cell), susceptibility=10).sources[1]
    cell = make()
    cell.polarization = (0, 0, 0)
    cell.H_ext = coil.getB(cell.centroid)
    J_ext = apply_demag(magpy.Collection(cell), susceptibility=10).sources[0]
    np.testing.assert_allclose(J_coil.polarization, J_ext.polarization, rtol=1e-12)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.1, np.array(0.2), 0.3), [[0.1] * 3, [0.2] * 3, [0.3] * 3]),
        ((0.1, (1.0, 2.0, 3.0), 0.3), [[0.1] * 3, [1.0, 2.0, 3.0], [0.3] * 3]),
    ],
    ids=["scalars", "mixed"],
)
def test_per_object_susceptibilities_stay_per_object(values, expected):
    """Three cells with their own values, scalars, 0-d arrays or 3-vectors, keep
    them, rather than one anisotropic value being repeated."""
    cells = [
        magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
        for _ in range(3)
    ]
    for cell, chi in zip(cells, values, strict=True):
        cell.susceptibility = chi
    np.testing.assert_allclose(
        get_susceptibilities(cells), np.ravel(expected, order="F")
    )


def test_copy_keeps_the_parents_material_attributes():
    """With inplace=False, a collection inside another one still sees the outer
    collection's H_ext and susceptibility, as an in-place solve does."""

    def scene():
        inner = magpy.Collection(
            magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
        )
        outer = magpy.Collection(inner)
        outer.H_ext = (0, 0, 0.01)
        outer.susceptibility = 100
        return inner

    copied = apply_demag(scene()).sources_all[0].polarization
    in_place = scene()
    apply_demag(in_place, inplace=True)
    assert copied[2] > 0.01
    np.testing.assert_allclose(copied, in_place.sources_all[0].polarization)


@pytest.mark.parametrize("solver", ["demag", "face charge"])
@pytest.mark.parametrize(
    ("h_ext", "kind", "match"),
    [
        (0.1, ValueError, "three finite numbers"),
        ((0, 0.1), ValueError, "three finite numbers"),
        ((0, 0, np.nan), ValueError, "three finite numbers"),
        (("0", "0", "0.01"), ValueError, "three finite numbers"),
        ((0, 0, 40), UserWarning, "not H in A/m"),
        (np.array([[0, 0, 0.01]]), None, None),
    ],
    ids=["scalar", "two numbers", "nan", "text", "A/m", "a row"],
)
def test_H_ext_is_checked_before_the_solve(solver, h_ext, kind, match):
    """On every part of two cubes. A scalar failed in apply_demag with a
    reshape error and passed in apply_face_charge; the Earth's 40 A/m taken as
    tesla gave 40 T unremarked. A row, as getB gives for one point in a list,
    is three numbers too. One warning, at the line that called the solver."""

    def solve(value):
        model = magpy.Collection()
        for x in (0, 0.02):
            soft = magpy.magnet.Cuboid(
                dimension=(0.01,) * 3, polarization=(0, 0, 0), position=(x, 0, 0)
            )
            soft.susceptibility = 1000
            model.add(mesh_Cuboid(soft, (2, 2, 2)) if solver == "demag" else soft)
        for part in model.sources_all:
            part.H_ext = value
        if solver == "demag":
            return apply_demag(model)
        return apply_face_charge(model, level=1)

    if kind is None:
        np.testing.assert_allclose(
            solve(h_ext).getB((0.01, 0, 0.01)),
            solve((0, 0, 0.01)).getB((0.01, 0, 0.01)),
        )
    elif issubclass(kind, Warning):
        with pytest.warns(kind, match=match) as caught:
            solve(h_ext)
        assert [w.filename for w in caught if match in str(w.message)] == [__file__]
    else:
        with pytest.raises(kind, match=match):
            solve(h_ext)


@pytest.mark.parametrize(("chi", "averaged"), [(0.5, False), (1000, True)])
def test_segment_cells_are_averaged_from_a_susceptibility_of_one(
    monkeypatch, chi, averaged
):
    """#100's cell average made a small meshed cylinder 5 times slower to
    solve, at any susceptibility; point matching goes wrong only once chi
    times its positive eigenvalues, at most 0.16, nears 1."""
    cylinder = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1e-3, 1e-3))

    def solve():
        solved = apply_demag(mesh_Cylinder(cylinder, 40), susceptibility=chi)
        return np.array([c.polarization for c in solved.sources_all])

    J = solve()
    monkeypatch.setattr(demag, "_segment_averages", lambda *_: None)
    assert np.array_equal(J, solve()) is not averaged


def test_thin_tube_keeps_its_moment_with_more_cells_across_its_wall():
    """A thin tube at high susceptibility beside a magnet, in cylinder-segment
    cells. Point matching on curved cells gave the interaction matrix positive
    eigenvalues, and the moment more than doubled from 1 to 2 cells across the
    wall (#92); averaging close segment pairs keeps it within 1 %."""

    def moment(across):
        magnet = magpy.magnet.Cuboid(
            polarization=(1, 0, 0), dimension=(0.01,) * 3, position=(0.015, 0, 0)
        )
        magnet.susceptibility = 0
        tube = magpy.Collection(
            *(
                magpy.magnet.CylinderSegment(
                    polarization=(0, 0, 0),
                    dimension=(a, b, 0.01, phi, phi + 60),
                    position=(0, 0, z),
                )
                for a, b in itertools.pairwise(np.linspace(0.004, 0.005, across + 1))
                for phi in range(0, 360, 60)
                for z in (-0.01, 0, 0.01)
            )
        )
        tube.susceptibility = 1000
        apply_demag(magpy.Collection(magnet, tube), inplace=True)
        return sum(cell.polarization[0] * cell.volume for cell in tube)

    assert moment(2) == pytest.approx(moment(1), rel=0.01)


@pytest.mark.parametrize("solver", ["demag", "face charge", "fem"])
def test_a_magnet_without_susceptibility_is_fixed_everywhere(solver):
    """The face-charge page's model: a magnet with no susceptibility over a soft
    plate. apply_face_charge took the magnet as fixed; apply_demag, mesh_study
    and fem_solve refused it, and could not check face charge."""
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(0.01,) * 3, position=(0, 0, 0.0075)
    )
    plate = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.03, 0.03, 0.005))
    plate.susceptibility = 1000
    model = magpy.Collection(magnet, plate)
    if solver == "demag":
        solved = apply_demag(mesh_uniform(model, 4))
    elif solver == "face charge":
        solved = apply_face_charge(model, level=1)
    else:
        fem = pytest.importorskip("magpylib_material_response.fem")
        solved = fem.fem_solve(model, order=2)
    reference = apply_face_charge(model, level=2).getB((0, 0, 0.02))[2]
    np.testing.assert_allclose(solved.getB((0, 0, 0.02))[2], reference, rtol=0.05)
