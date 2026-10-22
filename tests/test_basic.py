from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

import magpylib_material_response
from magpylib_material_response.demag import (
    apply_demag,
    demag_tensor,
    filter_distance,
    get_susceptibilities,
)
from magpylib_material_response.meshing import mesh_Cuboid


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
            "no_susceptibility",
            lambda: [magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))],
            "No susceptibility defined in any parent collection",
            id="no_susceptibility",
        ),
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
    "legacy",
    [{"pairs_matching": True}, {"max_dist": 1000}, {"split": 5}],
    ids=["pairs_matching", "max_dist", "split"],
)
def test_legacy_options_solve_the_same_system(legacy):
    """Nine cells in five chunks leave one chunk with a single cell."""

    def scene():
        magnet = magpy.magnet.Cuboid(
            dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1), position=(0, 0, 0.012)
        )
        magnet.susceptibility = 0
        soft = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
        soft.rotate_from_angax(30, "z")
        soft.susceptibility = 100
        return magpy.Collection(magnet, mesh_Cuboid(soft, 8))

    default = apply_demag(scene())
    out = apply_demag(scene(), **legacy)
    np.testing.assert_allclose(
        [s.polarization for s in out.sources_all],
        [s.polarization for s in default.sources_all],
        rtol=1e-9,
        atol=1e-12,
    )


def test_demag_tensor_legacy_options():
    """pairs_matching gives the full tensor; max_dist zeroes the pairs of cells
    farther apart than max_dist cell sizes."""
    cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
    cells = mesh_Cuboid(cube, 27).sources_all
    full = demag_tensor(cells)
    atol = 1e-12 * np.abs(full).max()
    np.testing.assert_allclose(
        demag_tensor(cells, pairs_matching=True), full, rtol=1e-9, atol=atol
    )
    kept = filter_distance(cells, 1.5).reshape(27, 27)
    assert 0 < kept.sum() < kept.size
    np.testing.assert_allclose(
        demag_tensor(cells, max_dist=1.5),
        full * kept[None, :, :, None],
        rtol=1e-9,
        atol=atol,
    )


def test_pairs_matching_keys_on_cell_dimensions_at_any_scale():
    """Pairs of different cell sizes, at nanometre scale, are not merged."""
    nm = 1e-9
    cubes = [
        magpy.magnet.Cuboid(
            polarization=(0, 0, 1), dimension=(d * nm,) * 3, position=(6 * nm * i, 0, 0)
        )
        for i, d in enumerate([1.0, 2.0, 2.0, 3.0])
    ]
    np.testing.assert_allclose(
        demag_tensor(cubes, pairs_matching=True), demag_tensor(cubes), rtol=1e-9
    )


@pytest.mark.parametrize(
    "legacy",
    [
        {"max_dist": -1},
        {"max_dist": 2, "pairs_matching": True},
        {"max_dist": 2, "split": 2},
        {"pairs_matching": True, "split": 2},
    ],
)
def test_legacy_options_refuse_bad_values(legacy):
    cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1))
    with pytest.raises(ValueError, match=r"max_dist|splitting"):
        apply_demag(mesh_Cuboid(cube, 8), susceptibility=1, **legacy)


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
