from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

import magpylib_material_response
from magpylib_material_response.demag import apply_demag, get_susceptibilities
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
    "test_case,susceptibility_input,expected_output",
    [
        pytest.param(
            "source_scalar",
            [(2.5,), (3.0,)],
            np.array([2.5, 3.0, 2.5, 3.0, 2.5, 3.0]),
            id="source_scalar"
        ),
        pytest.param(
            "source_vector", 
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            id="source_vector"
        ),
        pytest.param(
            "function_scalar",
            1.5,
            np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
            id="function_scalar"
        ),
        pytest.param(
            "function_vector",
            (2.0, 3.0, 4.0),
            np.array([2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
            id="function_vector"
        ),
        pytest.param(
            "function_list",
            [1.5, 2.5],
            np.array([1.5, 2.5, 1.5, 2.5, 1.5, 2.5]),
            id="function_list"
        ),
    ]
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
    "error_case,setup_func,error_message",
    [
        pytest.param(
            "no_susceptibility",
            lambda: [magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))],
            "No susceptibility defined in any parent collection",
            id="no_susceptibility"
        ),
        pytest.param(
            "invalid_format",
            lambda: [_create_zone_with_bad_susceptibility()],
            "susceptibility is not scalar or array of length 3",
            id="invalid_format"
        ),
        pytest.param(
            "wrong_length",
            lambda: [magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)) for _ in range(4)],
            "Apply_demag input susceptibility must be scalar, 3-vector, or same length as input Collection",
            id="wrong_length"
        ),
        pytest.param(
            "ambiguous_input",
            lambda: [magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)) for _ in range(3)],
            "Apply_demag input susceptibility is ambiguous",
            id="ambiguous_input"
        ),
    ]
)
def test_get_susceptibilities_errors(error_case, setup_func, error_message):
    """Test error cases for get_susceptibilities function"""
    sources = setup_func()
    
    if error_case == "wrong_length":
        with pytest.raises(ValueError, match=error_message):
            get_susceptibilities(sources, susceptibility=[1.0, 2.0, 3.0, 4.0, 5.0])
    elif error_case == "ambiguous_input":
        with pytest.raises(ValueError, match=error_message):
            get_susceptibilities(sources, susceptibility=(1.0, 2.0, 3.0))
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
