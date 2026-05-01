from __future__ import annotations

import json

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response import from_json, to_json
from magpylib_material_response.utils import (
    _deserialize_recursive,
    _serialize_recursive,
)

PROBE = np.array([(0.001, 0.002, 0.003), (-0.001, 0, 0.005)])


def _assert_field_equal(a, b, **kw):
    np.testing.assert_allclose(a.getB(PROBE), b.getB(PROBE), **kw)


def test_roundtrip_cuboid():
    src = magpy.magnet.Cuboid(
        polarization=(0.1, 0.2, 0.3),
        dimension=(0.001, 0.002, 0.003),
        position=(0.0, 0.001, 0.002),
    )
    src.rotate_from_angax(30, "z")
    src.susceptibility = 0.5
    src.style.label = "cube"

    dd = _serialize_recursive(src)
    assert dd["type"] == "magnet.Cuboid"
    assert dd["orientation"]["representation"] == "matrix"
    assert dd["susceptibility"]["value"] == 0.5

    out = _deserialize_recursive(dd)
    assert isinstance(out, magpy.magnet.Cuboid)
    assert out.susceptibility == 0.5
    assert out.style.label == "cube"
    _assert_field_equal(src, out)


def test_roundtrip_cylinder():
    src = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(0.002, 0.003))
    src.susceptibility = 100.0
    out = _deserialize_recursive(_serialize_recursive(src))
    assert isinstance(out, magpy.magnet.Cylinder)
    assert out.susceptibility == 100.0
    _assert_field_equal(src, out)


def test_roundtrip_cylinder_segment():
    src = magpy.magnet.CylinderSegment(
        polarization=(1, 0, 0), dimension=(0.001, 0.0015, 0.002, 0.0, 270.0)
    )
    out = _deserialize_recursive(_serialize_recursive(src))
    assert isinstance(out, magpy.magnet.CylinderSegment)
    np.testing.assert_allclose(out.dimension, src.dimension)
    _assert_field_equal(src, out)


def test_roundtrip_polyline():
    src = magpy.current.Polyline(
        current=2.5,
        vertices=[(0, 0, 0), (0.001, 0, 0), (0.001, 0.001, 0), (0, 0.001, 0)],
    )
    src.move((0.001, 0, 0))
    out = _deserialize_recursive(_serialize_recursive(src))
    assert isinstance(out, magpy.current.Polyline)
    assert out.current == 2.5
    np.testing.assert_allclose(out.vertices, src.vertices)
    _assert_field_equal(src, out)


def test_roundtrip_circle():
    src = magpy.current.Circle(current=1.5, diameter=0.005)
    src.rotate_from_angax(45, "x")
    out = _deserialize_recursive(_serialize_recursive(src))
    assert isinstance(out, magpy.current.Circle)
    assert out.current == 1.5
    assert out.diameter == 0.005
    _assert_field_equal(src, out)


def test_roundtrip_sensor():
    src = magpy.Sensor(pixel=[(0, 0, 0), (0.001, 0, 0)], position=(0, 0, 0.001))
    out = _deserialize_recursive(_serialize_recursive(src))
    assert isinstance(out, magpy.Sensor)
    np.testing.assert_allclose(out.pixel, src.pixel)


def test_roundtrip_collection_mixed():
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    cube.susceptibility = 0.5
    pline = magpy.current.Polyline(
        current=1.0, vertices=[(0, 0, 0), (0.001, 0, 0), (0.001, 0.001, 0)]
    )
    circle = magpy.current.Circle(current=2.0, diameter=0.002)
    sub = magpy.Collection(circle, style_label="sub")
    coll = magpy.Collection(cube, pline, sub, style_label="root")

    out = from_json(to_json(coll))[0]

    assert isinstance(out, magpy.Collection)
    assert out.style.label == "root"
    assert len(out.children) == 3
    assert isinstance(out.children[0], magpy.magnet.Cuboid)
    assert isinstance(out.children[1], magpy.current.Polyline)
    assert isinstance(out.children[2], magpy.Collection)
    assert out.children[2].style.label == "sub"
    _assert_field_equal(coll, out)


def test_path_roundtrip():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    src.move(np.linspace((0, 0, 0), (0, 0, 0.005), 5))
    src.rotate_from_angax(np.linspace(0, 90, 5), "z", start=0)
    out = _deserialize_recursive(_serialize_recursive(src))
    np.testing.assert_allclose(out.position, src.position)
    np.testing.assert_allclose(
        out.orientation.as_matrix(), src.orientation.as_matrix(), atol=1e-12
    )
    _assert_field_equal(src, out, atol=1e-15)


def test_to_from_json_string():
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    cube.susceptibility = 0.3
    pline = magpy.current.Polyline(current=1.0, vertices=[(0, 0, 0), (0.001, 0, 0)])
    coll = magpy.Collection(cube, pline)

    s = to_json(coll, indent=2)
    assert isinstance(s, str)
    # Output must be valid JSON with the documented type tags.
    decoded = json.loads(s)
    assert decoded[0]["type"] == "Collection"
    assert decoded[0]["children"][0]["type"] == "magnet.Cuboid"
    assert decoded[0]["children"][1]["type"] == "current.Polyline"

    out = from_json(s)[0]
    _assert_field_equal(coll, out)


def test_unsupported_type_raises():
    src = magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=0.001)
    with pytest.raises(TypeError, match="Unsupported"):
        _serialize_recursive(src)


def test_unknown_type_tag_raises():
    bad = {
        "type": "magnet.Bogus",
        "position": {"value": [0, 0, 0], "unit": "m"},
        "orientation": {
            "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "representation": "matrix",
        },
    }
    with pytest.raises(TypeError, match="Unknown type tag"):
        _deserialize_recursive(bad)


def test_bad_unit_raises():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    dd = _serialize_recursive(src)
    dd["position"]["unit"] = "mm"
    with pytest.raises(ValueError, match="Position unit"):
        _deserialize_recursive(dd)


def test_bad_orientation_representation_raises():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    dd = _serialize_recursive(src)
    dd["orientation"]["representation"] = "quaternion"
    with pytest.raises(ValueError, match="Orientation representation"):
        _deserialize_recursive(dd)


def test_parent_serialization_warns():
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    magpy.Collection(cube)  # sets cube.parent
    with pytest.warns(UserWarning, match="parent"):
        _serialize_recursive(cube)


def test_from_json_returns_multiple_objects():
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    sens = magpy.Sensor()
    out = from_json(to_json(cube, sens))
    assert len(out) == 2
    assert isinstance(out[0], magpy.magnet.Cuboid)
    assert isinstance(out[1], magpy.Sensor)
