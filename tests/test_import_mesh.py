"""Test meshing functions"""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.customMesh import import_mesh


def test_mesh_Cuboid():
    # create cuboid
    c = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(0.001, 0.001, 0.002),
        position=[0.00, 0.00, 0.00],
    )
    c.move((0, 0, 0.0005))
    c.style.label = "Cuboid 1"
    c.susceptibility = 3999

    cm = import_mesh(
        "tests/testdata/cuboid1.inp",
        scaling=1e-3,
        polarization=(0, 0, 1),
        succeptibility=3999,
    )
    cm.move((0, 0, -0.0005))

    # test if custom meshed cuboid collection provides the same field as original cuboid
    np.testing.assert_allclose(
        c.getB([0, 0, 0]), cm.getB([0, 0, 0]), rtol=7e-4, atol=2e-5
    )

    # test if all children sources got the right susceptibility value
    assert all(s.susceptibility == c.susceptibility for s in cm.sources_all)

    with pytest.raises(ValueError):
        import_mesh(
            "cuboid1.stl", scaling=1e-3, polarization=(0, 0, 1), succeptibility=3999
        )


def test_mesh_Cuboid2():
    # test for custom elements
    c = import_mesh(
        "tests/testdata/cuboid1.inp",
        scaling=1e-3,
        polarization=(0, 0, 1),
        succeptibility=3999,
    )
    num_elemets = 80
    assert num_elemets == len(c.sources_all)
