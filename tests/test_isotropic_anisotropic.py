from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from magpylib_material_response import demag, meshing


def test_isotropic_susceptibility():
    cells = 1000  # should be >=1000, otherwise discretization error too large

    magnet = magpy.magnet.Cuboid(dimension=(1e-3, 1e-3, 1e-3), polarization=(0, 0, 1.1))
    grid = np.loadtxt("tests/testdata/grid_points.pts")
    field_ansys = np.loadtxt("tests/testdata/isotropic_results_ansys.txt", skiprows=1)
    field_ansys = field_ansys[:, 3:]

    # isotropic
    magnet.susceptibility = 0.1
    magnet_meshed = meshing.mesh_Cuboid(magnet, cells)

    demag.apply_demag(magnet_meshed, inplace=True)

    field_magpylib = magnet_meshed.getB(grid)

    np.testing.assert_allclose(field_ansys, field_magpylib, rtol=0, atol=0.0012)


@pytest.mark.parametrize(
    "rotation",
    [R.identity(), R.from_rotvec((0.3, -0.5, 0.8))],
    ids=["unrotated", "rotated"],
)
def test_anisotropic_susceptibility(rotation):
    """The susceptibility acts along the magnet's own axes (#50): rotating the
    magnet and the grid together rotates the reference field."""
    cells = 1000  # should be >=1000, otherwise discretization error too large

    magnet = magpy.magnet.Cuboid(dimension=(1e-3, 1e-3, 1e-3), polarization=(0, 0, 1.1))
    magnet.rotate(rotation)
    grid = np.loadtxt("tests/testdata/grid_points.pts")
    field_ansys = np.loadtxt("tests/testdata/anisotropic_results_ansys.txt", skiprows=1)
    field_ansys = field_ansys[:, 3:]

    # anisotropic
    magnet.susceptibility = (0.3, 0.2, 0.1)
    magnet_meshed = meshing.mesh_Cuboid(magnet, cells)

    demag.apply_demag(magnet_meshed, inplace=True)

    field_magpylib = rotation.inv().apply(magnet_meshed.getB(rotation.apply(grid)))

    np.testing.assert_allclose(field_ansys, field_magpylib, rtol=0, atol=0.0012)


def test_negative_susceptibility():
    cells = 1000  # should be >=1000, otherwise discretization error too large

    magnet = magpy.magnet.Cuboid(
        dimension=(1e-3, 1e-3, 1e-3), polarization=(0, 0, -0.1)
    )
    grid = np.loadtxt("tests/testdata/grid_points.pts")
    field_ansys = np.loadtxt(
        "tests/testdata/negative_susceptibility_ansys.txt", skiprows=1
    )
    field_ansys = field_ansys[:, 3:]

    # isotropic
    magnet.susceptibility = -1.1
    magnet_meshed = meshing.mesh_Cuboid(magnet, cells)

    demag.apply_demag(magnet_meshed, inplace=True)

    field_magpylib = magnet_meshed.getB(grid)

    np.testing.assert_allclose(field_ansys, field_magpylib, rtol=0, atol=0.0065)
