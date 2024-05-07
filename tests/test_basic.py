import numpy as np
import magpylib as magpy
import magpylib_material_response
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid


def test_version():
    assert isinstance(magpylib_material_response.__version__, str)


def test_susceptibility_inputs():
    """
    test if different xi inputs give the same result
    """

    zone = magpy.magnet.Cuboid(
        dimension=(1,1,1),
        polarization=(0,0,1),
    )
    mesh = mesh_Cuboid(zone, (2,2,2))

    dm1 = apply_demag(mesh, susceptibility=4)
    dm2 = apply_demag(mesh, susceptibility=(4,4,4))
    dm3 = apply_demag(mesh, susceptibility=[4]*8)
    dm4 = apply_demag(mesh, susceptibility=[(4,4,4)]*8)

    zone = magpy.magnet.Cuboid(
        dimension=(1,1,1),
        polarization=(0,0,1),
    )
    zone.susceptibility = 4
    mesh = mesh_Cuboid(zone, (2,2,2))
    dm5 = apply_demag(mesh)

    zone = magpy.magnet.Cuboid(
        dimension=(1,1,1),
        polarization=(0,0,1),
    )
    zone.susceptibility = (4,4,4)
    mesh = mesh_Cuboid(zone, (2,2,2))
    dm6 = apply_demag(mesh)

    b1 = dm1.getB((1,2,3))
    for dm in [dm2,dm3,dm4,dm5,dm6]:
        bb = dm.getB((1,2,3))
        np.testing.assert_allclose(b1,bb)
