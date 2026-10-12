"""Test meshing functions"""

from __future__ import annotations

import inspect

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.demag import _matching_point, apply_demag
from magpylib_material_response.meshing import (
    mesh_all,
    mesh_Cuboid,
    mesh_Cylinder,
    mesh_thin_CylinderSegment_with_cuboids,
    mesh_TriangularMesh,
    slice_Cuboid,
    voxelize,
)
from magpylib_material_response.meshing_utils import trimesh_from_model3d


def test_mesh_Cuboid():
    # create cuboid with rotated path
    N = 5
    opacity = 0.5
    c = magpy.magnet.Cuboid(
        polarization=(1, 0, 0),
        dimension=(0.001, 0.002, 0.003),
        position=[0.004, 0.005, 0.006],
    )
    c.style.label = "Cuboid 1"
    c.style.opacity = 0  # will be overridden by mesh kwargs lower
    c.rotate_from_angax(np.linspace(0, 76, N), "z", anchor=0, start=0)
    c.move(np.linspace((0, 0, 0), (0, 0, 0.0105), N), start=0)
    c.susceptibility = 3999

    cm = mesh_Cuboid(c, target_elems=8, style_opacity=opacity)

    # test if meshed cuboid collection provides the same field as original cuboid
    np.testing.assert_allclose(c.getB([0, 0, 0]), cm.getB([0, 0, 0]))

    # test if all children sources got the right susceptibility value
    assert all(s.susceptibility == c.susceptibility for s in cm.sources_all)

    # test style kwargs
    assert cm.style.label == c.style.label
    assert cm.style.opacity == opacity

    with pytest.raises(TypeError):
        mesh_Cuboid(magpy.magnet.Cylinder(), target_elems=8)


def test_mesh_Cuboid2():
    # test for custom target elems per dimension
    c = magpy.magnet.Cuboid(dimension=(1, 1, 1))
    target_elems = [1, 2, 3]
    cm = mesh_Cuboid(c, target_elems=target_elems)
    poss = np.array([o.position for o in cm])
    assert target_elems == [len(np.unique(poss[:, i])) for i in range(3)]


def test_slice_Cuboid():
    # create cuboid with rotated path
    N = 5
    opacity = 0.5
    c = magpy.magnet.Cuboid(
        polarization=(1, 0, 0),
        dimension=(0.001, 0.002, 0.003),
        position=[0.004, 0.005, 0.006],
    )
    c.style.label = "Cuboid 1"
    c.style.opacity = 0  # will be overridden by mesh kwargs lower
    c.rotate_from_angax(np.linspace(0, 76, N), "z", anchor=0, start=0)
    c.move(np.linspace((0, 0, 0), (0, 0, 0.0105), N), start=0)
    c.susceptibility = 3999

    cm = slice_Cuboid(c, shift=0.2, axis="y", style_opacity=opacity)

    # test if meshed cuboid collection provides the same field as original cuboid
    np.testing.assert_allclose(c.getB([0, 0, 0]), cm.getB([0, 0, 0]))

    # test if all children sources got the right susceptibility value
    assert all(s.susceptibility == c.susceptibility for s in cm.sources_all)

    # test style kwargs
    assert cm.style.label == c.style.label
    assert cm.style.opacity == opacity

    with pytest.raises(TypeError):
        slice_Cuboid(magpy.magnet.Cylinder(), target_elems=8)


def test_slice_Cuboid2():
    c = magpy.magnet.Cuboid(dimension=(1, 1, 1))

    # axis x
    cm = slice_Cuboid(c, shift=0.5, axis="x")
    np.testing.assert_allclose(cm[0].dimension, [0.5, 1.0, 1.0])
    np.testing.assert_allclose(cm[1].dimension, [0.5, 1.0, 1.0])
    np.testing.assert_allclose(cm[0].position, [-0.25, 0.0, 0.0])
    np.testing.assert_allclose(cm[1].position, [0.25, 0.0, 0.0])

    # axis y
    cm = slice_Cuboid(c, shift=0.2, axis="y")
    np.testing.assert_allclose(cm[0].dimension, [1.0, 0.2, 1.0])
    np.testing.assert_allclose(cm[1].dimension, [1.0, 0.8, 1.0])
    np.testing.assert_allclose(cm[0].position, [0.0, -0.4, 0.0])
    np.testing.assert_allclose(cm[1].position, [0.0, 0.1, 0.0])
    cm = slice_Cuboid(c, shift=0.1, axis="z")

    # axis z
    np.testing.assert_allclose(cm[0].dimension, [1.0, 1.0, 0.1])
    np.testing.assert_allclose(cm[1].dimension, [1.0, 1.0, 0.9])
    np.testing.assert_allclose(cm[0].position, [0.0, 0.0, -0.45])
    np.testing.assert_allclose(cm[1].position, [0.0, 0.0, 0.05])

    with pytest.raises(ValueError, match=r"Shift must be between 0 and 1 *."):
        slice_Cuboid(c, shift=0, axis="y")


def test_mesh_Cylinder():
    # create CylinderSegment with rotated path
    N = 3
    opacity = 0.5
    c = magpy.magnet.CylinderSegment(
        polarization=(1, 0, 0),
        dimension=(1, 1.1, 1, 0, 270),
        position=[1, 2, 1],
    )
    c.style.label = "CylinderSegment 1"
    c.style.opacity = 0  # will be overridden by mesh kwargs lower
    c.rotate_from_angax(np.linspace(0, 90, N), "z", anchor=0, start=0)
    c.rotate_from_angax(np.linspace(0, 90, N), "x", anchor=0, start=0)
    c.move(np.linspace((0, 0, 0), (0, 0, 1), N), start=0)
    c.style.magnetization.show = False
    c.susceptibility = 3999

    cm = mesh_Cylinder(c, target_elems=20, style_opacity=opacity)

    # test if meshed cuboid collection provides the same field as original cuboid
    np.testing.assert_allclose(c.getB([0, 0, 0]), cm.getB([0, 0, 0]))

    # test if all children sources got the right susceptibility value
    assert all(s.susceptibility == c.susceptibility for s in cm.sources_all)

    # test style kwargs
    assert cm.style.label == c.style.label
    assert cm.style.opacity == opacity


def test_mesh_Cylinder2():
    # test for innermost is cylinder when closed and at least 3 layers
    c = magpy.magnet.Cylinder(dimension=(1, 1))
    cm = mesh_Cylinder(c, target_elems=200)
    cyls = [o for o in cm if isinstance(o, magpy.magnet.Cylinder)]
    cyl_segs = [o for o in cm if isinstance(o, magpy.magnet.CylinderSegment)]

    assert len(cyls) == 6
    assert len(cyl_segs) == 162


@pytest.mark.parametrize(
    ("susceptibility", "accepted"),
    [(0.5, True), ((0.1, 0.1, 2.0), True), (np.array((2.0, 0.1, 0.1)), False)],
)
def test_mesh_thin_CylinderSegment_with_cuboids_anisotropy(susceptibility, accepted):
    """Its cells are turned about z, so only a susceptibility equal along x and y
    keeps the segment's own axes (#50)."""
    seg = magpy.magnet.CylinderSegment(
        polarization=(1, 0, 0), dimension=(1, 1.1, 1, 0, 270)
    )
    seg.susceptibility = susceptibility
    if accepted:
        mesh_thin_CylinderSegment_with_cuboids(seg, target_elems=20)
    else:
        with pytest.raises(ValueError, match="same along x and y"):
            mesh_thin_CylinderSegment_with_cuboids(seg, target_elems=20)


def test_mesh_thin_CylinderSegment_with_cuboids():
    # create thin CylinderSegment with rotated path
    N = 3
    opacity = 0.5
    c = magpy.magnet.CylinderSegment(
        polarization=(1, 0, 0),
        dimension=(1, 1.1, 1, 0, 270),
        position=[1, 2, 1],
        style_opacity=0.2,
    )
    c.rotate_from_angax(np.linspace(0, 90, N), "z", anchor=0, start=0)
    c.rotate_from_angax(np.linspace(0, 90, N), "x", anchor=0, start=0)
    c.move(np.linspace((0, 0, 0), (0, 0, 1), N), start=0)
    c.susceptibility = 3999

    cm = mesh_thin_CylinderSegment_with_cuboids(
        c, target_elems=100, match_volume=False, style_opacity=opacity
    )

    # test if meshed cuboid collection provides the same field as original cuboid
    # needs rtol because it is an approximation.
    np.testing.assert_allclose(c.getB([0, 0, 0]), cm.getB([0, 0, 0]), rtol=0.06)

    cm = mesh_thin_CylinderSegment_with_cuboids(
        c, target_elems=100, style_opacity=opacity
    )

    # test if meshed cuboid collection provides the same field as original cuboid
    # needs rtol because it is an approximation.
    # by default match_volume=True, delivers better accuracy, hence lower rtol
    np.testing.assert_allclose(c.getB([0, 0, 0]), cm.getB([0, 0, 0]), rtol=0.01)

    # test if all children sources got the right susceptibility value
    assert all(s.susceptibility == c.susceptibility for s in cm.sources_all)

    # test style kwargs
    assert cm.style.label == c.style.label
    assert cm.style.opacity == opacity

    with pytest.raises(TypeError):
        mesh_Cuboid(magpy.magnet.Cylinder(), target_elems=8)


def test_mesh_all():
    susceptibility = 3999
    c = magpy.magnet.Cuboid(
        polarization=(1, 0, 0),
        dimension=(1, 2, 3),
        position=[4, 5, 6],
        style_label="C1",
    )
    c.susceptibility = susceptibility
    cy = magpy.magnet.Cylinder(polarization=(1, 1, 1), dimension=(4, 1))
    cy.susceptibility = c.susceptibility
    c2 = magpy.Collection(c.copy(dimension=c.dimension / 2).move((5, 0, 0)))
    c2.style.label = "C2 super"
    # currents and sensors and sensors should be just silently ignored by meshing
    curr = magpy.current.Circle(current=1, diameter=1)
    sens = magpy.Sensor()
    c0 = magpy.Collection(c, c2, cy, sens, curr)

    # fail on other unsupported magnets
    c1 = c0.copy()
    c1.add(magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1))
    with pytest.raises(TypeError):
        mesh_all(c1, target_elems=20)

    # mesh per total target elems (by default)
    cm = mesh_all(c0, target_elems=20)
    cubs = [o for o in cm.sources_all if isinstance(o, magpy.magnet.Cuboid)]
    cyls = [o for o in cm.sources_all if isinstance(o, magpy.magnet.Cylinder)]
    cylsegs = [o for o in cm.sources_all if isinstance(o, magpy.magnet.CylinderSegment)]
    assert len(cubs) == 12
    assert len(cyls) == 0
    assert len(cylsegs) == 12

    # mesh per child elems
    cm = mesh_all(c0, per_child_elems=True, target_elems=20)
    cubs = [o for o in cm.sources_all if isinstance(o, magpy.magnet.Cuboid)]
    cyls = [o for o in cm.sources_all if isinstance(o, magpy.magnet.Cylinder)]
    cylsegs = [o for o in cm.sources_all if isinstance(o, magpy.magnet.CylinderSegment)]
    assert len(cubs) == 24
    assert len(cyls) == 0
    assert len(cylsegs) == 14

    # mesh on copy (by default)
    cm = mesh_all(c0, target_elems=20)
    assert cm is not c0
    # test if all children sources got the right susceptibility value
    mags = [s for s in cm.sources_all if not isinstance(s, magpy.current.Circle)]
    assert all(s.susceptibility == c.susceptibility for s in mags)

    # mesh inplace
    cm = mesh_all(c0, target_elems=20, inplace=True)
    assert cm is c0
    # test if all children sources got the right susceptibility value
    mags = [s for s in cm.sources_all if not isinstance(s, magpy.current.Circle)]
    assert all(s.susceptibility == c.susceptibility for s in mags)


def test_mesh_TriangularMesh():
    """
    Build a TriangularMesh from a cuboid surface (via make_Cuboid), mesh it with
    mesh_TriangularMesh, and compare getB against the original Cuboid.
    """
    pytest.importorskip("tetgen", reason="tetgen optional dependency not installed")

    dim = (0.002, 0.003, 0.005)
    pol = (0.0, 0.0, 1.0)
    opacity = 0.5

    # Reference cuboid
    cuboid = magpy.magnet.Cuboid(polarization=pol, dimension=dim)
    cuboid.style.label = "Cuboid ref"
    cuboid.style.opacity = 0
    cuboid.susceptibility = 3999

    # Build a closed TriangularMesh from make_Cuboid surface output
    trimesh = trimesh_from_model3d("cuboid", pol, dimension=dim)
    trimesh.style.label = "Cuboid ref"
    trimesh.style.opacity = 0
    trimesh.susceptibility = cuboid.susceptibility
    assert not trimesh.status_open, "TriangularMesh surface must be closed"

    # Mesh into Tetrahedra and check field
    cm = mesh_TriangularMesh(trimesh, target_elems=50, style_opacity=opacity)

    # A cuboid has only flat faces — no curved surface is approximated — so the
    # tetrahedral mesh is geometrically exact and the field should match very closely.
    # Observer is off all symmetry axes so all three B components are non-zero
    # (avoids rtol instability when the reference value is ~0).
    obs = [0.01, 0.007, 0.004]
    np.testing.assert_allclose(
        cuboid.getB(obs),
        cm.getB(obs),
        rtol=1e-3,
        atol=1e-8,
        err_msg="mesh_TriangularMesh B-field deviates from reference Cuboid",
    )

    # Susceptibility propagated to all cells
    assert all(s.susceptibility == cuboid.susceptibility for s in cm.sources_all)

    # Style propagated
    assert cm.style.label == trimesh.style.label
    assert cm.style.opacity == opacity

    # TypeError on wrong input type
    with pytest.raises(TypeError):
        mesh_TriangularMesh(cuboid, target_elems=10)


def test_thin_cylinder_segment_extreme_aspect_ratio():
    """Regression: nphi/nh rounding to zero caused ZeroDivisionError."""
    seg = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 1), dimension=(0.0099, 0.01, 1.0, 0, 45)
    )
    coll = mesh_thin_CylinderSegment_with_cuboids(seg, 1)
    assert len(coll.sources_all) >= 1


def test_trimesh_from_model3d_tiny_scale():
    """Regression: absolute 1e-10 vertex rounding collapsed sub-1e-10 m shapes;
    dedup quantization is now relative to the geometry scale."""
    trimesh = trimesh_from_model3d("cuboid", (0, 0, 1), dimension=(1e-10, 1e-10, 1e-10))
    assert trimesh.vertices.shape == (8, 3)
    assert not trimesh.status_open


def test_voxelize_small_target_finite_cells():
    """Regression: grid_elems=1 caused division by zero, and an empty grid
    crashed with an opaque reshape error; both now behave cleanly."""
    sphere = magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=0.001)
    with pytest.raises(ValueError, match="no cell centres"):
        voxelize(sphere, 3)  # grid too coarse: clean error, no crash
    coll = voxelize(sphere, 100)
    dims = np.array([c.dimension for c in coll.sources_all])
    assert np.isfinite(dims).all()
    assert (dims > 0).all()


# magpylib 6 takes `meshing` on the getFT call; 5.2 keeps it on the targets
MESHING_ON_CALL = "meshing" in inspect.signature(magpy.getFT).parameters


def test_cells_are_force_targets():
    """Every cell gets one force point, so getFT accepts meshed and demagnetized
    collections; a meshed cuboid gives the force of the same grid on the whole."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01))
    src = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01), position=(0, 0, -0.011)
    )
    cyl = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(0.01, 0.01))
    for coll in (mesh_Cuboid(cube, 27), mesh_Cylinder(cyl, 30)):
        if MESHING_ON_CALL:
            F, T = magpy.getFT(src, coll, meshing=1)
        else:
            assert {cell.meshing for cell in coll.children} == {1}
            F, T = magpy.getFT(src, coll)
        assert np.all(np.isfinite(F))
        assert np.all(np.isfinite(T))

    if MESHING_ON_CALL:
        F_cells = magpy.getFT(src, mesh_Cuboid(cube, 27), meshing=1)[0]
        F_whole = magpy.getFT(src, cube, meshing=(3, 3, 3))[0]
    else:
        cube.meshing = (3, 3, 3)
        F_cells = magpy.getFT(src, mesh_Cuboid(cube, 27))[0]
        F_whole = magpy.getFT(src, cube)[0]
    np.testing.assert_allclose(F_cells, F_whole, rtol=1e-9, atol=1e-9)


def test_mesh_Cylinder_thin_ring_centroids_inside():
    """A wide sector of a thin ring has its centroid outside the material, where
    demagnetization matches the field: a 4-5 mm tube in 12 cells came out with
    the wrong sign in a transverse field."""
    tube = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0), dimension=(0.004, 0.005, 0.03, 0, 360)
    )
    for target in (3, 12, 24, 48):
        for cell in mesh_Cylinder(tube, target).children:
            r1, r2 = cell.dimension[:2]
            x, y, _ = _matching_point(cell)
            assert r1 <= np.hypot(x, y) <= r2

    src = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(0.01, 0.01, 0.01), position=(0.015, 0, 0)
    )
    src.susceptibility = 0

    def moment_x(target):
        cells = mesh_Cylinder(tube, target)
        cells.susceptibility = 1000
        out = apply_demag(magpy.Collection(src.copy(), cells)).children[1]
        return sum(c.polarization[0] * c.volume for c in out.children)

    reference = moment_x(800)
    for target in (12, 48):
        np.testing.assert_allclose(moment_x(target), reference, rtol=0.05)


def _core_makes_cells():
    """magpylib's force mesher returns cells from magpylib 6 on (magpylib#1023)."""
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    if MESHING_ON_CALL:
        return "cells" in cube._generate_mesh(1)
    cube.meshing = 1
    return "cells" in cube._generate_mesh()


def _segment_dimensions(cells):
    """(r1, r2, h, phi1, phi2) of each cell, a full ring written as 0 to 360."""
    dims = np.array(
        [
            (0, c.dimension[0] / 2, c.dimension[1], 0, 360)
            if isinstance(c, magpy.magnet.Cylinder)
            else tuple(c.dimension)
            for c in cells
        ]
        if not isinstance(cells, np.ndarray)
        else cells,
        dtype=float,
    )
    full = np.isclose(dims[:, 4] - dims[:, 3], 360)
    dims[full, 3], dims[full, 4] = 0, 360
    return dims


@pytest.mark.parametrize(
    ("magnet", "target_elems"),
    [
        (magpy.magnet.Cylinder(dimension=(1, 1)), 200),
        (magpy.magnet.CylinderSegment(dimension=(1, 1.1, 1, 0, 270)), 20),
        (magpy.magnet.CylinderSegment(dimension=(0, 1, 2, 10, 370)), 50),
        (magpy.magnet.CylinderSegment(dimension=(0, 1, 2, 10, 370)), 300),
        (magpy.magnet.CylinderSegment(dimension=(0.95, 1, 0.5, 0, 360)), 1),
        (magpy.magnet.Cylinder(dimension=(2, 1)), (8, 3, 2)),
        (magpy.magnet.Cuboid(dimension=(0.001, 0.002, 0.003)), 100),
        (magpy.magnet.Cuboid(dimension=(3, 2, 1)), 5),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), (2, 3, 4)),
    ],
    ids=[
        "Cylinder-200",
        "Segment-20",
        "Segment-50",
        "Disc-300",
        "ThinTube-1",
        "Cylinder-triple",
        "Cuboid-100",
        "Block-5",
        "Cuboid-triple",
    ],
)
def test_cells_match_magpylib_force_mesh(magnet, target_elems):
    """mesh_Cuboid and mesh_Cylinder cut a magnet into the same cells as magpylib's
    force mesher for the same target (magpylib#1023)."""
    if not _core_makes_cells():
        pytest.skip("this magpylib meshes points, not cells")
    magnet.polarization = (1, 0, 0)
    if MESHING_ON_CALL:
        mesh = magnet._generate_mesh(target_elems)
    else:
        magnet.meshing = target_elems
        mesh = magnet._generate_mesh()
    cells = mesh["cells"]
    scale = np.max(np.abs(magnet.dimension))
    if isinstance(magnet, magpy.magnet.Cuboid):
        coll = mesh_Cuboid(magnet, target_elems)
        dims = np.array([cell.dimension for cell in coll])
        expected = cells["dimensions"]
    else:
        coll = mesh_Cylinder(magnet, target_elems)
        dims = _segment_dimensions(coll)
        expected = _segment_dimensions(cells["dimensions"])
    np.testing.assert_allclose(
        [cell.position for cell in coll], cells["positions"], atol=1e-12 * scale
    )
    np.testing.assert_allclose(dims, expected, atol=1e-12 * scale)
