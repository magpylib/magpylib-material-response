"""Test meshing functions"""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import (
    mesh_all,
    mesh_Cuboid,
    mesh_Cylinder,
    mesh_thin_CylinderSegment_with_cuboids,
    mesh_TriangularMesh,
    slice_Cuboid,
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


def test_cells_are_force_targets():
    """Meshed and demagnetized collections are force targets with one point per
    cell; a meshed cuboid gives the force of the same grid on the whole."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01))
    src = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01), position=(0, 0, -0.011)
    )
    cyl = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(0.01, 0.01))
    for coll in (mesh_Cuboid(cube, 27), mesh_Cylinder(cyl, 30)):
        F, T = magpy.getFT(src, coll, meshing=1)
        assert np.all(np.isfinite(F))
        assert np.all(np.isfinite(T))

    F_cells = magpy.getFT(src, mesh_Cuboid(cube, 27), meshing=1)[0]
    F_whole = magpy.getFT(src, cube, meshing=(3, 3, 3))[0]
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
            x, y, _ = cell.centroid
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
    magnet.polarization = (1, 0, 0)
    mesh = magnet._generate_mesh(target_elems)
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


def test_H_ext_set_before_meshing_reaches_the_cells():
    """The applied field moves to the meshed collection, where a later change
    still reaches the cells."""

    def soft(H_ext=None):
        obj = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
        obj.susceptibility = 1000
        if H_ext is not None:
            obj.H_ext = H_ext
        return obj

    obj = soft(np.array([0, 0, 0.01]))
    mesh = mesh_Cuboid(obj, 8)
    mesh.H_ext *= 1  # an in-place edit of the mesh leaves the magnet alone
    assert mesh.H_ext is not obj.H_ext
    before = apply_demag(mesh)
    after = mesh_Cuboid(soft(), 8)
    after.H_ext = (0, 0, 0.01)
    after = apply_demag(after)
    J = np.array([c.polarization for c in before.sources_all])
    assert np.all(J[:, 2] > 0.01)
    np.testing.assert_allclose(J, [c.polarization for c in after.sources_all])

    changed = mesh_Cuboid(soft((0, 0, 0.01)), 8)
    changed.H_ext = (0, 0, 0.02)
    J2 = np.array([c.polarization for c in apply_demag(changed).sources_all])
    np.testing.assert_allclose(J2, 2 * J)


def test_mesh_all_returns_the_mesh_of_a_magnet_in_a_collection():
    magnet = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    outer = magpy.Collection(magnet)
    meshed = mesh_all(magnet, target_elems=8, inplace=True)
    assert isinstance(meshed, magpy.Collection)
    assert meshed.parent is outer
    assert len(meshed.sources_all) == 8


def test_mesh_Cuboid_grading_tiles_the_cuboid():
    """Graded cells fill the cuboid, shrinking toward the faces: with the same
    polarization they reproduce the field of the unmeshed magnet."""
    magnet = magpy.magnet.Cuboid(
        dimension=(0.02, 0.01, 0.03), polarization=(0.1, 0.2, 1), position=(0.1, 0, 0)
    )
    magnet.rotate_from_angax(30, "z")
    mesh = mesh_Cuboid(magnet, (4, 3, 5), grading=3)
    sides = np.array([c.dimension for c in mesh.sources_all])
    assert sides[:, 0].max() / sides[:, 0].min() == pytest.approx(3)
    probes = [(0.1, 0, 0.03), (0.13, 0.01, 0), (0.08, -0.02, 0.01)]
    np.testing.assert_allclose(mesh.getB(probes), magnet.getB(probes), rtol=1e-12)
    with pytest.raises(ValueError, match="grading"):
        mesh_Cuboid(magnet, 8, grading=0.5)


@pytest.mark.parametrize(("grading", "low", "high"), [(1, 1.5, 3.0), (3, 0.0, 0.5)])
def test_grading_cuts_the_error_of_a_soft_cube(grading, low, high):
    """Soft iron (susceptibility 1000) in 10 mT, 512 cells: identical cells miss
    the converged moment by over 1.5 %, cells three times smaller at the faces
    than in the middle by under 0.5 %. The converged moment, 0.02889 A m^2, is
    V * H0 / (N + 1/chi) with N = 1/3.6442, the polarizability of a perfectly
    permeable cube (that of a conducting cube, 3.6442 a^3)."""
    cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
    cube.susceptibility = 1000
    cube.H_ext = (0, 0, 0.01)
    cells = apply_demag(mesh_Cuboid(cube, 512, grading=grading)).sources_all
    moment = sum(c.polarization[2] * np.prod(c.dimension) for c in cells) / magpy.mu_0
    assert low < abs(moment - 0.02889) / 0.02889 * 100 < high
