from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("ngsolve")

from magpylib_material_response.fem import fem_study

B_EXT = 0.01


@pytest.mark.parametrize(
    ("chi", "polarization", "rotvec", "field"),
    [
        (1000.0, (0, 0, 0), (0, 0, 0), (0, 0, B_EXT)),
        ((1000.0, 10.0, 0.5), (0.3, 0, 1.0), (0.4, -0.3, 0.6), (B_EXT, 0, B_EXT)),
    ],
    ids=["soft", "anisotropic-magnet-turned"],
)
def test_sphere_matches_the_exact_field(chi, polarization, rotvec, field):
    """A sphere magnetizes uniformly: (I + chi/3) J = chi B_ext + J_r, and outside
    it is a dipole. Observers near it and far beyond the meshed ball, whose
    field comes through the Kelvin transform."""
    a = 0.005
    sphere = magpy.magnet.Sphere(
        diameter=2 * a,
        polarization=polarization,
        position=(0.001, -0.002, 0.0005),
        orientation=Rotation.from_rotvec(rotvec),
    )
    sphere.susceptibility = chi
    model = magpy.Collection(sphere)
    model.H_ext = field
    rot = sphere.orientation.as_matrix()
    chi_global = rot @ np.diag(np.broadcast_to(chi, 3)) @ rot.T
    J = np.linalg.solve(
        np.eye(3) + chi_global / 3,
        chi_global @ np.array(field) + sphere.orientation.apply(polarization),
    )
    dipole = magpy.misc.Dipole(
        moment=J * 4 / 3 * np.pi * a**3 / magpy.mu_0, position=sphere.position
    )
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(60, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = np.r_[rng.uniform(1.2, 3, 40), rng.uniform(5, 20, 20)] * a
    observers = sphere.position + directions * distances[:, None]

    study = fem_study(model, observers, (2, 3, 4), maxh=0.35 * a)

    exact = dipole.getB(observers)
    error = np.linalg.norm(study.estimate - exact) / np.linalg.norm(exact)
    assert error < study.error < 0.05
    moment_error = np.linalg.norm(study.solution.moment - dipole.moment)
    assert moment_error < 1e-4 * np.linalg.norm(dipole.moment)


def test_cube_converges_to_the_polarizability_from_above():
    """Helsing and Perfekt (2013): 3.644305 times its volume for a cube of
    infinite susceptibility, 1 / (1/3.644305 + 1e-6) at 1e6. The finite-element
    moment is an upper bound, so it falls toward it with the order. Turned, so
    that its edges are not along the axes."""
    a = 0.01
    cube = magpy.magnet.Cuboid(
        dimension=(a, a, a),
        polarization=(0, 0, 0),
        position=(0.002, 0, -0.001),
        orientation=Rotation.from_rotvec((0.3, -0.2, 0.5)),
    )
    cube.susceptibility = 1e6
    model = magpy.Collection(cube)
    model.H_ext = tuple(cube.orientation.apply((0, 0, B_EXT)))

    study = fem_study(model, lambda s: s.moment, (2, 3, 4), edge_layers=3)

    alphas = np.linalg.norm(study.values, axis=1) * magpy.mu_0 / (B_EXT * a**3)
    exact = 1 / (1 / 3.644305 + 1e-6)
    assert np.all(np.diff(alphas) < 0)
    assert alphas[-1] > exact
    assert alphas[-1] / exact - 1 < study.error < 0.01


def _icosahedron(radius):
    g = (1 + 5**0.5) / 2
    v = np.array(
        [
            (-1, g, 0),
            (1, g, 0),
            (-1, -g, 0),
            (1, -g, 0),
            (0, -1, g),
            (0, 1, g),
            (0, -1, -g),
            (0, 1, -g),
            (g, 0, -1),
            (g, 0, 1),
            (-g, 0, -1),
            (-g, 0, 1),
        ],
        dtype=float,
    )
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9)]
    faces += [(5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2)]
    faces += [(3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10)]
    faces += [(8, 6, 7), (9, 8, 1)]
    return magpy.magnet.TriangularMesh(
        vertices=radius * v / np.linalg.norm(v[0]), faces=faces, polarization=(0, 0, 0)
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: magpy.magnet.Cylinder(dimension=(0.01, 0.01), polarization=(0, 0, 0)),
        lambda: magpy.magnet.CylinderSegment(
            dimension=(0.003, 0.005, 0.01, 20, 140), polarization=(0, 0, 0)
        ),
        lambda: magpy.magnet.Tetrahedron(
            vertices=[(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)],
            polarization=(0, 0, 0),
        ),
        lambda: _icosahedron(0.005),
    ],
    ids=["cylinder", "segment", "tetrahedron", "icosahedron"],
)
def test_every_shape_solves(make):
    """Each shape takes its own path into a solid; an icosahedron, with five
    edges at each corner, broke netgen's edge layers."""
    soft = make()
    soft.susceptibility = 1000
    model = magpy.Collection(soft)
    model.H_ext = (0, 0, B_EXT)

    study = fem_study(model, lambda s: s.moment, (1, 2))

    assert np.all(study.values[:, 2] > 0)


def test_kept_solutions_stay_valid():
    """Every order curves the shared mesh; a solution kept from a study reads its
    field on its own geometry afterwards."""
    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
    sphere.susceptibility = 1000
    model = magpy.Collection(sphere)
    model.H_ext = (0, 0, B_EXT)
    observers = np.array([(0, 0, 0.008), (0.007, 0, 0.002)])
    kept = []

    def keep(solution):
        kept.append((solution, solution.getB(observers)))
        return solution.moment

    fem_study(model, keep, (1, 2, 3))

    for solution, at_solve in kept:
        np.testing.assert_allclose(solution.getB(observers), at_solve, rtol=1e-12)


def _two_fields():
    soft = [
        magpy.magnet.Cuboid(dimension=(1, 1, 1), position=(2 * i, 0, 0))
        for i in range(2)
    ]
    for part, field in zip(soft, [(0, 0, 0.1), (0, 0, 0.2)], strict=True):
        part.susceptibility, part.H_ext = 100, field
    return magpy.Collection(soft)


def _overlapping():
    cubes = [
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=(0.5 * i, 0, 0)
        )
        for i in range(2)
    ]
    for cube in cubes:
        cube.susceptibility = 0
    return magpy.Collection(cubes)


@pytest.mark.parametrize(
    ("model", "error"),
    [
        (
            lambda: magpy.Collection(
                magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)),
                magpy.current.Circle(diameter=3, current=1),
            ),
            NotImplementedError,
        ),
        (_two_fields, ValueError),
        (lambda: magpy.Collection(magpy.misc.Dipole(moment=(0, 0, 1))), TypeError),
        (
            lambda: magpy.Collection(
                magpy.misc.Triangle(
                    vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], polarization=(0, 0, 1)
                )
            ),
            TypeError,
        ),
        (_overlapping, ValueError),
    ],
    ids=["current", "two-applied-fields", "dipole", "triangle", "overlap"],
)
def test_refuses_what_it_cannot_model(model, error):
    with pytest.raises(error):
        fem_study(model(), [(0, 0, 5)])
