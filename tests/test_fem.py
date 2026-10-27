from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("ngsolve")

from magpylib_material_response.fem import _solve, fem_solve, fem_study

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
        lambda: magpy.magnet.Tetrahedron(
            vertices=[(0, 0, 0), (0, 0.01, 0), (0.01, 0, 0), (0, 0, 0.01)],
            polarization=(0, 0, 0),
        ),
        lambda: _icosahedron(0.005),
    ],
    ids=["cylinder", "segment", "tetrahedron", "tetrahedron-reversed", "icosahedron"],
)
def test_every_shape_solves(make):
    """Each shape takes its own path into a solid. A reversed tetrahedron came
    out inside-out and vanished from the mesh; an icosahedron, with five edges
    at each corner, broke netgen's edge layers."""
    soft = make()
    soft.susceptibility = 1000
    model = magpy.Collection(soft)
    model.H_ext = (0, 0, B_EXT)

    study = fem_study(model, lambda s: s.moment, (1, 2))

    assert np.all(study.values[:, 2] > 0)


def test_edge_layers_match_across_faces(monkeypatch):
    """On this cuboid, an observer above it made NGSolve's edge layers leave a
    prism's square face against two triangles, which no assembly accepts
    (#98); the mesh is made again at the next grading."""
    checks = []
    conforming = _solve._conforming
    monkeypatch.setattr(
        _solve,
        "_conforming",
        lambda mesh: checks.append(conforming(mesh)) or checks[-1],
    )
    box = magpy.magnet.Cuboid(
        dimension=(1.0935e-3, 1.4e-3, 0.8991e-3), polarization=(0, 0, 0)
    )
    box.susceptibility = 100
    model = magpy.Collection(box)
    model.H_ext = (B_EXT, 0, 0)
    observer = np.array([(0, 0, 2e-3)])

    solution = fem_solve(model, order=2, observers=observer)

    if checks[0]:
        pytest.skip("netgen meshed this cuboid matching at the first grading here")
    assert checks[-1]
    assert np.all(np.isfinite(solution.getB(observer)))


def test_kept_solutions_stay_valid():
    """Every order curves the shared mesh; a solution kept from a study reads its
    field on its own geometry afterwards. A lone observer gets one error."""
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
    assert fem_study(model, observers[0], (1, 2, 3)).errors.shape == (1,)


def test_no_source_gives_no_field():
    """No remanence and no applied field: a zero field, as from apply_demag."""
    sphere = magpy.magnet.Sphere(diameter=0.01, polarization=(0, 0, 0))
    sphere.susceptibility = 1000

    solution = fem_solve(magpy.Collection(sphere), 2)

    np.testing.assert_array_equal(solution.moment, 0)
    np.testing.assert_array_equal(solution.getB((0, 0, 0.008)), 0)


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


def _magnet():
    magnet = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    magnet.susceptibility = 0
    return magpy.Collection(magnet)


def _loop(*vertices):
    """A closed Polyline of 1 A through `vertices`, back to the first."""
    return magpy.current.Polyline(current=1, vertices=[*vertices, vertices[0]])


def _soft(make, chi):
    part = make()
    part.susceptibility = chi
    return part


def _cube(position=(0, 0, 0)):
    return _soft(
        lambda: magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 0), position=position
        ),
        10,
    )


def _ring():
    return _soft(
        lambda: magpy.magnet.CylinderSegment(
            dimension=(2, 3, 2, 0, 360), polarization=(0, 0, 0)
        ),
        1000,
    )


def _wired_magnet():
    """A wire straight through a magnet: its field is given in the air only."""
    return magpy.Collection(
        _cube(), _loop((-2, 0, 0), (2, 0, 0), (2, 0, 3), (-2, 0, 3))
    )


def _crossing_wire():
    """A wire crossing a magnet between its vertices: none of them is inside."""
    return magpy.Collection(
        _cube(), _loop((-1, 0, 0), (5, 0, 0), (5, 0, 4), (-1, 0, 4))
    )


def _clipped_corner():
    """A wire cutting 0.06 off a cube's corner, under the coarse spacing."""
    return magpy.Collection(
        _cube(), _loop((1.56, -0.6, 0), (-0.6, 1.56, 0), (-3, -3, 0))
    )


def _wired_tetrahedron():
    tetrahedron = _soft(
        lambda: magpy.magnet.Tetrahedron(
            vertices=[(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
            polarization=(0, 0, 0),
        ),
        10,
    )
    return magpy.Collection(
        tetrahedron, _loop((0.3, 0.3, -1), (0.3, 0.3, 1), (-2, 0.3, 1), (-2, 0.3, -1))
    )


def _open_wire():
    wire = magpy.current.Polyline(current=1, vertices=[(0, 0, 5), (0, 0, 50)])
    return magpy.Collection(_ring(), wire)


def _sheet():
    sheet = magpy.current.TriangleSheet(
        vertices=[(5, 0, 0), (6, 0, 0), (5, 1, 0)],
        faces=[(0, 1, 2)],
        current_densities=[(1, 0, 0)],
    )
    return magpy.Collection(_cube(), sheet)


def _linked_ring():
    """A loop through a soft ring's hole: its field circulates in the ring and
    has no potential there."""
    return magpy.Collection(
        _ring(), _loop((0, 0, -50), (0, 0, 50), (50, 0, 50), (50, 0, -50))
    )


def _lead_beside_a_coil():
    """1 A through a ring's hole beside a coil of 8,000 A that links nothing."""
    coil = magpy.current.Circle(current=8000, diameter=20, position=(0, 0, 10))
    lead = _loop((0, 0, -50), (0, 0, 50), (50, 0, 50), (50, 0, -50))
    return magpy.Collection(_ring(), coil, lead)


def _closed_core():
    """A coil round one bar of a closed square core: its current links the
    core however short the coil is."""
    bars = [
        magpy.magnet.Cuboid(dimension=d, polarization=(0, 0, 0), position=p)
        for d, p in (
            ((20, 4, 4), (0, -8, 0)),
            ((20, 4, 4), (0, 8, 0)),
            ((4, 12, 4), (-8, 0, 0)),
            ((4, 12, 4), (8, 0, 0)),
        )
    ]
    core = magpy.Collection(*bars)
    core.susceptibility = 1000
    coil = magpy.current.Circle(current=10, diameter=8, position=(0, -8, 0))
    coil.rotate_from_angax(90, "y")
    return magpy.Collection(core, coil)


def _band_round_a_cube():
    """A closed strip round a cube, 1 mm wide."""
    phi = np.linspace(0, 2 * np.pi, 25)[:-1]
    ring = np.column_stack([1.5 * np.cos(phi), 1.5 * np.sin(phi), np.zeros(24)])
    vertices = np.stack([ring, ring + np.array([0, 0, 1])], axis=1).reshape(-1, 3)
    strip = magpy.current.TriangleStrip(
        current=1, vertices=np.concatenate([vertices, vertices[:2]])
    )
    return magpy.Collection(_cube(), strip)


@pytest.mark.parametrize(
    "model",
    [
        # a coil hugging a ring from outside, 1e-4 apart, goes through no hole
        lambda: magpy.Collection(
            _ring(), magpy.current.Circle(current=1, diameter=6.0002)
        ),
        lambda: magpy.Collection(
            _ring(), magpy.current.Circle(current=1, diameter=2, position=(6, 0, 0))
        ),
        # cubes touching along an edge: the edge adds loops round no hole
        lambda: magpy.Collection(
            _cube(),
            _cube((1, 1, 0)),
            magpy.current.Circle(current=1, diameter=0.5, position=(3, 0, 0)),
        ),
        _band_round_a_cube,
    ],
    ids=[
        "coil-hugging-a-ring",
        "coil-beside-a-ring",
        "cubes-on-an-edge",
        "band-round-a-cube",
    ],
)
def test_solves_currents_that_link_nothing(model):
    observer = np.array([(0, 0, 5)])
    solution = fem_solve(model(), 1, observers=observer, edge_layers=0)
    assert np.all(np.isfinite(solution.getB(observer)))


@pytest.mark.parametrize("air_first", [True, False])
def test_magnet_faces_know_their_outward_normal(air_first):
    """A magnet face is named by the side the magnet is on, so that its normal
    can be turned outward of the magnet whichever region netgen puts first:
    the outward normal integrates x . n to three times the volume."""
    import ngsolve as ng  # noqa: PLC0415
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere  # noqa: PLC0415

    box = Box(Pnt(-1, -1, -1), Pnt(1, 1, 2)).mat("part0")
    box.faces.name = "surface"
    air = (Sphere(Pnt(0, 0, 0), 5) - box).mat("air")
    ngmesh = OCCGeometry(Glue([air, box] if air_first else [box, air])).GenerateMesh(
        maxh=1
    )
    assert _solve._name_magnet_faces(ngmesh) == 1
    mesh = ng.Mesh(ngmesh)
    x = ng.CF((ng.x, ng.y, ng.z))
    normal = _solve.outward(mesh, 1)
    flux = ng.Integrate(ng.InnerProduct(x, normal) * ng.ds("magnet_.*"), mesh)
    assert flux == pytest.approx(3 * 2 * 2 * 3, rel=1e-9)


def test_a_far_coil_acts_as_a_uniform_field():
    """A loop of 1 m round a model of 10 mm drives it as a uniform field does,
    to its own unevenness over the model, about 1e-5. With the currents the
    magnets take the total potential, with the field as H_ext the reduced one.
    Two soft cubes touch across the field's direction, a magnet sits beside
    them, and the field is read outside and inside them."""
    loop = magpy.current.Circle(current=B_EXT / magpy.mu_0, diameter=1.0)

    def model(*currents):
        cubes = [
            magpy.magnet.Cuboid(
                dimension=(0.005,) * 3, polarization=(0, 0, 0), position=(0, 0, z)
            )
            for z in (-0.0025, 0.0025)
        ]
        for cube in cubes:
            cube.susceptibility = 1000
        magnet = magpy.magnet.Cuboid(
            dimension=(0.003,) * 3, polarization=(0, 0, 0.5), position=(0, 0.006, 0)
        )
        magnet.susceptibility = 0.05
        return magpy.Collection(*cubes, magnet, *currents)

    # above the cubes, 0.1 mm inside the upper one, where the lift of the
    # applied potential lives, and in the magnet
    observers = np.array([(0, 0, 0.008), (0, 0, 0.0049), (0, 0.006, 0)])
    plain = model()
    plain.H_ext = tuple(magpy.getB(loop, np.zeros(3)))
    driven = fem_solve(model(loop), order=2, observers=observers)
    reference = fem_solve(plain, order=2, observers=observers)

    scale = np.abs(reference.getB(observers)).max()
    np.testing.assert_allclose(
        driven.getB(observers) - loop.getB(observers),
        reference.getB(observers),
        atol=2e-5 * scale,
    )
    np.testing.assert_allclose(
        driven.getH(observers) - loop.getH(observers),
        reference.getH(observers),
        atol=2e-5 * scale / magpy.mu_0,
    )
    np.testing.assert_allclose(
        driven.moment, reference.moment, atol=1e-4 * np.abs(reference.moment).max()
    )


@pytest.mark.parametrize(
    ("model", "kwargs", "error", "match"),
    [
        (_wired_magnet, {}, ValueError, "runs through a magnet"),
        (_crossing_wire, {}, ValueError, "runs through a magnet"),
        (_clipped_corner, {}, ValueError, "runs through a magnet"),
        (_wired_tetrahedron, {}, ValueError, "runs through a magnet"),
        (_open_wire, {}, ValueError, "Open current paths"),
        (_sheet, {}, TypeError, "TriangleSheet"),
        (_linked_ring, {}, ValueError, "links a magnet"),
        (_lead_beside_a_coil, {}, ValueError, "1 A"),
        (_closed_core, {}, ValueError, "links a magnet"),
        (_two_fields, {}, ValueError, "different applied fields"),
        (
            lambda: magpy.Collection(magpy.misc.Dipole(moment=(0, 0, 1))),
            {},
            TypeError,
            "Only magnets, currents",
        ),
        (
            lambda: magpy.Collection(
                magpy.misc.Triangle(
                    vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], polarization=(0, 0, 1)
                )
            ),
            {},
            TypeError,
            "with a volume",
        ),
        (_overlapping, {}, ValueError, "overlap"),
        (_magnet, {"edge_layers": -1}, ValueError, "edge_layers"),
        (_magnet, {"maxh": float("nan")}, ValueError, "maxh must be positive"),
        (
            _magnet,
            {"observer_maxh": 0.1, "observers": None},
            ValueError,
            "needs observers",
        ),
    ],
    ids=[
        "current-through-a-magnet",
        "current-crossing-a-magnet",
        "current-clipping-a-corner",
        "current-through-a-tetrahedron",
        "open-current",
        "sheet-current",
        "current-through-a-ring",
        "lead-beside-a-strong-coil",
        "coil-round-a-closed-core",
        "two-applied-fields",
        "dipole",
        "triangle",
        "overlap",
        "negative-edge-layers",
        "nan-maxh",
        "observer-maxh-without-observers",
    ],
)
def test_refuses_what_it_cannot_model(model, kwargs, error, match):
    kwargs = {"observers": [(0, 0, 5)], **kwargs}
    with pytest.raises(error, match=match):
        fem_solve(model(), 2, **kwargs)
