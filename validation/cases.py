"""The validation cases: each a model as magpylib objects and the sensors at
which B is compared, in named groups.

A case is data only. Its stored reference (``references/``) carries the case's
hash, so a reference left behind by a changed model is caught instead of
compared against. Lengths in m, polarizations and applied fields (``H_ext``,
mu0 H) in T.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import magpylib as magpy
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from magpylib_material_response.polyline import (
    create_polyline_fillet,
    move_grid_along_polyline,
)
from magpylib_material_response.utils import _serialize_recursive

if TYPE_CHECKING:
    from collections.abc import Callable

MM = 1e-3


@dataclass(frozen=True)
class Case:
    """One model and its sensors.

    Attributes
    ----------
    name : str
        Short identifier, also the reference's file name.
    title : str
        One line for tables.
    kind : str
        ``"first"``, ``"typical"`` or ``"edge"``: the group it is listed in.
    build : callable
        ``build()`` returns ``(collection, groups)``: the model, and a dict of
        sensor positions, each of shape (n, 3), by group name. Errors are
        reported per group, so a region where B is weak (under a plate that
        shields) is not hidden by one where it is strong.
    options : dict
        Method name to the options its solve takes here, such as
        ``{"MoM": {"interaction": "newell"}}``.
    reference : str, optional
        The case whose reference this one shares: the same model solved with
        other options.
    known_failures : dict
        Method name to the issue that explains why it fails here.
    waits : str, optional
        What no method can solve yet; the case is defined and drawn, not run.
    refused : str, optional
        Every method must refuse the case, with an error that says this.
    reference_orders : tuple of int
        The finite-element orders of the reference, lower where the highest
        default order would not fit in memory.
    """

    name: str
    title: str
    kind: str
    build: Callable[[], tuple[magpy.Collection, dict[str, np.ndarray]]]
    options: dict[str, dict] = field(default_factory=dict)
    reference: str | None = None
    known_failures: dict[str, str] = field(default_factory=dict)
    waits: str | None = None
    refused: str | None = None
    reference_orders: tuple[int, ...] = (3, 4, 5)

    def model(self):
        """The model and its sensors, as ``build`` returns them."""
        return self.build()

    def hash(self):
        """Hash of everything that defines the model: the collection as the
        package serializes it, without styles, and the sensors. Numbers are
        rounded to 12 decimals (1 pm in m), so the hash does not depend on the
        platform's last bit or on the sign of a zero."""
        collection, groups = self.build()
        data = {
            "model": _serialize_recursive(collection, parent="ignore"),
            "sensors": {name: points.tolist() for name, points in groups.items()},
        }
        text = json.dumps(_canonical(data), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def _canonical(value):
    """`value` with styles left out and every number rounded."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k != "style"}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return round(value, 12) + 0.0
    return value


def _soft(dimension, label, chi, **kwargs):
    """A cuboid that only responds: no polarization of its own. A `chi` of
    None leaves the susceptibility to a parent collection."""
    part = magpy.magnet.Cuboid(
        polarization=(0, 0, 0), dimension=dimension, style_label=label, **kwargs
    )
    if chi is not None:
        part.susceptibility = chi
    return part


def _line(start, stop, n):
    """`n` points from `start` to `stop`, both given in mm."""
    return np.linspace(np.multiply(start, MM), np.multiply(stop, MM), n)


# ── first cases ───────────────────────────────────────────────────────────────


def _cuboids():
    """Three cuboids, each magnetized and soft, turned against each other: the
    model of docs/examples/cuboids_demagnetization.md, its 301 sensors below."""
    cube1 = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1 * MM,) * 3, style_label="1 T, χ = 0.3"
    )
    cube1.move((-1.5 * MM, 0, 0))
    cube1.susceptibility = 0.3
    cube2 = magpy.magnet.Cuboid(
        polarization=(0.9, 0, 0), dimension=(1 * MM,) * 3, style_label="0.9 T, χ = 1"
    )
    cube2.rotate_from_angax(-45, "y").move((0, 0, 0.2 * MM))
    cube2.susceptibility = 1.0
    mx, my = 0.6 * np.sin(np.deg2rad(30)), 0.6 * np.cos(np.deg2rad(30))
    cube3 = magpy.magnet.Cuboid(
        polarization=(mx, my, 0),
        dimension=(1 * MM, 1 * MM, 2 * MM),
        style_label="0.6 T, χ = 0.5",
    )
    cube3.move((1.6 * MM, 0, 0.5 * MM)).rotate_from_angax(30, "z")
    cube3.susceptibility = 0.5
    groups = {"line below": _line((-4, 0, -1), (4, 0, -1), 301)}
    return magpy.Collection(cube1, cube2, cube3), groups


def _plate(magnet_height=4 * MM):
    """A magnet glued on an iron plate (20 x 20 x 5 mm, chi = 1000), off
    centre. Sensors 0.5 to 5 mm off four places: above the magnet, on the plate
    beside it, under the plate, where the plate shields, and off its side."""
    plate = _soft((20 * MM, 20 * MM, 5 * MM), "iron plate, χ = 1000", 1000)
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.2),
        dimension=(6 * MM, 6 * MM, magnet_height),
        style_label="magnet, 1.2 T, χ = 0.05",
    )
    magnet.move((2 * MM, 1 * MM, 2.5 * MM + magnet_height / 2))
    magnet.susceptibility = 0.05
    top = 2.5 * MM + magnet_height
    d = np.array([0.5, 1, 2, 5])[:, None] * MM
    groups = {
        "above the magnet": [2.5 * MM, 1.5 * MM, top] + d * [0, 0, 1],
        "beside the magnet": [7.5 * MM, 1 * MM, 2.5 * MM] + d * [0, 0, 1],
        "under the plate": [0, 0, -2.5 * MM] - d * [0, 0, 1],
        "off the side": [10 * MM, 3 * MM, 0] + d * [1, 0, 0],
    }
    return magpy.Collection(plate, magnet), groups


def _ucore():
    """The U-core of docs/examples/ucore_electromagnet.md: three iron cuboids
    (chi = 999) and a coil of 16 turns at 25 A around the middle bar. Sensors on
    the page's line, 30 mm above the poles."""
    a = 20 * MM
    core = magpy.Collection(
        _soft((100 * MM, a, a), "yoke", None),
        _soft((a, a, 60 * MM), "left leg", None, position=(-40 * MM, 0, 40 * MM)),
        _soft((a, a, 60 * MM), "right leg", None, position=(40 * MM, 0, 40 * MM)),
        style_label="iron core, χ = 999",
    )
    core.susceptibility = 999
    h = 14 * MM
    square = np.array(
        [[0, h, 0], [0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h], [0, h, 0]]
    )
    turn = create_polyline_fillet(square, max_radius=6 * MM, N=6)
    pitch = 2.2 * MM
    pack = np.array(
        [
            (x, y, 0)
            for x in (np.arange(8) - 3.5) * pitch
            for y in h + np.arange(2) * pitch
        ]
    )
    coil = magpy.Collection(
        *(
            magpy.current.Polyline(current=25, vertices=v)
            for v in move_grid_along_polyline(turn, pack)
        ),
        style_label="coil, 16 turns, 25 A",
    )
    groups = {"30 mm above the poles": _line((-80, 0, 100), (80, 0, 100), 161)}
    return magpy.Collection(core, coil), groups


# ── typical models ────────────────────────────────────────────────────────────


def _soft_magnets():
    """A magnet and a turned soft cube (chi = 3999): the model of
    docs/examples/soft_magnets.md, on its three sensor lines."""
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(1 * MM, 1 * MM, 2 * MM),
        style_label="magnet, 1 T, χ = 0.5",
    )
    magnet.move((0, 0, 0.5 * MM))
    magnet.susceptibility = 0.5
    cube = _soft((1 * MM,) * 3, "soft cube, χ = 3999", 3999)
    cube.rotate_from_angax(45, "y").move((1.5 * MM, 0, 0))
    groups = {f"{z} mm below": _line((-4, 0, -z), (6, 0, -z), 201) for z in (1, 3, 5)}
    return magpy.Collection(magnet, cube), groups


def _concentrator():
    """A bar (16 x 2 x 2 mm, chi = 4000) in the Earth's field along its axis:
    the flux concentrator of docs/examples/mesh_convergence.md. Sensors on the
    axis past its end, where the page reads the gain, and along its top."""
    bar = _soft((16 * MM, 2 * MM, 2 * MM), "bar, χ = 4000", 4000)
    bar.H_ext = (50e-6, 0, 0)
    groups = {
        "on the axis, past the end": _line((8.5, 0, 0), (10, 0, 0), 4),
        "1 mm above the bar": _line((-8, 0, 2), (8, 0, 2), 9),
    }
    return magpy.Collection(bar), groups


def _halbach():
    """Five 6 mm magnets with turning polarizations (chi = 0.05) on an iron
    back plate (chi = 1000): the Halbach array of
    docs/examples/mesh_convergence.md, its profile 2 mm above."""
    turns = [(1.3, 0, 0), (0, 0, 1.3), (-1.3, 0, 0), (0, 0, -1.3), (1.3, 0, 0)]
    magnets = []
    for k, (x, pol) in enumerate(zip((-12, -6, 0, 6, 12), turns, strict=True)):
        magnet = magpy.magnet.Cuboid(
            polarization=pol,
            dimension=(6 * MM,) * 3,
            position=(x * MM, 0, 0),
            style_label=f"magnet {k + 1}, 1.3 T, χ = 0.05",
        )
        magnet.susceptibility = 0.05
        magnets.append(magnet)
    plate = _soft(
        (30 * MM, 6 * MM, 3 * MM),
        "back iron, χ = 1000",
        1000,
        position=(0, 0, -4.5 * MM),
    )
    groups = {"2 mm above the array": _line((-15, 0, 5), (15, 0, 5), 21)}
    return magpy.Collection(*magnets, plate), groups


def _sheet():
    """A thin iron sheet (24 x 24 x 1 mm, chi = 1000) 3 mm over a magnet: the
    shielding sheet of docs/examples/mesh_convergence.md, sensors behind it."""
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.3), dimension=(6 * MM,) * 3, style_label="magnet, 1.3 T"
    )
    magnet.susceptibility = 0
    sheet = _soft(
        (24 * MM, 24 * MM, 1 * MM),
        "sheet, χ = 1000",
        1000,
        position=(0, 0, 6.5 * MM),
    )
    groups = {"2 mm behind the sheet": _line((0, 0, 9), (12, 0, 9), 3)}
    return magpy.Collection(magnet, sheet), groups


def _cylinder():
    """A soft cylinder (10 mm across and tall, chi = 1000) in 10 mT across its
    axis, meshed into cylinder segments by MoM: the face-charge study's step
    3a. Sensors 1 mm off the surface and 50 mm away."""
    cylinder = magpy.magnet.Cylinder(
        polarization=(0, 0, 0),
        dimension=(10 * MM, 10 * MM),
        style_label="cylinder, χ = 1000",
    )
    cylinder.susceptibility = 1000
    cylinder.H_ext = (0.01, 0, 0)
    groups = {
        "1 mm off the surface": np.array([(4, 0, 6), (6, 0, 3)]) * MM,
        "50 mm away": np.array([(0, 0, 50), (50, 0, 0)]) * MM,
    }
    return magpy.Collection(cylinder), groups


def _thin_ring():
    """A tube with a 1 mm wall (radius 4 to 5 mm, 30 mm long, chi = 1000)
    beside a 1 T cube magnet: issue #92. Sensors on the axis inside, where the
    tube shields, and on the far side, along x."""
    tube = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0),
        dimension=(4 * MM, 5 * MM, 30 * MM, 0, 360),
        style_label="tube, χ = 1000",
    )
    tube.susceptibility = 1000
    magnet = magpy.magnet.Cuboid(
        polarization=(1, 0, 0),
        dimension=(10 * MM,) * 3,
        position=(15 * MM, 0, 0),
        style_label="magnet, 1 T",
    )
    magnet.susceptibility = 0
    groups = {
        "inside, on the axis": _line((0, 0, -10), (0, 0, 10), 9),
        "far side": _line((-5.5, 0, 0), (-15, 0, 0), 9),
    }
    return magpy.Collection(tube, magnet), groups


def _cad_plate(n_fillet=6, n_hole=24):
    """Vertices and faces of a 20 x 12 x 4 mm plate with 3 mm fillets on its
    vertical edges and a 3 mm hole 3 mm off centre, exported as a CAD tool does:
    one facet per 15 degrees, each cap triangulated from its boundary vertices
    only, in long thin triangles."""
    a, b, t, r = 10 * MM, 6 * MM, 2 * MM, 3 * MM
    corners = ((a - r, b - r), (-a + r, b - r), (-a + r, -b + r), (a - r, -b + r))
    outline = []
    for k, (cx, cy) in enumerate(corners):
        phi = np.pi / 2 * (k + np.arange(n_fillet + 1) / n_fillet)
        outline += list(np.column_stack([cx + r * np.cos(phi), cy + r * np.sin(phi)]))
    outline = np.array(outline)
    phi = 2 * np.pi * np.arange(n_hole) / n_hole
    hole = np.column_stack([3 * MM + r * np.cos(phi), r * np.sin(phi)])
    points = np.concatenate([outline, hole])
    cap = Delaunay(points).simplices
    centre = points[cap].mean(axis=1)
    inside = np.hypot(centre[:, 0] - 3 * MM, centre[:, 1]) < r * np.cos(np.pi / n_hole)
    cap = cap[~inside]
    e1, e2 = (points[cap[:, k]] - points[cap[:, 0]] for k in (1, 2))
    clockwise = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
    cap[clockwise] = cap[clockwise][:, ::-1]
    n = len(points)
    vertices = np.concatenate(
        [np.column_stack([points, np.full(n, z)]) for z in (t, -t)]
    )
    faces = [*cap, *(tri[::-1] + n for tri in cap)]
    for first, count, outward in (
        (0, len(outline), True),
        (len(outline), len(hole), False),
    ):
        for k in range(count):
            i, j = first + k, first + (k + 1) % count
            quad = [(i + n, j + n, j), (i + n, j, i)]
            faces += quad if outward else [tri[::-1] for tri in quad]
    return vertices, np.array([list(face) for face in faces])


def _cad_part():
    """The CAD plate (``_cad_plate``) in iron (chi = 1000), in 10 mT along a
    direction no symmetry helps with: the face-charge study's step 3b. Sensors
    near the part and 60 mm away."""
    vertices, faces = _cad_plate()
    part = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0),
        vertices=vertices,
        faces=faces,
        style_label="CAD part, χ = 1000",
    )
    part.susceptibility = 1000
    part.H_ext = tuple(0.01 * np.array([0.8, 0.36, 0.48]))
    near = [(3, 0, 0), (8, 0, 3), (7 + 4 / 2**0.5, 3 + 4 / 2**0.5, 1), (0, 0, 3)]
    groups = {
        "near the part": np.array(near) * MM,
        "60 mm away": np.array([(60, 0, 0), (0, 0, 60)]) * MM,
    }
    return magpy.Collection(part), groups


# ── edge cases ────────────────────────────────────────────────────────────────


def _sides_off():
    """The magnet on the plate, the magnet 4.4 mm tall: no cell size fits both
    the plate's 5 mm and the magnet's 4.4 mm, so MoM's uniform cells never share
    one grid."""
    return _plate(magnet_height=4.4 * MM)


def _layer(sides, label, columns):
    """Soft cuboids of the given sides in a layer of `columns` per row, 2 mm or
    1.5 mm apart, sharing chi = 100 and 10 mT along the layer."""
    pitch = 1.5 * MM if columns == 10 else 2 * MM
    parts = [
        _soft(
            tuple(side),
            f"{label} {i + 1}",
            None,
            position=(
                (i % columns - (columns - 1) / 2) * pitch,
                (i // columns - (columns - 1) / 2) * pitch,
                0,
            ),
        )
        for i, side in enumerate(sides)
    ]
    layer = magpy.Collection(*parts, style_label=f"{len(parts)} {label}s, χ = 100")
    layer.susceptibility = 100
    layer.H_ext = (0.01, 0, 0)
    return layer


def _small_bodies():
    """100 soft cubes of 1 mm (chi = 100) in a 10 x 10 layer, 0.5 mm apart, in
    10 mT along the layer: many small bodies, none a grid with another."""
    layer = _layer(np.full((100, 3), 1 * MM), "cube", 10)
    groups = {
        "1 mm above the layer": _line((-8, 0.75, 1.5), (8, 0.75, 1.5), 17),
        "30 mm away": np.array([(30, 0, 0), (0, 0, 30)]) * MM,
    }
    return magpy.Collection(layer), groups


def _many_sizes():
    """36 soft cuboids (chi = 100) of 36 sizes, 0.6 to 1.4 mm a side, in a
    6 x 6 layer, in 10 mT along the layer: cells of many sizes, so MoM finds no
    grid at all."""
    # 108 sides in a fixed shuffled order (41 and 108 are coprime)
    sides = np.linspace(0.6, 1.4, 108)[np.arange(108) * 41 % 108] * MM
    layer = _layer(sides.reshape(36, 3), "cuboid", 6)
    groups = {
        "1 mm above the layer": _line((-6, 1, 1.7), (6, 1, 1.7), 13),
        "30 mm away": np.array([(30, 0, 0), (0, 0, 30)]) * MM,
    }
    return magpy.Collection(layer), groups


def _anisotropic():
    """A 1 mm magnet (1.1 T) with chi = (0.3, 0.2, 0.1) along its own axes,
    turned, so the susceptibility must turn with it: the model of
    tests/test_isotropic_anisotropic.py. Sensors 0.1 and 0.2 mm above its top
    face, turned with it."""
    turn = Rotation.from_rotvec((0.3, -0.5, 0.8))
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.1),
        dimension=(1 * MM,) * 3,
        style_label="magnet, 1.1 T, χ = (0.3, 0.2, 0.1)",
    )
    magnet.susceptibility = (0.3, 0.2, 0.1)
    magnet.rotate(turn)
    groups = {}
    for height in (0.6, 0.7):
        ends = ((0.7, 0, height), (0, 0.7, height), (0.7, 0.7, height))
        lines = [_line((0, 0, height), end, 71) for end in ends]
        groups[f"{height - 0.5:.1f} mm above the top"] = turn.apply(
            np.concatenate(lines)
        )
    return magpy.Collection(magnet), groups


def _iron_cube(field):
    """A 10 mm iron cube (chi = 1000) in a uniform field along z, sensors 1 mm
    over its top face and 30 mm above."""
    cube = _soft((10 * MM,) * 3, "iron cube, χ = 1000", 1000)
    cube.H_ext = (0, 0, field)
    groups = {
        "1 mm above the top": _line((-4, 0, 6), (4, 0, 6), 5),
        "30 mm above": np.array([(0, 0, 30)]) * MM,
    }
    return magpy.Collection(cube), groups


def _magnet_on_path():
    """A magnet moving along a path over an iron plate: each position is a
    different model, which no method solves in one call."""
    plate = _soft((20 * MM, 20 * MM, 2 * MM), "iron plate, χ = 1000", 1000)
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * MM,) * 3,
        position=[(x * MM, 0, 6 * MM) for x in (-5, 0, 5)],
        style_label="magnet on a path, 1 T",
    )
    magnet.susceptibility = 0
    groups = {"2 mm under the plate": _line((-5, 0, -3), (5, 0, -3), 3)}
    return magpy.Collection(plate, magnet), groups


ISSUES = "https://github.com/magpylib/magpylib-material-response/issues/"

CASES = {
    case.name: case
    for case in (
        Case("cuboids", "Three cuboids turned (case A)", "first", _cuboids),
        Case("plate", "Magnet on a soft plate", "first", _plate),
        Case("ucore", "U-core driven by a coil (case F)", "first", _ucore),
        Case(
            "soft_magnets", "Magnet and a soft cube (case B)", "typical", _soft_magnets
        ),
        Case("concentrator", "Flux concentrator (case C)", "typical", _concentrator),
        Case("halbach", "Halbach array on back iron (case D)", "typical", _halbach),
        Case("sheet", "Thin shielding sheet (case E)", "typical", _sheet),
        Case("cylinder", "Soft cylinder", "typical", _cylinder),
        Case(
            "thin_ring",
            "Thin ring beside a magnet",
            "typical",
            _thin_ring,
            known_failures={"MoM": ISSUES + "92"},
        ),
        # order 5 on its mesh would need about 11 GB
        Case(
            "cad_part",
            "CAD part, a triangle mesh",
            "typical",
            _cad_part,
            reference_orders=(2, 3, 4),
        ),
        Case(
            "concentrator_newell",
            "Flux concentrator, Newell interaction",
            "typical",
            _concentrator,
            options={"MoM": {"interaction": "newell"}},
            reference="concentrator",
        ),
        Case("sides_off", "Sides off the cell size", "edge", _sides_off),
        # 1,200 edges: FEM's default four layers toward them pass 7 GB at order 2
        Case(
            "small_bodies",
            "100 small bodies",
            "edge",
            _small_bodies,
            options={"FEM": {"edge_layers": 0}},
        ),
        # 432 edges: four layers toward them take 6.8 GB at order 3, two fit
        # orders 2 to 4 in 6.3 GB
        Case(
            "many_sizes",
            "Cuboids of many sizes",
            "edge",
            _many_sizes,
            options={"FEM": {"edge_layers": 2}},
            reference_orders=(2, 3, 4),
        ),
        Case("anisotropic", "Anisotropic susceptibility, turned", "edge", _anisotropic),
        Case(
            "gmres_stops",
            "GMRES does not converge",
            "edge",
            lambda: _iron_cube(0.01),
            options={"MoM": {"solver_tol": 1e-14, "max_iter": 1}},
        ),
        Case(
            "saturation",
            "Iron cube driven into saturation",
            "edge",
            lambda: _iron_cube(0.5),
            waits="B-H curves (issue #81): a constant chi puts this cube at 3 T",
        ),
        Case("path", "A magnet on a path", "edge", _magnet_on_path, refused="path"),
    )
}
