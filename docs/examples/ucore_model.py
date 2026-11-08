"""The U-core electromagnet of `ucore_electromagnet.md` and `saturation.md`: an
iron U and a coil of 16 windings around its yoke. The same model is the
validation case `ucore`."""

import magpylib as magpy
import numpy as np

from magpylib_material_response.polyline import (
    create_polyline_fillet,
    move_grid_along_polyline,
)

MM = 1e-3  # lengths in metres, written in millimetres


def coil(current=25.0):
    """16 windings around the middle of the yoke, `current` A each: 8 along
    it, in 2 layers, each a closed polyline with rounded corners."""
    h = 14 * MM  # half-width of a turn: 4 mm clear of the 20 mm core
    # one turn, a square that starts and ends mid-edge, away from the corners
    square = np.array(
        [[0, h, 0], [0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h], [0, h, 0]]
    )
    turn = create_polyline_fillet(square, max_radius=6 * MM, N=6)
    # the winding pack, across the turn at its start: along the coil's
    # axis, and outward from the turn
    pitch = 2.2 * MM
    pack = np.array(
        [(x, y) for x in (np.arange(8) - 3.5) * pitch for y in np.arange(2) * pitch]
    )
    windings = move_grid_along_polyline(turn, pack)  # one turn per position
    return magpy.Collection(
        *(magpy.current.Polyline(current=current, vertices=v) for v in windings),
        style_label=f"coil, 16 windings, {current:g} A",
    )


def core(susceptibility=999):
    """The U as three iron cuboids, 20 mm square: a 100 mm yoke and two
    60 mm legs."""
    a = 20 * MM
    parts = magpy.Collection(
        magpy.magnet.Cuboid(dimension=(100 * MM, a, a), style_label="yoke"),
        magpy.magnet.Cuboid(
            dimension=(a, a, 60 * MM),
            position=(-40 * MM, 0, 40 * MM),
            style_label="left leg",
        ),
        magpy.magnet.Cuboid(
            dimension=(a, a, 60 * MM),
            position=(40 * MM, 0, 40 * MM),
            style_label="right leg",
        ),
        style_label="iron core",
    )
    parts.susceptibility = susceptibility  # the parts inherit it
    return parts


def core_mesh(susceptibility=999):
    """The same U as one closed triangle mesh: its outline in the x-z plane,
    extruded 20 mm along y."""
    outline = MM * np.array(
        [
            (-50, -10),
            (50, -10),
            (50, 70),
            (30, 70),
            (30, 10),
            (-30, 10),
            (-30, 70),
            (-50, 70),
        ]
    )
    cap = [(0, 1, 4), (1, 3, 4), (1, 2, 3), (0, 4, 5), (0, 5, 7), (5, 6, 7)]
    n = len(outline)
    vertices = [(x, y, z) for y in (-10 * MM, 10 * MM) for x, z in outline]
    faces = cap + [tuple(i + n for i in tri) for tri in cap]
    for i in range(n):  # the side walls
        j = (i + 1) % n
        faces += [(i, j, j + n), (i, j + n, i + n)]
    u = magpy.magnet.TriangularMesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        reorient_faces=True,
        style_label="iron core",
    )
    u.susceptibility = susceptibility
    return u


def line_above_poles():
    """161 points 1 mm apart, 30 mm above the pole faces, along x at y = 0."""
    x = np.linspace(-80, 80, 161) * MM
    return np.stack([x, np.zeros_like(x), np.full_like(x, 100 * MM)], axis=-1)
