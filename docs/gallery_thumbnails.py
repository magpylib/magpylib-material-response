"""The examples gallery's pictures, drawn by magpylib-studio.

    uv run --group docs --with playwright python docs/gallery_thumbnails.py

writes docs/_static/gallery/NAME-light.jpg and NAME-dark.jpg for every example:
its model as the page builds it, soft parts solved by face charge so that they
show their poles, in the studio's view dressed as a picture (no panels), taken
at 400 x 300 CSS pixels, twice that in pixels, as magpylib-studio's own gallery
is. Run it after changing an example's model, and commit the pictures. Needs
`playwright install chromium` once; WebGL runs on SwiftShader, no GPU needed.
"""

from __future__ import annotations

import io
import pathlib
import sys
import tempfile
import time

import magpylib as magpy
import numpy as np
from magpylib_studio.widget import SceneWidget
from PIL import Image
from playwright.sync_api import sync_playwright

from magpylib_material_response import UniformField, solve_face_charge

HERE = pathlib.Path(__file__).parent
OUT = HERE / "_static" / "gallery"
WIDTH, HEIGHT = 400, 300
MM = 1e-3

#: What a picture leaves out, as in magpylib-studio's gallery: the view's
#: overlays, the page's margin and the view's frame.
DRESS = """<style>
  body { padding: 0 !important; margin: 0 !important; }
  .magpy-scene { display: flex !important; flex-direction: column;
    height: 100vh !important; border: 0 !important; border-radius: 0 !important; }
  .magpy-scene-stage { flex: 1; min-height: 0; }
  .magpy-scene-view { height: 100% !important; }
  .magpy-scene-legend, .magpy-colorbar, .magpy-scene-tools, .magpy-scene-tree,
  .magpy-scene-transport, .magpy-scene-counter, .magpy-scene-readout,
  .magpy-scene-notice { display: none !important; }
</style>
</head>"""


def soft(obj, chi=1000):
    obj.susceptibility = chi
    return obj


def soft_magnets():
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1 * MM, 1 * MM, 2 * MM)
    )
    magnet.move((0, 0, 0.5 * MM))
    cube = soft(
        magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1 * MM,) * 3), 3999
    )
    cube.rotate_from_angax(45, "y").move((1.5 * MM, 0, 0))
    return list(solve_face_charge(magpy.Collection(magnet, cube)).children)


def ucore_electromagnet():
    a = 0.02
    parts = [
        magpy.magnet.Cuboid(dimension=(0.10, a, a), polarization=(0, 0, 0)),
        magpy.magnet.Cuboid(
            dimension=(a, a, 0.06), position=(-0.04, 0, 0.04), polarization=(0, 0, 0)
        ),
        magpy.magnet.Cuboid(
            dimension=(a, a, 0.06), position=(0.04, 0, 0.04), polarization=(0, 0, 0)
        ),
    ]
    core = magpy.Collection(*parts)
    core.susceptibility = 999
    h = 0.014
    turns = []
    for x in np.linspace(-0.025, 0.025, 9):
        turns += [(x, -h, -h), (x, h, -h), (x, h, h), (x, -h, h), (x, -h, -h)]
    coil = magpy.current.Polyline(current=1, vertices=turns)
    return list(solve_face_charge(magpy.Collection(core, coil), level=2).children)


def mesh_convergence():
    turns = [(1.3, 0, 0), (0, 0, 1.3), (-1.3, 0, 0), (0, 0, -1.3), (1.3, 0, 0)]
    magnets = [
        magpy.magnet.Cuboid(
            dimension=(6 * MM,) * 3, position=(x * MM, 0, 0), polarization=p
        )
        for x, p in zip((-12, -6, 0, 6, 12), turns, strict=True)
    ]
    plate = soft(
        magpy.magnet.Cuboid(
            dimension=(30 * MM, 6 * MM, 3 * MM),
            position=(0, 0, -4.5 * MM),
            polarization=(0, 0, 0),
        )
    )
    return list(solve_face_charge(magpy.Collection(*magnets, plate), level=2).children)


def fem_check():
    line = np.linspace((-4, 0, -1), (6, 0, -1), 41) * MM
    sensors = magpy.Sensor(
        pixel=line,
        style_pixel_size=0.6,
        style_arrows_x_show=False,
        style_arrows_y_show=False,
        style_arrows_z_show=False,
    )
    return [*soft_magnets(), sensors]


def holding_magnet():
    magnet = magpy.magnet.Cylinder(
        dimension=(10 * MM, 5 * MM), position=(0, 0, 2.5 * MM), polarization=(0, 0, 1.3)
    )
    plate = soft(
        magpy.magnet.Cuboid(
            dimension=(40 * MM, 40 * MM, 6 * MM),
            position=(0, 0, -3 * MM),
            polarization=(0, 0, 0),
        )
    )
    return list(solve_face_charge(magpy.Collection(magnet, plate)).children)


def gear_tooth_sensor():
    base = magpy.magnet.Cuboid(
        dimension=(28 * MM, 6 * MM, 2 * MM),
        position=(0, 0, -1 * MM),
        polarization=(0, 0, 0),
    )
    teeth = [
        magpy.magnet.Cuboid(
            dimension=(2 * MM, 6 * MM, 3 * MM),
            position=(x * MM, 0, 1.5 * MM),
            polarization=(0, 0, 0),
        )
        for x in (-12, -8, -4, 0, 4, 8, 12)
    ]
    rack = soft(magpy.Collection(base, *teeth))
    magnet = magpy.magnet.Cylinder(
        dimension=(5 * MM, 4 * MM), position=(0, 0, 6.5 * MM), polarization=(0, 0, 1.2)
    )
    sensor = magpy.Sensor(position=(0, 0, 4 * MM))
    solved = solve_face_charge(magpy.Collection(rack, magnet), level=2)
    return [*solved.children, sensor]


def shield_tube():
    magnet = magpy.magnet.Cylinder(
        polarization=(0, 0, 1.22), dimension=(25 * MM, 10 * MM)
    )
    tube = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0),
        dimension=(22.5 * MM, 25 * MM, 130 * MM, 0, 360),
        position=(0, 0, 50 * MM),
        style_opacity=0.35,
    )
    return [magnet, tube]


def hall_current_sensor():
    sys.path.insert(0, str(HERE / "examples"))
    from hall_core import core_mesh  # noqa: PLC0415

    vertices, faces = core_mesh()
    core = soft(
        magpy.magnet.TriangularMesh(
            vertices=vertices * MM, faces=faces, polarization=(0, 0, 0)
        ),
        2000,
    )
    busbar = magpy.current.Polyline(current=100, vertices=[(0, 0, -0.02), (0, 0, 0.02)])
    closed = magpy.current.Polyline(
        current=100,
        vertices=[
            (0, 0, -0.5),
            (0, 0, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0, -0.5),
            (0, 0, -0.5),
        ],
    )
    solved = solve_face_charge(magpy.Collection(core, closed), level=2)
    return [solved.children[0], busbar, magpy.Sensor(position=(10.5 * MM, 0, 0))]


def angle_sensor():
    sys.path.insert(0, str(HERE / "examples"))
    from angle_sensor_study import angle_sensor as model  # noqa: PLC0415
    from angle_sensor_study import hall_plates  # noqa: PLC0415

    solved = solve_face_charge(model(30, 1.5, 5e-3, holder=True), level=2)
    shown = [c for c in solved.sources_all if not isinstance(c, UniformField)]
    return [*shown, *(magpy.Sensor(position=p) for p in hall_plates(1.5))]


def tolerance_monte_carlo():
    sys.path.insert(0, str(HERE / "examples"))
    from tolerance_study import assembly  # noqa: PLC0415

    model, _, sensor = assembly()
    return [*solve_face_charge(model, level=2).children, sensor]


def saturation():
    sys.path.insert(0, str(HERE / "examples"))
    from ucore_model import coil, core  # noqa: PLC0415

    model = magpy.Collection(core(), coil(400))
    return list(solve_face_charge(model, level=2).children)


EXAMPLES = [
    holding_magnet,
    gear_tooth_sensor,
    ucore_electromagnet,
    mesh_convergence,
    fem_check,
    shield_tube,
    hall_current_sensor,
    angle_sensor,
    tolerance_monte_carlo,
    saturation,
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2
        )
        for example in EXAMPLES:
            objects = example()
            for theme in ("light", "dark"):
                view = SceneWidget(*objects, theme=theme, axes=False, height=HEIGHT)
                html = pathlib.Path(tmp) / f"{example.__name__}-{theme}.html"
                html.write_text(
                    view.to_html(example.__name__).replace("</head>", DRESS, 1),
                    encoding="utf-8",
                )
                page.goto(html.as_uri(), wait_until="networkidle")
                time.sleep(2.5)
                shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                shot.save(OUT / f"{example.__name__}-{theme}.jpg", quality=85)
            print(example.__name__)  # noqa: T201
        browser.close()


if __name__ == "__main__":
    main()
