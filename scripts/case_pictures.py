"""Draw each case of the PR check as its method solves it, for the timing comment.

    python -m scripts.case_pictures           # those without a picture
    python -m scripts.case_pictures plate     # this case's, even if it has them

Needs the docs group (magpylib-studio) and Chrome (set CHROME to name another).
Each is drawn at its PR setting and at 800 x 600 pixels: the method of moments'
cells with their polarization, face charge's panels with their charge. Run it
from the repository's root after changing a case or a PR setting, and commit the
pictures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import magpylib as magpy
from PIL import Image

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.face_charge import apply_face_charge
from magpylib_material_response.meshing import mesh_uniform
from validation.cases import CASES
from validation.methods import METHODS
from validation.run import PR_CHECK
from validation.timing import PICTURES

CHROME = os.environ.get(
    "CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
#: What a picture leaves out of the view: as in magpylib-studio's own gallery
DRESS = """<style>
  body { padding: 0 !important; }
  .magpy-scene {
    display: flex !important;
    flex-direction: column;
    height: 100vh !important;
    border: 0 !important;
    border-radius: 0 !important;
  }
  .magpy-scene-stage { flex: 1; min-height: 0; }
  .magpy-scene-view { height: 100% !important; }
  .magpy-scene-legend, .magpy-colorbar, .magpy-scene-tools,
  .magpy-scene-transport, .magpy-scene-counter, .magpy-scene-readout,
  .magpy-scene-notice { display: none !important; }
</style>
</head>"""


def solved(case, method, setting):
    """The model of `case` as `method` solves it at `setting`."""
    collection, _ = case.model()
    options = case.options.get(method, {})
    if method == "MoM":
        cells = mesh_uniform(collection, setting, max_cells=METHODS[method].max_cells)
        return apply_demag(cells, **options)
    return apply_face_charge(collection, level=setting, **options)


def draw(case, method, setting):
    """Write the picture of `case` under `method`, in a page opened by headless
    Chrome."""
    view = magpy.show(solved(case, method, setting), backend="widget", return_fig=True)
    view.theme, view.axes = "light", False
    page = view.to_html(case.name).replace("</head>", DRESS, 1)
    out = PICTURES / f"{case.name}-{method}.png"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "page.html"
        path.write_text(page, encoding="utf-8")
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                f"--user-data-dir={folder}/profile",
                "--blink-settings=preferredColorScheme=1",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                # headless Chrome cuts off a window narrower than about 500
                "--window-size=800,600",
                "--virtual-time-budget=10000",
                f"--screenshot={out}",
                path.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    picture = Image.open(out).convert("RGB").resize((800, 600), Image.LANCZOS)
    picture.quantize(128).save(out, optimize=True)


def main(names):
    PICTURES.mkdir(exist_ok=True)
    for name, by_method in json.loads(PR_CHECK.read_text()).items():
        for method, entry in by_method.items():
            missing = not (PICTURES / f"{name}-{method}.png").exists()
            if name in names or (missing and not names):
                draw(CASES[name], method, entry["setting"])
                print(name, method, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
