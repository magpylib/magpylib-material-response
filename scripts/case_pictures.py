"""Draw each validation case as a picture, for the PR timing comment.

    python -m scripts.case_pictures           # the cases without a picture
    python -m scripts.case_pictures plate     # these, even if they have one

Needs the docs group (magpylib-studio) and Chrome (set CHROME to name another).
Each case is drawn as it is given, unsolved, at 800 x 600 pixels. Run it from the
repository's root after adding or changing a case, and commit the picture.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import magpylib as magpy
from PIL import Image

from validation.cases import CASES
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


def draw(case):
    """Write the picture of `case`, in a page opened by headless Chrome."""
    collection, _ = case.model()
    view = magpy.show(collection, backend="widget", return_fig=True)
    view.theme, view.axes = "light", False
    page = view.to_html(case.name).replace("</head>", DRESS, 1)
    out = PICTURES / f"{case.name}.png"
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
    # a case sharing another's model shares its picture; a refused or waiting
    # case is never timed
    cases = [
        case
        for case in CASES.values()
        if not (case.reference or case.refused or case.waits)
    ]
    for case in cases:
        if case.name in names or not (
            names or (PICTURES / f"{case.name}.png").exists()
        ):
            draw(case)
            print(case.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
