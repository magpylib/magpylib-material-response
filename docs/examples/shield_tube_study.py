"""The face-charge solves of `shield_tube.md`, stored in `shield_tube.json`: the
finest takes 20 s here and more than a minute on the 2 CPUs the docs are built
on.

    uv run --group docs python docs/examples/shield_tube_study.py

Run it after a change to `solve_face_charge`, and commit the JSON.
"""

import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np

from magpylib_material_response import solve_face_charge

MM = 1e-3  # lengths in metres, written in millimetres
#: the field is read on three lines, as in the validation case `shield_tube`
LINES = {
    "between magnet and wall": np.linspace((-18.45, 0, 0), (-18.45, 0, 100), 21) * MM,
    "on the axis": np.linspace((0, 0, 10), (0, 0, 100), 10) * MM,
    "outside the tube": np.linspace((30, 0, 0), (60, 0, 0), 7) * MM,
}
SETTINGS = {"level=2": {"level": 2}, "default": {}, "refine=1": {"refine": 1}}


def shield_tube():
    """A disc magnet inside an open steel tube, near one end."""
    magnet = magpy.magnet.Cylinder(
        polarization=(0, 0, 1.22), dimension=(25 * MM, 10 * MM)
    )
    magnet.susceptibility = 0.05
    tube = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0),
        dimension=(22.5 * MM, 25 * MM, 130 * MM, 0, 360),
        position=(0, 0, 50 * MM),
    )
    tube.susceptibility = 100
    return magnet, tube


def main():
    out = {}
    for name, settings in SETTINGS.items():
        t0 = time.perf_counter()
        solved = solve_face_charge(magpy.Collection(*shield_tube()), **settings)
        seconds = time.perf_counter() - t0
        out[name] = {
            "unknowns": solved.face_charge_info["unknowns"],
            "seconds": round(seconds, 2),
            "B": {k: magpy.getB(solved, p).tolist() for k, p in LINES.items()},
        }
    Path(__file__).with_name("shield_tube.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
