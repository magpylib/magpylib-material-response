"""The models and mesh studies of `mesh_convergence.md`. The studies take about
40 s here and two minutes on the 2 CPUs the docs are built on, so they run once
and are stored in `mesh_convergence.json`; the page rebuilds each with
`MeshStudy.from_values`, which gives the same estimate.

    uv run --group docs python docs/examples/mesh_convergence_studies.py

Run it after a change to `solve_cells` or to a model, and commit the JSON.
"""

import json
from pathlib import Path

import magpylib as magpy
import numpy as np

from magpylib_material_response import UniformField, study_cells

MM = 1e-3  # lengths in metres, written in millimetres
B0 = 50e-6  # T, the Earth's field, along the bar of the flux concentrator


def flux_concentrator():
    """A soft nickel-iron bar in the Earth's field, and its gain at three
    sensors beside its tip: the field there over the 50 µT around it."""
    bar = magpy.magnet.Cuboid(
        dimension=(16 * MM, 2 * MM, 2 * MM), polarization=(0, 0, 0)
    )
    bar.susceptibility = 4000
    model = magpy.Collection(bar, UniformField(B=(B0, 0, 0)))
    sensors = (
        np.array([(8.5, 0, 0), (9, 0, 0), (10, 0, 0)]) * MM
    )  # 0.5, 1, 2 mm from the tip

    def gain(solved):
        return magpy.getB(solved, sensors)[:, 0] / B0

    return model, gain


def halbach_array():
    """Five NdFeB cubes turning by 90° from one to the next, on a steel plate,
    and a line 2 mm above them."""
    turns = [(1.3, 0, 0), (0, 0, 1.3), (-1.3, 0, 0), (0, 0, -1.3), (1.3, 0, 0)]
    magnets = [
        magpy.magnet.Cuboid(
            dimension=(6 * MM,) * 3, position=(x * MM, 0, 0), polarization=p
        )
        for x, p in zip((-12, -6, 0, 6, 12), turns, strict=True)
    ]
    for magnet in magnets:
        magnet.susceptibility = 0.05  # NdFeB answers a little
    plate = magpy.magnet.Cuboid(
        dimension=(30 * MM, 6 * MM, 3 * MM),
        position=(0, 0, -4.5 * MM),
        polarization=(0, 0, 0),
    )
    plate.susceptibility = 1000
    x = np.linspace(-15, 15, 21)
    profile = np.column_stack([x, 0 * x, np.full(21, 5.0)]) * MM
    return magpy.Collection(*magnets, plate), profile


def steel_shield():
    """A 1 mm steel sheet 3 mm above a 6 mm magnet, and three sensors 2 mm
    behind it."""
    magnet = magpy.magnet.Cuboid(dimension=(6 * MM,) * 3, polarization=(0, 0, 1.3))
    magnet.susceptibility = 0  # only the sheet answers
    sheet = magpy.magnet.Cuboid(
        dimension=(24 * MM, 24 * MM, 1 * MM),
        position=(0, 0, 6.5 * MM),
        polarization=(0, 0, 0),
    )
    sheet.susceptibility = 1000
    behind = np.array([(0, 0, 9), (6, 0, 9), (12, 0, 9)]) * MM
    return magpy.Collection(magnet, sheet), behind


#: the studies: a model, what is compared, and the levels, cells across the
#: thinnest part that responds
STUDIES = {
    "concentrator": (flux_concentrator, (4, 8, 16)),
    "halbach": (halbach_array, (2, 4, 8)),
    "shield": (steel_shield, (1, 2, 4)),
}


def main():
    out = {}
    for name, (make, levels) in STUDIES.items():
        study = study_cells(*make(), levels)
        out[name] = {
            "cells": [int(n) for n in study.cells],
            "values": [np.asarray(v).tolist() for v in study.values],
            "seconds": [float(t) for t in study.seconds],
        }
    Path(__file__).with_name("mesh_convergence.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
