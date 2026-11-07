"""The stored results of `saturation.md`: the U-core of `ucore_model.py` on a
B-H curve, its coil at 25 to 400 A, solved by cells and by finite elements.
The finite-element solves take about a minute each, so they are stored in
`saturation.json`.

    uv run --group docs python docs/examples/saturation_study.py

Run it after a change to a solver or to the model, and commit the JSON.
"""

import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from loguru import logger
from ucore_model import coil, core, core_mesh, line_above_poles

from magpylib_material_response import BHCurve, solve_cells, solve_face_charge
from magpylib_material_response.fem import solve_fem
from magpylib_material_response.meshing import mesh_TriangularMesh

CHI0, JS = 999.0, 1.6  # initial susceptibility, saturation polarization in T
CURRENTS = (25, 50, 100, 200, 400)  # A in each of the 16 windings
FEM_CURRENTS = (25, 100, 400)
FEM_ORDERS = (2, 3)
TETRAHEDRA = (100, 700)  # target_elems of the two cell meshes


def iron_curve():
    """A stand-in for a soft iron: J = chi0 mu0 H / (1 + chi0 mu0 H / Js),
    initial susceptibility 999, saturating at 1.6 T."""
    H = np.r_[0.0, np.geomspace(1e-2, 1e6, 400)]  # A/m
    J = CHI0 * magpy.mu_0 * H / (1 + CHI0 * magpy.mu_0 * H / JS)
    return BHCurve(H=H, B=J + magpy.mu_0 * H)


def peak(field):
    """The largest |Bz| on the line above the poles, in mT."""
    return float(np.abs(field[:, 2]).max() * 1e3)


def regions(cells):
    """Mean polarization magnitude in T of the yoke's middle, the yoke's
    ends and the legs, from tetrahedra."""
    J = np.linalg.norm([c.polarization for c in cells], axis=1)
    x, z = np.array([c.centroid for c in cells]).T[[0, 2]]
    yoke = z < 10e-3
    return {
        "yoke middle": float(J[yoke & (np.abs(x) < 10e-3)].mean()),
        "yoke ends": float(J[yoke & (np.abs(x) > 30e-3)].mean()),
        "legs": float(J[~yoke].mean()),
    }


def main():
    solves = []  # how many linear solves each B-H iteration took, from its log
    logger.enable("magpylib_material_response")
    logger.remove()
    logger.add(
        lambda m: solves.append(int(m.record["extra"]["solves"])),
        filter=lambda r: "solves" in r["extra"],
        level="INFO",
    )
    line = line_above_poles()
    out = {"currents": CURRENTS, "linear": [], "cells": {}, "fem": {}}

    for current in CURRENTS:  # linear, chi = 999: face charge
        solved = solve_face_charge(magpy.Collection(core(CHI0), coil(current)))
        out["linear"].append(peak(magpy.getB(solved, line)))

    for target in TETRAHEDRA:
        mesh = mesh_TriangularMesh(core_mesh(iron_curve()), target_elems=target)
        rows = []
        for current in CURRENTS:
            solves.clear()
            t0 = time.perf_counter()
            solved = solve_cells(magpy.Collection(mesh.copy(), coil(current)))
            seconds = time.perf_counter() - t0
            rows.append(
                {
                    "peak": peak(magpy.getB(solved, line)),
                    "seconds": round(seconds, 2),
                    "solves": solves[-1],
                    "J": regions(solved.children[0].sources_all),
                }
            )
        linear = mesh.copy()
        linear.susceptibility = CHI0
        t0 = time.perf_counter()
        solve_cells(magpy.Collection(linear, coil(25)))
        out["cells"][str(len(mesh.sources_all))] = {
            "runs": rows,
            "linear_seconds": round(time.perf_counter() - t0, 2),
        }

    for order in FEM_ORDERS:
        rows = []
        for current in FEM_CURRENTS:
            t0 = time.perf_counter()
            solution = solve_fem(
                magpy.Collection(core(iron_curve()), coil(current)),
                order=order,
                observers=line,
            )
            rows.append(
                {
                    "current": current,
                    "peak": peak(solution.getB(line)),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
        out["fem"][str(order)] = rows

    Path(__file__).with_name("saturation.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
