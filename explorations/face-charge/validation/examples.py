"""The documented examples of material-response, built as their pages build them.

Each case gives the model, the observers, how to read the compared quantity
from B there, the reference with its source, and how the docs run the MoM.
"""

import glob
import json
import os
import time

import magpylib as magpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3


def _timed_mom(coll, observers, **mesh_kw):
    from magpylib_material_response.demag import apply_demag
    from magpylib_material_response.meshing import mesh_all

    meshed = mesh_all(coll, **mesh_kw)
    t0 = time.perf_counter()
    solved = apply_demag(meshed)
    t = time.perf_counter() - t0
    return magpy.getB(solved, observers, sumup=True), len(meshed.sources_all), t


def _docs_loop(build, observers):
    """The MoM as the ANSYS pages run it: one target per magnet, 1 to 256."""
    out = []
    for target in [1, 2, 8, 16, 32, 64, 128, 256]:
        B, cells, t = _timed_mom(build(), observers, target_elems=target, per_child_elems=True, min_elems=1)
        out.append({"label": f"{target} per magnet", "cells": cells, "solve_s": t, "B": B})
    return out


def _study(build, quantity, levels, observers):
    """The MoM as the mesh-convergence page runs it, and B at the observers."""
    from magpylib_material_response.convergence import mesh_study

    study = mesh_study(build(), quantity, levels)
    out = []
    for n, value, t in zip(study.cells, study.values, study.seconds, strict=True):
        out.append({"label": f"{n} cells", "cells": int(n), "solve_s": float(t), "Q": np.asarray(value)})
    est = None if study.estimate is None else np.asarray(study.estimate)
    return out, est, None if study.errors is None else np.asarray(study.errors)


# ── A: three magnets against ANSYS ────────────────────────────────────────────


def build_A():
    cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    cube1.move((-0.0015, 0, 0))
    cube1.susceptibility = 0.3
    cube2 = magpy.magnet.Cuboid(polarization=(0.9, 0, 0), dimension=(0.001, 0.001, 0.001))
    cube2.rotate_from_angax(-45, "y").move((0, 0, 0.0002))
    cube2.susceptibility = 1.0
    mx, my = 0.6 * np.sin(np.deg2rad(30)), 0.6 * np.cos(np.deg2rad(30))
    cube3 = magpy.magnet.Cuboid(polarization=(mx, my, 0), dimension=(0.001, 0.001, 0.002))
    cube3.move((0.0016, 0, 0.0005)).rotate_from_angax(30, "z")
    cube3.susceptibility = 0.5
    return magpy.Collection(cube1, cube2, cube3)


def case_A():
    from magpylib_material_response import get_dataset

    obs = np.linspace((-0.004, 0, -0.001), (0.004, 0, -0.001), 301)
    v = get_dataset("FEMdata_test_cuboids")["results"][0]["value"]
    return {
        "title": "Three magnets, χ 0.3 to 1, two rotated (cuboids_demagnetization)",
        "build": build_A, "observers": obs, "comps": [0, 1, 2],
        "reference": np.stack([v["Bx"], v["By"], v["Bz"]], 1),
        "ref_source": "ANSYS (FEMdata_test_cuboids), 301 points",
        "fc_levels": [2, 3, 4], "mom": lambda: _docs_loop(build_A, obs),
    }


# ── B: hard magnet and soft cube against ANSYS ────────────────────────────────


def build_B():
    cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.002))
    cube1.move((0, 0, 0.0005))
    cube1.susceptibility = 0.5
    cube2 = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.001, 0.001, 0.001))
    cube2.rotate_from_angax(angle=45, axis="y").move((0.0015, 0, 0))
    cube2.susceptibility = 3999
    return magpy.Collection(cube1, cube2)


def case_B():
    from magpylib_material_response import get_dataset

    obs = np.concatenate([np.linspace((-0.004, 0, z), (0.006, 0, z), 1001) for z in (-0.001, -0.003, -0.005)])
    v = get_dataset("FEMdata_test_softmag")["results"][0]["value"]
    # The page gives each line as a 1001-step path; its dataframe, and so the
    # dataset, interleaves the three lines step by step. Here they are in blocks.
    lines = lambda a: np.asarray(a).reshape(1001, 3).T.reshape(-1)  # noqa: E731
    ref = np.stack([lines(v["Bx"]), np.full(3003, np.nan), lines(v["Bz"])], 1)
    return {
        "title": "Hard magnet (χ 0.5) and a turned soft cube (χ 3999) (soft_magnets)",
        "build": build_B, "observers": obs, "comps": [0, 2],
        "reference": ref, "ref_source": "ANSYS (FEMdata_test_softmag), Bx and Bz, 3 lines of 1001 points",
        "fc_levels": [3, 4, 6], "mom": lambda: _docs_loop(build_B, obs),
    }


# ── C: flux concentrator ──────────────────────────────────────────────────────

B0 = 50e-6
C_SENSORS = np.array([(8.5, 0, 0), (9, 0, 0), (10, 0, 0)]) * MM


def build_C():
    bar = magpy.magnet.Cuboid(dimension=(16 * MM, 2 * MM, 2 * MM), polarization=(0, 0, 0))
    bar.susceptibility = 4000
    bar.H_ext = (B0, 0, 0)
    return magpy.Collection(bar)


def case_C():
    gain_of_B = lambda B: (B[:, 0] + B0) / B0  # noqa: E731
    gain = lambda solved: (magpy.getB(solved, C_SENSORS)[:, 0] + B0) / B0  # noqa: E731
    return {
        "title": "Flux concentrator, χ 4000 bar in the Earth's field (mesh_convergence)",
        "build": build_C, "observers": C_SENSORS, "quantity": gain_of_B, "unit": "gain",
        "reference": np.array([4.259, 3.346, 2.168]),
        "ref_source": "docs: MoM study on 4,096 to 262,144 cells, within 0.14 %",
        "fc_levels": [3, 4, 6], "mom_study": lambda: _study(build_C, gain, (4, 8, 16), C_SENSORS),
        "study_gives": "quantity",
    }


# ── D: Halbach array on back iron ─────────────────────────────────────────────

D_X = np.linspace(-15, 15, 21)
D_PROFILE = np.column_stack([D_X, 0 * D_X, np.full(21, 5.0)]) * MM


def build_D():
    turns = [(1.3, 0, 0), (0, 0, 1.3), (-1.3, 0, 0), (0, 0, -1.3), (1.3, 0, 0)]
    magnets = [
        magpy.magnet.Cuboid(dimension=(6 * MM,) * 3, position=(x * MM, 0, 0), polarization=p)
        for x, p in zip((-12, -6, 0, 6, 12), turns, strict=True)
    ]
    for magnet in magnets:
        magnet.susceptibility = 0.05
    plate = magpy.magnet.Cuboid(dimension=(30 * MM, 6 * MM, 3 * MM), position=(0, 0, -4.5 * MM), polarization=(0, 0, 0))
    plate.susceptibility = 1000
    return magpy.Collection(*magnets, plate)


def case_D():
    return {
        "title": "Halbach array of 5 magnets (χ 0.05) on back iron (χ 1000) (mesh_convergence)",
        "build": build_D, "observers": D_PROFILE, "comps": [0, 1, 2],
        "reference": "mom_estimate",
        "ref_source": "MoM study (2, 4, 8) extrapolated; docs: good to 0.03 % against 245,760 cells",
        "fc_levels": [3, 4], "mom_study": lambda: _study(build_D, D_PROFILE, (2, 4, 8), D_PROFILE),
    }


# ── E: shielding sheet ────────────────────────────────────────────────────────

E_BEHIND = np.array([(0, 0, 9), (6, 0, 9), (12, 0, 9)]) * MM


def build_E():
    magnet = magpy.magnet.Cuboid(dimension=(6 * MM,) * 3, polarization=(0, 0, 1.3))
    magnet.susceptibility = 0
    sheet = magpy.magnet.Cuboid(dimension=(24 * MM, 24 * MM, 1 * MM), position=(0, 0, 6.5 * MM), polarization=(0, 0, 0))
    sheet.susceptibility = 1000
    return magpy.Collection(magnet, sheet)


def case_E():
    return {
        "title": "Shielding sheet (24×24×1 mm, χ 1000) behind a magnet (mesh_convergence)",
        "build": build_E, "observers": E_BEHIND, "quantity": lambda B: np.linalg.norm(B, axis=1) * 1e3, "unit": "|B| mT",
        "reference": np.array([5.52, 5.85, 8.80]),
        "ref_source": "docs: MoM study on 4,608 to 294,912 cells, within 3.3 %, 1.1 %, 0.09 %",
        "fc_levels": [2, 3], "mom_study": lambda: _study(build_E, E_BEHIND, (1, 2, 4), E_BEHIND),
    }


# ── F: U-core electromagnet ───────────────────────────────────────────────────

F_LINE = np.stack([np.linspace(-0.08, 0.08, 161), np.zeros(161), np.full(161, 0.10)], axis=-1)


def build_F():
    from magpylib_material_response.polyline import create_polyline_fillet, move_grid_along_polyline

    a = 0.02
    yoke = magpy.magnet.Cuboid(dimension=(0.10, a, a))
    leg_left = magpy.magnet.Cuboid(dimension=(a, a, 0.06), position=(-0.04, 0, 0.04))
    leg_right = magpy.magnet.Cuboid(dimension=(a, a, 0.06), position=(0.04, 0, 0.04))
    core = magpy.Collection(yoke, leg_left, leg_right)
    core.susceptibility = 999
    h = 0.014
    square = np.array([[0, h, 0], [0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h], [0, h, 0]])
    turn = create_polyline_fillet(square, max_radius=0.006, N=6)
    pitch = 0.0022
    pack = np.array([(xt, yl, 0) for xt in (np.arange(8) - 3.5) * pitch for yl in h + np.arange(2) * pitch])
    coil = magpy.Collection(*(magpy.current.Polyline(current=25, vertices=v) for v in move_grid_along_polyline(turn, pack)))
    return magpy.Collection(core, coil)


def fem_ucore_reference():
    """FEM Bz along the line (mT), extrapolated over its three finest meshes."""
    runs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(HERE, "ref", "fem-ucore-*-mu1000.json"))),
                  key=lambda d: -d["h_mm"])
    h = np.array([d["h_mm"] for d in runs[-3:]])
    b = np.array([d["Bz_mT"] for d in runs[-3:]])
    d12, d23 = np.linalg.norm(b[1] - b[0]), np.linalg.norm(b[2] - b[1])
    from scipy.optimize import brentq

    p = brentq(lambda p: (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - d12 / d23, 0.2, 8)
    ext = b[2] + (b[2] - b[1]) * h[2] ** p / (h[1] ** p - h[2] ** p)
    return ext, p, [d["h_mm"] for d in runs], [d["peak_mT"] for d in runs]


def case_F():
    ref, p, hs, peaks = fem_ucore_reference()
    return {
        "title": "U-core electromagnet (3 cuboids, χ 999) driven by a 16-winding coil (ucore_electromagnet)",
        "build": build_F, "observers": F_LINE, "quantity": lambda B: B[:, 2] * 1e3, "unit": "Bz mT",
        "reference": ref,
        "ref_source": f"FEM (scikit-fem P2, explore/cells-and-fem), h = {hs} mm, extrapolated over the three finest (order {p:.2f})",
        "fc_levels": [3, 4, 6],
        "mom": lambda: [dict(zip(("B", "cells", "solve_s"), _timed_mom(build_F(), F_LINE, target_elems=250)), label="docs: 250 target")],
    }


CASES = {"A": case_A, "B": case_B, "C": case_C, "D": case_D, "E": case_E, "F": case_F}
