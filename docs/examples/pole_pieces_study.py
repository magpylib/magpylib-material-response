"""The stored results of `pole_pieces.md`: two ferrite discs with iron discs
between them, solved by finite elements, face charge and cells. Together they
take several minutes on an Apple M4 Pro, so they are stored in
`pole_pieces.json`.

    uv run --group docs python docs/examples/pole_pieces_study.py            # all
    uv run --group docs python docs/examples/pole_pieces_study.py cells fem  # these

Run it after a change to a solver or to the model, and commit the JSON.
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

import magpylib as magpy
import numpy as np

from magpylib_material_response import BHCurve, solve_cells, study_face_charge
from magpylib_material_response.fem import solve_fem, study_fem
from magpylib_material_response.meshing import mesh_uniform, voxelize

MM = 1e-3  # lengths in metres, written in millimetres
DIAMETER, THICKNESS, GAP = 120 * MM, 20 * MM, 40 * MM
FERRITE = 0.39  # T, the polarization of a Y30BH ferrite
IRON = 3999  # the iron discs' susceptibility
WINDOW = 5 * MM  # the uniformity is judged over +-5 mm from the centre
#: the field is read on two lines through the centre, at points 1 mm apart
LINES = {
    "along the axis": np.linspace((0, 0, -15), (0, 0, 15), 31) * MM,
    "across the gap": np.linspace((-25, 0, 0), (25, 0, 0), 51) * MM,
}
#: and in the gap's middle plane y = 0, a quarter, mirrored when shown
MAP_X, MAP_Z = np.linspace(0, 25, 21) * MM, np.linspace(0, 15, 13) * MM
#: TEAM problem 13's measured steel (H in A/m, B in T), a structural steel:
#: a stand-in for pure iron, whose datasheet gives a steeper curve
TEAM13_H = [0, 16, 30, 54, 93, 143, 191, 210, 222, 233, 247, 258, 272, 289, 313,
            342, 377, 433, 509, 648, 933, 1228, 1934, 2913, 4993, 7189, 9423]  # fmt: skip
TEAM13_B = [0, 0.0025, 0.005, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.6, 1.65, 1.7,
            1.75, 1.8]  # fmt: skip
ORDER = 4  # finite elements for the stored comparisons


def pole_pieces(iron=IRON, angle=0.0, strength=1.0, ferrite=None):
    """Two ferrite discs polarized along z, an iron disc on the inner face of
    each, a gap between the iron discs. `iron` is their susceptibility (None:
    no iron discs). The top magnet's polarization is turned `angle` degrees
    off the axis, about y, and is `strength` times the bottom one's. `ferrite`
    is the magnets' own susceptibility (None: fixed polarization)."""
    parts = []
    for side in (-1, 1):
        magnet = magpy.magnet.Cylinder(
            dimension=(DIAMETER, THICKNESS),
            position=(0, 0, side * (GAP / 2 + 1.5 * THICKNESS)),
            polarization=(0, 0, FERRITE),
        )
        if side == 1:
            a = np.radians(angle)
            magnet.polarization = (
                strength * FERRITE * np.array((np.sin(a), 0, np.cos(a)))
            )
        if ferrite is not None:
            magnet.susceptibility = ferrite
        parts.append(magnet)
        if iron is not None:
            disc = magpy.magnet.Cylinder(
                dimension=(DIAMETER, THICKNESS),
                position=(0, 0, side * (GAP / 2 + THICKNESS / 2)),
                polarization=(0, 0, 0),
            )
            disc.susceptibility = iron
            parts.append(disc)
    return magpy.Collection(*parts)


def team13():
    """TEAM problem 13's steel as a B-H curve."""
    return BHCurve(H=np.array(TEAM13_H, float), B=np.array(TEAM13_B))


def lines():
    """The points of both lines, one array."""
    return np.concatenate(list(LINES.values()))


def by_line(B):
    """Split a field on `lines()` back into the lines, as lists."""
    out, start = {}, 0
    for name, points in LINES.items():
        out[name] = np.asarray(B)[start : start + len(points)].tolist()
        start += len(points)
    return out


def variation(B, line):
    """Peak-to-peak Bz over the window, relative to Bz at the centre, in ppm."""
    points = LINES[line]
    coordinate = 2 if line == "along the axis" else 0
    window = np.abs(points[:, coordinate]) <= WINDOW + 1e-9
    Bz = np.asarray(B)[:, 2]
    return float(np.ptp(Bz[window]) / Bz[len(points) // 2] * 1e6)


def _timed(solve):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = solve()
    seconds = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, round(seconds, 2), round(peak / 1e9, 2)


def fem():
    """Finite elements at orders 3 to 5: the reference."""
    study = study_fem(pole_pieces(), lines(), orders=(3, 4, 5))
    return {
        "orders": study.orders.tolist(),
        "ndofs": study.ndofs.tolist(),
        "seconds": np.round(study.seconds + study.mesh_seconds, 2).tolist(),
        "B": [by_line(v) for v in study.values],
    }


def face_charge():
    """Face charge at levels 1 to 3, with study_face_charge's estimate."""
    study = study_face_charge(
        pole_pieces(), lines(), levels=(1, 2, 3), max_unknowns=30_000
    )
    return {
        "levels": study.levels.tolist(),
        "unknowns": study.unknowns.tolist(),
        "seconds": np.round(study.seconds, 2).tolist(),
        "error": float(study.error),
        "B": [by_line(v) for v in study.values],
    }


def cells():
    """Cells: cylinder cells across the discs, and cubes on one lattice."""
    out = {}
    for n in (1, 2):
        mesh = mesh_uniform(pole_pieces(), n)
        solved, seconds, gb = _timed(lambda mesh=mesh: solve_cells(mesh))
        out[f"cylinder cells, {n} across"] = {
            "cells": len(mesh.sources_all), "seconds": seconds, "GB": gb,
            "B": by_line(magpy.getB(solved, lines())),
        }  # fmt: skip
    for target in (2_000, 8_000, 30_000):
        model = pole_pieces()
        parts = []
        for part in model.children:
            if getattr(part, "susceptibility", None) is None:
                parts.append(part.copy())
            else:
                cubes = voxelize(part, target)
                cubes.susceptibility = part.susceptibility
                parts.append(cubes)
        mesh = magpy.Collection(*parts)
        solved, seconds, gb = _timed(lambda mesh=mesh: solve_cells(mesh))
        per_disc = (len(mesh.sources_all) - 2) // 2  # the magnets stay whole
        out[f"cubes, {per_disc:,} per disc"] = {
            "cells": len(mesh.sources_all), "seconds": seconds, "GB": gb,
            "B": by_line(magpy.getB(solved, lines())),
        }  # fmt: skip
    return out


def field_map():
    """Finite elements at ORDER on the quarter of the middle plane."""
    X, Z = np.meshgrid(MAP_X, MAP_Z, indexing="ij")
    points = np.stack([X, 0 * X, Z], axis=-1).reshape(-1, 3)
    solved = solve_fem(pole_pieces(), order=ORDER, observers=points)
    return {"Bz": magpy.getB(solved, points)[:, 2].reshape(X.shape).tolist()}


#: the top magnet's variants that the page sweeps, checked here by finite elements
CHECKED_VARIANTS = ({"angle": 1.0, "strength": 1.0}, {"angle": 0.0, "strength": 0.98},
                    {"angle": 2.0, "strength": 1.02})  # fmt: skip


def variants():
    """Finite elements at ORDER for some of the top magnet's variants."""
    out = []
    for kwargs in CHECKED_VARIANTS:
        solved = solve_fem(pole_pieces(**kwargs), order=ORDER, observers=lines())
        out.append({**kwargs, "B": by_line(magpy.getB(solved, lines()))})
    return out


def mesh():
    """Finite elements at ORDER on a finer mesh: half the element size in the
    discs, and 1 mm at the observers, a sixth to a half of the default
    there."""
    solved = solve_fem(
        pole_pieces(),
        order=ORDER,
        observers=lines(),
        maxh=10 * MM,
        observer_maxh=1 * MM,
    )
    return {"ndof": solved.ndof, "B": by_line(magpy.getB(solved, lines()))}


def ferrite():
    """Finite elements at ORDER with the ferrite's own susceptibility, 0.05."""
    solved = solve_fem(pole_pieces(ferrite=0.05), order=ORDER, observers=lines())
    return {"B": by_line(magpy.getB(solved, lines()))}


def iron():
    """Finite elements at ORDER for other susceptibilities and on the TEAM 13
    curve, and the largest |B| in the iron discs, sampled to 0.1 mm from
    their faces, with where it is; and the largest more than 1 mm inside the
    rim, away from its edges, where the field rises without bound."""
    inside = np.array(
        [(r, 0, z) for r in np.linspace(0, 59.9, 300) * MM for z in np.linspace(20.1, 39.9, 100) * MM]
    )  # fmt: skip
    out = {}
    for name, chi in (
        ("χ = 300", 300),
        ("χ = 1000", 1000),
        ("χ = 3999", IRON),
        ("TEAM 13 curve", team13()),
    ):
        t0 = time.perf_counter()
        solved = solve_fem(
            pole_pieces(iron=chi), order=ORDER, observers=lines(), solver_tol=1e-8
        )
        seconds = time.perf_counter() - t0
        B = np.linalg.norm(magpy.getB(solved, inside), axis=1)
        out[name] = {
            "seconds": round(seconds, 1),
            "B": by_line(magpy.getB(solved, lines())),
            "max |B| in iron": float(B.max()),
            "where (r, z) in mm": (inside[B.argmax()][[0, 2]] / MM).round(1).tolist(),
            "max |B| 1 mm inside the rim": float(
                B[inside[:, 0] <= DIAMETER / 2 - 1 * MM].max()
            ),
        }
    return out


SECTIONS = {
    f.__name__: f
    for f in (fem, mesh, ferrite, face_charge, cells, field_map, variants, iron)
}


def main(names):
    path = Path(__file__).with_name("pole_pieces.json")
    # all sections start afresh; named ones replace theirs in the stored file
    stored = json.loads(path.read_text()) if names and path.exists() else {}
    for name in names or SECTIONS:
        t0 = time.perf_counter()
        stored[name] = SECTIONS[name]()
        print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)  # noqa: T201
    # written once at the end, so that an interrupted run leaves the file as it was
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main(sys.argv[1:])
