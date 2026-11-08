"""Face charge on each part, faceted three ways, against the exact reference.

- export:  a CAD export's faceting, fixed whatever the level: 12, 24, 48 or
           96 facets per turn (30, 15, 7.5, 3.75 degrees), vertices on the
           curve. Solved at levels 1, 2, 3.
- brep:    the solver facets the exact curve itself, 12 (level + 1) per turn,
           as it does a Cylinder today: "widened" keeps the exact volume,
           "on the curve" does not.

Each row: error against the exact reference per sensor group, as
||B - B_ref|| / ||B_ref|| and relative to the part's own field, unknowns, and
the solve's wall time (best of two, under the machine-wide perf lock).

    python arms.py plate|pole   -> results/arms_<part>.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np

from geometry import FIELD, SENSORS, as_magnet, faceted

from magpylib_material_response import UniformField, solve_face_charge

HERE = Path(__file__).parent
LOCK = Path("/Users/alexandreboisselet/Documents/GitRepos.nosync/.perf-lock")
LEVELS = (1, 2, 3)


def configs():
    for n in (12, 24, 48, 96):
        for level in LEVELS:
            yield "export", n, False, level
    for level in LEVELS:
        for widen in (True, False):
            yield "brep", 12 * (level + 1), widen, level


def errors(B, ref):
    out = {}
    first = 0
    for group, v in ref["groups"].items():
        Bref = np.array(v["B"])
        rows = slice(first, first + len(Bref))
        first += len(Bref)
        d = np.linalg.norm(B[rows] - Bref)
        out[group] = {
            "of_B": float(d / np.linalg.norm(Bref)),
            "of_response": float(d / np.linalg.norm(Bref - FIELD)),
        }
    return out


def run(part):
    ref = json.loads((HERE / "results" / f"reference_{part}.json").read_text())
    sensors = np.concatenate(list(SENSORS[part].values()))
    rows = []
    for arm, n, widen, level in configs():
        V, F = faceted(part, n, widen)
        magnet = as_magnet(V, F, label=f"{part} {arm} {n}")
        model = magpy.Collection(magnet, UniformField(B=FIELD))
        best = None
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                solved = solve_face_charge(model, level=level)
                dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        B = magpy.getB(solved, sensors)
        row = {
            "part": part,
            "arm": arm,
            "facets_per_turn": n,
            "degrees": 360 / n,
            "widened": widen,
            "level": level,
            "triangles_in": len(F),
            "volume_mm3": float(magnet.volume / 1e-9),
            "unknowns": solved.face_charge_info["unknowns"],
            "seconds": round(best, 4),
            "errors": errors(B, ref),
        }
        rows.append(row)
        e = row["errors"]["near the curves"]
        print(f"{arm:6s} {n:3d}/turn widen={widen!s:5} L{level}: {row['unknowns']:6d} unknowns "
              f"{best:6.2f} s  near {100 * e['of_B']:.3f} % of B, {100 * e['of_response']:.2f} % of response",
              flush=True)
    path = HERE / "results" / f"arms_{part}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    LOCK.mkdir()  # fails if another session holds it: wait, do not time beside it
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        run(sys.argv[1])
    finally:
        LOCK.rmdir()
