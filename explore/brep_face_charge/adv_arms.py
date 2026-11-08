"""Face charge in the adversarial scenarios, against their exact references.

Arms: an export (15, 7.5, 3.75 degree facets) at levels 2 and 3; the solver's
own facets from the exact curve, widened (levels 1-3) or not (2, 3); and the
export widened from the mesh alone (15 and 7.5 degrees, levels 2 and 3): what
a solver could do without the B-rep. Errors relative to the part's own field
(B minus the other sources'). Under the machine-wide perf lock.

    python adv_arms.py plate|pole   -> results/adv_arms_<part>.jsonl
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np

from geometry import as_magnet, export_mesh_widened, faceted
from scenarios import scenarios

from magpylib_material_response import solve_face_charge

HERE = Path(__file__).parent
LOCK = Path("/Users/alexandreboisselet/Documents/GitRepos.nosync/.perf-lock")


def arms(part):
    for n in (24, 48, 96):
        for level in (2, 3):
            yield ("export", n, level), faceted(part, n, False)
    for level in (1, 2, 3):
        yield ("B-rep, widened", 12 * (level + 1), level), faceted(part, 12 * (level + 1), True)
    for level in (2, 3):
        yield ("B-rep, on the curve", 12 * (level + 1), level), faceted(part, 12 * (level + 1), False)
    for n in (24, 48):
        for level in (2, 3):
            yield ("export, widened from the mesh", n, level), export_mesh_widened(part, n)


def run(part):
    refs = json.loads((HERE / "results" / f"adv_reference_{part}.json").read_text())
    rows = []
    for (arm, n, level), (V, F) in arms(part):
        for name, (others, groups) in scenarios(part).items():
            ref = refs[name]["groups"]
            sensors = np.concatenate(list(groups.values()))
            model = magpy.Collection(as_magnet(V, F, label=f"{part} {arm} {n}"), *others)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                solved = solve_face_charge(model, level=level)
                dt = time.perf_counter() - t0
            B = magpy.getB(solved, sensors)
            errs, first = {}, 0
            for group, v in ref.items():
                Bref, bg = np.array(v["B"]), np.array(v["background"])
                rows_ = slice(first, first + len(Bref))
                first += len(Bref)
                errs[group] = float(np.linalg.norm(B[rows_] - Bref) / np.linalg.norm(Bref - bg))
            rows.append({"part": part, "scenario": name, "arm": arm, "facets_per_turn": n,
                         "level": level, "unknowns": solved.face_charge_info["unknowns"],
                         "seconds": round(dt, 4), "error_of_own_field": errs})
            print(f"{name:24s} {arm:30s} {n:3d}/turn L{level}: {rows[-1]['unknowns']:6d} unk "
                  f"{dt:5.2f} s near {100 * errs['near the curves']:.3f} %", flush=True)
    (HERE / "results" / f"adv_arms_{part}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    LOCK.mkdir()
    try:
        run(sys.argv[1])
    finally:
        LOCK.rmdir()
