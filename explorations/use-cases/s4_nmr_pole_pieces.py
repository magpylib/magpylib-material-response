"""Use case 4: comparing designs of an NMR magnet (#85, after #33): two ferrite
discs (0.38 T, an assumed Y30BH value) with iron discs (chi 3999) between them,
all 120 mm across, ferrite and iron 20 mm thick, a 40 mm gap. Design A has
20 mm iron discs, design B 24 mm, both facing the same gap.

The question is a difference: does B make the field more uniform than A? On
#85 the absolute error (0.6 %) is larger than the variation judged (1,138 ppm).
This solves both designs at face-charge levels 1 to 4 and reads Bz at the
centre and its variation along the axis over +-5 mm; mesh_study takes one model
and one quantity, so a study of a difference is written by hand.

    python s4_nmr_pole_pieces.py [max_level] [max_unknowns]
"""

import sys
import time

import magpylib as magpy
import numpy as np

from common import done, mm, solve_timed, write_result
from magpylib_material_response.face_charge import apply_face_charge

MAX_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_UNKNOWNS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
NAME = f"s4_nmr_levels1-{MAX_LEVEL}_max{MAX_UNKNOWNS}"
AXIS = np.stack([np.zeros(21), np.zeros(21), np.linspace(-5, 5, 21)], axis=1) * mm
GAP = 40 * mm


def model(iron):
    parts = []
    for side in (1, -1):
        disc = magpy.magnet.Cylinder(
            polarization=(0, 0, 0),
            dimension=(120 * mm, iron),
            position=(0, 0, side * (GAP / 2 + iron / 2)),
        )
        disc.susceptibility = 3999
        ferrite = magpy.magnet.Cylinder(
            polarization=(0, 0, 0.38),
            dimension=(120 * mm, 20 * mm),
            position=(0, 0, side * (GAP / 2 + iron + 10 * mm)),
        )
        parts += [disc, ferrite]
    return magpy.Collection(*parts)


def read(solved):
    Bz = magpy.getB(solved, AXIS)[:, 2]
    centre = Bz[len(Bz) // 2]
    return float(centre), float(np.ptp(Bz) / abs(centre) * 1e6)


if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

rows = []
start = time.perf_counter()
for level in range(1, MAX_LEVEL + 1):
    row = {"level": level}
    for key, iron in (("A", 20 * mm), ("B", 24 * mm)):
        try:
            solved, total, _ = solve_timed(
                apply_face_charge, model(iron), level=level, max_unknowns=MAX_UNKNOWNS
            )
        except ValueError as err:  # over max_unknowns
            row[key] = {"refused": str(err).split(":")[0]}
            continue
        centre, ppm = read(solved)
        row[key] = {
            "Bz_centre_mT": centre * 1e3,
            "variation_ppm": ppm,
            "unknowns": solved.face_charge_info["unknowns"],
            "seconds": total,
        }
    rows.append(row)
    print(row, flush=True)
    if any("refused" in row[k] for k in "AB"):
        break
    row["B_minus_A_ppm"] = row["B"]["variation_ppm"] - row["A"]["variation_ppm"]
    row["B_minus_A_centre_mT"] = row["B"]["Bz_centre_mT"] - row["A"]["Bz_centre_mT"]

write_result(NAME, {"wall_s": time.perf_counter() - start, "rows": rows})
