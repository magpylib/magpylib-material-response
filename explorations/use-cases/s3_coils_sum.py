"""Use case 3: coils on soft iron, linear in their currents. A U-core (yoke
100 x 20 x 20 mm, legs 20 x 20 x 60 mm, chi 999, as in the docs) with one coil
of five loops on each leg; Bz on a line 30 mm above the poles.

Today every set of currents is a new solve. The response is linear in the
currents, so one solve per coil at 1 A gives any set as a sum: electromagnetic
navigation systems model their fields as linear in the currents. Here the user
writes that bookkeeping by hand.

    python s3_coils_sum.py [level] [sets]
"""

import sys
import time

import magpylib as magpy
import numpy as np

from common import backsub_seconds, done, mm, solve_timed, write_result
from magpylib_material_response.face_charge import apply_face_charge

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SETS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NAME = f"s3_coils_level{LEVEL}_sets{SETS}"
LINE = np.stack([np.linspace(-80, 80, 81), np.zeros(81), np.full(81, 100.0)], axis=1) * mm


def model(i_left, i_right):
    yoke = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(100 * mm, 20 * mm, 20 * mm))
    legs = [
        magpy.magnet.Cuboid(
            polarization=(0, 0, 0),
            dimension=(20 * mm, 20 * mm, 60 * mm),
            position=(x * mm, 0, 40 * mm),
        )
        for x in (-40, 40)
    ]
    core = magpy.Collection(yoke, *legs)
    core.susceptibility = 999
    # opposite windings, so the two legs are opposite poles at equal currents
    coils = [
        magpy.Collection(
            *[
                magpy.current.Circle(current=sign * current, diameter=32 * mm, position=(x * mm, 0, z * mm))
                for z in (25, 35, 45, 55, 65)
            ]
        )
        for x, sign, current in ((-40, 1, i_left), (40, -1, i_right))
    ]
    return magpy.Collection(core, *coils)


if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

rng = np.random.default_rng(2)
currents = rng.uniform(-10, 10, (SETS, 2))

# today: a solve per set of currents
t0 = time.perf_counter()
direct, totals, sources = [], [], []
for i_left, i_right in currents:
    solved, total, src = solve_timed(apply_face_charge, model(i_left, i_right), level=LEVEL)
    direct.append(magpy.getB(solved, LINE))
    totals.append(total)
    sources.append(src)
today = time.perf_counter() - t0
unknowns = solved.face_charge_info["unknowns"]

# by hand: one solve per coil at 1 A, then sums
t0 = time.perf_counter()
unit = [
    magpy.getB(apply_face_charge(model(*one), level=LEVEL), LINE) for one in ((1, 0), (0, 1))
]
summed = [a * unit[0] + b * unit[1] for a, b in currents]
by_sums = time.perf_counter() - t0

worst = max(np.abs(s - d).max() / np.abs(d).max() for s, d in zip(summed, direct))
write_result(
    NAME,
    {
        "level": LEVEL,
        "sets": SETS,
        "unknowns": unknowns,
        "today_s": today,
        "per_set_s": float(np.mean(totals)),
        "per_set_source_field_s": float(np.mean(sources)),
        "kept_factorization_s": totals[0]
        + (SETS - 1) * (float(np.mean(sources)) + backsub_seconds(unknowns)),
        "by_sums_s": by_sums,
        "sums_vs_solves_worst_rel": worst,
        "Bz_peak_mT": float(np.abs(np.array(direct)[:, :, 2]).max() * 1e3),
    },
)
