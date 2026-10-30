"""Use case 1: a gradiometric angle sensor over a magnet on a steel shaft, in a
uniform stray field, after Infineon US 12,237,730 B2 (shaft 20 mm across,
mu_r 4000, 5 mT stray field, 1.5 mm air gap). The magnet (6 x 3 mm, 0.5 T
diametric) and the soft holder ring around the sensor are stand-ins.

The rotor (shaft and magnet) turns; the sensor, and the holder when present,
stay. Four Hall plates on a 1 mm circle read Bz; the angle is atan2 of the two
differences, so a uniform field cancels and only the shaft's distortion of it
remains. Today every angle, air gap and stray field is a new solve.

    python s1_shaft_sensor.py [level] [angles]
"""

import sys
import time

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import backsub_seconds, done, mm, solve_timed, write_result
from magpylib_material_response.face_charge import apply_face_charge

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ANGLES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NAME = f"s1_shaft_level{LEVEL}_angles{ANGLES}"
CHI = 3999
PLATE_R = 1 * mm
MAGNET_TOP = 3 * mm


def model(angle, gap, stray, holder):
    turn = R.from_euler("z", angle, degrees=True)
    shaft = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(20 * mm, 40 * mm), position=(0, 0, -20 * mm)
    )
    shaft.susceptibility = CHI
    magnet = magpy.magnet.Cylinder(
        polarization=(0.5, 0, 0), dimension=(6 * mm, 3 * mm), position=(0, 0, 1.5 * mm)
    )
    rotor = magpy.Collection(shaft, magnet).rotate(turn, anchor=0)
    parts = [rotor]
    if holder:
        # stocky on purpose: a 2 mm ring, as a holder would be, takes 16,044
        # unknowns alone at level 2 (measured below), too many for a grid
        ring = magpy.magnet.CylinderSegment(
            polarization=(0, 0, 0),
            dimension=(3 * mm, 8 * mm, 4 * mm, 0, 360),
            position=(0, 0, MAGNET_TOP + gap + 1 * mm),
        )
        ring.susceptibility = CHI
        parts.append(ring)
    coll = magpy.Collection(*parts)
    coll.H_ext = (stray, 0, 0)
    return coll


def plates(gap):
    z = MAGNET_TOP + gap
    return np.array([[PLATE_R, 0, z], [-PLATE_R, 0, z], [0, PLATE_R, z], [0, -PLATE_R, z]])


def angle_error(Bz, angle):
    est = np.degrees(np.arctan2(Bz[2] - Bz[3], Bz[0] - Bz[1]))
    return (est - angle + 180) % 360 - 180


if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

angles = np.linspace(0, 360, ANGLES, endpoint=False)
rows, totals, sources, unknowns = [], [], [], []
start = time.perf_counter()
for holder in (False, True):
    for gap in (1.5 * mm, 2.5 * mm):
        for stray in (0.0, 5e-3):
            errors = []
            for angle in angles:
                coll = model(angle, gap, stray, holder)
                solved, total, src = solve_timed(apply_face_charge, coll, level=LEVEL)
                # getB leaves out H_ext (#71); the plates add it back, though a
                # uniform field cancels in the differences anyway
                B = magpy.getB(solved, plates(gap)) + np.array([stray, 0, 0])
                errors.append(angle_error(B[:, 2], angle))
                totals.append(total)
                sources.append(src)
                unknowns.append(solved.face_charge_info["unknowns"])
            errors = np.array(errors)
            rows.append(
                {
                    "holder": holder,
                    "gap_mm": gap / mm,
                    "stray_mT": stray * 1e3,
                    "error_pp_deg": float(np.ptp(errors)),
                    "error_max_deg": float(np.abs(errors - errors.mean()).max()),
                }
            )
            print(rows[-1], flush=True)

thin = magpy.magnet.CylinderSegment(
    polarization=(0, 0, 0), dimension=(3 * mm, 12 * mm, 2 * mm, 0, 360)
)
thin.susceptibility = CHI
thin_solved, thin_s, _ = solve_timed(apply_face_charge, magpy.Collection(thin), level=2)

n = len(totals) // 2  # solves per variant, without and with the holder
plain, held = slice(0, n), slice(n, 2 * n)
n_plain = max(unknowns[plain])
write_result(
    NAME,
    {
        "level": LEVEL,
        "angles": ANGLES,
        "solves": 2 * n,
        "wall_s": time.perf_counter() - start,
        "unknowns": sorted(set(unknowns)),
        "seconds": totals,
        "source_field_seconds": sources,
        # without the holder the shaft is the only solved part: turning it with
        # the magnet, moving the sensor or changing the stray field leaves its
        # matrix as it is, so one factorization serves the whole grid; each
        # further configuration costs its source field and a back-substitution
        "no_holder_today_s": float(np.sum(totals[plain])),
        "no_holder_kept_factorization_s": totals[0]
        + (n - 1) * (float(np.mean(sources[plain])) + backsub_seconds(n_plain)),
        "with_holder_today_s": float(np.sum(totals[held])),
        "thin_ring_level2": {
            "unknowns": thin_solved.face_charge_info["unknowns"],
            "seconds": thin_s,
        },
        "rows": rows,
    },
)
