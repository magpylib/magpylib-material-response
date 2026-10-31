"""The turning shaft of s1 through apply_face_charge with the cache: the rotor
turns as one, the air gap moves only sensors, the stray field is H_ext, so the
grid without a holder has one matrix. With the holder the rotor turns against
it, and only repeats of an angle can hit. Against no cache on every call.

    python c4_shaft_cache.py [level] [angles] [keep_after]
"""

import sys
import time

import cached
import magpylib as magpy
import numpy as np
from common import done, mm, write_result
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.face_charge import apply_face_charge

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ANGLES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cached.KEEP_AFTER = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NAME = f"c4_shaft_cache_level{LEVEL}_angles{ANGLES}_keep{cached.KEEP_AFTER}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")
CHI, PLATE_R, MAGNET_TOP = 3999, 1 * mm, 3 * mm


def model(angle, gap, stray, holder):
    turn = R.from_euler("z", angle, degrees=True)
    shaft = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(20 * mm, 40 * mm), position=(0, 0, -20 * mm)
    )
    shaft.susceptibility = CHI
    magnet = magpy.magnet.Cylinder(
        polarization=(0.5, 0, 0), dimension=(6 * mm, 3 * mm), position=(0, 0, 1.5 * mm)
    )
    parts = [magpy.Collection(shaft, magnet).rotate(turn, anchor=0)]
    if holder:
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
    return np.array(
        [[PLATE_R, 0, z], [-PLATE_R, 0, z], [0, PLATE_R, z], [0, -PLATE_R, z]]
    )


def grid(holder):
    angles = np.linspace(0, 360, ANGLES, endpoint=False)
    t, out = time.perf_counter(), []
    for gap in (1.5 * mm, 2.5 * mm):
        for stray in (0.0, 5e-3):
            for a in angles:
                solved = apply_face_charge(model(a, gap, stray, holder), level=LEVEL)
                out.append(magpy.getB(solved, plates(gap)) + np.array([stray, 0, 0]))
    return time.perf_counter() - t, np.array(out), solved.face_charge_info["unknowns"]


rows = []
for holder in (False, True):
    cached.uninstall()
    plain_s, plain, unknowns = grid(holder)
    cached.install()
    cached.clear()
    kept_s, kept, _ = grid(holder)
    row = {
        "holder": holder,
        "unknowns": unknowns,
        "configurations": len(plain),
        "plain_s": plain_s,
        "cached_s": kept_s,
        "hits": cached.stats["hits"],
        "misses": cached.stats["misses"],
        "relative_error": float(np.abs(plain - kept).max() / np.abs(plain).max()),
    }
    rows.append(row)
    print(row, flush=True)
write_result(NAME, {"level": LEVEL, "angles": ANGLES, "rows": rows})
