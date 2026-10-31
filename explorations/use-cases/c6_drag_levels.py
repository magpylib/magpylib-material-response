"""One solve per position of a magnet moving over the plate, the time a re-solve
would take on each frame of a drag (studio #96 quotes 58, 143 and 357 ms at
levels 2, 3 and 4). Without and with the cache, with the sheets' read of one
sensor; the frame itself would also send the sheets to the view.

    python c6_drag_levels.py
"""

import sys
import time

import cached
import magpylib as magpy
import numpy as np
from common import done, mm, write_result

from magpylib_material_response.face_charge import apply_face_charge

NAME = "c6_drag_levels"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")
FRAMES = 30
SENSOR = np.array([3, 0, 1.5]) * mm
cached.KEEP_AFTER = 2


def model(k):
    # a path that stays a mm or more off the plate, so its panels stay the same
    x = 2 * mm * np.sin(k / 5)
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(5 * mm,) * 3, position=(x, x / 2, 4.5 * mm)
    )
    plate = magpy.magnet.Cuboid(
        polarization=(0, 0, 0), dimension=(10 * mm, 10 * mm, 2 * mm)
    )
    plate.susceptibility = 1000
    return magpy.Collection(magnet, plate)


def frames(level, n):
    out, times = [], []
    for k in range(n):
        t = time.perf_counter()
        solved = apply_face_charge(model(k), level=level)
        out.append(magpy.getB(solved, SENSOR))
        times.append(time.perf_counter() - t)
    return np.array(out), np.array(times), solved.face_charge_info["unknowns"]


rows = []
for level in (2, 3, 4, 5):
    cached.uninstall()
    plain, plain_t, unknowns = frames(level, FRAMES)
    cached.install()
    cached.clear()
    kept, kept_t, _ = frames(level, FRAMES)
    rows.append(
        {
            "level": level,
            "unknowns": unknowns,
            "plain_ms": float(np.median(plain_t) * 1e3),
            "first_two_ms": (kept_t[:2] * 1e3).tolist(),
            "hit_ms": float(np.median(kept_t[2:]) * 1e3),
            "hit_slowest_ms": float(kept_t[2:].max() * 1e3),
            "hits": cached.stats["hits"],
            "relative_error": float(np.abs(plain - kept).max() / np.abs(plain).max()),
        }
    )
    print(rows[-1], flush=True)
write_result(NAME, {"frames": FRAMES, "rows": rows})
