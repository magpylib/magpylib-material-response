"""The cache above 2,000 unknowns, where apply_face_charge factors in single
precision and refines: it keeps a double LU instead. The plate of the Monte
Carlo at rising levels: the plain call, the first cached call, the later ones,
memory, and the answer against the plain call. One level a call, each result
its own file; stops itself above 6 GB.

    python c5_cache_size.py level
"""

import os
import resource
import sys
import threading
import time

import cached
import magpylib as magpy
import numpy as np
from common import done, write_result
from p_model import draw, model, sensor_at

from magpylib_material_response.face_charge import apply_face_charge

LEVEL = int(sys.argv[1])
NAME = f"c5_cache_size_level{LEVEL}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")


def guard():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6e9:
            print("over 6 GB, stopping", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=guard, daemon=True).start()
d = draw(8)
cached.MAX_BYTES = 8 * 10**9  # the size is what is measured here
cached.KEEP_AFTER = 1


def call(i):
    t = time.perf_counter()
    solved = apply_face_charge(model(d, i), level=LEVEL, max_unknowns=30_000)
    t1 = time.perf_counter() - t
    return t1, magpy.getB(solved, sensor_at(d, i)), solved.face_charge_info["unknowns"]


plain = [call(i) for i in range(3)]
cached.install()
cached.clear()
kept = [call(i) for i in range(3)]
hits = [call(i)[0] for i in range(3, 8)]
write_result(
    NAME,
    {
        "level": LEVEL,
        "unknowns": plain[0][2],
        "plain_call_s": float(np.median([p[0] for p in plain])),
        "first_cached_call_s": kept[0][0],
        "hit_call_s": float(np.median(hits)),
        "lu_gb": 8 * (plain[0][2]) ** 2 / 1e9,
        "stats": cached.stats,
        "relative_error": max(
            float(np.abs(a[1] - b[1]).max() / np.abs(a[1]).max())
            for a, b in zip(plain, kept, strict=True)
        ),
    },
)
