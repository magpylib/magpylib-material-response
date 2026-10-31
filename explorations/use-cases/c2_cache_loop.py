"""The tolerance Monte Carlo through apply_face_charge itself, with the cache
installed: the real cost of a call, to replace the estimate by subtraction, and
where what remains goes. Checks each sample against no cache on the first ones.

    python c2_cache_loop.py [samples]
"""

import cProfile
import pstats
import sys
import time

import cached
import magpylib as magpy
import numpy as np
from common import done, write_result
from p_model import draw, model, sensor_at

import magpylib_material_response
from magpylib_material_response.face_charge import apply_face_charge

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CHECKED = 50
NAME = f"c2_cache_loop_{SAMPLES}_magpylib-{magpy.__version__}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

d = draw(SAMPLES)
cached.KEEP_AFTER = 1


def run(n):
    return np.array(
        [
            magpy.getB(apply_face_charge(model(d, i)), sensor_at(d, i))[2]
            for i in range(n)
        ]
    )


plain = run(CHECKED)
cached.install()
with_cache = run(CHECKED)
error = float(np.abs(plain - with_cache).max())

t = time.perf_counter()
Bz = run(SAMPLES)
total = time.perf_counter() - t

prof = cProfile.Profile()
prof.enable()
run(100)
prof.disable()
rows = []
for (file, line, name), (_, calls, _, cum, _) in pstats.Stats(prof).stats.items():
    if "magpylib_material_response" in file or name in ("getB", "lu_solve"):
        rows.append(
            (round(cum / 100 * 1e3, 2), calls // 100, name, file.split("/")[-1], line)
        )
rows.sort(reverse=True)

write_result(
    NAME,
    {
        "samples": SAMPLES,
        "magpylib": magpy.__version__,
        "material_response": magpylib_material_response.__version__,
        "per_sample_ms": total / SAMPLES * 1e3,
        "checked": CHECKED,
        "check_max_abs_error_T": error,
        "stats": cached.stats,
        "Bz_mean_mT": float(Bz.mean() * 1e3),
        "where_ms_per_call_profiled": rows[:22],
    },
)
