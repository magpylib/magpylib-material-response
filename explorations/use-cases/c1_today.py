"""The tolerance Monte Carlo of s2 (a) as a user writes it today, a solve per
sample, on the released packages: what the kept factorization and the cache are
measured against.

    python c1_today.py [samples]
"""

import sys
import time

import magpylib as magpy
import numpy as np
from common import done, write_result
from p_model import draw, model, sensor_at

import magpylib_material_response
from magpylib_material_response.face_charge import apply_face_charge

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NAME = f"c1_today_{SAMPLES}_magpylib-{magpy.__version__}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

d = draw(SAMPLES)
t = time.perf_counter()
Bz = [
    magpy.getB(apply_face_charge(model(d, i)), sensor_at(d, i))[2]
    for i in range(SAMPLES)
]
total = time.perf_counter() - t
write_result(
    NAME,
    {
        "samples": SAMPLES,
        "magpylib": magpy.__version__,
        "material_response": magpylib_material_response.__version__,
        "per_sample_ms": total / SAMPLES * 1e3,
        "Bz_mean_mT": float(np.mean(Bz) * 1e3),
    },
)
