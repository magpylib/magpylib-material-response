"""Is the cache safe when a user's own threads call apply_face_charge at once?
Tasks whose answers are known from a serial run without the cache, then the same
tasks from a thread pool, in a shuffled order and with Python switching threads
far more often than usual, to bring races out. A hang dumps the stacks and exits.

    python c8_threads.py label plain|cached [keep_after]
"""

import faulthandler
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cached
import magpylib as magpy
import numpy as np
from common import done, mm, write_result

from magpylib_material_response.face_charge import _bodies, apply_face_charge

LABEL, MODE = sys.argv[1], sys.argv[2]
KEEP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NAME = f"c8_threads_{LABEL}_{MODE}_keep{KEEP}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")
faulthandler.dump_traceback_later(900, exit=True)
OBS = np.array([[12, 0, 0], [0, 0, 12], [7, 7, 7], [0, 0, -9]], float) * mm
TASKS = 40
ROUNDS = 2
original_solve = _bodies._solve


def magnet(k):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * mm,) * 3,
        position=(np.sin(k) * mm, np.cos(k) * mm, 4.5 * mm),
    )


def soft(length=10.0, chi=1000.0):
    p = magpy.magnet.Cuboid(
        polarization=(0, 0, 0), dimension=(length * mm, 10 * mm, 2 * mm)
    )
    p.susceptibility = chi
    return p


def cylinder(k):
    c = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(8 * mm, 2 * mm), position=(0, 0, 0)
    )
    c.susceptibility = 500
    c.rotate_from_angax(7 * k, "z")
    return magpy.Collection(magnet(k), c)


WORKLOADS = {
    "one matrix, hits": lambda k: magpy.Collection(magnet(k), soft()),
    "two matrices in turn": lambda k: magpy.Collection(
        magnet(k), soft(chi=(1000, 500)[k % 2])
    ),
    "eight matrices, evicting": lambda k: magpy.Collection(
        magnet(k), soft(length=10 + 0.1 * (k % 8))
    ),
    "a new matrix each time": lambda k: magpy.Collection(magnet(k), soft(chi=100 + k)),
    "a mesh and cuboids mixed": lambda k: (
        cylinder(k) if k % 2 else magpy.Collection(magnet(k), soft())
    ),
}


def read(workload, k):
    return magpy.getB(apply_face_charge(workload(k), level=2), OBS)


rows = []
for name, workload in WORKLOADS.items():
    truth = [read(workload, k) for k in range(TASKS)]
    if MODE == "cached":
        cached.install()
        cached.KEEP_AFTER = KEEP
        cached.MAX_BYTES = (
            3 * 8 * 800**2 if name == "eight matrices, evicting" else 2 * 10**9
        )
    for workers in (1, 4, 16):
        sys.setswitchinterval(1e-5 if workers > 1 else 5e-3)
        wrong = failed = 0
        errors = set()
        worst = 0.0
        if MODE == "cached":
            cached.clear()
        t = time.perf_counter()
        for rnd in range(ROUNDS):
            order = list(range(TASKS))
            random.Random(rnd).shuffle(order)
            with ThreadPoolExecutor(workers) as pool:
                futures = [(k, pool.submit(read, workload, k)) for k in order]
                for k, f in futures:
                    try:
                        B = f.result()
                    except Exception as e:  # pylint: disable=broad-except
                        failed += 1
                        errors.add(f"{type(e).__name__}: {e}"[:120])
                        continue
                    err = float(np.abs(B - truth[k]).max() / np.abs(truth[k]).max())
                    worst = max(worst, err)
                    wrong += err > 1e-9
        row = {
            "workload": name,
            "workers": workers,
            "calls": TASKS * ROUNDS,
            "failed": failed,
            "wrong": wrong,
            "worst_relative_error": worst,
            "errors": sorted(errors),
            "ms_per_call": (time.perf_counter() - t) / (TASKS * ROUNDS) * 1e3,
            "stats": dict(cached.stats) if MODE == "cached" else None,
            "solve_left_patched": _bodies._solve
            not in (original_solve, getattr(cached, "_dispatch", None)),
        }
        rows.append(row)
        print(
            {
                k: row[k]
                for k in (
                    "workload",
                    "workers",
                    "failed",
                    "wrong",
                    "worst_relative_error",
                    "ms_per_call",
                    "solve_left_patched",
                    "errors",
                )
            },
            flush=True,
        )
    cached.uninstall()
write_result(
    NAME,
    {"label": LABEL, "mode": MODE, "keep_after": KEEP, "rows": rows},
)
