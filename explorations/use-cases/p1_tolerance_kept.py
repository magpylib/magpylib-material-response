"""Prototype 1: the tolerance Monte Carlo of s2 (a), the magnet a fixed source
over a soft plate, with the factorization kept (kept.py) instead of a solve
per sample. Replaces s2's estimate with a measurement, two ways:

- loop: one sample at a time, as a loop over a kept solver would run;
- path: all samples at once, the magnet and the sensor along paths of every
  pose, as a solver that accepts paths could run.

First checks the kept solver against apply_face_charge sample by sample.
Which magpylib runs is whatever imports first. The older results ran on main
and on #1059's branch; 6.1.0 has those threads, and threads limits them.

    python p1_tolerance_kept.py [samples] [loop|path] [threads]
"""

import sys
import time

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import done, mm, write_result
from kept import Kept
from magpylib_material_response.face_charge import apply_face_charge

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MODE = sys.argv[2] if len(sys.argv) > 2 else "path"
CHECKED = 20
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else magpy.defaults.threads
magpy.defaults.threads = THREADS
MAGPYLIB = f"{magpy.__version__}, {THREADS} threads"
NAME = f"p1_tolerance_kept_{MODE}_{SAMPLES}_magpylib-{magpy.__version__}_t{THREADS}"
SENSOR = np.array([3, 0, 1.5]) * mm
rng = np.random.default_rng(1)


def magnet(xy, tilt):
    """The magnet at one pose, or along a path of poses (xy (n, 2), tilt (n, 2))."""
    xy = np.atleast_2d(xy)
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * mm, 5 * mm, 5 * mm),
        position=np.c_[xy, np.full(len(xy), 4.5 * mm)],
        orientation=R.from_euler("xy", np.atleast_2d(tilt), degrees=True),
    )


def plate():
    p = magpy.magnet.Cuboid(
        polarization=(0, 0, 0), dimension=(10 * mm, 10 * mm, 2 * mm)
    )
    p.susceptibility = 1000
    return p


def H_along(source, points):
    """B / mu0 of a source along its path at fixed points: (steps, points, 3)."""
    out = magpy.getB(source, points, squeeze=False)
    return out.reshape(out.shape[1], -1, 3) / magpy.mu_0


if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

d = {
    "magnet_xy": rng.normal(0, 0.1 * mm, (SAMPLES, 2)),
    "magnet_tilt": rng.normal(0, 1.0, (SAMPLES, 2)),
    "sensor_xy": rng.normal(0, 0.1 * mm, (SAMPLES, 2)),
    "sensor_tilt": rng.normal(0, 1.0, (SAMPLES, 2)),
}
sensor_pos = SENSOR + np.c_[d["sensor_xy"], np.zeros(SAMPLES)]
sensor_rot = R.from_euler("xy", d["sensor_tilt"], degrees=True)

t0 = time.perf_counter()
kept = Kept(magpy.Collection(magnet((0, 0), (0, 0)), plate()))


def field(points, sl):
    return H_along(magnet(d["magnet_xy"][sl], d["magnet_tilt"][sl]), points)


# the kept solver's load of the nominal magnet is the solve's own
nominal = kept.loads(lambda P, _sl: H_along(magnet((0, 0), (0, 0)), P), 1)[0]
load_error = float(
    np.abs(nominal - kept.nominal_rhs).max() / np.abs(kept.nominal_rhs).max()
)


def read(sl, x):
    """Bz in each sample's own sensor frame: the plate's field plus the magnet's."""
    plate_B = sensor_rot[sl].apply(
        magpy.mu_0 * kept.field(sensor_pos[sl], x), inverse=True
    )
    # a sensor along a path pairs its step i with the magnet's, in its own frame
    sensor = magpy.Sensor(position=sensor_pos[sl], orientation=sensor_rot[sl])
    magnet_B = np.atleast_2d(
        magpy.getB(magnet(d["magnet_xy"][sl], d["magnet_tilt"][sl]), sensor)
    )
    return (plate_B + magnet_B)[:, 2]


# the check: apply_face_charge per sample against the kept solver
full = []
for i in range(CHECKED):
    solved = apply_face_charge(
        magpy.Collection(magnet(d["magnet_xy"][i], d["magnet_tilt"][i]), plate())
    )
    sensor = magpy.Sensor(position=sensor_pos[i], orientation=sensor_rot[i])
    full.append(magpy.getB(solved, sensor)[2])
x = kept.solve(kept.loads(field, CHECKED))
check = read(slice(0, CHECKED), x)
check_error = float(np.abs(check - np.array(full)).max())

timing = {"first_solve_s": kept.first_s, "factor_s": kept.factor_s}
t = time.perf_counter()
if MODE == "loop":
    Bz = np.empty(SAMPLES)
    for i in range(SAMPLES):
        sl = slice(i, i + 1)
        Bz[i] = read(sl, kept.solve(kept.loads(lambda P, _, sl=sl: field(P, sl), 1)))[0]
else:
    t1 = time.perf_counter()
    rhs = kept.loads(field, SAMPLES)
    t2 = time.perf_counter()
    x = kept.solve(rhs)
    t3 = time.perf_counter()
    Bz = read(slice(0, SAMPLES), x)
    timing |= {
        "loads_s": t2 - t1,
        "solves_s": t3 - t2,
        "read_s": time.perf_counter() - t3,
    }
timing["samples_s"] = time.perf_counter() - t
timing["total_s"] = time.perf_counter() - t0

write_result(
    NAME,
    {
        "samples": SAMPLES,
        "mode": MODE,
        "magpylib": MAGPYLIB,
        "unknowns": int(kept.offsets[-1]),
        "load_points": len(kept.points),
        "nominal_load_rel_error": load_error,
        "checked_samples": CHECKED,
        "check_max_abs_error_T": check_error,
        "check_Bz_range_T": [float(min(full)), float(max(full))],
        "per_sample_ms": timing["samples_s"] / SAMPLES * 1e3,
        **timing,
        "Bz_mean_mT": float(Bz.mean() * 1e3),
        "Bz_std_mT": float(Bz.std() * 1e3),
    },
)
