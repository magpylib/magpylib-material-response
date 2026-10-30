"""Use case 2: a tolerance Monte Carlo of a sensor in the gap between a magnet
and a soft plate, over 8 degrees of freedom as in Ergun et al. 2019: the
magnet's and the sensor's x and y (sigma 0.1 mm) and tilts about x and y
(sigma 1 deg). People run 1e4 to 1e5 samples; this times 40 and extrapolates.

(a) today: every magnet pose is a new solve;
(b) the sensor's tolerances alone need no new solve: one solve, getB at all
    sensor poses, which magpylib does today;
(c) a magnet with a recoil susceptibility of 0.05 is solved too, so moving it
    changes the coupling to the plate, not only the load.

    python s2_tolerance_mc.py [samples]
"""

import sys
import time

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import backsub_seconds, done, mm, solve_timed, write_result
from magpylib_material_response.face_charge import apply_face_charge

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TARGET = 10_000
NAME = f"s2_tolerance_samples{SAMPLES}"
SENSOR = np.array([3, 0, 1.5]) * mm
rng = np.random.default_rng(1)


def draw(n):
    """Offsets (m) and tilts (deg) of the magnet and the sensor."""
    return {
        "magnet_xy": rng.normal(0, 0.1 * mm, (n, 2)),
        "magnet_tilt": rng.normal(0, 1.0, (n, 2)),
        "sensor_xy": rng.normal(0, 0.1 * mm, (n, 2)),
        "sensor_tilt": rng.normal(0, 1.0, (n, 2)),
    }


def model(dxy=(0, 0), tilt=(0, 0), magnet_chi=0.0):
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * mm, 5 * mm, 5 * mm),
        position=(dxy[0], dxy[1], 4.5 * mm),
        orientation=R.from_euler("xy", tilt, degrees=True),
    )
    if magnet_chi:
        magnet.susceptibility = magnet_chi
    plate = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(10 * mm, 10 * mm, 2 * mm))
    plate.susceptibility = 1000
    return magpy.Collection(magnet, plate)


def sensor(dxy, tilt):
    """A sensor at one pose, or along a path of poses (dxy of shape (2, n))."""
    dxy = np.asarray(dxy)
    offset = np.stack([dxy[0], dxy[1], np.zeros_like(dxy[0])], axis=-1)
    return magpy.Sensor(
        position=SENSOR + offset, orientation=R.from_euler("xy", tilt, degrees=True)
    )


if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

d = draw(SAMPLES)
out = {"samples": SAMPLES, "target": TARGET}

# (a) and (c): one solve per sample
for key, chi in (("a_magnet_fixed", 0.0), ("c_magnet_recoil", 0.05)):
    totals, sources, Bz, unknowns = [], [], [], []
    for i in range(SAMPLES):
        coll = model(d["magnet_xy"][i], d["magnet_tilt"][i], chi)
        solved, total, src = solve_timed(apply_face_charge, coll)
        t = time.perf_counter()
        Bz.append(magpy.getB(solved, sensor(d["sensor_xy"][i], d["sensor_tilt"][i]))[2])
        totals.append(total + time.perf_counter() - t)
        sources.append(src)
        unknowns.append(solved.face_charge_info["unknowns"])
    n = max(unknowns)
    out[key] = {
        "unknowns": n,
        "per_sample_s": float(np.mean(totals)),
        "per_sample_source_field_s": float(np.mean(sources)),
        "today_10k_s": float(np.mean(totals)) * TARGET,
        "Bz_mean_mT": float(np.mean(Bz)) * 1e3,
        "Bz_std_mT": float(np.std(Bz)) * 1e3,
    }
    if not chi:
        # the plate is the only solved part: a magnet pose changes the load only,
        # so one factorization serves every sample, the back-substitutions all at once
        out[key]["kept_factorization_10k_s"] = float(
            totals[0]
            + TARGET * np.mean(sources)
            + backsub_seconds(n, rhs=TARGET, repeat=1)
        )
    print(key, out[key], flush=True)

# (b): sensor tolerances only, one solve and one getB call for every pose
solved, total, _ = solve_timed(apply_face_charge, model())
many = draw(TARGET)
t = time.perf_counter()
# one sensor with a path of all poses, as magpylib's docs advise
B = magpy.getB(solved, sensor(many["sensor_xy"].T, many["sensor_tilt"]))
out["b_sensor_only"] = {
    "solve_s": total,
    "getB_10k_poses_s": time.perf_counter() - t,
    "Bz_std_mT": float(np.std(B[:, 2])) * 1e3,
}
print("b_sensor_only", out["b_sensor_only"], flush=True)
write_result(NAME, out)
