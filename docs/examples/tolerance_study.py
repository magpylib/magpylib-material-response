"""The model and the stored results of `tolerance_monte_carlo.md`: a magnet over
a steel back plate, a Hall sensor between them, and the spread of its reading
over the assembly's tolerances, after Ergun et al. 2019: the magnet's and the
sensor's positions within sigma 0.1 mm and tilts within sigma 1 degree, eight
degrees of freedom. 10,000 samples take a few minutes here, so the results are
stored in `tolerance.json`.

    uv run --group docs python docs/examples/tolerance_study.py

Run it after a change to `solve_face_charge`, and commit the JSON.
"""

import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

from magpylib_material_response import solve_face_charge

MM = 1e-3  # lengths in metres, written in millimetres
SAMPLES = 10_000
SIGMA = {"position": 0.1 * MM, "tilt": 1.0}  # m and degrees
#: the eight degrees of freedom: what moves, along what
DOF = [
    ("magnet", "position", 0),
    ("magnet", "position", 1),
    ("magnet", "tilt", 0),
    ("magnet", "tilt", 1),
    ("sensor", "position", 0),
    ("sensor", "position", 1),
    ("sensor", "tilt", 0),
    ("sensor", "tilt", 1),
]


def assembly():
    """A 5 mm magnet 1 mm over a 10 x 10 x 2 mm steel plate, and the sensor in
    the gap, just outside the magnet's edge."""
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * MM, 5 * MM, 5 * MM),
        position=(0, 0, 4.5 * MM),
    )
    plate = magpy.magnet.Cuboid(
        polarization=(0, 0, 0), dimension=(10 * MM, 10 * MM, 2 * MM)
    )
    plate.susceptibility = 1000
    sensor = magpy.Sensor(position=(3 * MM, 0, 1.5 * MM))
    return magpy.Collection(magnet, plate), magnet, sensor


def draw(samples, seed=1):
    """Each sample's deviations: (samples, 8), in the order of DOF."""
    rng = np.random.default_rng(seed)
    scale = [SIGMA[kind] for _, kind, _ in DOF]
    return rng.normal(0, 1, (samples, len(DOF))) * scale


def pose(obj, nominal, deviation, who):
    """Put `obj` at its nominal position, moved and tilted by the deviations of
    `who`, the magnet or the sensor."""
    shift, tilt = np.zeros(3), np.zeros(2)
    for value, (part, kind, axis) in zip(deviation, DOF, strict=True):
        if part == who:
            if kind == "position":
                shift[axis] = value
            else:
                tilt[axis] = value
    obj.position = nominal + shift
    obj.orientation = Rotation.from_euler("xy", tilt, degrees=True)


def main():
    model, magnet, sensor = assembly()
    magnet_at, sensor_at = magnet.position.copy(), sensor.position.copy()
    deviations = draw(SAMPLES)

    # with the plate: a solve per sample, the factorization kept, since only
    # the magnet, a fixed source, moves
    bz, solved = np.empty(SAMPLES), True
    t0 = time.perf_counter()
    for i, deviation in enumerate(deviations):
        pose(magnet, magnet_at, deviation, "magnet")
        pose(sensor, sensor_at, deviation, "sensor")
        solved = solve_face_charge(model, level=2, reuse=solved)
        bz[i] = magpy.getB(solved, sensor)[2]
    seconds = time.perf_counter() - t0
    unknowns = solved.face_charge_info["unknowns"]

    # Magpylib alone: the magnet without the plate
    alone = np.empty(SAMPLES)
    for i, deviation in enumerate(deviations):
        pose(magnet, magnet_at, deviation, "magnet")
        pose(sensor, sensor_at, deviation, "sensor")
        alone[i] = magpy.getB(magnet, sensor)[2]

    # one tolerance at a time, +- one sigma: the linear share of each
    share = []
    for k, (_, kind, _) in enumerate(DOF):
        ends = []
        for sign in (-1, 1):
            deviation = np.zeros(len(DOF))
            deviation[k] = sign * SIGMA[kind]
            pose(magnet, magnet_at, deviation, "magnet")
            pose(sensor, sensor_at, deviation, "sensor")
            ends.append(magpy.getB(solve_face_charge(model, level=2), sensor)[2])
        share.append((ends[1] - ends[0]) / 2)
    pose(magnet, magnet_at, np.zeros(len(DOF)), "magnet")
    pose(sensor, sensor_at, np.zeros(len(DOF)), "sensor")
    nominal = magpy.getB(solve_face_charge(model, level=2), sensor)[2]
    nominal_alone = magpy.getB(magnet, sensor)[2]

    out = {
        "samples": SAMPLES,
        "unknowns": unknowns,
        "seconds": round(seconds, 1),
        "nominal": nominal,
        "nominal_alone": nominal_alone,
        "dof": [f"{part} {kind} {'xy'[axis]}" for part, kind, axis in DOF],
        "share": share,
        "bz": np.round(bz, 7).tolist(),
        "alone": np.round(alone, 7).tolist(),
    }
    Path(__file__).with_name("tolerance.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
