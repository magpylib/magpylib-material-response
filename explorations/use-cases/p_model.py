"""The tolerance Monte Carlo's model of s2 (a): a magnet, a fixed source, over a
soft plate, and a sensor, with the tolerances of each sample."""

import magpylib as magpy
import numpy as np
from common import mm
from scipy.spatial.transform import Rotation as R

SENSOR = np.array([3, 0, 1.5]) * mm


def draw(samples, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "magnet_xy": rng.normal(0, 0.1 * mm, (samples, 2)),
        "magnet_tilt": rng.normal(0, 1.0, (samples, 2)),
        "sensor_xy": rng.normal(0, 0.1 * mm, (samples, 2)),
        "sensor_tilt": rng.normal(0, 1.0, (samples, 2)),
    }


def plate(length=10 * mm):
    p = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(length, 10 * mm, 2 * mm))
    p.susceptibility = 1000
    return p


def model(d, i):
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * mm, 5 * mm, 5 * mm),
        position=(*d["magnet_xy"][i], 4.5 * mm),
        orientation=R.from_euler("xy", d["magnet_tilt"][i], degrees=True),
    )
    return magpy.Collection(magnet, plate())


def sensor_at(d, i):
    return magpy.Sensor(
        position=SENSOR + np.array([*d["sensor_xy"][i], 0]),
        orientation=R.from_euler("xy", d["sensor_tilt"][i], degrees=True),
    )
