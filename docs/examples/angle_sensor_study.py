"""The model and the stored results of `angle_sensor.md`: a gradiometric angle
sensor over a magnet on a steel shaft, in a uniform stray field, after Infineon's
US 12,237,730 B2 (shaft 20 mm across, mu_r 4000, a 5 mT stray field, a 1.5 mm
air gap). The magnet and the soft holder ring around the sensor are stand-ins.
With the holder, every angle is a new solve, about three minutes in all here, so
the results are stored in `angle_sensor.json`.

    uv run --group docs python docs/examples/angle_sensor_study.py

Run it after a change to `solve_face_charge`, and commit the JSON.
"""

import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

from magpylib_material_response import UniformField, solve_face_charge

MM = 1e-3  # lengths in metres, written in millimetres
CHI = 3999  # the shaft's steel, mu_r 4000
MAGNET_TOP = 3.0  # mm above the shaft's end
ANGLES = np.linspace(0, 360, 12, endpoint=False)
GAPS = (1.5, 2.5)  # mm between the magnet and the sensor
STRAYS = (0.0, 5e-3)  # T, along x


def angle_sensor(angle, gap, stray, holder=False):
    """The shaft and its magnet turned by `angle` degrees, the stray field, and,
    with `holder`, a soft ring around the sensor, which stays put."""
    shaft = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(20 * MM, 40 * MM), position=(0, 0, -20 * MM)
    )
    shaft.susceptibility = CHI
    magnet = magpy.magnet.Cylinder(  # 6 x 3 mm, diametric
        polarization=(0.5, 0, 0), dimension=(6 * MM, 3 * MM), position=(0, 0, 1.5 * MM)
    )
    rotor = magpy.Collection(shaft, magnet)
    rotor.rotate(Rotation.from_euler("z", angle, degrees=True), anchor=0)
    parts = [rotor, UniformField(B=(stray, 0, 0))]
    if holder:
        ring = magpy.magnet.CylinderSegment(
            polarization=(0, 0, 0),
            dimension=(3 * MM, 8 * MM, 4 * MM, 0, 360),
            position=(0, 0, (MAGNET_TOP + gap + 1) * MM),
        )
        ring.susceptibility = CHI
        parts.append(ring)
    return magpy.Collection(*parts)


def hall_plates(gap):
    """Four Hall plates on a 1 mm circle, `gap` mm above the magnet."""
    z = (MAGNET_TOP + gap) * MM
    return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]) * MM + (0, 0, z)


def angle_error(bz, angle):
    """The angle the plates read, atan2 of their two differences, less the
    true one, in degrees: a uniform Bz cancels in each difference."""
    read = np.degrees(np.arctan2(bz[2] - bz[3], bz[0] - bz[1]))
    return (read - angle + 180) % 360 - 180


def main():
    out = {"angles": ANGLES.tolist(), "runs": []}
    for holder in (False, True):
        solved = True
        for gap in GAPS:
            for stray in STRAYS:
                errors, seconds = [], []
                for angle in ANGLES:
                    model = angle_sensor(angle, gap, stray, holder)
                    t0 = time.perf_counter()
                    # without the holder the shaft is the only solved part and
                    # turns as a whole: one factorization serves every solve;
                    # with it, the shaft turns against the ring
                    solved = solve_face_charge(
                        model, level=2, reuse=False if holder else solved
                    )
                    seconds.append(time.perf_counter() - t0)
                    bz = magpy.getB(solved, hall_plates(gap))[:, 2]
                    errors.append(angle_error(bz, angle))
                out["runs"].append(
                    {
                        "holder": holder,
                        "gap": gap,
                        "stray": stray,
                        "errors": [float(e) for e in errors],
                        "seconds": round(float(np.sum(seconds)), 2),
                        "unknowns": solved.face_charge_info["unknowns"],
                    }
                )
    Path(__file__).with_name("angle_sensor.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
