"""Adversarial scenarios: conditions chosen to break the first result.

- dense sensors all round the curved faces, 1 mm off them (or 0.5 mm in the
  hole), not four chosen points;
- three drives: the first run's oblique 10 mT, 10 mT along z, 10 mT along x;
- a 4 mm cube magnet, 1 T, its nearest corner 1 mm off a curved face, no
  applied field.

Each scenario: (sources besides the part, sensors by group). Lengths mm here.
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np

from geometry import FIELD, MM

from magpylib_material_response import UniformField

DRIVES = {
    "oblique": FIELD,
    "along z": np.array([0, 0, 0.01]),
    "along x": np.array([0.01, 0, 0]),
}


def dense(part):
    """Sensors all round the curved faces: 'near the curves' (mm)."""
    pts = []
    if part == "plate":
        for z in (-1.5, 0, 1.5):  # in the hole, 0.5 mm off its wall
            pts += [(3 + np.cos(a), np.sin(a), z) for a in np.radians(np.arange(0, 360, 60))]
        centres = [(7, 3), (-7, 3), (-7, -3), (7, -3)]
        for k, (cx, cy) in enumerate(centres):  # 1 mm off each fillet, mid-arc
            a = np.pi / 4 + k * np.pi / 2
            pts += [(cx + 4 * np.cos(a), cy + 4 * np.sin(a), z) for z in (-1.5, 0, 1.5)]
    else:
        for b in np.radians([22.5, 45, 67.5]):  # 1 mm off the fillet
            r, z = 7 + 4 * np.cos(b), 2 + 4 * np.sin(b)
            pts += [(r * np.cos(a), r * np.sin(a), z) for a in np.radians(np.arange(0, 360, 60))]
        for z in (-3, 0):  # 1 mm off the side
            pts += [(11 * np.cos(a), 11 * np.sin(a), z) for a in np.radians([30, 120, 210, 300])]
    return {
        "near the curves": np.array(pts) * MM,
        "60 mm away": np.array([(60, 0, 0), (0, 0, 60)]) * MM,
    }


def magnet(part):
    """A 4 mm cube, 1 T toward the part, a corner pointing at a curve's centre
    and 1 mm off it, between the dense sensors (not on one)."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    d = 2 * 2**0.5  # half the cube's diagonal in its own plane
    if part == "plate":  # the (+x, +y) fillet, at 20 degrees on its arc
        a = np.radians(20)
        u = np.array([np.cos(a), np.sin(a), 0])
        corner = np.array([7, 3, 0]) + 4 * u
        centre = corner + d * u
        rot = R.from_euler("z", 20 - 225, degrees=True)  # local (1, 1) corner along -u, at the part
        pol = (1 / 2**0.5, 1 / 2**0.5, 0)  # local, toward the part once turned
    else:  # the rim fillet at 45 degrees on its profile, azimuth 30 degrees
        b = np.radians(45)
        prof = np.array([np.cos(b), 0, np.sin(b)])
        corner = np.array([7, 0, 2]) + 4 * prof
        centre = corner + 2 * np.array([1, 0, 1])  # local (-1, *, -1) edge toward the fillet
        rot = R.from_euler("z", 30, degrees=True)
        corner, centre = rot.apply(corner), rot.apply(centre)
        pol = (-1 / 2**0.5, 0, -1 / 2**0.5)
    m = magpy.magnet.Cuboid(polarization=pol, dimension=(4 * MM,) * 3,
                            position=centre * MM, orientation=rot, style_label="magnet")
    return m


def scenarios(part):
    """name -> (list of other sources, sensors by group)."""
    out = {name: ([UniformField(B=B)], dense(part)) for name, B in DRIVES.items()}
    out["magnet 1 mm off a curve"] = ([magnet(part)], dense(part))
    return out
