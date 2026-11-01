"""Probe, values only, no timing: the tolerance Monte Carlo's plate model
(p_model.py on origin/explore/use-cases). Bz at the sensor and its slope per
mm of magnet shift, from magpylib alone, from face charge solved at each
position, and from one solve kept while the magnet is moved afterwards in
magpylib. Then two API checks: a sensor path on a solved collection, and the
path refusals of apply_face_charge and apply_demag."""

import warnings

import magpylib as magpy
import numpy as np

magpy.defaults.threads = 1
warnings.filterwarnings("ignore")

from magpylib_material_response.demag import apply_demag  # noqa: E402
from magpylib_material_response.face_charge import apply_face_charge  # noqa: E402
from magpylib_material_response.meshing import mesh_Cuboid  # noqa: E402

mm = 1e-3
SENSOR = np.array([3, 0, 1.5]) * mm
H = 0.05 * mm  # half-step of the central difference in the magnet's x


def magnet(x=0.0):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(5 * mm,) * 3, position=(x, 0, 4.5 * mm)
    )


def plate():
    p = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(10 * mm, 10 * mm, 2 * mm))
    p.susceptibility = 1000
    return p


def bz_alone(x):
    return magpy.getB(magnet(x), SENSOR)[2]


def solve(x, level):
    return apply_face_charge(magpy.Collection(magnet(x), plate()), level=level)


def bz_solved(x, level):
    return magpy.getB(solve(x, level), SENSOR)[2]


def bz_moved_after(solved, x):
    """The solve kept as it is; only the magnet moved, in magpylib."""
    c = solved.copy()
    mags = [s for s in c.sources_all if isinstance(s, magpy.magnet.Cuboid)]
    assert len(mags) == 1, "the plate should have become a ChargeSheet"
    mags[0].position = (x, 0, 4.5 * mm)
    return magpy.getB(c, SENSOR)[2]


# T/m equals mT/mm numerically
print("Bz at the sensor (3, 0, 1.5) mm and its slope per mm of magnet shift:")
s_alone = (bz_alone(H) - bz_alone(-H)) / (2 * H)
print(f"  magpylib alone, no plate      : {bz_alone(0) * 1e3:8.2f} mT  {s_alone:8.2f} mT/mm")

for level in (3, 4):
    solved = solve(0.0, level)
    n = solved.face_charge_info["unknowns"]
    b = magpy.getB(solved, SENSOR)[2]
    s = (bz_solved(H, level) - bz_solved(-H, level)) / (2 * H)
    print(f"  face charge level {level}, {n:5d} unknowns, a solve per position:"
          f" {b * 1e3:8.2f} mT  {s:8.2f} mT/mm")
    if level == 3:
        s_kept = (bz_moved_after(solved, H) - bz_moved_after(solved, -H)) / (2 * H)
        print(f"  level 3 solved once, magnet moved afterwards in magpylib:"
              f" {b * 1e3:8.2f} mT  {s_kept:8.2f} mT/mm")
        print(f"  the plate's share: {(b - bz_alone(0)) * 1e3:.2f} mT of Bz,"
              f" {s - s_alone:.2f} mT/mm of the slope")
        solved3 = solved
        sheet = next(s for s in solved.sources_all if type(s).__name__ == "ChargeSheet")
        print(f"  the solved plate at level 3: mean polarization {np.linalg.norm(sheet.mean_polarization):.3f} T"
              f" (saturation check, bulk average only), jump indicator {sheet.jump_indicator:.2e}")

print()
print("API checks on main:")
rng = np.random.default_rng(1)
sensor = magpy.Sensor(position=SENSOR + np.c_[rng.normal(0, 0.1 * mm, (100, 2)), np.zeros(100)])
B = magpy.getB(solved3, sensor)
print(f"  getB of the solved collection at a sensor path of 100 poses: shape {B.shape},"
      f" Bz spread {np.ptp(B[:, 2]) * 1e3:.1f} mT")

m = magnet()
m.position = [(0, 0, 4.5 * mm), (H, 0, 4.5 * mm)]
try:
    apply_face_charge(magpy.Collection(m, plate()))
    print("  apply_face_charge accepted a magnet path (unexpected)")
except Exception as e:  # noqa: BLE001
    print(f"  apply_face_charge on a magnet path: {type(e).__name__}: {e}")
m2 = magnet()
m2.position = [(0, 0, 4.5 * mm), (H, 0, 4.5 * mm)]
try:
    apply_demag(magpy.Collection(m2, mesh_Cuboid(plate(), target_elems=8)))
    print("  apply_demag accepted a magnet path (unexpected)")
except Exception as e:  # noqa: BLE001
    print(f"  apply_demag on a magnet path: {type(e).__name__}: {e}")
