"""What goes wrong if a path passes the "rigid" test and everything is kept from
its first step: the prototype kept.py solving a magnet that moves near a plate,
against a full solve at each position, at sensors around the plate. A magnet
within 1 % of the parts' size (here 0.025 mm) grades the plate's panels toward
its footprint; within a panel's size it takes a finer load rule. Also a solved
part's own polarization per step: kept from the first step, then per step.

    python c7_frozen_layout.py
"""

import sys

import magpylib as magpy
import numpy as np
from common import done, mm, write_result
from kept import Kept

from magpylib_material_response.face_charge import apply_face_charge

NAME = "c7_frozen_layout"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")
SENSORS = np.array([[7, 0, 1.5], [0, 0, -1.5], [3, 0, -1.5]]) * mm
PLATE_TOP = 1.0  # mm


def magnet(x, gap):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(5 * mm,) * 3,
        position=(x * mm, 0, (PLATE_TOP + gap + 2.5) * mm),
    )


def plate(J=(0, 0, 0), chi=1000):
    p = magpy.magnet.Cuboid(polarization=J, dimension=(10 * mm, 10 * mm, 2 * mm))
    p.susceptibility = chi
    return p


def full(m, p):
    solved = apply_face_charge(magpy.Collection(m, p))
    return magpy.getB(solved, SENSORS), solved.face_charge_info["unknowns"]


def frozen(kept, m, plate_J_load=None):
    """The kept solver at a new position: B at the sensors."""

    def field(P, _sl):
        return (magpy.getB(m, P) / magpy.mu_0)[None]

    R = kept.loads(field, 1)
    if plate_J_load is not None:
        R = R + plate_J_load
    x = kept.solve(R)
    X = SENSORS
    plate_B = magpy.mu_0 * kept.field(X, np.repeat(x, len(X), axis=0))
    return plate_B + magpy.getB(m, X)


rows = []


def path_of(name, poses):
    kept = Kept(magpy.Collection(magnet(*poses[0]), plate()))
    base, n0 = full(magnet(*poses[0]), plate())
    prev = base
    for pose in poses:
        truth, n = full(magnet(*pose), plate())
        ours = frozen(kept, magnet(*pose))
        rows.append(
            {
                "case": name,
                "x_mm": pose[0],
                "gap_mm": pose[1],
                "unknowns_first": n0,
                "unknowns_here": n,
                "error_mT": float(np.linalg.norm(ours - truth, axis=1).max() * 1e3),
                "error_share_of_B": float(
                    (
                        np.linalg.norm(ours - truth, axis=1)
                        / np.linalg.norm(truth, axis=1)
                    ).max()
                ),
                "move_changes_B_mT": float(
                    np.linalg.norm(truth - prev, axis=1).max() * 1e3
                ),
            }
        )
        prev = truth
        print(rows[-1], flush=True)


path_of("slides, far: 1 mm off", [(x, 1.0) for x in (0, 1, 2, 3)])
path_of("slides, 0.1 mm off: only the load rule", [(x, 0.1) for x in (0, 1, 2, 3)])
path_of(
    "slides, 0.01 mm off: its footprint grades the panels",
    [(x, 0.01) for x in (0, 1, 2, 3)],
)
path_of("comes down from 1 mm", [(0, g) for g in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)])

# a solved part's own polarization: its load is the remanence's, in proportion
chi = 0.05
kept = Kept(magpy.Collection(magnet(0, 1.0), plate((0, 0, 0.2), chi)))
body = kept.bodies[0]
unit = body.load_remanence() / body.chi / body.J0[2]
extra = np.zeros(kept.size)
polar = []
for Jz in (0.2, 0.5, -1.0):
    truth, _ = full(magnet(0, 1.0), plate((0, 0, Jz), chi))
    as_first = frozen(kept, magnet(0, 1.0))
    extra[: len(unit)] = (Jz - 0.2) * unit
    per_step = frozen(kept, magnet(0, 1.0), extra[None])
    polar.append(
        {
            "Jz_T": Jz,
            "kept_from_first_step_mT": float(
                np.linalg.norm(as_first - truth, axis=1).max() * 1e3
            ),
            "per_step_mT": float(np.linalg.norm(per_step - truth, axis=1).max() * 1e3),
        }
    )
    print(polar[-1], flush=True)

write_result(NAME, {"rows": rows, "solved_part_polarization": polar})
