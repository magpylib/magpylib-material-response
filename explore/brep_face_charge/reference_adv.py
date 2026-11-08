"""Exact finite-element references for the adversarial scenarios
(scenarios.py), made as reference.py makes the first ones. The part's own
field is B minus that of the other sources, which every error is also
measured against.

    python reference_adv.py plate|pole   -> results/adv_reference_<part>.json
"""

from __future__ import annotations

import json
import sys
import time

import magpylib as magpy
import numpy as np

import reference as R  # the exact-solid patch, applied on import
from geometry import EXACT, as_magnet, fine
from scenarios import scenarios
from validation.references import FINER, SAFETY_FACTOR, _finer_at_sensors

from magpylib_material_response.fem import solve_fem, study_fem


def make(part, orders=(3, 4, 5)):
    label = f"{part}, exact"
    R.EXACT_BY_LABEL[label] = EXACT[part]()
    V, F = fine(part)
    out = {}
    for name, (others, groups) in scenarios(part).items():
        collection = magpy.Collection(as_magnet(V, F, label=label), *others)  # others: fresh objects per scenario
        sensors = np.concatenate(list(groups.values()))
        background = magpy.getB(list(others), sensors, sumup=True)
        t0 = time.perf_counter()
        with _finer_at_sensors(FINER):
            study = study_fem(collection, sensors, orders=orders)
        coarse = solve_fem(collection, order=orders[-1], observers=sensors).getB(sensors)
        change = study.values[-1] - study.values[-2]
        res, first = {}, 0
        for group, points in groups.items():
            rows = slice(first, first + len(points))
            first += len(points)
            B = study.estimate[rows]
            own = np.linalg.norm(B - background[rows])
            res[group] = {
                "B": B.tolist(),
                "background": background[rows].tolist(),
                "error_of_own_field": float(
                    max(SAFETY_FACTOR * np.linalg.norm(change[rows]), np.linalg.norm(coarse[rows] - B)) / own
                ),
            }
        out[name] = {"seconds": round(time.perf_counter() - t0, 1), "groups": res}
        print(part, name, out[name]["seconds"], "s:",
              {g: f"{100 * v['error_of_own_field']:.3f} %" for g, v in res.items()}, flush=True)
    path = R.HERE / "results" / f"adv_reference_{part}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    make(sys.argv[1])
