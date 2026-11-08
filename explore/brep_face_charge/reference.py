"""Finite-element reference of each part on its EXACT solid: the B-rep, not a
faceting. As validation/references.py makes one (orders 3, 4, 5 on a mesh
finer at the sensors, then the highest order on the default mesh; the error
is the larger of the two changes), except that the solver is handed the
OpenCASCADE solid itself, so netgen curves its elements onto the real
fillets and bore.

    python reference.py plate|pole      -> results/reference_<part>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from netgen import occ

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))  # the repository, for validation/

from geometry import EXACT, FIELD, MM, SENSORS, as_magnet, fine  # noqa: E402
from validation.references import FINER, SAFETY_FACTOR, _finer_at_sensors  # noqa: E402

from magpylib_material_response import UniformField  # noqa: E402
from magpylib_material_response.fem import _geometry, _solve, solve_fem, study_fem  # noqa: E402

EXACT_BY_LABEL = {}  # label -> OCC solid in mm; looked up, not carried, so copies keep it

_plain = _geometry.solid


def _exact_solid(src, center, length):
    exact = EXACT_BY_LABEL.get(src.style.label)
    if exact is None:
        return _plain(src, center, length)
    shape = exact.Scale(occ.Pnt(0, 0, 0), MM / length)
    shape = shape.Move(occ.Vec(*((np.ravel(src.position) - center) / length)))
    _geometry._mark_sharp(shape)  # noqa: SLF001
    return shape


_geometry.solid = _solve.solid = _exact_solid


def model(part):
    label = f"{part}, exact"
    exact = EXACT[part]()
    EXACT_BY_LABEL[label] = exact
    V, F = fine(part)
    return magpy.Collection(as_magnet(V, F, label=label), UniformField(B=FIELD))


def make(part, orders=(3, 4, 5)):
    collection = model(part)
    groups = SENSORS[part]
    sensors = np.concatenate(list(groups.values()))
    start = time.perf_counter()
    with _finer_at_sensors(FINER):
        study = study_fem(collection, sensors, orders=orders)
    coarse = solve_fem(collection, order=orders[-1], observers=sensors).getB(sensors)
    seconds = time.perf_counter() - start
    change = study.values[-1] - study.values[-2]
    out, first = {}, 0
    for group, points in groups.items():
        rows = slice(first, first + len(points))
        first += len(points)
        B = study.estimate[rows]
        size = np.linalg.norm(B)
        response = B - FIELD  # the part's own field
        by_order = SAFETY_FACTOR * np.linalg.norm(change[rows]) / size
        by_mesh = np.linalg.norm(coarse[rows] - B) / size
        out[group] = {
            "B": B.tolist(),
            "error": float(max(by_order, by_mesh)),
            "order_change": float(by_order),
            "mesh_change": float(by_mesh),
            # the same, relative to the part's own field: the harder measure
            "error_of_response": float(
                max(SAFETY_FACTOR * np.linalg.norm(change[rows]), np.linalg.norm(coarse[rows] - B))
                / np.linalg.norm(response)
            ),
        }
    data = {
        "part": part,
        "geometry": "exact B-rep (netgen.occ), curved elements",
        "orders": list(orders),
        "sensor_elements": FINER,
        "unknowns": study.ndofs.tolist(),
        "seconds": round(seconds, 1),
        "groups": out,
    }
    path = HERE / "results" / f"reference_{part}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    return data


if __name__ == "__main__":
    d = make(sys.argv[1])
    print(d["part"], d["seconds"], "s", d["unknowns"])
    for g, v in d["groups"].items():
        print(f"  {g}: error {v['error']:.2e} of B, {v['error_of_response']:.2e} of the part's own field "
              f"(order {v['order_change']:.1e}, mesh {v['mesh_change']:.1e})")
