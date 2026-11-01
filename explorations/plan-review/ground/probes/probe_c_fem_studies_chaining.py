"""Probe C: FemSolution on a tiny model, the studies, sensor paths inside the
model, and what happens when one solver's output is fed to another.

Same plate model as probes A and B. Correctness and API behaviour only; the
FEM's seconds are masked.
"""

import warnings

import magpylib as magpy

magpy.defaults.threads = 1

import numpy as np  # noqa: E402

from magpylib_material_response import to_json  # noqa: E402
from magpylib_material_response.convergence import compare, mesh_study  # noqa: E402
from magpylib_material_response.demag import apply_demag  # noqa: E402
from magpylib_material_response.face_charge import ChargeSheet, apply_face_charge  # noqa: E402
from magpylib_material_response.meshing import mesh_uniform  # noqa: E402

MM = 1e-3
np.set_printoptions(precision=4, suppress=True)


def model():
    plate = magpy.magnet.Cuboid(
        dimension=(10 * MM, 10 * MM, 2 * MM), polarization=(0, 0, 0), style_label="plate"
    )
    plate.susceptibility = 1000
    magnet = magpy.magnet.Cuboid(
        dimension=(5 * MM, 5 * MM, 5 * MM),
        polarization=(0, 0, 1.0),
        position=(0, 0, 4.5 * MM),
        style_label="magnet",
    )
    sensor = magpy.Sensor(position=(0, 0, 1.5 * MM), style_label="sensor")
    return magpy.Collection(plate, magnet, sensor, style_label="model"), plate, magnet, sensor


print("== a sensor path inside the model: accepted by both solvers ==")
coll, plate, magnet, sensor = model()
sensor.position = np.linspace((0, 0, 1.5 * MM), (3 * MM, 0, 1.5 * MM), 10)
fc = apply_face_charge(coll, level=2)
print("face charge: solved; getB(solved, its sensor) shape", magpy.getB(fc, fc.sensors_all[0]).shape)
mm = apply_demag(mesh_uniform(coll, 2))
print("method of moments: solved; getB shape", magpy.getB(mm, mm.sensors_all[0]).shape)

print("== feeding one solver's output to another ==")
coll, plate, magnet, sensor = model()
mesh = mesh_uniform(coll, cells_across=1)


def plate_of(c):
    return next(x for x in c.children if isinstance(x, magpy.Collection))


print("a cell mesh (cells_across=1):", len(plate_of(mesh).sources_all), "cells, sealed =", getattr(plate_of(mesh), "sealed", None))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fc_on_mesh = apply_face_charge(mesh, level=1)
print("apply_face_charge on the cell mesh: accepted; unknowns", fc_on_mesh.face_charge_info["unknowns"], "| sheets:", sum(isinstance(s, ChargeSheet) for s in fc_on_mesh.sources_all), "| plate collection sealed now:", getattr(plate_of(fc_on_mesh), "sealed", None), "| warnings:", [str(x.message)[:80] for x in w])
print("   B at the sensor mT:", magpy.getB(fc_on_mesh, sensor) * 1e3, "| the unmeshed plate solved at level 1:", magpy.getB(apply_face_charge(coll, level=1), sensor) * 1e3)
solved_fc = apply_face_charge(coll, level=2)
try:
    apply_demag(solved_fc)
except Exception as e:  # noqa: BLE001
    print("apply_demag on a face-charge result ->", type(e).__name__, ":", e)
try:
    mesh_uniform(solved_fc, 2)
except Exception as e:  # noqa: BLE001
    print("mesh_uniform on a face-charge result ->", type(e).__name__, ":", e)

print("== mesh_study on the plate ==")
coll, plate, magnet, sensor = model()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    study = mesh_study(coll, sensor, levels=(1, 2, 4))
print("type:", type(study).__name__, "| fields:", [f for f in study.__dataclass_fields__])
print("cells:", study.cells, "| values (mT):", study.values.reshape(len(study.cells), -1) * 1e3, "| order:", study.order, "| error:", study.error, "| last_change:", f"{study.last_change:.3%}", "| estimate mT:", study.estimate * 1e3)
print("warnings:", [str(x.message)[:120] for x in w])
print("the study holds no solved collection:", "solution" in study.__dataclass_fields__ or "solved" in study.__dataclass_fields__)
cmp_ = compare(study, study)
print("compare(study, study):", type(cmp_).__name__, "agreement", cmp_.agreement, "relative_difference", cmp_.relative_difference)

print("== finite elements on a tiny version (order 2, edge_layers 1) ==")
try:
    from magpylib_material_response.fem import FemSolution, fem_solve

    coll, plate, magnet, sensor = model()
    sol = fem_solve(coll, order=2, edge_layers=1, observers=sensor)
    print("type:", type(sol).__name__, "| isinstance FemSolution:", isinstance(sol, FemSolution), "| order", sol.order, "| ndof", sol.ndof, "| elements", sol.elements, "| seconds keys", sorted(sol.seconds))
    print("public attributes:", [a for a in dir(sol) if not a.startswith("_")])
    print("moment (A m2):", sol.moment)
    print("getB(sensor) mT:", sol.getB(sensor) * 1e3, "| getB(point) mT:", sol.getB((0, 0, 1.5 * MM)) * 1e3)
    print("getB inside the plate (0,0,0) mT:", sol.getB((0, 0, 0)) * 1e3, "| mu0 getH there mT:", magpy.mu_0 * sol.getH((0, 0, 0)) * 1e3)
    s2 = magpy.Sensor(position=np.linspace((0, 0, 1.5 * MM), (3 * MM, 0, 1.5 * MM), 5))
    print("getB on a 5-step sensor path shape:", sol.getB(s2).shape)
    for name, call in (
        ("magpy.getB(solution, sensor)", lambda: magpy.getB(sol, sensor)),
        ("magpy.show(solution)", lambda: magpy.show(sol, backend="plotly", return_fig=True)),
        ("to_json(solution)", lambda: to_json(sol)),
        ("solution.getJ", lambda: sol.getJ((0, 0, 0))),
    ):
        try:
            call()
            print(name, ": ok")
        except Exception as e:  # noqa: BLE001
            print(name, "->", type(e).__name__, ":", str(e)[:100])
    coll3, _, m3, _ = model()
    m3.position = np.linspace((0, 0, 4.5 * MM), (2 * MM, 0, 4.5 * MM), 3)
    try:
        fem_solve(coll3, order=1)
    except Exception as e:  # noqa: BLE001
        print("fem_solve with a magnet path ->", type(e).__name__, ":", e)
    coll4, p4, m4, _ = model()
    p4.H_ext = (0, 0, 0.005)
    m4.susceptibility = 0.05
    m4.H_ext = (0, 0, 0.0)
    try:
        fem_solve(coll4, order=1)
    except Exception as e:  # noqa: BLE001
        print("fem_solve with two H_ext values on responding parts ->", type(e).__name__, ":", e)
except ImportError as e:
    print("fem not available:", e)
