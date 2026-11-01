"""Probe A: what apply_face_charge returns on the brief's plate model.

A 5 mm cube magnet 1 mm above a 10 x 10 x 2 mm plate of chi = 1000, a sensor in
the gap. Correctness and API behaviour only; no timing (seconds are masked).
"""

import warnings

import magpylib as magpy

magpy.defaults.threads = 1

import numpy as np  # noqa: E402

from magpylib_material_response import from_json, to_json  # noqa: E402
from magpylib_material_response.face_charge import (  # noqa: E402
    ChargeSheet,
    apply_face_charge,
)

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


def hide_seconds(info):
    return {k: ("<masked>" if k == "seconds" else v) for k, v in info.items()}


coll, plate, magnet, sensor = model()
print("== input ==")
print("children:", [type(c).__name__ for c in coll.children])
print(
    "sources_all:",
    [type(c).__name__ for c in coll.sources_all],
    "| sensors_all:",
    [type(c).__name__ for c in coll.sensors_all],
)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    solved = apply_face_charge(coll, level=2)
print("warnings during the solve:", [str(x.message) for x in w])

print("== result ==")
print("type:", type(solved).__name__, "| is the input object:", solved is coll)
print("face_charge_info:", hide_seconds(solved.face_charge_info))
print("sealed:", getattr(solved, "sealed", "<no attribute>"))
print("children:", [(type(c).__name__, c.style.label) for c in solved.children])
print("sensors kept:", [type(c).__name__ for c in solved.sensors_all])
print("input plate untouched:", type(coll.children[0]).__name__, "| input magnet:", type(coll.children[1]).__name__)
sheet = next(s for s in solved.sources_all if isinstance(s, ChargeSheet))
print("sheet MRO:", [k.__name__ for k in type(sheet).__mro__[:4]])
print(
    "sheet vertices", sheet.vertices.shape, "faces", sheet.faces.shape,
    "charge_densities", sheet.charge_densities.shape,
    "panels", None if sheet.panels is None else len(np.unique(sheet.panels)),
    "triangles", sheet.triangles.shape,
)
print("unknowns by hand, 3 per panel:", 3 * len(np.unique(sheet.panels)))
print("sheet.mean_polarization (T):", sheet.mean_polarization, "| norm:", f"{np.linalg.norm(sheet.mean_polarization):.4f} T")
print("sheet.jump_indicator:", f"{sheet.jump_indicator:.3%}", "| panel_indicator shape:", sheet.panel_indicator().shape)
print("sheet.style.charge.level (A/m):", sheet.style.charge.level)
print("sheet.susceptibility:", getattr(sheet, "susceptibility", "<no attribute>"), "| sheet.H_ext:", getattr(sheet, "H_ext", "<no attribute>"))
print("sheet.position:", sheet.position, "| orientation rotvec:", sheet.orientation.as_rotvec())
print(
    "sheet has .volume:", hasattr(sheet, "volume"), "| .moment:", hasattr(sheet, "moment"),
    "| .getB:", hasattr(sheet, "getB"), "| .polarization:", hasattr(sheet, "polarization"),
    "| .dimension:", hasattr(sheet, "dimension"),
)
T = sheet.triangles
vol = abs(np.einsum("fd,fd->f", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()) / 6
print(
    f"volume from the sheet's triangles: {vol:.4e} m3 (the plate: 2.0000e-07); "
    f"moment by hand J V / mu0 = {np.linalg.norm(sheet.mean_polarization) * vol / magpy.mu_0:.4e} A m2"
)

print("== fields ==")
B_solved = magpy.getB(solved, sensor)
print("magpy.getB(solved, sensor) mT:", B_solved * 1e3)
try:
    solved.getB(sensor)
except Exception as e:  # noqa: BLE001
    print("solved.getB(sensor) ->", type(e).__name__, ":", str(e)[:130])
print("solved.getB() with no inputs, its own sensor, equal:", np.allclose(solved.getB(), B_solved))
print("magnet alone at the sensor mT:", magpy.getB(magnet, sensor) * 1e3)
print("sheet alone at the sensor mT:", magpy.getB(sheet, sensor) * 1e3)
inside = (0, 0, 0)
B_in, H_in = magpy.getB(solved, inside), magpy.getH(solved, inside)
J_in, M_in = magpy.getJ(solved, inside), magpy.getM(solved, inside)
print("inside the plate (0,0,0): B mT", B_in * 1e3, "| mu0 H mT", magpy.mu_0 * H_in * 1e3, "| J", J_in, "| M", M_in)
print(
    "   B == mu0 H inside:", np.allclose(B_in, magpy.mu_0 * H_in),
    f"(the part's mean polarization, {np.linalg.norm(sheet.mean_polarization):.3f} T, is missing from B)",
)
below = (0, 0, 0.7 * MM)  # 0.3 mm below the top face
print("0.3 mm below the top face: B mT", magpy.getB(solved, below) * 1e3)
s2 = magpy.Sensor(position=np.linspace((0, 0, 1.5 * MM), (3 * MM, 0, 1.5 * MM), 5))
print("observer path of 5 steps, getB shape:", magpy.getB(solved, s2).shape)
sheet_copy = sheet.copy()
sheet_copy.position = np.linspace((0, 0, 0), (1 * MM, 0, 0), 3)
print("a sheet given its own 3-step path, getB shape:", magpy.getB(sheet_copy, sensor).shape)

print("== show ==")
fig = magpy.show(solved, backend="plotly", return_fig=True)
print("plotly traces:", len(fig.data), [t.type for t in fig.data])
import matplotlib  # noqa: E402

matplotlib.use("Agg")
fig2 = magpy.show(solved, backend="matplotlib", return_fig=True)
print("matplotlib figure:", type(fig2).__name__)

print("== serialization ==")
s = to_json(solved)
back = from_json(s)[0]
print("json chars:", len(s), "| round-trip type:", type(back).__name__, "| children:", [type(c).__name__ for c in back.children])
print("face_charge_info survives:", getattr(back, "face_charge_info", None), "| sealed:", getattr(back, "sealed", "<none>"))
bs = next(x for x in back.sources_all if isinstance(x, ChargeSheet))
print("style.charge.level survives:", bs.style.charge.level, "| mean_polarization equal:", np.allclose(bs.mean_polarization, sheet.mean_polarization))
print("B after the round trip equal:", np.allclose(magpy.getB(back, sensor), B_solved))
print("sensor survives:", [type(c).__name__ for c in back.sensors_all])
try:
    to_json(sheet)
    print("to_json(sheet) alone: ok")
except Exception as e:  # noqa: BLE001
    print("to_json(sheet) alone ->", type(e).__name__, e)

print("== re-solve trap: move the magnet inside the solved collection, solve again ==")
moved = solved.copy()
m2 = next(s for s in moved.sources_all if isinstance(s, magpy.magnet.Cuboid))
m2.position = (2 * MM, 0, 4.5 * MM)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    again = apply_face_charge(moved, level=2)
print("warnings:", [str(x.message)[:100] for x in w])
print("info:", hide_seconds(again.face_charge_info))
fresh_coll, _, fresh_magnet, _ = model()
fresh_magnet.position = (2 * MM, 0, 4.5 * MM)
fresh = apply_face_charge(fresh_coll, level=2)
sens = magpy.Sensor(position=(0, 0, 1.5 * MM))
print("B from re-solving the solved copy mT:", magpy.getB(again, sens) * 1e3)
print("B from a fresh solve of the moved model mT:", magpy.getB(fresh, sens) * 1e3)

print("== refusals and fallbacks ==")
c3, _, m3, _ = model()
m3.position = np.linspace((0, 0, 4.5 * MM), (2 * MM, 0, 4.5 * MM), 3)
try:
    apply_face_charge(c3)
except Exception as e:  # noqa: BLE001
    print("magnet with a 3-step path ->", type(e).__name__, ":", e)
c4, p4, _, _ = model()
p4.susceptibility = (1000, 1000, 10)
try:
    apply_face_charge(c4)
except Exception as e:  # noqa: BLE001
    print("anisotropic chi ->", type(e).__name__, ":", str(e)[:140])
c5 = model()[0]
try:
    apply_face_charge(c5, level=3, max_unknowns=1000)
except Exception as e:  # noqa: BLE001
    print("level=3 over max_unknowns=1000 ->", type(e).__name__, ":", str(e)[:200])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r6 = apply_face_charge(model()[0], max_unknowns=1000)
print("no level, max_unknowns=1000 -> info:", hide_seconds(r6.face_charge_info), "| warning:", [str(x.message)[:140] for x in w])
c7, p7, m7, s7 = model()
m7.susceptibility = 0.05  # a recoil magnet: solved too
r7 = apply_face_charge(c7, level=2)
print("magnet with chi=0.05: children", [(type(c).__name__, c.style.label) for c in r7.children], "| unknowns", r7.face_charge_info["unknowns"])
c8 = model()[0]
c8.H_ext = (0, 0, 0.005)  # a stray field on the whole model
r8 = apply_face_charge(c8, level=2)
print("H_ext=5 mT on the collection: B at the sensor mT", magpy.getB(r8, sensor) * 1e3, "(field excludes H_ext itself)")

print("== unknowns and B per level ==")
for lv in (1, 2, 3):
    r = apply_face_charge(model()[0], level=lv)
    print(f"level {lv}: unknowns {r.face_charge_info['unknowns']}, B at the sensor mT {magpy.getB(r, sensor) * 1e3}, jump {next(s for s in r.sources_all if isinstance(s, ChargeSheet)).jump_indicator:.2%}")
