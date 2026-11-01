"""Probe B: what mesh_uniform and apply_demag return on the brief's plate model.

Same model as probe A. Correctness and API behaviour only; no timing.
"""

import warnings

import magpylib as magpy

magpy.defaults.threads = 1

import numpy as np  # noqa: E402

from magpylib_material_response import (  # noqa: E402
    configure_logging,
    disable_logging,
    from_json,
    to_json,
)
from magpylib_material_response.demag import apply_demag  # noqa: E402
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


def plate_of(coll):
    """The meshed plate: the one child that is a Collection."""
    return next(x for x in coll.children if isinstance(x, magpy.Collection))


def magnet_of(coll):
    return next(x for x in coll.children if isinstance(x, magpy.magnet.Cuboid))


coll, plate, magnet, sensor = model()
meshed = mesh_uniform(coll, cells_across=2)
print("== mesh ==")
print("type:", type(meshed).__name__, "| is the input:", meshed is coll)
print("children order, input:", [c.style.label for c in coll.children], "| meshed:", [(type(c).__name__, c.style.label) for c in meshed.children])
plate_mesh = plate_of(meshed)
cells = list(plate_mesh.sources_all)
print("plate mesh: sealed =", getattr(plate_mesh, "sealed", "<none>"), "| cells:", len(cells), type(cells[0]).__name__, "| cell size mm:", cells[0].dimension / MM)
print("cell.susceptibility:", getattr(cells[0], "susceptibility", "<none>"), "| plate_mesh.susceptibility:", getattr(plate_mesh, "susceptibility", "<none>"), "| plate_mesh.H_ext:", getattr(plate_mesh, "H_ext", "<none>"))
print("magnet in the mesh:", type(magnet_of(meshed)).__name__, "| its susceptibility:", getattr(magnet_of(meshed), "susceptibility", "<none>"))
print("sensors in the mesh:", [type(c).__name__ for c in meshed.sensors_all])
print("input plate untouched:", type(coll.children[0]).__name__)
print("meshed collection has face_charge_info:", hasattr(meshed, "face_charge_info"))

log = []
configure_logging(sink=lambda m: log.append(m.record["message"]), min_log_time=0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    solved = apply_demag(meshed)
disable_logging()
print("== result ==")
print("warnings:", [str(x.message)[:100] for x in w])
print("log lines naming the solver:", [line for line in log if "auto" in line or "Cell structure" in line])
print("type:", type(solved).__name__, "| is the input:", solved is meshed, "| info attributes:", [a for a in ("face_charge_info", "demag_info", "info") if hasattr(solved, a)])
sp = plate_of(solved)
print("solved plate: sealed =", getattr(sp, "sealed", "<none>"), "| cells:", len(sp.sources_all), "| first cell polarization (T):", sp.sources_all[0].polarization)
print("input mesh cells still at zero polarization:", np.allclose([c.polarization for c in plate_of(meshed).sources_all], 0))
print("solved cells keep susceptibility:", getattr(sp.sources_all[0], "susceptibility", "<none>"))
J = np.array([c.orientation.apply(c.polarization) for c in sp.sources_all])
vols = np.array([c.volume for c in sp.sources_all])
print(f"moment by hand (A m2): {(vols[:, None] * J).sum(0) / magpy.mu_0} | mean J (T): {(vols[:, None] * J).sum(0) / vols.sum()} | max |J| of a cell (T): {np.linalg.norm(J, axis=1).max():.3f}")

print("== fields ==")
B = magpy.getB(solved, sensor)
print("B at the sensor mT:", B * 1e3, "| magnet alone:", magpy.getB(magnet, sensor) * 1e3)
inside = (0, 0, 0)
B_in, H_in, J_in = magpy.getB(solved, inside), magpy.getH(solved, inside), magpy.getJ(solved, inside)
print("inside the plate (0,0,0): B mT", B_in * 1e3, "| mu0 H mT", magpy.mu_0 * H_in * 1e3, "| J T", J_in, "| B == mu0 H + J:", np.allclose(B_in, magpy.mu_0 * H_in + J_in))
below = (0, 0, 0.7 * MM)
print("0.3 mm below the top face: B mT", magpy.getB(solved, below) * 1e3)
s2 = magpy.Sensor(position=np.linspace((0, 0, 1.5 * MM), (3 * MM, 0, 1.5 * MM), 5))
print("observer path of 5 steps, getB shape:", magpy.getB(solved, s2).shape)

print("== show ==")
fig = magpy.show(solved, backend="plotly", return_fig=True)
print("plotly traces:", len(fig.data))

print("== serialization ==")
s = to_json(solved)
back = from_json(s)[0]
print("json chars:", len(s), "| children:", [type(c).__name__ for c in back.children], "| plate sealed:", getattr(plate_of(back), "sealed", "<none>"), "| cell susceptibility:", getattr(plate_of(back).sources_all[0], "susceptibility", "<none>"), "| collection susceptibility:", getattr(plate_of(back), "susceptibility", "<none>"))
print("B after the round trip equal:", np.allclose(magpy.getB(back, sensor), B))

print("== re-solve trap: apply_demag on an already solved mesh ==")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    twice = apply_demag(solved)
print("B solved once mT:", B * 1e3, "| solved twice:", magpy.getB(twice, sensor) * 1e3, "| warnings:", [str(x.message)[:80] for x in w])

print("== inplace ==")
m2 = mesh_uniform(model()[0], cells_across=2)
r = apply_demag(m2, inplace=True)
print("returns:", r, "| the input's cells are now polarized:", not np.allclose([c.polarization for c in plate_of(m2).sources_all], 0))

print("== direct vs auto ==")
d = apply_demag(meshed, solver="direct")
print("direct B mT:", magpy.getB(d, sensor) * 1e3, "| largest difference to auto (T):", f"{np.abs(magpy.getB(d, sensor) - B).max():.2e}")

print("== refusals and what is accepted ==")
c3, _, m3, _ = model()
m3.position = np.linspace((0, 0, 4.5 * MM), (2 * MM, 0, 4.5 * MM), 3)
try:
    apply_demag(mesh_uniform(c3, cells_across=2))
except Exception as e:  # noqa: BLE001
    print("magnet with a 3-step path ->", type(e).__name__, ":", e)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    u = apply_demag(model()[0])
print("unmeshed model accepted: the plate is one cell; B at the sensor mT", magpy.getB(u, sensor) * 1e3, "| warnings:", [str(x.message)[:80] for x in w])
c5 = model()[0]
c5.add(magpy.misc.Dipole(moment=(0, 0, 1e-3), position=(0, 0, 10 * MM)))
try:
    apply_demag(mesh_uniform(c5, 2))
except Exception as e:  # noqa: BLE001
    print("a Dipole source ->", type(e).__name__, ":", e)
c6 = model()[0]
c6.children[0].susceptibility = 0
try:
    mesh_uniform(c6, 2)
except Exception as e:  # noqa: BLE001
    print("no responding part ->", type(e).__name__, ":", e)
c9, p9, _, _ = model()
p9.susceptibility = (1000, 1000, 10)
r9 = apply_demag(mesh_uniform(c9, 2))
print("anisotropic chi accepted: B at the sensor mT", magpy.getB(r9, sensor) * 1e3)
c8 = model()[0]
c8.H_ext = (0, 0, 0.005)
r8 = apply_demag(mesh_uniform(c8, 2))
print("H_ext=5 mT on the collection: B at the sensor mT", magpy.getB(r8, sensor) * 1e3)

print("== cells and B per cells_across ==")
for ca in (1, 2, 4):
    m = mesh_uniform(model()[0], cells_across=ca)
    r = apply_demag(m)
    print(f"cells_across {ca}: cells {len(plate_of(m).sources_all)}, B at the sensor mT {magpy.getB(r, sensor) * 1e3}")
