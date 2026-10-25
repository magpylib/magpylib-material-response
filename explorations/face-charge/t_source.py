"""Checks of source.py's ChargeSheet: against magpylib's own Triangle, against
the solver it is built from, and as a magpylib source (paths, show).
Prints one line per check; a few seconds."""
import os
import sys
import time

import magpylib as magpy
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "validation"):
    sys.path.insert(0, os.path.join(HERE, sub))
from source import ChargeSheet  # noqa: E402
import fc  # noqa: E402

rng = np.random.default_rng(3)
V = np.array([[0.0, 0.0, 0.0], [0.004, 0.001, 0.0], [0.001, 0.003, 0.0005]])  # a tilted triangle (m)
n = np.cross(V[1] - V[0], V[2] - V[0])
n /= np.linalg.norm(n)
cen = V.mean(0)
obs = np.concatenate([cen + rng.normal(size=(40, 3)) * 0.01, cen + rng.normal(size=(40, 3)) * 0.0008])


def one(V, q, **kw):
    """A sheet of one face."""
    return ChargeSheet(V, [[0, 1, 2]], [q], **kw)


def rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()


# 1. uniform charge = magpylib's Triangle, sigma = J.n / mu0
J = np.array([0.3, -0.2, 0.9])
sigma = J @ n / magpy.mu_0
src = one(V, [sigma] * 3)
ref = magpy.misc.Triangle(vertices=V, polarization=J)
print(f"uniform charge vs magpy.misc.Triangle: H {rel(src.getH(obs), ref.getH(obs)):.1e}, B {rel(src.getB(obs), ref.getB(obs)):.1e}")

# 2. linear charge = limit of uniform-charge pieces at their centroids
q = np.array([1e5, -3e4, 6e4])
src = one(V, q)
far = obs[:40]
H = src.getH(far)
for k in (3, 4, 5):
    bary = np.eye(3)[None]
    for _ in range(k):
        a, b, c = bary[:, 0], bary[:, 1], bary[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        bary = np.concatenate([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    pieces = [magpy.misc.Triangle(vertices=bv @ V, polarization=magpy.mu_0 * (bv.mean(0) @ q) * n) for bv in bary]
    print(f"linear charge vs {len(pieces):4d} uniform pieces: {rel(H, magpy.getH(pieces, far, sumup=True)):.1e}")

# 3. a magpylib collection solved by face charge, as magpylib sources
soft = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.004), position=(0, 0, 0), polarization=(0, 0, 0))
soft.susceptibility = 1000.0
soft.H_ext = (0.0, 0.0, 0.005)
magnet = magpy.magnet.Cuboid(dimension=(0.004, 0.004, 0.003), position=(0, 0, 0.0045), polarization=(0, 0, 1.2))
magnet.susceptibility = 0.0  # exactly 0: stays a magpylib Cuboid, drawn as one
coll = magpy.Collection(soft, magnet)
sensors = np.array([[0.0, 0.0, 0.009], [0.008, 0.0, 0.0], [0.0, 0.0, -0.005], [0.02, 0.01, 0.01]])
B_fc, info = fc.solve(coll, 3, sensors)
solved = fc.apply_face_charge(coll, 3)
kinds = sorted(type(s).__name__ for s in solved.sources_all)
sheet = next(s for s in solved.sources_all if isinstance(s, ChargeSheet))
print(f"apply_face_charge: {kinds}; getB vs fc.solve {rel(solved.getB(sensors), B_fc):.1e}; input untouched: {coll.sources_all[0] is soft}")
print(f"the soft cuboid's sheet: {len(sheet.faces)} faces on {len(sheet.vertices)} vertices (corners shared, {3 * len(sheet.faces)} unshared); "
      f"jump indicator {sheet.jump_indicator:.2%}, fc.solve reports {info['jump_indicator']:.2%}")
B_r, info_r = fc.solve(coll, 3, sensors, refine=2)
sizes = " -> ".join(str(r["unknowns"]) for r in info_r["rounds"])
jumps = " -> ".join(f"{r['jump_indicator']:.2%}" for r in info_r["rounds"])
print(f"refine=2: unknowns {sizes}, jump indicator {jumps}; apply_face_charge's getB vs fc.solve "
      f"{rel(fc.apply_face_charge(coll, 3, refine=2).getB(sensors), B_r):.1e}")
cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.002
cube_faces = [[0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6]]
Jc = np.array([0.2, -0.5, 1.1])
nrm = np.cross(cube[cube_faces][:, 1] - cube[cube_faces][:, 0], cube[cube_faces][:, 2] - cube[cube_faces][:, 0])
uniform = ChargeSheet(cube, cube_faces, np.repeat((nrm / np.linalg.norm(nrm, axis=1)[:, None] @ Jc / magpy.mu_0)[:, None], 3, axis=1))
print(f"mean_polarization: a uniformly magnetized cube's {rel(uniform.mean_polarization, Jc):.0e}; the soft cuboid's "
      f"{np.linalg.norm(sheet.mean_polarization) * 1e3:.1f} mT, as fc.solve reports {info['polarization'][0] * 1e3:.1f} mT")

# 4. paths and orientation behave as for any magpylib source
p = one(V, q, position=[(0, 0, 0), (0.01, 0.0, 0.002)], orientation=R.from_euler("z", [[0], [30]], degrees=True))
x = np.array([0.012, 0.004, 0.003])
r1 = R.from_euler("z", 30, degrees=True)
manual = r1.apply(one(V, q).getH(r1.inv().apply(x - np.array([0.01, 0.0, 0.002]))))
print(f"path of 2 positions, turned 30 deg: getH shape {p.getH(x).shape}, second step vs by hand {rel(p.getH(x)[1], manual):.1e}")

# copy() rebinds the field to the copy; two sources in one getB keep their own charge
a = one(V, q)
b = a.copy()
b.charge_densities = [2 * q]
Ha, Hb = a.getH(far), b.getH(far)
both = magpy.getH([a, b], far, sumup=True)
print(f"copy with doubled charge: {rel(Hb, 2 * Ha):.1e}; original kept: {rel(a.getH(far), H):.1e}; both in one getB: {rel(both, 3 * Ha):.1e}")

# 5. show() draws it coloured by the charge, and a big source evaluates in reasonable time
pv.OFF_SCREEN = True
for backend in ("plotly", "matplotlib", "pyvista"):
    fig = magpy.show(solved, backend=backend, return_fig=True)
mesh = next(t for t in magpy.show(sheet, backend="plotly", return_fig=True).data if t.type == "mesh3d")
q = sheet._smoothed()
moved = np.abs(q - sheet.charge_densities.ravel()).max() / np.abs(sheet.charge_densities).max()
print(f"show(): plotly, matplotlib and pyvista draw it; plotly colours by the square root of the smoothed charge: "
      f"{rel(mesh.intensity, np.sign(q) * np.sqrt(np.abs(q))):.0e}; smoothing moves the charge by up to {moved:.1%} of its peak")
grid = [t for t in magpy.show(sheet, backend="plotly", style_mesh_grid_show=True, return_fig=True).data if t.type == "scatter3d"]
F = sheet.faces
tri_edges = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
print(f"style.mesh.grid: {len(grid[0].x) // 3} edges drawn, the {tri_edges} triangle edges less one diagonal per panel ({len(F) // 2})")
X = rng.normal(size=(1000, 3)) * 0.02 + 0.03
t = time.perf_counter()
sheet.getB(X)
print(f"getB of {len(sheet.faces)} faces at 1,000 observers: {time.perf_counter() - t:.2f} s")
