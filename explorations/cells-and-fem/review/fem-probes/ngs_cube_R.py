"""Copy of S/ngs_cube.py with air radius R and outer maxh as parameters (review probe only).
usage: ngs_cube_R.py <h_mm> <grading> <order> <mur> <R_mm> <maxh_outer_mm>
"""
import sys
import time

import magpylib as magpy
import numpy as np
from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere
from ngsolve import (
    CF, H1, VOL, BilinearForm, CGSolver, GridFunction, Integrate, LinearForm, Mesh,
    Preconditioner, TaskManager, dx, ds, grad, sqrt, x, y, z,
)

MU0 = 1.25663706127e-6
h, grading, order, mur = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
R, maxh_out = float(sys.argv[5]), float(sys.argv[6])
a = 10.0
J, B0 = (1.0, 0.0) if mur == 1 else (0.0, 0.01)

t0 = time.perf_counter()
body = Box(Pnt(-a / 2, -a / 2, -a / 2), Pnt(a / 2, a / 2, a / 2)).mat("body")
body.maxh = h
air = Sphere(Pnt(0, 0, 0), R).mat("air")
air.faces.name = "outer"
mesh = Mesh(OCCGeometry(Glue([air - body, body])).GenerateMesh(maxh=maxh_out, grading=grading))
mesh.Curve(order)
t1 = time.perf_counter()
nbody = sum(1 for el in mesh.Elements(VOL) if el.mat == "body")

fes = H1(mesh, order=order)
u, v = fes.TnT()
mu = mesh.MaterialCF({"body": mur}, default=1)
M = CF((0, 0, J / MU0 + (mur - 1) * B0 / MU0))
lhs = BilinearForm(mu * grad(u) * grad(v) * dx + 2 / sqrt(x * x + y * y + z * z) * u * v * ds("outer"))
rhs = LinearForm(M * grad(v) * dx("body"))
pre = Preconditioner(lhs, "bddc")
psi = GridFunction(fes)
with TaskManager():
    lhs.Assemble()
    rhs.Assemble()
    psi.vec.data = CGSolver(lhs.mat, pre.mat, tol=1e-12, maxiter=1000) * rhs.vec
t2 = time.perf_counter()

head = f"R={R:g} maxh={maxh_out:g} h={h:.2f} g={grading:.2f} p={order} mur={mur:g} tets={mesh.ne} (body {nbody}) dofs={fes.ndof} mesh {t1 - t0:.1f}s solve {t2 - t1:.1f}s  "
if mur == 1:
    d = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    r = np.linspace(7.5, 50, 30)
    X = r[:, None] * d
    B = np.array([-MU0 * np.array(grad(psi)(mesh(*p))) for p in X])
    Bx = magpy.magnet.Cuboid(dimension=(a * 1e-3,) * 3, polarization=(0, 0, J)).getB(X * 1e-3)
    e = np.linalg.norm(B - Bx, axis=1) / np.linalg.norm(Bx, axis=1)
    print(head + f"worst {100 * e.max():.3f} % at r={r[e.argmax()]:.1f}  near {100 * e[0]:.3f}  mid(10-30) {100 * e[(r > 10) & (r < 30)].max():.3f}  far(30-50) {100 * e[-12:].max():.3f} %")
    print("   e% per probe:", " ".join(f"{100 * v:.3f}" for v in e))
else:
    m = Integrate((mur - 1) * (B0 / MU0 - grad(psi)[2]), mesh, definedon=mesh.Materials("body")) * 1e-9
    ref = a**3 * 1e-9 * B0 / MU0 / (1 / 3.6442 + 1 / (mur - 1))
    print(head + f"moment {m:.6f} vs {ref:.6f}: {100 * (m / ref - 1):+.3f} %")
