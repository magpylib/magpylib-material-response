"""Copy of S/ngs_plate.py with mesh diagnostics (review probe only).
usage: ngs_plate_diag.py <h_mm> <grading> <order> [R_mm]
"""
import sys
import time

import numpy as np
from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere
from ngsolve import (
    CF, H1, VOL, BilinearForm, CGSolver, GridFunction, Integrate, LinearForm, Mesh,
    Preconditioner, TaskManager, dx, ds, grad, sqrt, x, y, z,
)

MU0 = 1.25663706127e-6
h, grading, order = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
R = float(sys.argv[4]) if len(sys.argv) > 4 else 400.0
mur, L, T, B0 = 1001.0, 40.0, 1.0, 0.01

t0 = time.perf_counter()
body = Box(Pnt(-L / 2, -L / 2, -T / 2), Pnt(L / 2, L / 2, T / 2)).mat("body")
body.maxh = h
air = Sphere(Pnt(0, 0, 0), R).mat("air")
air.faces.name = "outer"
mesh = Mesh(OCCGeometry(Glue([air - body, body])).GenerateMesh(maxh=R / 4, grading=grading))
mesh.Curve(order)
t1 = time.perf_counter()
nbody = sum(1 for el in mesh.Elements(VOL) if el.mat == "body")
pts = np.array([v.point for v in mesh.vertices])
inner = ((np.abs(pts[:, 0]) < L / 2 - 1e-9) & (np.abs(pts[:, 1]) < L / 2 - 1e-9) & (np.abs(pts[:, 2]) < T / 2 - 1e-9)).sum()
onface = ((np.abs(pts[:, 0]) < L / 2 - 1e-9) & (np.abs(pts[:, 1]) < L / 2 - 1e-9) & (np.abs(np.abs(pts[:, 2]) - T / 2) < 1e-9)).sum()

fes = H1(mesh, order=order)
u, v = fes.TnT()
mu = mesh.MaterialCF({"body": mur}, default=1)
M = CF(((mur - 1) * B0 / MU0, 0, 0))
lhs = BilinearForm(mu * grad(u) * grad(v) * dx + 2 / sqrt(x * x + y * y + z * z) * u * v * ds("outer"))
rhs = LinearForm(M * grad(v) * dx("body"))
pre = Preconditioner(lhs, "bddc")
psi = GridFunction(fes)
with TaskManager():
    lhs.Assemble()
    rhs.Assemble()
    psi.vec.data = CGSolver(lhs.mat, pre.mat, tol=1e-12, maxiter=1000) * rhs.vec
t2 = time.perf_counter()

m = Integrate((mur - 1) * (B0 / MU0 - grad(psi)[0]), mesh, definedon=mesh.Materials("body")) * 1e-9
print(f"R={R:g} h={h:.2f} g={grading:.2f} p={order} tets={mesh.ne} (body {nbody}; vertices strictly inside thickness {inner}, on top/bottom faces {onface}) dofs={fes.ndof} mesh {t1 - t0:.1f}s solve {t2 - t1:.1f}s  m_x={m:.6f} A m^2")
