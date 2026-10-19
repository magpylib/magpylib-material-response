"""Same scalar-potential formulation as the scikit-fem spike, on NGSolve.

Lengths in mm (the equation is scale-free); fields in SI.
usage: ngs_cube.py <h_mm> <grading> <order> <mur>
"""

import sys
import time

import magpylib as magpy
import numpy as np
from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere
from ngsolve import (
    CF, H1, BilinearForm, CGSolver, GridFunction, Integrate, LinearForm, Mesh,
    Preconditioner, TaskManager, dx, ds, grad, sqrt, x, y, z,
)

MU0 = 1.25663706127e-6
h, grading, order, mur = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
a, R = 10.0, 100.0
J, B0 = (1.0, 0.0) if mur == 1 else (0.0, 0.01)  # magnet, or soft cube in 10 mT

t0 = time.perf_counter()
body = Box(Pnt(-a / 2, -a / 2, -a / 2), Pnt(a / 2, a / 2, a / 2)).mat("body")
body.maxh = h
air = Sphere(Pnt(0, 0, 0), R).mat("air")
air.faces.name = "outer"
mesh = Mesh(OCCGeometry(Glue([air - body, body])).GenerateMesh(maxh=R / 4, grading=grading))
mesh.Curve(order)
t1 = time.perf_counter()

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

head = f"h={h:.2f}mm grading={grading:.2f} p={order} mur={mur:g}  tets={mesh.ne:>7} dofs={fes.ndof:>8}  mesh {t1 - t0:4.1f}s solve {t2 - t1:4.1f}s  "
if mur == 1:
    d = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    X = np.linspace(7.5, 50, 30)[:, None] * d
    B = np.array([-MU0 * np.array(grad(psi)(mesh(*p))) for p in X])
    Bx = magpy.magnet.Cuboid(dimension=(a * 1e-3,) * 3, polarization=(0, 0, J)).getB(X * 1e-3)
    e = np.linalg.norm(B - Bx, axis=1) / np.linalg.norm(Bx, axis=1)
    print(head + f"worst pointwise {100 * e.max():.3f} %  near(7.5mm) {100 * e[0]:.3f} %  far(30-50mm) {100 * e[-12:].max():.3f} %")
else:
    m = Integrate((mur - 1) * (B0 / MU0 - grad(psi)[2]), mesh, definedon=mesh.Materials("body")) * 1e-9
    ref = a**3 * 1e-9 * B0 / MU0 / (1 / 3.6442 + 1 / (mur - 1))
    print(head + f"moment {m:.5f} vs {ref:.5f} A m^2: {100 * (m / ref - 1):+.2f} %")
