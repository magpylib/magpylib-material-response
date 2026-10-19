"""Thin soft-iron plate 40 x 40 x 1 mm, chi = 1000, 10 mT along x, on NGSolve.

Lengths in mm (the equation is scale-free); fields in SI.
usage: ngs_plate.py <h_mm> <grading> <order>
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
h, grading, order = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
mur, L, T, R, B0 = 1001.0, 40.0, 1.0, 400.0, 0.01

t0 = time.perf_counter()
body = Box(Pnt(-L / 2, -L / 2, -T / 2), Pnt(L / 2, L / 2, T / 2)).mat("body")
body.maxh = h
air = Sphere(Pnt(0, 0, 0), R).mat("air")
air.faces.name = "outer"
mesh = Mesh(OCCGeometry(Glue([air - body, body])).GenerateMesh(maxh=R / 4, grading=grading))
mesh.Curve(order)
t1 = time.perf_counter()

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
Bx = -MU0 * grad(psi)(mesh(L / 2 + 2, 0, 0))[0] * 1e3
print(f"h={h:.2f}mm grading={grading:.2f} p={order}  tets={mesh.ne:>7} dofs={fes.ndof:>8}  mesh {t1 - t0:4.1f}s solve {t2 - t1:4.1f}s  m_x={m:.6f} A m^2  Bx(+2mm)={Bx:.4f} mT")
