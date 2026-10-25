"""Soft sphere (radius a, chi) in a uniform field H0 z: total scalar potential,
air sphere of radius R with phi = -H0 z on it. Exact outside: dipole with
m = 4 pi a^3 chi/(chi+3) H0. Error of B at points 1.5a to 3a from the centre."""
import time
import numpy as np
import ngsolve as ng
from netgen.occ import OCCGeometry, Pnt, Sphere, Glue

ng.ngsglobals.msg_level = 0
a, R, chi, H0 = 1.0, 20.0, 1000.0, 1.0
pts = np.array([[0, 0, 1.5], [1.5, 0, 0], [1.2, 1.2, 1.2], [0, 0, 3.0], [2.0, 0, 1.0]])

def exact_H(x):
    m = 4 * np.pi * a**3 * chi / (chi + 3) * H0 * np.array([0, 0, 1.0])
    r = np.linalg.norm(x, axis=1)[:, None]
    n = x / r
    return np.array([0, 0, H0]) + (3 * n * (n @ m)[:, None] - m) / (4 * np.pi * r**3)

for maxh_in, order in ((0.5, 2), (0.25, 2), (0.5, 3), (0.25, 3), (0.5, 4), (0.35, 4)):
    t0 = time.perf_counter()
    outer = Sphere(Pnt(0, 0, 0), R); outer.faces.name = "outer"
    iron = Sphere(Pnt(0, 0, 0), a); iron.mat("iron"); iron.faces.name = "iface"; iron.maxh = maxh_in
    near = Sphere(Pnt(0, 0, 0), 4 * a) - iron; near.mat("air"); near.maxh = maxh_in
    far = outer - Sphere(Pnt(0, 0, 0), 4 * a); far.mat("air")
    mesh = ng.Mesh(OCCGeometry(Glue([far, near, iron])).GenerateMesh(maxh=4.0, grading=0.3))
    mesh.Curve(order)
    t1 = time.perf_counter()
    fes = ng.H1(mesh, order=order, dirichlet="outer")
    u, v = fes.TnT()
    mu = mesh.MaterialCF({"iron": 1 + chi}, default=1.0)
    A = ng.BilinearForm(mu * ng.grad(u) * ng.grad(v) * ng.dx)
    pre = ng.Preconditioner(A, "bddc")
    A.Assemble()
    gf = ng.GridFunction(fes)
    gf.Set(-H0 * ng.z, ng.BND)
    rhs = -(A.mat * gf.vec)
    inv = ng.CGSolver(A.mat, pre.mat, tol=1e-12, maxiter=2000)
    gf.vec.data += inv * rhs
    t2 = time.perf_counter()
    H = -ng.grad(gf)
    Hp = np.array([H(mesh(*p)) for p in pts])
    t3 = time.perf_counter()
    err = np.linalg.norm(Hp - exact_H(pts), axis=1) / np.linalg.norm(exact_H(pts), axis=1)
    print(f"maxh {maxh_in} order {order}: {fes.ndof:7d} dofs, mesh {t1-t0:5.2f} s, solve {t2-t1:5.2f} s, eval {1e3*(t3-t2)/len(pts):.2f} ms/pt, max err {err.max():.1e}", flush=True)
