"""Solve side: the point-rule MoM operator of the U-core as FMM (cells as dipoles) + exact near correction.

usage: fmm_solve.py <tetgen_target> [near multiples of the max edge, default 3] [dense: 0|1]
Compares the matvec with #52's dense operator (if dense=1), then solves with GMRES and checks the peak field
against the stock solution of the same mesh.
"""

import json
import sys
import time

import fmm3dpy
import magpylib as magpy
import numpy as np
import pymagba.fields as pf
import scipy.sparse as sp
from adapt import CHI, COIL, LINE
from adapt_tet import tetgen_mesh
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial import cKDTree

MU0 = magpy.mu_0
target = int(sys.argv[1])
NEAR = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
DENSE = len(sys.argv) > 3 and sys.argv[3] == "1"

V = tetgen_mesh(target)
n = len(V)
cent = V.mean(1)
vol = np.abs(np.linalg.det(V[:, 1:] - V[:, :1])) / 6
edges = np.array([np.linalg.norm(V[:, a] - V[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)]).max(0)
out = {"cells": n, "near_mult": NEAR, "max_edge_mm": round(edges.max() * 1e3, 2)}

# ---- near pairs and the correction C = exact - dipole (self: exact)
t0 = time.perf_counter()
tree = cKDTree(cent)
balls = tree.query_ball_point(cent, r=NEAR * edges, workers=-1)  # d < NEAR * e_i, own size
ii = np.repeat(np.arange(n), [len(b) for b in balls]); jj = np.concatenate([np.asarray(b) for b in balls])
key = np.unique(np.concatenate([ii * n + jj, jj * n + ii]))  # symmetrize: d < NEAR * max(e_i, e_j); includes self
I, Jc = key // n, key % n  # observer, source
npair = len(I)
obs = cent[I]
T_ex = np.empty((npair, 3, 3))  # mu0 H at obs from source cell with unit polarization e_m, column m
for s0 in range(0, npair, 300_000):
    sl = slice(s0, s0 + 300_000)
    for m in range(3):
        pol = np.zeros((len(obs[sl]), 3))
        pol[:, m] = 1
        T_ex[sl, :, m] = magpy.func.tetrahedron_field("H", obs[sl], V[Jc[sl]], pol) * MU0
r = obs - cent[Jc]
d = np.linalg.norm(r, axis=1)
nonself = I != Jc
rh = r[nonself] / d[nonself, None]
D = vol[Jc[nonself], None, None] / (4 * np.pi * d[nonself, None, None] ** 3) * (3 * rh[:, :, None] * rh[:, None, :] - np.eye(3))
C_blocks = T_ex.copy()
C_blocks[nonself] -= D
rows = (3 * I[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2).ravel()
cols = (3 * Jc[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1).ravel()
C = sp.csr_matrix((C_blocks.ravel(), (rows, cols)), shape=(3 * n, 3 * n))
out.update({"near_pairs_per_cell": round(npair / n, 1), "near_build_s": round(time.perf_counter() - t0, 2), "C_MB": round(C.data.nbytes * 1.5 / 1e6, 1)})

EPS = 1e-8


def mu0H(Jflat):
    Jv = Jflat.reshape(n, 3)
    o = fmm3dpy.lfmm3d(eps=EPS, sources=cent.T.copy(), dipvec=(Jv * vol[:, None] / MU0).T.copy(), pg=2)
    return (-MU0 * o.grad.T).ravel() + C @ Jflat


# ---- matvec check against #52's dense operator
if DENSE:
    from magpylib_material_response.demag import demag_tensor

    cells = [magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V]
    t0 = time.perf_counter()
    T4 = demag_tensor(cells) * MU0
    Td = T4.swapaxes(2, 3).reshape((3 * n, 3 * n)).T  # #52 layout, vectors in F order x1..xn, y1..yn, z1..zn
    t_dense = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    Jr = rng.standard_normal((n, 3))
    t0 = time.perf_counter()
    ref = Td @ Jr.reshape(-1, order="F")
    t_dmv = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = mu0H(Jr.ravel()).reshape(n, 3).reshape(-1, order="F")
    t_fmv = time.perf_counter() - t0
    out["matvec"] = {"dense_assembly_s": round(t_dense, 2), "dense_matvec_s": round(t_dmv, 4), "fmm_matvec_s": round(t_fmv, 3),
                     "rel_err": float(np.abs(got - ref).max() / np.abs(ref).max()), "rms_rel_err": float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2)))}
    del Td, T4

# ---- GMRES solve of (I - chi T) J = chi B_ext, block-Jacobi preconditioner from the self blocks
B_ext = COIL.getB(cent)
rhs = (CHI * B_ext).ravel()
self_blocks = T_ex[~nonself]  # (n, 3, 3), in cell order
Minv = np.linalg.inv(np.eye(3) - CHI * self_blocks)
A = LinearOperator((3 * n, 3 * n), matvec=lambda x: x - CHI * mu0H(x), dtype=float)
M = LinearOperator((3 * n, 3 * n), matvec=lambda x: np.einsum("nab,nb->na", Minv, x.reshape(n, 3)).ravel(), dtype=float)
it = [0]
t0 = time.perf_counter()
Jsol, info = gmres(A, rhs, M=M, rtol=1e-8, restart=60, maxiter=400, callback=lambda r: it.__setitem__(0, it[0] + 1), callback_type="pr_norm")
t_solve = time.perf_counter() - t0
J = Jsol.reshape(n, 3)
B = np.zeros_like(LINE)
for j in range(n):
    B += pf.tetrahedron_B(LINE, polarization=J[j], vertices=V[j].tolist())
Bz = (B + COIL.getB(LINE))[:, 2]
out.update({"gmres_info": int(info), "gmres_iters": it[0], "solve_s": round(t_solve, 2), "peak_mT": float(np.abs(Bz).max() * 1e3)})
print(json.dumps(out), flush=True)
