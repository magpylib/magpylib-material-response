"""Material response with real linear-cell kernels (#70), timed end to end.

usage: p1real.py cube <n> <basis>     basis p0 (centre collocation) | p1 (8 points, least squares)
       p1real.py ucore <h_mm> <basis>
"""

import json
import sys
import time

import magpylib as magpy
import numpy as np
from lincell import tensor
from p1emul import COLLOC, P, grid

MU0 = magpy.mu_0


def assemble_solve(pos, dim, chi, b_ext, basis, chunk_pairs=400_000):
    n, half = len(pos), dim / 2
    linear = basis == "p1"
    import os
    mode = os.environ.get("COLLOC", "lsq8")
    g = 1 / np.sqrt(3)
    gauss8 = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * g
    nb, xi_p = (12, gauss8 if mode == "galerkin" else COLLOC["lsq8"]) if linear else (3, COLLOC["centre"])
    npc = len(xi_p)
    X = (pos[:, None, :] + xi_p[None] * half[:, None, :]).reshape(-1, 3)
    A = np.empty((len(X), 3, n, nb))
    step = max(1, chunk_pairs // n)
    for s in range(0, len(X), step):
        rel = X[s : s + step, None, :] - pos[None]
        T = tensor(rel.reshape(-1, 3), np.broadcast_to(half, rel.shape).reshape(-1, 3), linear=linear)
        A[s : s + step] = -chi * T.reshape(len(rel), n, 3, nb).transpose(0, 2, 1, 3)
    A = A.reshape(len(X) * 3, n * nb)
    Pp = P(xi_p, nb).reshape(-1, nb)
    for i in range(n):
        A[i * npc * 3 : (i + 1) * npc * 3, i * nb : (i + 1) * nb] += Pp
    b = chi * b_ext(X).reshape(-1)
    if linear and mode == "galerkin":  # project each cell's 24 residuals onto its 4 shape functions x 3 components
        psi = np.concatenate([np.ones((8, 1)), xi_p], axis=1)  # (8 points, 4 functions), equal Gauss weights
        Wc = np.kron(psi.T, np.eye(3))  # (12, 24)
        A = np.einsum("ab,nbk->nak", Wc, A.reshape(n, 24, -1)).reshape(n * 12, -1)
        b = np.einsum("ab,nb->na", Wc, b.reshape(n, 24)).reshape(-1)
    if A.shape[0] == A.shape[1]:
        u = np.linalg.solve(A, b)
    else:  # least squares through the normal equations
        u = np.linalg.solve(A.T @ A, A.T @ b)
    return u.reshape(n, nb)


def cube(n, basis):
    pos, dim = grid((0.01,) * 3, (0, 0, 0), (n, n, n))
    t0 = time.perf_counter()
    u = assemble_solve(pos, dim, 1000.0, lambda X: np.broadcast_to([0, 0, 0.01], X.shape), basis)
    t = time.perf_counter() - t0
    m = (u[:, 2] * np.prod(dim, 1)).sum() / MU0
    return {"case": "cube", "basis": basis, "cells": len(pos), "unknowns": u.size, "err_pct": 100 * (m / 0.02889 - 1), "s": round(t, 2)}


def ucore(h, basis):
    from adapt import BLOCKS, CHI, COIL, LINE
    parts = [grid(d, c, np.maximum(np.round(np.array(d) / h).astype(int), 1)) for d, c in BLOCKS]
    pos, dim = np.vstack([p for p, _ in parts]), np.vstack([d for _, d in parts])
    t0 = time.perf_counter()
    u = assemble_solve(pos, dim, CHI, COIL.getB, basis)
    t = time.perf_counter() - t0
    line = LINE + (0, 1e-6, 0)  # off the edge-line extensions the probe spacing hits (the corner formulas are 0/0 there)
    rel = line[:, None, :] - pos[None]
    T = tensor(rel.reshape(-1, 3), np.broadcast_to(dim / 2, rel.shape).reshape(-1, 3), linear=basis == "p1")
    B = np.einsum("lncb,nb->lc", T.reshape(len(line), len(pos), 3, -1), u) + COIL.getB(line)
    peak = float(np.abs(B[:, 2]).max() * 1e3)
    return {"case": "ucore", "basis": basis, "cells": len(pos), "unknowns": u.size, "peak_mT": peak,
            "err_pct": 100 * (peak / 2.153 - 1), "s": round(t, 2)}


if __name__ == "__main__":
    f = cube if sys.argv[1] == "cube" else ucore
    arg = int(sys.argv[2]) if sys.argv[1] == "cube" else float(sys.argv[2]) * 1e-3
    print(json.dumps(f(arg, sys.argv[3])), flush=True)
