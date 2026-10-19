"""Emulate #70: cuboid cells whose polarization varies linearly inside them.

A linear cell (12 unknowns: J = a + G xi, xi in [-1, 1]^3) is represented by s^3 uniform
sub-cuboids carrying the linear function at their centres; its field is their sum.
usage: p1emul.py check
       p1emul.py cube <n> <basis> <colloc>         basis p0|p1, colloc centre|lsq8|tet4
       p1emul.py ucore <h_mm> <basis> <colloc>
"""

import json
import sys
import time

import magpylib as magpy
import numpy as np

MU0 = magpy.mu_0
S = int(__import__("os").environ.get("S", 5))  # sub-cells per side; odd, so the centre collocation point avoids every face
XI_Q = ((np.arange(S) + 0.5) / S * 2 - 1)
XI_Q = np.stack(np.meshgrid(XI_Q, XI_Q, XI_Q, indexing="ij"), -1).reshape(-1, 3)
COLLOC = {
    "centre": np.zeros((1, 3)),
    "lsq8": np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * 0.4,
    "tet4": np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) * 0.4,
}


def P(xi, nb):
    """(k, 3, nb): polarization at xi per unit coefficient; u = [a(3), G row-major(9)]."""
    M = np.zeros((len(xi), 3, 12))
    M[:, [0, 1, 2], [0, 1, 2]] = 1
    for r in range(3):
        M[:, r, 3 + 3 * r : 6 + 3 * r] = xi
    return M[:, :, :nb]


def h_unit(X, subpos, subdim):
    """mu0 H at X from each sub-cuboid with unit polarization: (len(X), 3, Q, 3)."""
    p, q = len(X), len(subpos)
    obs, pos = np.repeat(X, q, axis=0), np.tile(subpos, (p, 1))
    dims = np.broadcast_to(subdim, (p * q, 3))
    out = np.empty((p, 3, q, 3))
    for m in range(3):
        pol = np.zeros((p * q, 3))
        pol[:, m] = 1
        H = magpy.func.cuboid_field("H", obs, dims, pol, positions=pos)
        out[:, :, :, m] = H.reshape(p, q, 3).transpose(0, 2, 1)
    return out * MU0


def solve(pos, dim, chi, b_ext, basis, colloc):
    """b_ext(X) -> mu0 H_applied (T) at points. Returns u (n, nb) and the residual norm."""
    n, nb, xi_p = len(pos), (3 if basis == "p0" else 12), COLLOC[colloc]
    npc = len(xi_p)
    X = (pos[:, None, :] + xi_p[None] * dim[:, None, :] / 2).reshape(-1, 3)
    Pq, Pp = P(XI_Q, nb), P(xi_p, nb)
    A = np.zeros((len(X) * 3, n * nb))
    for j in range(n):
        F = h_unit(X, pos[j] + XI_Q * dim[j] / 2, dim[j] / S)
        A[:, j * nb : (j + 1) * nb] = -chi * np.einsum("pcqm,qmb->pcb", F, Pq).reshape(-1, nb)
    for i in range(n):
        A[i * npc * 3 : (i + 1) * npc * 3, i * nb : (i + 1) * nb] += Pp.reshape(-1, nb)
    b = chi * b_ext(X).reshape(-1)
    u = np.linalg.solve(A, b) if A.shape[0] == A.shape[1] else np.linalg.lstsq(A, b, rcond=None)[0]
    return u.reshape(n, nb)


def sub_sources(pos, dim, u):
    """All sub-cuboids with their polarization: positions, dims, pols."""
    nb = u.shape[1]
    J = np.einsum("qcb,nb->nqc", P(XI_Q, nb), u)
    subpos = (pos[:, None, :] + XI_Q[None] * dim[:, None, :] / 2).reshape(-1, 3)
    subdim = np.repeat(dim / S, len(XI_Q), axis=0)
    return subpos, subdim, J.reshape(-1, 3)


def grid(d, c, n):
    cell = np.array(d) / n
    axes = [np.linspace(-dd / 2 + s / 2, dd / 2 - s / 2, k) for dd, s, k in zip(d, cell, n)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c
    return g, np.tile(cell, (len(g), 1))


def cube(n, basis, colloc):
    a, chi, B0 = 0.01, 1000.0, np.array([0, 0, 0.01])
    pos, dim = grid((a, a, a), (0, 0, 0), (n, n, n))
    t0 = time.perf_counter()
    u = solve(pos, dim, chi, lambda X: np.broadcast_to(B0, X.shape), basis, colloc)
    m = (u[:, :3] * np.prod(dim, 1)[:, None]).sum(0) / MU0
    return {"case": "cube", "cells": len(pos), "unknowns": u.size, "basis": basis, "colloc": colloc,
            "moment": m[2], "err_pct": 100 * (m[2] / 0.02889 - 1), "s": round(time.perf_counter() - t0, 1)}


def ucore(h, basis, colloc):
    from adapt import BLOCKS, CHI, COIL, LINE
    parts = [grid(d, c, np.maximum(np.round(np.array(d) / h).astype(int), 1)) for d, c in BLOCKS]
    pos, dim = np.vstack([p for p, _ in parts]), np.vstack([d for _, d in parts])
    t0 = time.perf_counter()
    u = solve(pos, dim, CHI, lambda X: COIL.getB(X), basis, colloc)
    t = time.perf_counter() - t0
    sp, sd, sJ = sub_sources(pos, dim, u)
    B = np.array([magpy.func.cuboid_field("B", np.broadcast_to(x, sp.shape), sd, sJ, positions=sp).sum(0) for x in LINE])
    Bz = (B + COIL.getB(LINE))[:, 2]
    peak = float(np.abs(Bz).max() * 1e3)
    return {"case": "ucore", "cells": len(pos), "unknowns": u.size, "basis": basis, "colloc": colloc,
            "peak_mT": peak, "err_pct": 100 * (peak / 2.153 - 1), "s": round(t, 1)}


if __name__ == "__main__":
    if sys.argv[1] == "check":  # p0 + centre must equal apply_demag exactly
        from magpylib_material_response.demag import apply_demag
        r = cube(3, "p0", "centre")
        pos, dim = grid((0.01,) * 3, (0, 0, 0), (3, 3, 3))
        coll = magpy.Collection(*(magpy.magnet.Cuboid(dimension=d, position=p, polarization=(0, 0, 0)) for p, d in zip(pos, dim)))
        coll.susceptibility, coll.H_ext = 1000.0, (0, 0, 0.01)
        ref = sum(s.polarization[2] * s.volume for s in apply_demag(coll).sources_all) / MU0
        print(f"cube 27 cells: emulated p0 moment {r['moment']:.10f}, apply_demag {ref:.10f}, rel diff {abs(r['moment'] / ref - 1):.1e}")
        print(json.dumps(ucore(0.01, "p0", "centre")))
    elif sys.argv[1] == "cube":
        print(json.dumps(cube(int(sys.argv[2]), sys.argv[3], sys.argv[4])), flush=True)
    else:
        print(json.dumps(ucore(float(sys.argv[2]) * 1e-3, sys.argv[3], sys.argv[4])), flush=True)
