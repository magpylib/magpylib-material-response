"""Material response with linear tetrahedra, timed end to end; plus the cost per pair.

usage: p1tet.py bench
       p1tet.py ucore <tetgen_target> <basis>    basis p0 (centroid) | p1 (8 child centroids, least squares)
       p1tet.py cube <tetgen_target> <basis>
"""

import itertools
import json
import os
import sys
import time

import magpylib as magpy
import numpy as np
from lintet import tensor
from p1emul import P

MU0 = magpy.mu_0


def child_centroids(V):
    from adapt_tet import split
    return split(V).mean(axis=2)  # (n, 8, 3)


def assemble_solve(V, chi, b_ext, basis, chunk_pairs=200_000):
    n, linear = len(V), basis == "p1"
    c = V.mean(1)
    scale = np.sqrt(((V - c[:, None]) ** 2).sum(-1)).max(1)
    nb = 12 if linear else 3
    mode = os.environ.get("COLLOC", "child8")
    if not linear:
        Xp = c[:, None, :]
    elif mode == "child8":
        Xp = child_centroids(V)
    else:  # points halfway from the centroid to each vertex, optionally with the centroid
        f = float(os.environ.get("FRAC", 0.5))
        Xp = c[:, None, :] + f * (V - c[:, None, :])
        if mode == "corner4c":
            Xp = np.concatenate([Xp, c[:, None, :]], axis=1)
    npc = Xp.shape[1]
    X = Xp.reshape(-1, 3)
    A = np.empty((len(X), 3, n, nb))
    step = max(1, chunk_pairs // n)
    for s in range(0, len(X), step):
        xs = X[s : s + step]
        T = tensor(np.repeat(xs, n, axis=0), np.tile(V, (len(xs), 1, 1)), np.tile(scale, len(xs)), linear=linear)
        A[s : s + step] = -chi * T.reshape(len(xs), n, 3, nb).transpose(0, 2, 1, 3)
    A = A.reshape(len(X) * 3, n * nb)
    for i in range(n):
        Pp = P((Xp[i] - c[i]) / scale[i], nb).reshape(-1, nb)
        A[i * npc * 3 : (i + 1) * npc * 3, i * nb : (i + 1) * nb] += Pp
    b = chi * b_ext(X).reshape(-1)
    u = np.linalg.solve(A, b) if A.shape[0] == A.shape[1] else np.linalg.solve(A.T @ A, A.T @ b)
    return u.reshape(n, nb), scale


def field(V, u, scale, pts, linear):
    n = len(V)
    T = tensor(np.repeat(pts, n, axis=0), np.tile(V, (len(pts), 1, 1)), np.tile(scale, len(pts)), linear=linear)
    return np.einsum("pncb,nb->pc", T.reshape(len(pts), n, 3, -1), u)


def ucore(target, basis):
    from adapt import CHI, COIL, LINE
    from adapt_tet import tetgen_mesh
    V = tetgen_mesh(target)
    t0 = time.perf_counter()
    u, scale = assemble_solve(V, CHI, COIL.getB, basis)
    t = time.perf_counter() - t0
    line = LINE + (0, 1e-6, 0)
    B = field(V, u, scale, line, basis == "p1") + COIL.getB(line)
    peak = float(np.abs(B[:, 2]).max() * 1e3)
    return {"case": "ucore", "basis": basis, "cells": len(V), "unknowns": u.size, "peak_mT": peak,
            "err_pct": 100 * (peak / 2.153 - 1), "s": round(t, 2)}


def cube(target, basis):
    from magpylib_material_response.meshing import mesh_TriangularMesh
    corners = np.array(list(itertools.product((-0.005, 0.005), repeat=3)))
    box = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
    V = np.array([s.vertices for s in mesh_TriangularMesh(box, target).sources_all])
    t0 = time.perf_counter()
    u, _ = assemble_solve(V, 1000.0, lambda X: np.broadcast_to([0, 0, 0.01], X.shape), basis)
    t = time.perf_counter() - t0
    vol = np.abs(np.linalg.det(V[:, 1:] - V[:, :1])) / 6
    m = (u[:, 2] * vol).sum() / MU0
    return {"case": "cube", "basis": basis, "cells": len(V), "unknowns": u.size, "err_pct": 100 * (m / 0.02889 - 1), "s": round(t, 2)}


def bench():
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    n = 200_000
    V = rng.uniform(-1, 1, (n, 4, 3))
    X = rng.uniform(-3, 3, (n, 3))
    sc = np.ones(n)

    def best(f, k=3):
        f()
        ts = []
        for _ in range(k):
            t0 = time.perf_counter()
            jax.block_until_ready(f())
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e9 / n

    pols = [np.tile(e, (n, 1)) for e in np.eye(3)]
    t_mp = best(lambda: np.stack([magpy.func.tetrahedron_field("H", X, V, p) for p in pols], -1))
    t_u = best(lambda: tensor(X, V, sc, linear=False))
    t_l = best(lambda: tensor(X, V, sc, linear=True))
    ju = jax.jit(lambda a, b, c: tensor(a, b, c, linear=False, xp=jnp))
    jl = jax.jit(lambda a, b, c: tensor(a, b, c, linear=True, xp=jnp))
    Xj, Vj, sj = jnp.asarray(X), jnp.asarray(V), jnp.asarray(sc)
    t_ju, t_jl = best(lambda: ju(Xj, Vj, sj)), best(lambda: jl(Xj, Vj, sj))
    diff = float(np.abs(np.asarray(jl(Xj, Vj, sj)) - tensor(X, V, sc)).max())
    print(f"n={n:,} pairs | magpylib tetrahedron 3x3: {t_mp:.0f} ns | this uniform 3x3: numpy {t_u:.0f}, jit {t_ju:.0f} ns"
          f" | linear 3x12: numpy {t_l:.0f}, jit {t_jl:.0f} ns (jit vs numpy {diff:.0e})")


if __name__ == "__main__":
    if sys.argv[1] == "bench":
        bench()
    else:
        f = ucore if sys.argv[1] == "ucore" else cube
        print(json.dumps(f(int(sys.argv[2]), sys.argv[3])), flush=True)
