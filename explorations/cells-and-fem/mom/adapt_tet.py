"""Tetrahedral U-core: uniform TetGen meshes, and adaptive 1 -> 8 splitting.

usage: adapt_tet.py uniform <target>...
       adapt_tet.py adaptive <start_target> <max_cells>
"""

import json
import sys
import time

import magpylib as magpy
import numpy as np
from adapt import CHI, COIL, LINE, MU0
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_TriangularMesh

# the U-core as one closed surface, as in the docs page's tetrahedral section
_u = [(-0.05, -0.01), (0.05, -0.01), (0.05, 0.07), (0.03, 0.07), (0.03, 0.01), (-0.03, 0.01), (-0.03, 0.07), (-0.05, 0.07)]
_cap = [(0, 1, 4), (1, 3, 4), (1, 2, 3), (0, 4, 5), (0, 5, 7), (5, 6, 7)]
_n = len(_u)
_faces = list(_cap) + [tuple(i + _n for i in t) for t in _cap]
for _i in range(_n):
    _j = (_i + 1) % _n
    _faces += [(_i, _j, _j + _n), (_i, _j + _n, _i + _n)]
U_IRON = magpy.magnet.TriangularMesh(
    vertices=np.array([(x, y, z) for y in (-0.01, 0.01) for x, z in _u]), faces=np.array(_faces), reorient_faces=True
)


def tetgen_mesh(target):
    return np.array([c.vertices for c in mesh_TriangularMesh(U_IRON, target).sources_all])


def split(V):
    """Midpoint subdivision: 4 corner tetrahedra plus the octahedron cut along m02-m13."""
    v0, v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2], V[:, 3]
    m01, m02, m03 = (v0 + v1) / 2, (v0 + v2) / 2, (v0 + v3) / 2
    m12, m13, m23 = (v1 + v2) / 2, (v1 + v3) / 2, (v2 + v3) / 2
    kids = [(v0, m01, m02, m03), (v1, m01, m12, m13), (v2, m02, m12, m23), (v3, m03, m13, m23),
            (m02, m13, m01, m03), (m02, m13, m03, m23), (m02, m13, m23, m12), (m02, m13, m12, m01)]
    return np.stack([np.stack(k, axis=1) for k in kids], axis=1)  # (m, 8, 4, 3)


def volume(V):
    return np.abs(np.linalg.det(V[..., 1:, :] - V[..., :1, :])) / 6


def solve(V):
    core = magpy.Collection(*(magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V))
    core.susceptibility = CHI
    solved = apply_demag(magpy.Collection(core, COIL, override_parent=True))
    mags = [s for s in solved.sources_all if isinstance(s, magpy.magnet.Tetrahedron)]
    return solved, np.array([m.polarization for m in mags])


def run(V, extra=None):
    t0 = time.perf_counter()
    solved, J = solve(V)
    t1 = time.perf_counter()
    Bz = solved.getB(LINE)[:, 2]
    i = int(np.argmax(np.abs(Bz)))
    out = {"cells": len(V), "peak_mT": abs(Bz[i]) * 1e3, "pole_mT": Bz[120] * 1e3, "solve_s": round(t1 - t0, 2)}
    return out | (extra or {}), solved, J


def residual(V, solved, J, chunk=100):
    X = split(V).mean(axis=2).reshape(-1, 3)  # child centroids
    H = np.vstack([solved.getH(X[s : s + chunk]) for s in range(0, len(X), chunk)])
    return CHI * MU0 * H.reshape(len(V), 8, 3) - J[:, None, :]


if __name__ == "__main__":
    if sys.argv[1] == "check":
        V = tetgen_mesh(50)
        kids = split(V)
        print(f"{len(V)} tets; children volume sum / parent: {volume(kids).sum(1).max() / volume(V).min():.3f} "
              f"max |sum - parent| {np.abs(volume(kids).sum(1) - volume(V)).max():.1e}, "
              f"child/parent ratio range {(volume(kids) / volume(V)[:, None]).min():.4f}-{(volume(kids) / volume(V)[:, None]).max():.4f}")
    elif sys.argv[1] == "uniform":
        for target in sys.argv[2:]:
            print(json.dumps(run(tetgen_mesh(int(target)), {"target": int(target)})[0]), flush=True)
    else:
        V, max_cells = tetgen_mesh(int(sys.argv[2])), int(sys.argv[3])
        while True:
            out, solved, J = run(V)
            t0 = time.perf_counter()
            r = residual(V, solved, J)
            eta = volume(V) * np.sqrt((r**2).sum(-1).mean(-1))
            print(json.dumps(out | {"indicator_s": round(time.perf_counter() - t0, 2)}), flush=True)
            budget = (max_cells - len(V)) // 7
            if budget < 1:
                np.savez("mesh-tet.npz", V=V)
                break
            order = np.argsort(eta)[::-1]
            k = min(int(np.searchsorted(np.cumsum(eta[order]), 0.5 * eta.sum()) + 1), budget)
            mark = np.zeros(len(V), bool)
            mark[order[:k]] = True
            V = np.vstack([V[~mark], split(V[mark]).reshape(-1, 4, 3)])
