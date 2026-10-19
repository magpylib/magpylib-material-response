"""Adaptive cuboid refinement for material-response, tested on the U-core docs example.

usage: adapt.py uniform <h_mm>...                 uniform near-cubic grids
       adapt.py adaptive <indicator> <max_cells>  indicator: residual | targeted
Prints one JSON line per solve.
"""

import json
import sys
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response.demag import apply_demag, demag_tensor
from magpylib_material_response.polyline import create_polyline_fillet, move_grid_along_polyline
from scipy.linalg import lu_factor, lu_solve

MU0 = magpy.mu_0
CHI = 999.0
A = 0.02
BLOCKS = [((0.10, A, A), (0, 0, 0)), ((A, A, 0.06), (-0.04, 0, 0.04)), ((A, A, 0.06), (0.04, 0, 0.04))]

# coil and probe line, verbatim from docs/examples/ucore_electromagnet.md
_h = 0.014
_square = np.array([[0, _h, 0], [0, _h, _h], [0, -_h, _h], [0, -_h, -_h], [0, _h, -_h], [0, _h, 0]])
_turn = create_polyline_fillet(_square, max_radius=0.006, N=6)
_pitch = 0.0022
_pack = np.array([(xt, yl, 0) for xt in (np.arange(8) - 3.5) * _pitch for yl in _h + np.arange(2) * _pitch])
COIL = magpy.Collection(*(magpy.current.Polyline(current=25, vertices=v) for v in move_grid_along_polyline(_turn, _pack)))
LINE = np.stack([np.linspace(-0.08, 0.08, 161), np.zeros(161), np.full(161, 0.10)], axis=-1)
CHILD = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) / 4


def uniform_mesh(h):
    pos, dim = [], []
    for d, c in BLOCKS:
        n = np.maximum(np.round(np.array(d) / h).astype(int), 1)
        cell = np.array(d) / n
        axes = [np.linspace(-dd / 2 + s / 2, dd / 2 - s / 2, k) for dd, s, k in zip(d, cell, n)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c
        pos.append(grid)
        dim.append(np.tile(cell, (len(grid), 1)))
    return np.vstack(pos), np.vstack(dim)


def cells(pos, dim, J=None):
    J = np.zeros_like(pos) if J is None else J
    return [magpy.magnet.Cuboid(dimension=d, position=p, polarization=j) for p, d, j in zip(pos, dim, J)]


def solve(pos, dim):
    """Dense point-matched system as apply_demag builds it; LU kept for the adjoint."""
    n = len(pos)
    T = demag_tensor(cells(pos, dim)) * MU0
    T = T.swapaxes(2, 3).reshape((3 * n, 3 * n)).T
    Q = np.eye(3 * n) - CHI * T
    del T
    rhs = CHI * np.reshape(COIL.getB(pos), (3 * n,), order="F")
    lu = lu_factor(Q, overwrite_a=True)
    J = np.reshape(lu_solve(lu, rhs), (n, 3), order="F")
    return J, lu


def field_line(pos, dim, J):
    return magpy.getB(cells(pos, dim, J) + [COIL], LINE, sumup=True)


def adjoint(pos, dim, lu, p_star):
    """z = Q^-T g, g = d Bz(p*) / d J (per cell and component)."""
    n = len(pos)
    g = np.concatenate([magpy.getB(cells(pos, dim, np.tile(e, (n, 1))), p_star)[:, 2] for e in np.eye(3)])
    return np.reshape(lu_solve(lu, g, trans=1), (n, 3), order="F")


def residual(pos, dim, J, chunk=400):
    """r_ik = chi mu0 H(child k of cell i) - J_i: what splitting cell i would correct."""
    X = (pos[:, None, :] + CHILD[None] * dim[:, None, :]).reshape(-1, 3)
    srcs = cells(pos, dim, J) + [COIL]
    H = np.vstack([magpy.getH(srcs, X[s : s + chunk], sumup=True) for s in range(0, len(X), chunk)])
    return CHI * MU0 * H.reshape(len(pos), 8, 3) - J[:, None, :]


def run(pos, dim, extra=None):
    t0 = time.perf_counter()
    J, lu = solve(pos, dim)
    t1 = time.perf_counter()
    Bz = field_line(pos, dim, J)[:, 2]
    i = int(np.argmax(np.abs(Bz)))
    out = {"cells": len(pos), "peak_mT": abs(Bz[i]) * 1e3, "pole_mT": Bz[120] * 1e3, "x_peak_mm": LINE[i, 0] * 1e3,
           "solve_s": round(t1 - t0, 2)}
    return out | (extra or {}), J, lu, LINE[i : i + 1], np.sign(Bz[i])


def adaptive(indicator, max_cells, theta=0.5):
    pos, dim = uniform_mesh(0.01)
    while True:
        out, J, lu, p_star, sgn = run(pos, dim)
        t0 = time.perf_counter()
        z = adjoint(pos, dim, lu, p_star) * sgn  # sensitivity of |Bz| at the peak
        del lu
        r = residual(pos, dim, J)
        vol = np.prod(dim, axis=1)
        zr = np.einsum("ia,ika->ik", z, r) / 8  # per child, predicted change of |Bz(p*)|
        if indicator == "residual":
            eta = vol * np.sqrt((r**2).sum(-1).mean(-1))
        else:
            eta = np.abs(zr).sum(-1)
        out |= {"est_signed_mT": zr.sum() * 1e3, "est_abs_mT": np.abs(zr).sum() * 1e3,
                "indicator_s": round(time.perf_counter() - t0, 2)}
        print(json.dumps(out), flush=True)
        budget = (max_cells - len(pos)) // 7
        if budget < 1:
            np.savez(f"mesh-{indicator}.npz", pos=pos, dim=dim)
            return
        order = np.argsort(eta)[::-1]
        k = min(int(np.searchsorted(np.cumsum(eta[order]), theta * eta.sum()) + 1), budget)
        mark = np.zeros(len(pos), bool)
        mark[order[:k]] = True
        kids_pos = (pos[mark][:, None, :] + CHILD[None] * dim[mark][:, None, :]).reshape(-1, 3)
        kids_dim = np.repeat(dim[mark] / 2, 8, axis=0)
        pos, dim = np.vstack([pos[~mark], kids_pos]), np.vstack([dim[~mark], kids_dim])


if __name__ == "__main__":
    if sys.argv[1] == "check":  # my assembly against apply_demag on the coarse mesh
        pos, dim = uniform_mesh(0.01)
        J, _ = solve(pos, dim)
        core = magpy.Collection(*cells(pos, dim))
        core.susceptibility = CHI
        solved = apply_demag(magpy.Collection(core, COIL))
        ref = np.array([s.polarization for s in solved.sources_all if isinstance(s, magpy.magnet.Cuboid)])
        print(f"cells {len(pos)}, max |J - apply_demag| = {np.abs(J - ref).max():.2e} T of max |J| {np.abs(ref).max():.3f} T")
    elif sys.argv[1] == "apply":  # apply_demag on a uniform grid: adapt.py apply <solver> <h_mm>
        pos, dim = uniform_mesh(float(sys.argv[3]) * 1e-3)
        core = magpy.Collection(*cells(pos, dim))
        core.susceptibility = CHI
        t0 = time.perf_counter()
        solved = apply_demag(magpy.Collection(core, COIL), solver=sys.argv[2])
        t = time.perf_counter() - t0
        Bz = solved.getB(LINE)[:, 2]
        print(json.dumps({"cells": len(pos), "solver": sys.argv[2], "peak_mT": float(np.abs(Bz).max() * 1e3), "solve_s": round(t, 2)}))
    elif sys.argv[1] == "uniform":
        for h in sys.argv[2:]:
            print(json.dumps(run(*uniform_mesh(float(h) * 1e-3), {"h_mm": float(h)})[0]), flush=True)
    else:
        adaptive(sys.argv[2], int(sys.argv[3]))
