"""How much does fem.py's P0 post-processing (element-mean J, then uniformly polarized cells) err
on the U-core line field? Prescribe a smooth J(x) = mu0*1000*H_coil(x), take exact cell means
(2x2x2 Gauss), evaluate the line field from uniformly polarized cuboid cells of side s, and
Richardson-extrapolate in s. Only magpylib field sums, no solve."""
import sys
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response.polyline import create_polyline_fillet, move_grid_along_polyline

MU0 = magpy.mu_0
A = 0.02
BLOCKS = [((0.10, A, A), (0, 0, 0)), ((A, A, 0.06), (-0.04, 0, 0.04)), ((A, A, 0.06), (0.04, 0, 0.04))]
_h = 0.014
_square = np.array([[0, _h, 0], [0, _h, _h], [0, -_h, _h], [0, -_h, -_h], [0, _h, -_h], [0, _h, 0]])
_turn = create_polyline_fillet(_square, max_radius=0.006, N=6)
_pitch = 0.0022
_pack = np.array([(xt, yl, 0) for xt in (np.arange(8) - 3.5) * _pitch for yl in _h + np.arange(2) * _pitch])
COIL = magpy.Collection(*(magpy.current.Polyline(current=25, vertices=v) for v in move_grid_along_polyline(_turn, _pack)))
LINE = np.stack([np.linspace(-0.08, 0.08, 161), np.zeros(161), np.full(161, 0.10)], axis=-1)
G = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) / (2 * np.sqrt(3))


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


def Jsmooth(X, chunk=1500):
    return MU0 * 1000 * np.vstack([magpy.getH(COIL, X[k : k + chunk]) for k in range(0, len(X), chunk)])


def field(cells, chunk=300):
    return sum(magpy.getB(cells[k : k + chunk], LINE, sumup=True) for k in range(0, len(cells), chunk))


Bcoil = COIL.getB(LINE)
rows = []
for s in [float(v) for v in sys.argv[1:]] or [10, 5, 4, 2.5, 2]:
    t = time.perf_counter()
    pos, dim = uniform_mesh(s * 1e-3)
    X = (pos[:, None, :] + G[None] * dim[:, None, :]).reshape(-1, 3)
    Jbar = Jsmooth(X).reshape(len(pos), 8, 3).mean(axis=1)
    cells = [magpy.magnet.Cuboid(dimension=d, position=p, polarization=j) for p, d, j in zip(pos, dim, Jbar)]
    B = field(cells)
    Bz = B[:, 2] + Bcoil[:, 2]
    m = (Jbar * dim.prod(axis=1)[:, None]).sum(axis=0) / MU0
    rows.append((s, len(pos), np.abs(Bz).max(), Bz[int(np.abs(Bz).argmax())], B[42, 2], B[80, 2], m[0]))
    print(f"s={s:5.2f} mm cells={len(pos):6d} peak|Bz|={rows[-1][2] * 1e3:.6f} mT  Bz_cells(x=-38)={rows[-1][4] * 1e3:.6f} Bz_cells(x=0)={rows[-1][5] * 1e3:.6f}  m_x={m[0]:.6f}  {time.perf_counter() - t:.1f}s", flush=True)
r = np.array(rows)
# Richardson in s^2 from the two finest
s1, s2 = r[-2, 0], r[-1, 0]
for name, col in (("peak", 2), ("Bz_cells(x=-38)", 4)):
    f1, f2 = r[-2, col], r[-1, col]
    lim = f2 + (f2 - f1) / ((s1 / s2) ** 2 - 1)
    print(f"{name}: s^2-Richardson limit {lim * 1e3:.6f} mT; error of each grid vs limit (%):", " ".join(f"s={row[0]:g}:{100 * (row[col] / lim - 1):+.4f}" for row in r))
    if len(r) >= 3:
        d = np.diff(r[:, col])
        for i in range(len(d) - 1):
            p = np.log(d[i] / d[i + 1]) / np.log(r[i, 0] * r[i + 2, 0] / r[i + 1, 0] ** 2) if False else None
        # observed order from successive triples assuming f = L + C s^p: solve for p by bisection
        for i in range(len(r) - 2):
            sa, sb, sc = r[i:i + 3, 0]
            fa, fb, fc = r[i:i + 3, col]
            ratio = (fa - fb) / (fb - fc)
            lo, hi = 0.1, 6.0
            for _ in range(60):
                mid = (lo + hi) / 2
                val = (sa**mid - sb**mid) / (sb**mid - sc**mid)
                lo, hi = (mid, hi) if val < ratio else (lo, mid)
            print(f"   observed order from s=({sa:g},{sb:g},{sc:g}): p={mid:.2f}")
