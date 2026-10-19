"""Sensitivity of the U-core peak |Bz| and m_x to a uniform x-field (the image field of fem.py's
Dirichlet phi=0 truncation at R_d ~ 0.4-0.5 m is H_c = m_x/(4 pi R_d^3) along x). MoM on uniform
cuboid grids of 10 and 5 mm (88 / 704 cells), dense direct solve as in adapt.py."""
import sys
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response.demag import demag_tensor
from magpylib_material_response.polyline import create_polyline_fillet, move_grid_along_polyline

MU0 = magpy.mu_0
CHI = 999.0
A = 0.02
BLOCKS = [((0.10, A, A), (0, 0, 0)), ((A, A, 0.06), (-0.04, 0, 0.04)), ((A, A, 0.06), (0.04, 0, 0.04))]
_h = 0.014
_square = np.array([[0, _h, 0], [0, _h, _h], [0, -_h, _h], [0, -_h, -_h], [0, _h, -_h], [0, _h, 0]])
_turn = create_polyline_fillet(_square, max_radius=0.006, N=6)
_pitch = 0.0022
_pack = np.array([(xt, yl, 0) for xt in (np.arange(8) - 3.5) * _pitch for yl in _h + np.arange(2) * _pitch])
COIL = magpy.Collection(*(magpy.current.Polyline(current=25, vertices=v) for v in move_grid_along_polyline(_turn, _pack)))
LINE = np.stack([np.linspace(-0.08, 0.08, 161), np.zeros(161), np.full(161, 0.10)], axis=-1)


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


def cells(pos, dim, J):
    return [magpy.magnet.Cuboid(dimension=d, position=p, polarization=j) for p, d, j in zip(pos, dim, J)]


for h in [float(v) for v in sys.argv[1:]] or [10, 5]:
    t = time.perf_counter()
    pos, dim = uniform_mesh(h * 1e-3)
    n = len(pos)
    T = demag_tensor(cells(pos, dim, np.zeros_like(pos))) * MU0
    T = T.swapaxes(2, 3).reshape((3 * n, 3 * n)).T
    Q = np.eye(3 * n) - CHI * T
    Hc = 1.0  # A/m along x
    rhs = np.stack([CHI * np.reshape(COIL.getB(pos), (3 * n,), order="F"),
                    CHI * np.concatenate([np.full(n, MU0 * Hc), np.zeros(n), np.zeros(n)])], axis=1)
    sol = np.linalg.solve(Q, rhs)
    Jc, Ju = (np.reshape(sol[:, k], (n, 3), order="F") for k in (0, 1))
    vol = dim.prod(axis=1)[:, None]
    mc, mu_ = (Jc * vol).sum(0) / MU0, (Ju * vol).sum(0) / MU0
    Bc = magpy.getB(cells(pos, dim, Jc) + [COIL], LINE, sumup=True)[:, 2]
    Bu = magpy.getB(cells(pos, dim, Ju), LINE, sumup=True)[:, 2]
    i = int(np.abs(Bc).argmax())
    print(f"h={h:g} mm cells={n}: coil solve m_x={mc[0]:.4f} A m^2, peak|Bz|={abs(Bc[i]) * 1e3:.4f} mT at x={LINE[i, 0] * 1e3:g}; "
          f"per 1 A/m uniform x-field: dm_x={mu_[0]:.3e} A m^2/(A/m), dBz(peak)={Bu[i] * np.sign(Bc[i]) * 1e3:.3e} mT/(A/m)  {time.perf_counter() - t:.1f}s")
    for Rd in (0.4, 0.45, 0.5):
        Hc_img = 6.097 / (4 * np.pi * Rd**3)  # fem.py's m_x at h=1 mm
        print(f"   R_d={Rd} m: H_c={Hc_img:.2f} A/m -> dm_x/m_x={100 * mu_[0] * Hc_img / mc[0]:+.3f} %, dpeak/peak={100 * Bu[i] * np.sign(Bc[i]) * Hc_img / abs(Bc[i]):+.3f} %")
