"""Does m(chi) = V H0 / (1/3.6442 + 1/chi) hold at chi = 1000 for a cube? Compare the MoM ratio
m(1000)/m(1e6) on one fixed cell mesh (discretisation error largely cancels) with the formula's ratio."""
import sys
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response.demag import demag_tensor

MU0 = magpy.mu_0
a, B0 = 0.01, 0.01
H0 = B0 / MU0
V = a**3
formula = lambda chi: V * H0 / (1 / 3.6442 + 1 / chi)
print(f"formula: m(1000)={formula(1000):.6f} m(1e6)={formula(1e6):.6f} ratio={formula(1000) / formula(1e6):.6f}")
for n in [int(v) for v in sys.argv[1:]] or [4, 6, 8, 10]:
    t = time.perf_counter()
    cell = a / n
    axes = np.linspace(-a / 2 + cell / 2, a / 2 - cell / 2, n)
    pos = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(-1, 3)
    N = len(pos)
    T = demag_tensor([magpy.magnet.Cuboid(dimension=(cell,) * 3, position=p) for p in pos]) * MU0
    T = T.swapaxes(2, 3).reshape((3 * N, 3 * N)).T
    ms = {}
    for chi in (1000.0, 1e6, 1e-3):
        Q = np.eye(3 * N) - chi * T
        rhs = chi * np.concatenate([np.zeros(N), np.zeros(N), np.full(N, B0)])
        J = np.reshape(np.linalg.solve(Q, rhs), (N, 3), order="F")
        ms[chi] = J[:, 2].sum() * cell**3 / MU0
    print(f"n={n:2d} cells={N:5d}: m(1000)={ms[1000.0]:.6f} ({100 * (ms[1000.0] / formula(1000) - 1):+.2f} % vs formula)  m(1e6)={ms[1e6]:.6f} ({100 * (ms[1e6] / formula(1e6) - 1):+.2f} %)  "
          f"ratio m(1000)/m(1e6)={ms[1000.0] / ms[1e6]:.6f} (formula {formula(1000) / formula(1e6):.6f}; formula/MoM-1 = {100 * (formula(1000) / formula(1e6) / (ms[1000.0] / ms[1e6]) - 1):+.4f} %)  "
          f"chi->0: m/(V H0 chi)={ms[1e-3] / (V * H0 * 1e-3):.5f} (1-chi/3={1 - 1e-3 / 3:.5f})  {time.perf_counter() - t:.1f}s", flush=True)
