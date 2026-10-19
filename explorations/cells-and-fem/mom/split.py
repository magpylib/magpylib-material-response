"""Where material-response's dense solve spends its time: interaction assembly vs LU. #52 at 4901d1d."""

import sys
import time

import magpylib as magpy
import numpy as np
import scipy.linalg
from adapt import CHI, cells, uniform_mesh
from adapt_tet import tetgen_mesh
from magpylib_material_response.demag import demag_tensor

kind, size = sys.argv[1], sys.argv[2]
if kind == "cub":
    pos, dim = uniform_mesh(float(size) * 1e-3)
    srcs = cells(pos, dim)
else:
    srcs = [magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in tetgen_mesh(int(size))]
n = len(srcs)
t0 = time.perf_counter()
T = demag_tensor(srcs) * magpy.mu_0
t1 = time.perf_counter()
Q = np.eye(3 * n) - CHI * T.swapaxes(2, 3).reshape((3 * n, 3 * n)).T
del T
t2 = time.perf_counter()
scipy.linalg.lu_factor(Q, overwrite_a=True, check_finite=False)
t3 = time.perf_counter()
print(f"{kind} {n:>5} cells: assembly {t1 - t0:6.2f} s, reshape {t2 - t1:5.2f} s, LU {t3 - t2:6.2f} s"
      f"  -> kernels {100 * (t1 - t0) / (t3 - t0):.0f} % of the solve")
