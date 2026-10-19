"""Thin soft-iron plate, 40 x 40 x 1 mm, chi = 1000, 10 mT applied along x.

usage: plate.py cub <nx> <nz>     cuboid cells nx * nx * nz
       plate.py tet <target>      TetGen tetrahedra
Prints cell count, mesh and solve time, moment m_x and Bx 2 mm past the +x edge.
"""

import itertools
import os
import sys
import time

import magpylib as magpy
import numpy as np
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid, mesh_TriangularMesh

L, T, CHI, B0 = 0.04, 0.001, 1000.0, 0.01
plate = magpy.magnet.Cuboid(dimension=(L, L, T), polarization=(0, 0, 0))

t0 = time.perf_counter()
if sys.argv[1] == "cub":
    nx, nz = int(sys.argv[2]), int(sys.argv[3])
    cells = mesh_Cuboid(plate, (nx, nx, nz))
else:
    corners = np.array(list(itertools.product((-L / 2, L / 2), (-L / 2, L / 2), (-T / 2, T / 2))))
    box = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
    relaxed = {"minratio": 2.0, "mindihedral": 5.0} if "relaxed" in sys.argv else {}
    cells = mesh_TriangularMesh(box, int(sys.argv[2]), **relaxed)
cells.susceptibility = CHI
cells.H_ext = (B0, 0, 0)
t1 = time.perf_counter()
solved = apply_demag(cells, **({"solver": os.environ["SOLVER"]} if "SOLVER" in os.environ else {}))
t2 = time.perf_counter()

srcs = solved.sources_all
m = sum(s.polarization * s.volume for s in srcs) / magpy.mu_0
Bx = solved.getB((L / 2 + 0.002, 0, 0))[0]
print(f"{sys.argv[1]} {' '.join(sys.argv[2:]):>8}  cells={len(srcs):>6}  mesh {t1 - t0:5.1f}s"
      f"  solve {t2 - t1:6.1f}s  m_x={m[0]:.6f} A m^2  Bx(+2mm)={Bx * 1e3:.4f} mT")
