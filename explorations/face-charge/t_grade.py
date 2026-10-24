import numpy as np, magpylib as magpy
from engine import CuboidResponse, solve_in_field
from cases import SHAPES, observers, source
from conv2 import extrap
ref = np.load("p1_g3_6_8_12_16_20_24.npy", allow_pickle=True).item()
refp = np.load("lay_plate_g3_3_4_6_8.npy", allow_pickle=True).item()
for shape, ci, R in [("cube", 0, ref), ("cube", 4, ref), ("plate", 3, refp)]:
    half = SHAPES[shape]; obs, tags = observers(half); src = source(half)
    far = np.array([d >= 0.5e-3 for (_, d) in tags]); one = np.array([d >= 1e-3 for (_, d) in tags])
    ext, p, _ = extrap(R["res"], ci, R["levels"])
    _, chi, J0, _ = R["cases"][ci]
    for m in [3, 4, 6]:
        for g in [2.0, 2.5, 3.0]:
            r = CuboidResponse(2*half, m=m, g=g)
            e = np.linalg.norm(r.B(obs, solve_in_field(r, chi, [src], J0=J0))-ext, axis=1)/np.linalg.norm(ext, axis=1)
            print(f"{shape} chi={chi} m={m} g={g} dofs={r.s.Nd}: max err d>=1mm {e[one].max():.1e}  d>=0.5mm {e[far].max():.1e}  d<0.5mm {e[~far].max():.1e}", flush=True)
