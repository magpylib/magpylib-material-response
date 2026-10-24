import numpy as np, magpylib as magpy
from engine import CuboidResponse, solve_in_field
from cases import SHAPES, observers, source
from conv2 import extrap
ref = np.load("p1_g3_6_8_12_16_20_24.npy", allow_pickle=True).item()
half = SHAPES["cube"]; obs, tags = observers(half); src = source(half)
far = np.array([d >= 0.5e-3 for (_, d) in tags])
ext, p, _ = extrap(ref["res"], 0, ref["levels"])
for m in [4, 6]:
    for nqr in [2, 3, 4, 6]:
        r = CuboidResponse(2*half, m=m, nq_rhs=nqr)
        e = np.linalg.norm(r.B(obs, solve_in_field(r, 1000., [src]))-ext, axis=1)/np.linalg.norm(ext, axis=1)
        print(f"m={m} nq_rhs={nqr}: max err d>=0.5mm {e[far].max():.1e}  d<0.5mm {e[~far].max():.1e}   per-tag:", " ".join(f"{x:.0e}" for x in e))
