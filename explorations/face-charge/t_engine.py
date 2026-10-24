import numpy as np, time, magpylib as magpy
from engine import CuboidResponse, solve_in_field, MU0
from cases import SHAPES, observers, source
from conv2 import extrap
ref = np.load("p1_g3_6_8_12_16_20_24.npy", allow_pickle=True).item()
cases = ref["cases"]
def bench(f, k=20):
    f(); t = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter()-t)/k*1e3
for m in [3, 4, 6]:
    half = SHAPES["cube"]; obs, tags = observers(half); src = source(half)
    r = CuboidResponse(2*half, m=m, verbose=True)
    for ci in [0, 1, 4]:
        _, chi, J0, _ = cases[ci]
        c = solve_in_field(r, chi, [src], J0=J0)
        B = r.B(obs, c)
        ext, p, _ = extrap(ref["res"], ci, ref["levels"])
        e = np.linalg.norm(B-ext, axis=1)/np.linalg.norm(ext, axis=1)
        far = np.array([d >= 0.5e-3 for (_, d) in tags])
        print(f"  chi={chi:7.2f} J0={J0}: max err d>=0.5mm {e[far].max():.1e}, d<0.5mm {e[~far].max():.1e}")
    t_samp = bench(lambda: magpy.getH(src, r.sample_points()))
    H = magpy.getH(src, r.sample_points())
    t_solve = bench(lambda: r.solve(1000., H))
    t_eval = bench(lambda: r.B(obs, c), 5)
    print(f"  online: sample {len(H)} pts {t_samp:.2f} ms, solve {t_solve:.2f} ms, B at {len(obs)} obs {t_eval:.2f} ms ({t_eval/len(obs)*1e3:.0f} us/obs)")
# cross-check against the bordered direct solve
from p1 import P1
r = CuboidResponse(SHAPES["cube"]*2, m=4)
H = magpy.getH(source(SHAPES["cube"]), r.sample_points())
for chi in [0.05, 10., 1e3, 1e9]:
    c = r.solve(chi, H)
    F = r.load(np.einsum("gnd,nd->gn", H.reshape(len(r._S), r.pan.N, 3), r.pan.normal))
    G = r.s.assemble(); cd = r.s.solve(chi, G, F)
    print(f"chi={chi:g}: |c_schur - c_direct|/|c| = {np.linalg.norm(c-cd)/np.linalg.norm(cd):.1e}")
