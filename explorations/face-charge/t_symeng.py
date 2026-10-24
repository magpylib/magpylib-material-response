import numpy as np, time, magpylib as magpy
from engine import CuboidResponse, SymCuboidResponse, solve_in_field
from cases import SHAPES, observers, source
def bench(f, k=30):
    f(); t = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter()-t)/k*1e3
for shape, m in [("cube", 3), ("cube", 4), ("cube", 6), ("plate", 4)]:
    half = SHAPES[shape]; obs, tags = observers(half); src = source(half)
    rs = SymCuboidResponse(2*half, m=m, verbose=True)
    r = CuboidResponse(2*half, m=m, g=2.5)
    H = magpy.getH(src, rs.sample_points())
    for chi in [0.05, 1000., 1e9]:
        a, b = rs.solve(chi, H), r.solve(chi, H)
        print(f"   chi={chi:g}: sym vs full |dc|/|c| {np.linalg.norm(a-b)/np.linalg.norm(b):.1e}")
    ts = bench(lambda: magpy.getH(src, rs.sample_points()))
    tsol = bench(lambda: rs.solve(1000., H))
    c = rs.solve(1000., H)
    tB = bench(lambda: rs.B(obs, c), 10)
    print(f"   online: sample {len(H)} pts {ts:.2f} ms | solve {tsol:.2f} ms (full Schur {bench(lambda: r.solve(1000., H)):.2f}) | B {tB/len(obs)*1e3:.0f} us/observer")
