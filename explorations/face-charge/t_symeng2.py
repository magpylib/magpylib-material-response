import numpy as np, time, magpylib as magpy
from engine import SymCuboidResponse, CuboidResponse
from cases import SHAPES, observers, source
def bench(f, k=50):
    f(); t = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter()-t)/k*1e3
for shape, m in [("cube", 3), ("cube", 4), ("cube", 6), ("plate", 4)]:
    half = SHAPES[shape]; obs, tags = observers(half); src = source(half)
    rs = SymCuboidResponse(2*half, m=m, verbose=True)
    r = CuboidResponse(2*half, m=m, g=2.5)
    H = magpy.getH(src, rs.sample_points())
    a, b = rs.solve(1000., H), r.solve(1000., H)
    t = time.perf_counter(); rs.fix_chi(1000.); tf = (time.perf_counter()-t)*1e3
    a2 = rs.solve(1000., H)
    print(f"   sym vs full {np.linalg.norm(a-b)/np.linalg.norm(b):.1e}, fixed-chi vs Schur {np.linalg.norm(a2-a)/np.linalg.norm(a):.1e}")
    print(f"   solve any chi {bench(lambda: rs.solve(999., H)):.2f} ms | fix_chi once {tf:.0f} ms, then solve {bench(lambda: rs.solve(1000., H)):.2f} ms | sample {len(H)} pts {bench(lambda: magpy.getH(src, rs.sample_points())):.2f} ms")
