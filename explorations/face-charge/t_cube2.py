import numpy as np, time
from bem import Panels, solve, moment, kprime_galerkin
half = np.array([1., 1., 1.]); V = 8.0
for kind, g in [("power", 2.0), ("power", 3.0)]:
    for n in [8, 12, 16, 24]:
        t = time.perf_counter()
        pan = Panels(half, n, kind, g)
        K = kprime_galerkin(pan, 4); ta = time.perf_counter() - t
        f = pan.normal[:, 2] * 1.0
        out = []
        for chi in [1e12, 1000., 10., 1.]:
            s = solve(pan, chi, f, K=K)
            out.append(moment(pan, s)[2] / V)
        print(f"GAL {kind}{g:.0f} n={n:3d} N={pan.N:5d} asm {ta:5.2f}s alpha=", " ".join(f"{a:.6f}" for a in out), f" err_inf={out[0]/3.644305-1:+.2e}", flush=True)
