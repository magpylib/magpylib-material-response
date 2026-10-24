import numpy as np, time
from bem import geo_panels, kprime_gal, solve, moment
half = np.array([1., 1., 1.]); V = 8.
for L, r in [(3, 0.25), (6, 0.25), (6, 0.15), (9, 0.15)]:
    for h in [0.5, 1/3, 0.25, 1/6, 0.125]:
        pan = geo_panels(half, h, L, r)
        t = time.perf_counter(); K = kprime_gal(pan, 3); ta = time.perf_counter()-t
        a = [moment(pan, solve(pan, c, pan.normal[:, 2], K=K))[2]/V for c in (1e12, 1000., 1.)]
        print(f"L={L} r={r} h={h:.3f} N={pan.N:5d} ({ta:4.1f}s) alpha_inf={a[0]:.6f} err={a[0]/3.644305-1:+.1e}  chi1000={a[1]:.6f} chi1={a[2]:.6f}", flush=True)
