import numpy as np, time
from bem import Panels, kprime_galerkin, kprime_gal, solve, moment
half = np.array([1., 1., 1.]); V = 8.
for n in [16, 24]:
    pan = Panels(half, n, "power", 2.0)
    t = time.perf_counter(); K4 = kprime_galerkin(pan, 4); t4 = time.perf_counter()-t
    for nq in [3, 4, 5]:
        t = time.perf_counter(); K = kprime_gal(pan, nq); tq = time.perf_counter()-t
        s = solve(pan, 1e12, pan.normal[:, 2], K=K)
        print(n, nq, f"{tq:.2f}s (old nq4 {t4:.2f}s)", "alpha", moment(pan, s)[2]/V, "max|K-K4|", np.abs(K-K4).max() if nq == 4 else "")
