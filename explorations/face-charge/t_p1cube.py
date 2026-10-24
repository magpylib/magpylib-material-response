import numpy as np, time
from bem import Panels
from p1 import P1
half = np.array([1., 1., 1.]); V = 8.
for g in [2.0, 3.0]:
    for n in [6, 8, 12, 16]:
        pan = Panels(half, n, "power", g); s = P1(pan, nq=4)
        t = time.perf_counter(); G = s.assemble(); ta = time.perf_counter()-t
        F = s.rhs(lambda x: np.tile([0, 0, 1.], (len(x), 1)))
        a = [s.moment(s.solve(c, G, F))[2]/V for c in (1e12, 1000., 1.)]
        print(f"P1 power{g:.0f} n={n:2d} dofs={s.Nd:5d} asm {ta:5.2f}s alpha_inf={a[0]:.6f} err={a[0]/3.644305-1:+.1e} chi1000={a[1]:.6f} chi1={a[2]:.6f}", flush=True)
