import numpy as np, time, sys
from cases import *
CASES = [("cube", 1000.0, None, None), ("cube", 10.0, None, None), ("cube", 1.0, None, None),
         ("plate", 1000.0, None, None), ("cube", 0.05, (0, 0, 1.2), None),
         ("cube", 1000.0, None, (0.004, 0.002, 0.009))]
levels = [int(a) for a in sys.argv[1:]] or [16, 24, 32]
res = {}
for shape in ["cube", "plate"]:
    obs, tags = observers(SHAPES[shape])
    for n in levels:
        t = time.perf_counter()
        pan = Panels(SHAPES[shape], n, "power", 2.0); K = kprime_gal(pan, 3)
        ta = time.perf_counter() - t
        for ci, (sh, chi, J0, uni) in enumerate(CASES):
            if sh != shape: continue
            t = time.perf_counter()
            _, s, _ = bem_case(shape, chi, n, J0=J0, uniform=uni, K=K, pan=pan)
            B = body_B(pan, s, obs)
            res[(ci, n)] = B
            print(f"case {ci} {shape} chi={chi} n={n} N={pan.N} asm {ta:.1f}s solve+eval {time.perf_counter()-t:.1f}s", flush=True)
np.save("ref_%s.npy" % "_".join(map(str, levels)), {"res": res, "levels": levels, "cases": CASES}, allow_pickle=True)
