import numpy as np, time, sys, magpylib as magpy
from bem import Panels, layer_panels
from p1 import P1
from cases import SHAPES, observers, source, MU0
CASES = [("cube", 1000.0, None, None), ("cube", 10.0, None, None), ("cube", 1.0, None, None),
         ("plate", 1000.0, None, None), ("cube", 0.05, (0, 0, 1.2), None),
         ("cube", 1000.0, None, (0.004, 0.002, 0.009))]
SHAPE_LIST = sys.argv[1].split(","); g = float(sys.argv[2]); levels = [int(a) for a in sys.argv[3:]]
res = {}; timing = {}
for shape in SHAPE_LIST:
    half = SHAPES[shape]; obs, tags = observers(half); src = source(half)
    for n in levels:
        pan = layer_panels(half, n, g); s = P1(pan, nq=4)
        t = time.perf_counter(); G = s.assemble(); ta = time.perf_counter()-t
        for ci, (sh, chi, J0, uni) in enumerate(CASES):
            if sh != shape: continue
            t = time.perf_counter()
            if uni is None: F = s.rhs(lambda x: magpy.getH(src, x))
            else: F = s.rhs(lambda x: np.tile(np.array(uni)/MU0, (len(x), 1)))
            F0 = None if J0 is None else s.rhs_const_normal(pan.normal @ (np.array(J0)/MU0))
            c = s.solve(chi, G, F, F0)
            B = MU0*s.field(obs, c)
            res[(ci, n)] = B; timing[(ci, n)] = (ta, time.perf_counter()-t, s.Nd)
            print(f"case {ci} {shape} chi={chi} n={n} dofs={s.Nd} asm {ta:.1f}s solve+eval {time.perf_counter()-t:.2f}s", flush=True)
np.save(f"lay_{sys.argv[1]}_g{g:.0f}_" + "_".join(map(str, levels)) + ".npy", {"res": res, "levels": levels, "cases": CASES, "timing": timing}, allow_pickle=True)
