import numpy as np, sys
from scipy.optimize import brentq
from cases import observers, SHAPES
files = sys.argv[1:] or ["ref_16_24_32.npy"]
res = {}; cases = None
for fn in files:
    d = np.load(fn, allow_pickle=True).item(); res.update(d["res"]); cases = d["cases"]
levels = sorted({n for (_, n) in res})
print("levels", levels)
for ci, (shape, chi, J0, uni) in enumerate(cases):
    obs, tags = observers(SHAPES[shape])
    B = np.array([res[(ci, n)] for n in levels])   # (L, M, 3)
    n1, n2, n3 = levels[-3:]
    b1, b2, b3 = B[-3], B[-2], B[-1]
    # global order from the norm of differences (fit a + c n^-p over the last three levels)
    d12 = np.linalg.norm(b2-b1); d23 = np.linalg.norm(b3-b2)
    g = lambda p: (n1**-p - n2**-p)/(n2**-p - n3**-p) - d12/d23
    try: p = brentq(g, 0.3, 5)
    except ValueError: p = np.nan
    ext = b3 + (b3-b2)*n3**-p/(n2**-p - n3**-p)
    print(f"\ncase {ci} {shape} chi={chi} J0={J0} uniform={uni}: observed order p={p:.2f}")
    for lvl, b in zip(levels, B):
        e = np.linalg.norm(b-ext, axis=1)/np.linalg.norm(ext, axis=1)
        print(f"  n={lvl:2d} rel err face/edge/corner at d=0.1..5mm:",
              " | ".join(" ".join(f"{e[k]:.0e}" for k in range(j, len(tags), 3)) for j in range(3)))
