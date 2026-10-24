"""Errors of each level against the Richardson limit of the last three; cross-check vs another reference."""
import numpy as np, sys
from scipy.optimize import brentq
from cases import observers, SHAPES

def load(fn):
    d = np.load(fn, allow_pickle=True).item(); return d

def extrap(res, ci, levels):
    n1, n2, n3 = levels[-3:]
    b1, b2, b3 = (res[(ci, n)] for n in (n1, n2, n3))
    d12 = np.linalg.norm(b2-b1); d23 = np.linalg.norm(b3-b2)
    g = lambda p: (n1**-p - n2**-p)/(n2**-p - n3**-p) - d12/d23
    try: p = brentq(g, 0.3, 8)
    except ValueError: p = np.nan
    return b3 + (b3-b2)*n3**-p/(n2**-p - n3**-p), p, b3

if __name__ == "__main__":
    d = load(sys.argv[1]); res, levels, cases = d["res"], d["levels"], d["cases"]
    other = None
    if len(sys.argv) > 2:
        o = load(sys.argv[2]); ro = dict(o["res"])
        for f in sys.argv[3:]: ro.update(load(f)["res"])
        other = (ro, sorted({n for (_, n) in ro}))
    for ci, (shape, chi, J0, uni) in enumerate(cases):
        if (ci, levels[-1]) not in res: continue
        obs, tags = observers(SHAPES[shape])
        ext, p, fin = extrap(res, ci, levels)
        print(f"\ncase {ci} {shape} chi={chi} J0={J0} uniform={uni}: order p={p:.2f}; |limit - finest| max rel {np.max(np.linalg.norm(ext-fin,axis=1)/np.linalg.norm(ext,axis=1)):.1e}")
        for lvl in levels:
            e = np.linalg.norm(res[(ci, lvl)]-ext, axis=1)/np.linalg.norm(ext, axis=1)
            print(f"  n={lvl:2d} err face|edge|corner d=0.1..5mm:", " | ".join(" ".join(f"{e[k]:.0e}" for k in range(j, len(tags), 3)) for j in range(3)))
        if other:
            ro, lo = other
            ext2, p2, _ = extrap(ro, ci, lo)
            e = np.linalg.norm(ext2-ext, axis=1)/np.linalg.norm(ext, axis=1)
            print(f"  other-ref limit vs this limit:", " | ".join(" ".join(f"{e[k]:.0e}" for k in range(j, len(tags), 3)) for j in range(3)))
