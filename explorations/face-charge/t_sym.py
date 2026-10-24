import numpy as np, time
from bem import layer_panels
from p1 import P1
from sym import irrep_bases
for half in [np.array([5e-3]*3), np.array([10e-3, 5e-3, 1.25e-3])]:
    pan = layer_panels(half, 4, 2.5); s = P1(pan)
    G = s.assemble(); d = np.sqrt(s.mass); Gh = G/d[:, None]/d[None, :]
    t = time.perf_counter(); B = irrep_bases(pan); tb = time.perf_counter()-t
    Bf = np.hstack([b.toarray() for b in B])
    print("sizes", [b.shape[1] for b in B], f"build {tb:.2f}s", "orthonormal", np.abs(Bf.T@Bf - np.eye(Bf.shape[1])).max())
    blocks = Bf.T @ Gh @ Bf
    off = blocks.copy(); o = 0
    for b in B:
        n = b.shape[1]; off[o:o+n, o:o+n] = 0; o += n
    print("  off-block / max:", np.abs(off).max()/np.abs(blocks).max())
