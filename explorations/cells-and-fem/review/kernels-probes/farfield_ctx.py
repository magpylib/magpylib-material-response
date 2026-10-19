import sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy
from lincell import tensor as cub_tensor
from lintet import tensor as tet_tensor
MU0 = magpy.mu_0
hc = np.array([[1.0, 0.7, 1.3]])
dirn = np.array([0.3, -0.5, 0.8]); dirn /= np.linalg.norm(dirn)
Vol = 8 * np.prod(hc)
print("uniform cuboid columns vs point dipole (rel): r | magpylib cuboid | lincell")
for r in (100.0, 1e3, 1e4):
    x = (r * dirn)[None]
    Kd = 3 * np.outer(x[0], x[0]) / r**5 - np.eye(3) / r**3
    dip = Kd * Vol / (4 * np.pi)
    mp = np.stack([magpy.func.cuboid_field("H", x, 2 * hc, e[None], squeeze=False).reshape(3) * MU0 for e in np.eye(3)], -1)
    lc = cub_tensor(x, hc, linear=False)[0]
    print(f"    r={r:6.0f}: magpylib {np.abs(mp - dip).max() / np.abs(dip).max():.1e} | lincell {np.abs(lc - dip).max() / np.abs(dip).max():.1e}  (dipole truncation ~ (h/r)^2 = {(1.3 / r) ** 2:.0e})")
Vt = np.array([[[0, 0, 0], [1.0, 0, 0], [0.3, 0.9, 0], [0.4, 0.3, 0.8]]])
ct = Vt.mean(1); st = np.sqrt(((Vt - ct[:, None]) ** 2).sum(-1)).max(1); vol = abs(np.linalg.det(Vt[0, 1:] - Vt[0, 0])) / 6
print("uniform tet columns vs point dipole (rel): r | magpylib tetrahedron | lintet")
for r in (100.0, 1e3, 1e4):
    x = ct + r * dirn
    rr = x[0] - ct[0]; Kd = 3 * np.outer(rr, rr) / r**5 - np.eye(3) / r**3
    dip = Kd * vol / (4 * np.pi)
    mp = np.stack([magpy.func.tetrahedron_field("H", x, Vt, e[None]).reshape(3) * MU0 for e in np.eye(3)], -1)
    lt = tet_tensor(x, Vt, st, linear=False)[0]
    print(f"    r={r:6.0f}: magpylib {np.abs(mp - dip).max() / np.abs(dip).max():.1e} | lintet {np.abs(lt - dip).max() / np.abs(dip).max():.1e}")
