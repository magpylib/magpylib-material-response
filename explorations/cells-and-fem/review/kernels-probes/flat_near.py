import sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/review/kernels")
import magpylib as magpy
from indep_tet import inside_mask
from indep_tet4 import surface2
from lintet import tensor as tet_tensor
MU0 = magpy.mu_0
Vf = np.array([[[0, 0, 0], [1.0, 0, 0], [0.3, 0.9, 0], [0.4, 0.3, 1e-3]]])
sf = np.sqrt(((Vf - Vf.mean(1)[:, None]) ** 2).sum(-1)).max(1)
Xn = np.array([[0.4, 0.4, 0.002], [0.4, 0.4, -0.002], [0.4, 0.4, 0.0005], [0.9, 0.1, 0.0003], [0.4, 0.3, 0.0012]])
V4, s4 = np.tile(Vf, (5, 1, 1)), np.tile(sf, 5)
Tn = tet_tensor(Xn, V4, s4)
Sn = surface2(Xn, V4, s4)
Sn2 = surface2(Xn, V4, s4, lv=13, lw=11, ng=8)
ref = np.stack([magpy.func.tetrahedron_field("H", Xn, V4, np.tile(e, (5, 1))) * MU0 for e in np.eye(3)], -1)
print(f"flat tet (height 1e-3): observers z=+0.002, -0.002 (outside), (0.4,0.4,0.0005), (0.9,0.1,0.0003) (inside), (0.4,0.3,0.0012) (just above apex)")
print(f"  inside flags {inside_mask(Xn, V4)}; kernel vs 2x-graded surface rule max|diff| per point {np.abs(Tn - Sn).max(axis=(1, 2))} (scale {np.abs(Sn).max():.2f}; rule self-consistency {np.abs(Sn - Sn2).max():.1e}); magpylib uniform vs rule {np.abs(ref - Sn[:, :, :3]).max():.1e}; uniform cols vs magpylib {np.abs(Tn[:, :, :3] - ref).max():.1e}")
