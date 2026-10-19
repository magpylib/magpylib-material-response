"""E3: float32 precision of the cuboid kernel vs observer distance (unit cube, J = z)."""
import os, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/pr854/overlay")
import numpy as np
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield as pr
rng = np.random.default_rng(1)
dirs = rng.normal(size=(2000, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
print("distance/side   max rel err float32 (vs float64)   median")
for d in (1.5, 3, 10, 30, 100, 300, 1000):
    obs = dirs * d; dim = np.ones((2000, 3)); pol = np.tile([0, 0, 1.0], (2000, 1))
    b64 = pr(obs, dim, pol)
    b32 = pr(obs.astype(np.float32), dim.astype(np.float32), pol.astype(np.float32))
    rel = np.linalg.norm(b32.astype(np.float64) - b64, axis=1) / np.linalg.norm(b64, axis=1)
    print(f"{d:>8}        {rel.max():.1e}   {np.median(rel):.1e}   (out dtype {b32.dtype})")
