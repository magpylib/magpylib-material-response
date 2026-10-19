"""D8: which S put the lsq8 collocation points (+-0.4) on sub-cell centres, and how large is the staircase
emulation's error of the self-cell field at those points, for S = 3, 5, 7, 9, 15, 25."""
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy  # noqa: E402
from lincell import tensor  # noqa: E402

MU0 = magpy.mu_0
colloc = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * 0.4
h = np.ones((8, 3))
exact = tensor(colloc, h)  # (8, 3, 12) self-cell field at the collocation points


def P(xi):
    M = np.zeros((len(xi), 3, 12))
    M[:, [0, 1, 2], [0, 1, 2]] = 1
    for r in range(3):
        M[:, r, 3 + 3 * r : 6 + 3 * r] = xi
    return M


for S in (3, 5, 7, 9, 15, 25):
    xq = (np.arange(S) + 0.5) / S * 2 - 1
    has = np.any(np.isclose(xq, 0.4))
    # distance of 0.4 from the nearest sub-cell centre, in units of the sub-cell half width
    off = np.min(np.abs(xq - 0.4)) / (1 / S)
    XI = np.stack(np.meshgrid(xq, xq, xq, indexing="ij"), -1).reshape(-1, 3)
    q = len(XI)
    obs = np.repeat(colloc, q, axis=0)
    pos = np.tile(XI, (8, 1))
    dims = np.full((8 * q, 3), 2.0 / S)
    F = np.empty((8, 3, q, 3))
    for m in range(3):
        pol = np.zeros((8 * q, 3))
        pol[:, m] = 1
        F[:, :, :, m] = (magpy.func.cuboid_field("H", obs, dims, pol, positions=pos) * MU0).reshape(8, q, 3).transpose(0, 2, 1)
    emul = np.einsum("pcqm,qmb->pcb", F, P(XI))
    d = np.abs(emul - exact)
    rel_u = d[:, :, :3].max() / np.abs(exact[:, :, :3]).max()
    rel_l = d[:, :, 3:].max() / np.abs(exact[:, :, 3:]).max()
    print(f"S={S:2d}: 0.4 on a sub-cell centre: {str(has):5s} (offset {off:.2f} half-widths); self-field error at the 8 colloc points:"
          f" uniform cols {rel_u:.1e}, linear cols {rel_l:.1e} (rel to max |exact|); worst linear col {d[:, :, 3:].max(axis=(0, 1)).argmax() + 3}")
