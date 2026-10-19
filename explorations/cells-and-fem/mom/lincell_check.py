"""Validate lincell.tensor: uniform columns against magpylib, linear columns against slab quadrature."""

import magpylib as magpy
import numpy as np
from lincell import tensor

rng = np.random.default_rng(1)
MU0 = magpy.mu_0
n = 300
h = rng.uniform(0.5, 2.0, (n, 3))
X = rng.uniform(-3, 3, (n, 3)) * h  # mixes inside (|X| < h) and outside points
inside = np.all(np.abs(X) < h, axis=1)


def unif(Xo, cen, half, m):
    pol = np.zeros((len(Xo), 3))
    pol[:, m] = 1
    return magpy.func.cuboid_field("H", Xo, 2 * half, pol, positions=cen, squeeze=False).reshape(-1, 3) * MU0


T = tensor(X, h)
U = tensor(X, h, linear=False)
ref = np.stack([unif(X, np.zeros((n, 3)), h, m) for m in range(3)], axis=-1)
print(f"{inside.sum()} of {n} points inside their cell")
print(f"uniform columns vs magpylib: max |diff| {np.abs(T[:, :, :3] - ref).max():.1e} (scale {np.abs(ref).max():.2f}); uniform-only call {np.abs(U - ref).max():.1e}")

gx, gw = np.polynomial.legendre.leggauss(64)


def slab(m, k):
    """Field of J_m = x'_k / h_k by quadrature over uniform slabs [t, h_k] along k."""
    out = np.zeros((n, 3))
    for r in range(n):
        lo, hi = -h[r, k], h[r, k]
        cuts = [lo] + ([X[r, k]] if lo < X[r, k] < hi else []) + [hi]
        acc = np.zeros(3)
        for a, b in zip(cuts[:-1], cuts[1:]):
            t = (a + b) / 2 + (b - a) / 2 * gx
            cen = np.zeros((len(t), 3))
            cen[:, k] = (t + hi) / 2
            half = np.tile(h[r], (len(t), 1))
            half[:, k] = (hi - t) / 2
            acc += ((b - a) / 2 * gw[:, None] * unif(np.tile(X[r], (len(t), 1)), cen, half, m)).sum(0)
        out[r] = acc / h[r, k] - unif(X[r : r + 1], np.zeros((1, 3)), h[r : r + 1], m)[0]
    return out


worst = 0
for m in range(3):
    for k in range(3):
        d = np.abs(T[:, :, 3 + 3 * m + k] - slab(m, k))
        worst = max(worst, d.max())
        print(f"  J_{'xyz'[m]} = xi_{'xyz'[k]}: max |diff| inside {d[inside].max():.1e}, outside {d[~inside].max():.1e}")
print(f"linear columns vs slab quadrature: worst {worst:.1e}; finite: {np.isfinite(T).all()}")
