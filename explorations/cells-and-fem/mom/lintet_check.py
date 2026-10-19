"""Validate lintet.tensor: uniform columns against magpylib; linear columns against the validated linear-cuboid
kernel, with the cuboid split into 6 tetrahedra carrying the same linear polarization."""

import itertools

import magpylib as magpy
import numpy as np
from lincell import tensor as cub_tensor
from lintet import tensor as tet_tensor

rng = np.random.default_rng(3)
MU0 = magpy.mu_0

# 1. uniform columns vs magpylib, random tetrahedra, points inside and outside
n = 400
V = rng.uniform(-1, 1, (n, 4, 3))
X = rng.uniform(-1.5, 1.5, (n, 3))
X[:40] = V[:40].mean(1) + 0.1 * (V[:40, 0] - V[:40].mean(1))  # some points inside
T = tet_tensor(X, V, np.ones(n))
ref = np.stack([magpy.func.tetrahedron_field("H", X, V, np.tile(e, (n, 1))) * MU0 for e in np.eye(3)], -1)
print(f"uniform columns vs magpylib tetrahedron: max |diff| {np.abs(T[:, :, :3] - ref).max():.1e} (scale {np.abs(ref).max():.1f})")

# 2. linear columns: cuboid = 6 Kuhn tetrahedra; same J = a + G (x' - c)/s on all
h = np.array([0.7, 1.1, 0.5])
corner = lambda bits: (2 * np.array(bits) - 1) * h  # noqa: E731
tets = []
for perm in itertools.permutations(range(3)):
    bits = [0, 0, 0]
    vs = [corner(bits)]
    for ax in perm:
        bits[ax] = 1
        vs.append(corner(bits))
    tets.append(vs)
tets = np.array(tets)  # (6, 4, 3), box centred at the origin
m = 300
Xo = np.vstack([rng.uniform(-1, 1, (100, 3)) * h, rng.uniform(-3, 3, (200, 3)) * h])
inside = np.all(np.abs(Xo) < h, axis=1)
Tc = cub_tensor(Xo, np.tile(h, (m, 1)))  # basis xi = x'/h (per axis)
acc = np.zeros((m, 3, 12))
s = 1.0
for tv in tets:
    Tt = tet_tensor(Xo, np.tile(tv, (m, 1, 1)), np.full(m, s))  # basis (x' - c_t)/s
    ct = tv.mean(0)
    # re-express in the cuboid's basis: J_m = x'_k / h_k = (s (x' - c_t)_k / s + c_t,k) / h_k
    conv = np.zeros((m, 3, 12))
    conv[:, :, :3] = Tt[:, :, :3]
    for mm in range(3):
        for k in range(3):
            conv[:, :, 3 + 3 * mm + k] = (s * Tt[:, :, 3 + 3 * mm + k] + ct[k] * Tt[:, :, mm]) / h[k]
    acc += conv
d = np.abs(acc - Tc)
print(f"6 tetrahedra vs linear cuboid: max |diff| inside {d[inside].max():.1e}, outside {d[~inside].max():.1e} (scale {np.abs(Tc).max():.2f}); finite {np.isfinite(acc).all()}")
