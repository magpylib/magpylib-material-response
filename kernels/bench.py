"""``charge_field`` and the cuboid, numpy against the compiled kernels (one
thread, all threads), at the sizes of #208's baseline. Best of ``REPEAT``.

    PYTHONPATH=src:<mmr_kernels install> python kernels/bench.py
"""

import os
import time

import mmr_kernels
import numpy as np
from magpylib.func import cuboid_field

from magpylib_material_response.face_charge import _triangle as T

REPEAT = 3


def best(f, *args, **kwargs):
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = f(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times), out


def rel(a, b):
    return np.max(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1))


def main():
    rng = np.random.default_rng(0)
    print(f"threads available: {os.cpu_count()}")
    print(
        "charge_field          numpy        rust 1 thread        rust all threads    max rel diff"
    )
    for n_tri, n_obs in ((500, 2000), (2000, 2000), (2000, 8000)):
        c0 = rng.uniform(-0.01, 0.01, (n_tri, 3))
        V = c0[:, None, :] + rng.uniform(-0.001, 0.001, (n_tri, 3, 3))
        fr, q = T.frames(V), rng.normal(size=(n_tri, 3))
        X = rng.uniform(-0.02, 0.02, (n_obs, 3))
        pairs = n_tri * n_obs
        t_np, F_np = best(T.charge_field, X, V, fr, q)
        t_1, F_1 = best(mmr_kernels.charge_field, X, V, q)
        t_n, F_n = best(mmr_kernels.charge_field, X, V, q, threads=True)
        print(
            f"{n_tri:5d} x {n_obs:5d}  {t_np / pairs * 1e9:6.1f} ns/pair  "
            f"{t_1 / pairs * 1e9:6.1f} ({t_np / t_1:4.1f}x)  "
            f"{t_n / pairs * 1e9:6.2f} ({t_np / t_n:5.1f}x)   "
            f"{max(rel(F_1, F_np), rel(F_n, F_np)):.1e}"
        )
    n = 100_000
    X = rng.uniform(-0.03, 0.03, (n, 3))
    dim = np.tile((0.01, 0.02, 0.03), (n, 1))
    pol = np.tile((0.3, -0.5, 1.2), (n, 1))
    t_mp, B_mp = best(cuboid_field, "B", X, dim, pol)
    t_1, B_1 = best(mmr_kernels.cuboid_field, "B", X, dim, pol)
    t_n, B_n = best(mmr_kernels.cuboid_field, "B", X, dim, pol, threads=True)
    print(
        f"cuboid B, {n} points: magpylib {t_mp / n * 1e9:.0f} ns/pt, rust {t_1 / n * 1e9:.0f} "
        f"({t_mp / t_1:.1f}x), all threads {t_n / n * 1e9:.1f} ({t_mp / t_n:.0f}x), "
        f"max rel diff {max(rel(B_1, B_mp), rel(B_n, B_mp)):.1e}"
    )


if __name__ == "__main__":
    main()
