"""Cuboid core field: magpylib main (NumPy) vs #854 (Array API) on NumPy, JAX eager and JAX jit. float64, CPU."""

import importlib.util
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
HERE = "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/pr854"
sys.path.insert(0, os.path.join(HERE, "overlay"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield as pr  # noqa: E402

spec = importlib.util.spec_from_file_location("main_cuboid", os.path.join(HERE, "main_field_BH_cuboid.py"))
main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_mod)
main = main_mod.magnet_cuboid_Bfield


def best(f, *a, k=3):
    out = f(*a)
    jax.block_until_ready(out) if not isinstance(out, np.ndarray) else None
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        out = f(*a)
        if not isinstance(out, np.ndarray):
            jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return min(ts), np.asarray(out)


rng = np.random.default_rng(0)
jit_pr = jax.jit(pr)
print(f"threads: jax devices {jax.devices()}, cpu count {os.cpu_count()}")
for n in (10_000, 100_000):
    obs = rng.uniform(-2, 2, (n, 3))
    dim = rng.uniform(0.5, 1.5, (n, 3))
    pol = rng.uniform(-1, 1, (n, 3))
    t_main, ref = best(main, obs, dim, pol)
    t_np, b_np = best(pr, obs, dim, pol)
    jo, jd, jp = (jnp.asarray(a) for a in (obs, dim, pol)); print("   dtypes:", obs.dtype, jo.dtype, "jit out:", jit_pr(jo, jd, jp).dtype, "x64:", jax.config.jax_enable_x64)
    t_eager, b_eager = best(pr, jo, jd, jp)
    try:
        t_jit, b_jit = best(jit_pr, jo, jd, jp)
        jit = f"{t_jit * 1e9 / n:6.0f} ns/pair ({t_main / t_jit:4.1f}x), max diff {np.abs(b_jit - ref).max():.1e}"
    except Exception as e:  # noqa: BLE001
        jit = f"fails: {type(e).__name__}: {str(e)[:120]}"
    print(f"n={n:>9,}  main {t_main * 1e9 / n:6.0f} ns/pair | #854 numpy {t_np * 1e9 / n:6.0f} (diff {np.abs(b_np - ref).max():.1e})"
          f" | jax eager {t_eager * 1e9 / n:6.0f} | jax jit {jit}", flush=True)
