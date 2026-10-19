"""E1: is the jax.jit number single- or multi-core? cpu time / wall time over 30 calls at n = 100,000."""
import os, sys, time
os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/pr854/overlay")
import jax, jax.numpy as jnp, numpy as np
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield as pr
n = 100_000; rng = np.random.default_rng(0)
obs, dim, pol = rng.uniform(-2, 2, (n, 3)), rng.uniform(0.5, 1.5, (n, 3)), rng.uniform(-1, 1, (n, 3))
jo, jd, jp = (jnp.asarray(a) for a in (obs, dim, pol)); f = jax.jit(pr); jax.block_until_ready(f(jo, jd, jp))
print("XLA_FLAGS =", os.environ.get("XLA_FLAGS"))
for name, fn in (("numpy #854", lambda: pr(obs, dim, pol)), ("jax.jit", lambda: jax.block_until_ready(f(jo, jd, jp))),
                 ("jax.jit incl. numpy->jax transfer", lambda: jax.block_until_ready(f(jnp.asarray(obs), jnp.asarray(dim), jnp.asarray(pol))))):
    fn()
    t0, c0 = time.perf_counter(), time.process_time()
    for _ in range(30): fn()
    t, c = time.perf_counter() - t0, time.process_time() - c0
    print(f"  {name:36s} wall {t / 30 * 1e9 / n:5.0f} ns/pair, cpu/wall = {c / t:.2f}")
