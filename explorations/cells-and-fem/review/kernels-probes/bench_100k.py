"""D3 timing re-run: n = 100,000 pairs only. Same arrays/dtype as bench_lincell.py plus fairness variants:
 - magpylib uniform via 3 calls (as #52 does) vs one call with 3n stacked pairs
 - magpylib core function (no input checks / masks) via magpy.core.magnet_cuboid_Bfield (if available)
 - JAX jit with XLA threads as configured by XLA_FLAGS (set from the command line) and thread count printed
"""
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, "/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib/src")
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import magpylib as magpy  # noqa: E402
import numpy as np  # noqa: E402
from lincell import tensor  # noqa: E402


def best(f, k=5):
    f()
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return min(ts), np.median(ts)


rng = np.random.default_rng(2)
n = 100_000
X = rng.uniform(-3, 3, (n, 3))
h = rng.uniform(0.5, 1.5, (n, 3))
pols = [np.tile(e, (n, 1)) for e in np.eye(3)]
ns = lambda t: t * 1e9 / n  # noqa: E731
print(f"magpylib {magpy.__version__}, jax {jax.__version__}, devices {jax.devices()}, cpu_count {os.cpu_count()}, XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
t_mp = best(lambda: np.stack([magpy.func.cuboid_field("H", X, 2 * h, p) for p in pols], -1))
X3, h3, p3 = np.tile(X, (3, 1)), np.tile(h, (3, 1)), np.vstack(pols)
t_mp1 = best(lambda: magpy.func.cuboid_field("H", X3, 2 * h3, p3))
t_mpB = best(lambda: np.stack([magpy.func.cuboid_field("B", X, 2 * h, p) for p in pols], -1))
try:
    core = magpy.core.magnet_cuboid_Bfield
    t_core = best(lambda: np.stack([core(observers=X, dimensions=2 * h, polarizations=p) for p in pols], -1))
except Exception as e:  # noqa: BLE001
    t_core = (float("nan"), float("nan"))
    print("core:", e)
t_uni = best(lambda: tensor(X, h, linear=False))
t_lin = best(lambda: tensor(X, h, linear=True))
print(f"numpy, per pair: magpylib H 3 calls {ns(t_mp[0]):5.0f} ns (median {ns(t_mp[1]):.0f}) | 1 call 3n pairs {ns(t_mp1[0]):5.0f} | B 3 calls {ns(t_mpB[0]):5.0f} | core 3 calls {ns(t_core[0]):5.0f}"
      f" | this uniform 3x3 {ns(t_uni[0]):5.0f} (median {ns(t_uni[1]):.0f}) | linear 3x12 {ns(t_lin[0]):5.0f} (median {ns(t_lin[1]):.0f})", flush=True)
jit_lin = jax.jit(lambda X, h: tensor(X, h, linear=True, xp=jnp))
jit_uni = jax.jit(lambda X, h: tensor(X, h, linear=False, xp=jnp))
Xj, hj = jnp.asarray(X), jnp.asarray(h)
t_juni = best(lambda: jit_uni(Xj, hj))
t_jlin = best(lambda: jit_lin(Xj, hj))
diff = float(np.abs(np.asarray(jit_lin(Xj, hj)) - tensor(X, h)).max())
print(f"jit, per pair: uniform {ns(t_juni[0]):4.0f} ns (median {ns(t_juni[1]):.0f}) | linear {ns(t_jlin[0]):4.0f} ns (median {ns(t_jlin[1]):.0f}) | jit vs numpy max diff {diff:.0e}", flush=True)
# is the jit result dominated by compute or by the output allocation? time an elementwise op of the same output size
out = jit_lin(Xj, hj)
t_copy = best(lambda: out * 2.0)
print(f"reference: jnp elementwise op on the (n,3,12) output {ns(t_copy[0]):4.0f} ns/pair; numpy same {ns(best(lambda: np.asarray(out) * 2.0)[0]):4.0f} ns/pair")
