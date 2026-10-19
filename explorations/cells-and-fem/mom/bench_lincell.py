"""Cost per source-observer pair: magpylib's uniform cuboid kernel vs the linear-cell kernel, NumPy and jax.jit."""

import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
if os.environ.get("MAGPYLIB_SRC"):  # a magpylib checkout, when the JAX env has none installed
    sys.path.insert(0, os.environ["MAGPYLIB_SRC"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import magpylib as magpy  # noqa: E402
import numpy as np  # noqa: E402
from lincell import tensor  # noqa: E402


def best(f, k=3):
    f()
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return min(ts)


rng = np.random.default_rng(2)
jit_lin = jax.jit(lambda X, h: tensor(X, h, linear=True, xp=jnp))
jit_uni = jax.jit(lambda X, h: tensor(X, h, linear=False, xp=jnp))
print(f"magpylib {magpy.__version__}, jax {jax.__version__}, float64, CPU ({os.cpu_count()} cores)")
for n in (100_000, 500_000):
    X = rng.uniform(-3, 3, (n, 3))
    h = rng.uniform(0.5, 1.5, (n, 3))
    pols = [np.tile(e, (n, 1)) for e in np.eye(3)]
    t_mp = best(lambda: np.stack([magpy.func.cuboid_field("H", X, 2 * h, p) for p in pols], -1))
    t_uni = best(lambda: tensor(X, h, linear=False))
    t_lin = best(lambda: tensor(X, h, linear=True))
    Xj, hj = jnp.asarray(X), jnp.asarray(h)
    t_juni = best(lambda: jit_uni(Xj, hj))
    t_jlin = best(lambda: jit_lin(Xj, hj))
    diff = float(np.abs(np.asarray(jit_lin(Xj, hj)) - tensor(X, h)).max())
    ns = lambda t: t * 1e9 / n  # noqa: E731
    print(f"n={n:>7,} pairs | magpylib uniform 3x3: {ns(t_mp):5.0f} ns | this uniform 3x3: numpy {ns(t_uni):5.0f}, jit {ns(t_juni):4.0f} ns"
          f" | linear 3x12: numpy {ns(t_lin):5.0f}, jit {ns(t_jlin):4.0f} ns (jit vs numpy max diff {diff:.0e})", flush=True)
