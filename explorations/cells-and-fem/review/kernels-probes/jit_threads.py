"""Does jax.jit's 67 ns/pair use one core or 14? Compare process CPU time with wall time."""
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, "/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib/src")
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from lincell import tensor  # noqa: E402

rng = np.random.default_rng(2)
n = 100_000
X = rng.uniform(-3, 3, (n, 3))
h = rng.uniform(0.5, 1.5, (n, 3))
Xj, hj = jnp.asarray(X), jnp.asarray(h)
jit_lin = jax.jit(lambda X, h: tensor(X, h, linear=True, xp=jnp))
jax.block_until_ready(jit_lin(Xj, hj))
reps = 50
for name, f in (("numpy linear", lambda: tensor(X, h)), ("jit linear", lambda: jax.block_until_ready(jit_lin(Xj, hj)))):
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(reps):
        f()
    c1, w1 = time.process_time(), time.perf_counter()
    print(f"{name:14s}: wall {(w1 - w0) / reps * 1e9 / n:6.0f} ns/pair, process CPU {(c1 - c0) / reps * 1e9 / n:6.0f} ns/pair, CPU/wall {(c1 - c0) / (w1 - w0):.1f}  (XLA_FLAGS={os.environ.get('XLA_FLAGS', '')})")
