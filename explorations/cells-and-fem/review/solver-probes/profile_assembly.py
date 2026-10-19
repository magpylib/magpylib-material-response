"""Attack 7: what fraction of demag_tensor's assembly is kernel vs Python overhead. Small cases only."""
import cProfile, pstats, io, sys, time
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy
from loguru import logger
logger.remove()
from adapt import cells, uniform_mesh
from adapt_tet import tetgen_mesh
from magpylib_material_response.demag import demag_tensor
from magpylib_material_response import newell

def run(label, srcs, n_rep=3):
    demag_tensor(srcs)  # warm
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); demag_tensor(srcs); ts.append(time.perf_counter() - t0)
    t = min(ts)
    n = len(srcs)
    pr = cProfile.Profile(); pr.enable(); demag_tensor(srcs); pr.disable()
    st = pstats.Stats(pr)
    tot = st.total_tt
    def tt(pattern):
        return sum(v[3] for k, v in st.stats.items() if pattern in k[2] or pattern in k[0])  # cumulative time of matching functions
    print(f"\n== {label}: {n} cells, {n*n:,} pairs, best of {n_rep}: {t:.3f} s = {t / (n * n) * 1e9:,.0f} ns/pair (profiled total {tot:.3f} s)")
    for pat in ("point_block", "_point_block_exact", "_far_field_block", "BHJM_magnet_cuboid", "magnet_cuboid", "_dedup_blocks", "np.unique", "unique",
                "getH", "getBH_level", "BHJM_magnet_tetrahedron", "magnet_tetrahedron", "triangle", "BHJM_triangle", "_point_block_T", "_pair_block",
                "_analyze_collection", "analyze_collection", "as_quat", "from_quat", "apply", "polarization", "format_src_inputs", "check_format_input"):
        v = tt(pat)
        if v > 0.0005:
            print(f"   cum {v:7.3f} s  {100 * v / tot:5.1f} %  {pat}")
    s = io.StringIO(); st.stream = s; st.sort_stats("cumulative").print_stats(18); print("\n".join(s.getvalue().splitlines()[6:30]))

pos, dim = uniform_mesh(0.01); run("U-core cuboids h=10 (point rule, 3 grid clusters)", cells(pos, dim))
pos, dim = uniform_mesh(0.005); run("U-core cuboids h=5", cells(pos, dim), 2)
V = tetgen_mesh(50); run("U-core tets target 50 (getH path)", [magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V])
V = tetgen_mesh(200); run("U-core tets target 200", [magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V], 2)
# raw kernel cost for comparison
n = 200_000
rng = np.random.default_rng(0)
d = rng.uniform(-0.05, 0.05, (n, 3)); dim1 = np.array([0.01, 0.01, 0.01])
newell.point_block(d, dim1)
t0 = time.perf_counter(); newell.point_block(d, dim1); t = time.perf_counter() - t0
print(f"\nraw newell.point_block kernel: {t / n * 1e9:,.0f} ns/pair ({n:,} pairs)")
V = rng.uniform(-0.01, 0.01, (n, 4, 3)); X = rng.uniform(-0.03, 0.03, (n, 3)); pol = np.tile([0, 0, 1.0], (n, 1))
magpy.func.tetrahedron_field("H", X, V, pol)
t0 = time.perf_counter(); magpy.func.tetrahedron_field("H", X, V, pol); t = time.perf_counter() - t0
print(f"raw magpylib tetrahedron_field (one pol): {t / n * 1e9:,.0f} ns/pair -> x3 pols = {3 * t / n * 1e9:,.0f} ns/pair")
