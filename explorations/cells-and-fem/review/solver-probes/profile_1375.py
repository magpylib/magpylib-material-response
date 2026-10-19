"""E2: demag_tensor split at the 1,375-cell cuboid U-core that split.py used (kernel vs np.unique dedup vs rest)."""
import cProfile, pstats, sys, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
from loguru import logger; logger.remove()
from adapt import cells, uniform_mesh
from magpylib_material_response.demag import demag_tensor
pos, dim = uniform_mesh(0.004); srcs = cells(pos, dim); n = len(srcs)
t0 = time.perf_counter(); demag_tensor(srcs); t = time.perf_counter() - t0
pr = cProfile.Profile(); pr.enable(); demag_tensor(srcs); pr.disable()
st = pstats.Stats(pr); tot = st.total_tt
tt = lambda pat: sum(v[3] for k, v in st.stats.items() if pat in k[2])
print(f"{n} cells, {n*n:,} pairs: demag_tensor {t:.2f} s (profiled {tot:.2f} s)")
for pat in ("point_block", "_dedup_blocks", "unique", "_pair_block", "_assemble_T_dense", "_analyze_collection"):
    print(f"   cum {tt(pat):6.3f} s {100 * tt(pat) / tot:5.1f} %  {pat}")
