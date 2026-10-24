"""material-response MoM on the same problem: 10 mm soft cube (chi=1000) near a magnet."""
import numpy as np, time, sys, magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from cases import SHAPES, observers, source
from conv2 import extrap
ref = np.load("p1_g3_6_8_12_16_20_24.npy", allow_pickle=True).item()
half = SHAPES["cube"]; obs, tags = observers(half)
far = np.array([d >= 0.5e-3 for (_, d) in tags]); one = np.array([d >= 1e-3 for (_, d) in tags])
ext, p, _ = extrap(ref["res"], 0, ref["levels"])
inter = sys.argv[1]
for n in [int(a) for a in sys.argv[2:]]:
    src = source(half); src.susceptibility = 0.0
    soft = magpy.magnet.Cuboid(dimension=2*half, polarization=(0, 0, 0)); soft.susceptibility = 1000.0
    mesh = mesh_Cuboid(soft, target_elems=n)
    coll = magpy.Collection(src, mesh)
    t = time.perf_counter()
    out = apply_demag(coll, interaction=inter)
    ts = time.perf_counter() - t
    softd = out.children[1]
    t = time.perf_counter(); B = magpy.getB(softd, obs, sumup=True); te = time.perf_counter() - t
    e = np.linalg.norm(B-ext, axis=1)/np.linalg.norm(ext, axis=1)
    print(f"MR {inter} cells={len(softd.sources_all):6d}: apply_demag {ts:7.2f}s, getB {te*1e3:6.1f} ms; max err d>=1mm {e[one].max():.1e}  d>=0.5mm {e[far].max():.1e}  d<0.5mm {e[~far].max():.1e}", flush=True)
