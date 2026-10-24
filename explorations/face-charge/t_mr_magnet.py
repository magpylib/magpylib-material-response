import numpy as np, time, magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from engine import SymCuboidResponse, solve_in_field
from cases import SHAPES, observers, source
half = SHAPES["cube"]; obs, tags = observers(half)
one = np.array([d >= 1e-3 for (_, d) in tags])
r = SymCuboidResponse(2*half, m=3)
src = source(half)
c = solve_in_field(r, 0.05, [src], J0=(0, 0, 1.2))
Bb = r.B(obs, c)
B0 = magpy.getB(magpy.magnet.Cuboid(dimension=2*half, polarization=(0, 0, 1.2)), obs)
print("demag correction |B_chi - B_rigid| / |B_rigid| up to", np.max(np.linalg.norm(Bb-B0, axis=1)/np.linalg.norm(B0, axis=1)))
for n in [512, 4096, 32768]:
    s = source(half); s.susceptibility = 0.0
    mag = magpy.magnet.Cuboid(dimension=2*half, polarization=(0, 0, 1.2)); mag.susceptibility = 0.05
    t = time.perf_counter(); out = apply_demag(magpy.Collection(s, mesh_Cuboid(mag, target_elems=n))); ts = time.perf_counter()-t
    Bm = magpy.getB(out.children[1], obs, sumup=True)
    e = np.linalg.norm(Bm-Bb, axis=1)/np.linalg.norm(Bb, axis=1)
    corr = np.linalg.norm(Bb-B0, axis=1)
    ec = np.linalg.norm(Bm-Bb, axis=1)/corr
    print(f"MR {n:6d} cells ({ts:.1f}s): |MR - BEM|/|B| max d>=1mm {e[one].max():.1e}, all {e.max():.1e};  relative to the demag correction itself: d>=1mm {ec[one].max():.2f}")
