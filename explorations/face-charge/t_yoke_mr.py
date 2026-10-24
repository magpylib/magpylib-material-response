import numpy as np, time, sys, magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from yoke_case import YOKE, MAG, sensors
obs, tags = sensors()
res = {}
for h in [float(a) for a in sys.argv[1:]]:
    yoke = magpy.magnet.Cuboid(dimension=YOKE["dimension"], position=YOKE["position"], polarization=(0, 0, 0)); yoke.susceptibility = YOKE["chi"]
    mag = magpy.magnet.Cuboid(dimension=MAG["dimension"], position=MAG["position"], polarization=MAG["J0"]); mag.susceptibility = MAG["chi"]
    ny = int(round(np.prod(YOKE["dimension"]) / (h*1e-3)**3)); nm = int(round(np.prod(MAG["dimension"]) / (h*1e-3)**3))
    coll = magpy.Collection(mesh_Cuboid(yoke, ny), mesh_Cuboid(mag, nm))
    ncell = len(coll.sources_all)
    t = time.perf_counter(); out = apply_demag(coll); ts = time.perf_counter() - t
    B = magpy.getB(out, obs, sumup=True); res[h] = B
    print(f"MR h={h} mm: {ncell} cells, apply_demag {ts:.1f}s, Bz above magnet 1 mm {B[4,2]*1e3:.4f} mT", flush=True)
np.save("yoke_mr_" + "_".join(sys.argv[1:]) + ".npy", res, allow_pickle=True)
