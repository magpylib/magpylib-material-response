import sys, time, numpy as np, magpylib as magpy
sys.path.insert(0, sys.argv[1])
import magpylib_material_response as mr
from magpylib_material_response.logging_config import configure_logging
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from yoke_case import YOKE, MAG, sensors
configure_logging(level="INFO")
print("package:", mr.__file__, flush=True)
h = float(sys.argv[2])
yoke = magpy.magnet.Cuboid(dimension=YOKE["dimension"], position=YOKE["position"], polarization=(0, 0, 0)); yoke.susceptibility = YOKE["chi"]
mag = magpy.magnet.Cuboid(dimension=MAG["dimension"], position=MAG["position"], polarization=MAG["J0"]); mag.susceptibility = MAG["chi"]
ny = int(round(np.prod(YOKE["dimension"]) / (h*1e-3)**3)); nm = int(round(np.prod(MAG["dimension"]) / (h*1e-3)**3))
coll = magpy.Collection(mesh_Cuboid(yoke, ny), mesh_Cuboid(mag, nm))
t = time.perf_counter(); out = apply_demag(coll, min_log_time=0); ts = time.perf_counter() - t
obs, tags = sensors()
B = magpy.getB(out, obs, sumup=True)
np.save(sys.argv[3], {h: B}, allow_pickle=True)
print(f"h={h} mm: {len(coll.sources_all)} cells, apply_demag {ts:.1f}s", flush=True)
