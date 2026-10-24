"""Magnet turned 30 degrees about z on the plate: face-charge levels and MoM levels.
usage: t_yoke_turn.py fc 3 4 6  |  t_yoke_turn.py mom 1 0.5"""
import numpy as np, time, sys, magpylib as magpy
from scipy.spatial.transform import Rotation as Rot
from yoke_case import YOKE, MAG, sensors
TURN = Rot.from_euler("z", 30, degrees=True)
obs, tags = sensors()
res = {}
if sys.argv[1] == "fc":
    from multibody import Body, add_contact_lines, solve_system, field_B
    for m in [int(a) for a in sys.argv[2:]]:
        t = time.perf_counter()
        bodies = [Body(m=m, **YOKE), Body(m=m, orientation=TURN, **MAG)]
        add_contact_lines(bodies)
        for b in bodies: b.build()
        c = solve_system(bodies)
        res[m] = field_B(bodies, c, obs)
        print(f"face charge m={m}: dofs {bodies[0].Nd}+{bodies[1].Nd}, {time.perf_counter()-t:.1f}s, Bz above magnet 1 mm {res[m][4,2]*1e3:.4f} mT", flush=True)
else:
    from magpylib_material_response.demag import apply_demag
    from magpylib_material_response.meshing import mesh_Cuboid
    for h in [float(a) for a in sys.argv[2:]]:
        yoke = magpy.magnet.Cuboid(dimension=YOKE["dimension"], position=YOKE["position"], polarization=(0, 0, 0)); yoke.susceptibility = YOKE["chi"]
        mag = magpy.magnet.Cuboid(dimension=MAG["dimension"], position=MAG["position"], orientation=TURN, polarization=MAG["J0"]); mag.susceptibility = MAG["chi"]
        ny = int(round(np.prod(YOKE["dimension"]) / (h*1e-3)**3)); nm = int(round(np.prod(MAG["dimension"]) / (h*1e-3)**3))
        coll = magpy.Collection(mesh_Cuboid(yoke, ny), mesh_Cuboid(mag, nm))
        t = time.perf_counter(); out = apply_demag(coll); ts = time.perf_counter() - t
        res[h] = magpy.getB(out, obs, sumup=True)
        print(f"MoM h={h} mm: {len(coll.sources_all)} cells, apply_demag {ts:.1f}s, Bz above magnet 1 mm {res[h][4,2]*1e3:.4f} mT", flush=True)
np.save(f"turn_{sys.argv[1]}_" + "_".join(sys.argv[2:]) + ".npy", res, allow_pickle=True)
