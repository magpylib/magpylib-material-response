import numpy as np, time, sys
from multibody import Body, add_contact_lines, solve_system, field_B, moment
from yoke_case import YOKE, MAG, sensors
obs, tags = sensors()
res = {}
for m in [int(a) for a in sys.argv[1:]]:
    t = time.perf_counter()
    bodies = [Body(m=m, **YOKE), Body(m=m, **MAG)]
    add_contact_lines(bodies)
    for b in bodies: b.build()
    tb = time.perf_counter() - t
    t = time.perf_counter(); c = solve_system(bodies); ts = time.perf_counter() - t
    B = field_B(bodies, c, obs); res[m] = B
    print(f"m={m}: dofs yoke {bodies[0].Nd} + magnet {bodies[1].Nd}, build {tb:.1f}s, cross+solve {ts:.1f}s, Bz above magnet 1 mm {B[4,2]*1e3:.4f} mT", flush=True)
np.save("yoke_fc_" + "_".join(sys.argv[1:]) + ".npy", res, allow_pickle=True)
