"""Magnet on plate: where the time goes, repeat solves after one factorization,
and the error if the plate is meshed without the magnet's footprint lines."""
import numpy as np, time, scipy.linalg
from multibody import Body, add_contact_lines, cross_block, field_B, MU0
from yoke_case import YOKE, MAG, sensors
from yoke_compare import richardson
obs, tags = sensors()
fc = np.load("yoke_fc_3_4_6.npy", allow_pickle=True).item()
ref, _ = richardson(sorted(fc), [fc[m] for m in sorted(fc)])
far = np.array([d >= 0.5e-3 for _, d in tags])

def system(bodies, cross):
    offs = np.cumsum([0] + [b.Nd for b in bodies]); N = offs[-1]; nb = len(bodies)
    A = np.zeros((N + nb, N + nb))
    for a_, a in enumerate(bodies):
        sl = slice(offs[a_], offs[a_ + 1]); idx = np.arange(offs[a_], offs[a_ + 1])
        A[sl, sl] = -a.G; A[idx, idx] += (1 / a.chi + 0.5) * a.s.mass
        e = np.zeros(a.Nd); e[0::3] = a.pan.area
        A[sl, N + a_] = e; A[N + a_, sl] = e / a.pan.area.sum()
        for b_, b in enumerate(bodies):
            if b_ != a_: A[sl, offs[b_]:offs[b_ + 1]] = -cross[(a_, b_)]
    return A, offs

def rhs(bodies, offs, J0):
    r = np.zeros(offs[-1] + len(bodies))
    for a_, (a, j) in enumerate(zip(bodies, J0)):
        if j is not None:
            r[offs[a_]:offs[a_ + 1]] = a.s.rhs_const_normal(a.pan.normal @ (np.asarray(j) / MU0)) / a.chi
    return r

for m, contact in [(3, True), (4, True), (4, False), (6, False)]:
    bodies = [Body(m=m, **YOKE), Body(m=m, **MAG)]
    if contact: add_contact_lines(bodies)
    t0 = time.perf_counter(); bodies[0].build(); t1 = time.perf_counter(); bodies[1].build(); t2 = time.perf_counter()
    cross = {(0, 1): cross_block(bodies[0], bodies[1]), (1, 0): cross_block(bodies[1], bodies[0])}; t3 = time.perf_counter()
    A, offs = system(bodies, cross)
    lu = scipy.linalg.lu_factor(A, check_finite=False); t4 = time.perf_counter()
    x = scipy.linalg.lu_solve(lu, rhs(bodies, offs, [None, MAG["J0"]]), check_finite=False)
    t = time.perf_counter(); k = 20
    for i in range(k):  # e.g. a new magnet polarization each time
        x = scipy.linalg.lu_solve(lu, rhs(bodies, offs, [None, (0.1 * np.sin(i), 0, 1.2)]), check_finite=False)
    tsol = (time.perf_counter() - t) / k
    x = scipy.linalg.lu_solve(lu, rhs(bodies, offs, [None, MAG["J0"]]), check_finite=False)
    c = [x[offs[i]:offs[i + 1]] for i in range(2)]
    t = time.perf_counter(); B = field_B(bodies, c, obs); tB = (time.perf_counter() - t) / len(obs)
    e = np.linalg.norm(B - ref, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"m={m} footprint lines={contact}: dofs {bodies[0].Nd}+{bodies[1].Nd} | plate self {t1-t0:.1f}s, magnet self {t2-t1:.2f}s, "
          f"cross {t3-t2:.1f}s, LU {t4-t3:.1f}s | repeat solve {tsol*1e3:.1f} ms, B {tB*1e6:.0f} us/sensor | max err {e.max():.1e} "
          f"(above magnet {e[0::4].max():.0e}, beside {e[1::4].max():.0e}, under plate {e[2::4].max():.0e}, side {e[3::4].max():.0e})", flush=True)
