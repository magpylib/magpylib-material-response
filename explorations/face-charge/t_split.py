"""Two touching cubes with equal chi must reproduce the merged 20x10x10 block."""
import numpy as np, time
from multibody import Body, add_contact_lines, solve_system, field_B, moment, MU0
Hu = np.array([0.3, 0.2, 1.0]) * 1e-2 / MU0  # 10 mT, oblique
H_ext = lambda x: np.tile(Hu, (len(x), 1))
obs = np.array([[0, 0, 5.5e-3], [3e-3, 2e-3, 6e-3], [10.5e-3, 0, 0], [11e-3, 5.5e-3, 5.5e-3], [-4e-3, 6e-3, 1e-3], [0, 5.25e-3, 5.25e-3], [0.5e-3, 0, 5.25e-3]])
for chi in [1000.0, 1.0]:
    print(f"chi = {chi}")
    for m in [3, 4, 6]:
        t = time.perf_counter()
        merged = [Body((20e-3, 10e-3, 10e-3), chi=chi, m=m, lines=[[0.0], [], []]).build()]
        cm = solve_system(merged, H_ext)
        Bm = field_B(merged, cm, obs); mm = moment(merged, cm)
        tm = time.perf_counter() - t
        t = time.perf_counter()
        pair = [Body((10e-3,)*3, (-5e-3, 0, 0), chi=chi, m=m), Body((10e-3,)*3, (5e-3, 0, 0), chi=chi, m=m)]
        add_contact_lines(pair)
        for b in pair: b.build()
        cp = solve_system(pair, H_ext)
        Bp = field_B(pair, cp, obs); mp = moment(pair, cp)
        tp = time.perf_counter() - t
        e = np.linalg.norm(Bp - Bm, axis=1) / np.linalg.norm(Bm, axis=1)
        # interface: sum of the two coplanar charges should vanish
        a, b = pair
        ia = np.flatnonzero(a.pan.face == 0)  # a's +x face (at x = 0)
        ib = np.flatnonzero(b.pan.face == 1)  # b's -x face
        sa = cp[0].reshape(-1, 3)[ia, 0]; sb = cp[1].reshape(-1, 3)[ib, 0]
        print(f"  m={m}: dofs merged {merged[0].Nd} ({tm:.1f}s) pair {pair[0].Nd}+{pair[1].Nd} ({tp:.1f}s) | "
              f"moment diff {np.linalg.norm(mp-mm)/np.linalg.norm(mm):.1e} | B diff max {e.max():.1e} | "
              f"interface |sa+sb|/|sa| {np.linalg.norm(sa+sb)/np.linalg.norm(sa):.1e}", flush=True)
