"""Non-conforming contact: the two halves meshed differently must still converge to the merged block."""
import numpy as np, time
from multibody import Body, add_contact_lines, solve_system, field_B, moment, MU0
Hu = np.array([0.3, 0.2, 1.0]) * 1e-2 / MU0
H_ext = lambda x: np.tile(Hu, (len(x), 1))
obs = np.array([[0, 0, 5.5e-3], [3e-3, 2e-3, 6e-3], [10.5e-3, 0, 0], [11e-3, 5.5e-3, 5.5e-3], [-4e-3, 6e-3, 1e-3], [0, 5.25e-3, 5.25e-3], [0.5e-3, 0, 5.25e-3]])
chi = 1000.0
ref = [Body((20e-3, 10e-3, 10e-3), chi=chi, m=8, lines=[[0.0], [], []]).build()]
cr = solve_system(ref, H_ext); Br = field_B(ref, cr, obs); mr = moment(ref, cr)
for ma, mb, gb in [(3, 4, 2.0), (4, 6, 2.0), (6, 8, 2.0), (4, 4, 2.0)]:
    pair = [Body((10e-3,)*3, (-5e-3, 0, 0), chi=chi, m=ma), Body((10e-3,)*3, (5e-3, 0, 0), chi=chi, m=mb, g=gb)]
    add_contact_lines(pair)
    for b in pair: b.build()
    cp = solve_system(pair, H_ext)
    e = np.linalg.norm(field_B(pair, cp, obs) - Br, axis=1) / np.linalg.norm(Br, axis=1)
    conf = [ma, mb, gb]
    same = [Body((10e-3,)*3, (-5e-3, 0, 0), chi=chi, m=ma), Body((10e-3,)*3, (5e-3, 0, 0), chi=chi, m=ma)]
    add_contact_lines(same)
    for b in same: b.build()
    es = np.linalg.norm(field_B(same, solve_system(same, H_ext), obs) - Br, axis=1) / np.linalg.norm(Br, axis=1)
    print(f"halves m={ma} | m={mb},g={gb} (non-conforming): B err vs merged m=8: max {e.max():.1e}, moment {np.linalg.norm(moment(pair, cp)-mr)/np.linalg.norm(mr):.1e} | both m={ma} (conforming): max {es.max():.1e}", flush=True)
