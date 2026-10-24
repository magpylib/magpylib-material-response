"""Rotated bodies: exact consistency checks of multibody.py."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from multibody import Body, add_contact_lines, solve_system, field_B, moment, MU0
from yoke_case import YOKE, MAG, sensors
obs, tags = sensors()
rel = lambda B, R: np.max(np.linalg.norm(B - R, axis=1) / np.linalg.norm(R, axis=1))

def assembly(m, G=None, mag_turn=None, mag_dim=None):
    """Magnet on plate; G turns the whole assembly about the origin, mag_turn turns the magnet about its centre."""
    G = Rot.identity() if G is None else G
    mt = Rot.identity() if mag_turn is None else mag_turn
    bodies = [Body(m=m, position=G.apply(YOKE["position"]), orientation=G, dimension=YOKE["dimension"], chi=YOKE["chi"]),
              Body(m=m, position=G.apply(MAG["position"]), orientation=G * mt,
                   dimension=MAG["dimension"] if mag_dim is None else mag_dim, chi=MAG["chi"], J0=MAG["J0"])]
    add_contact_lines(bodies)
    for b in bodies: b.build()
    return bodies

# 1. regression: axis-aligned result unchanged (yoke_fc_3_4_6.npy holds the old m=3 result)
old = np.load("yoke_fc_3_4_6.npy", allow_pickle=True).item()[3]
bodies = assembly(3); c = solve_system(bodies); B0 = field_B(bodies, c, obs)
print(f"1. axis-aligned magnet on plate, m=3, polygon contact vs old rectangle contact: max rel diff {rel(B0, old):.1e}")

# 2. one rotated body in a uniform field: B(G x; G body, G H) = G B(x)
G = Rot.from_euler("zyx", [37, -21, 63], degrees=True)
Hu = np.array([0.3, 0.2, 1.0]) * 1e-2 / MU0
x = np.array([[0, 0, 4e-3], [12e-3, 3e-3, 1e-3], [-2e-3, 11e-3, -4e-3]])
plain = [Body((20e-3, 10e-3, 5e-3), chi=1000.0, m=3).build()]
Bp = field_B(plain, solve_system(plain, lambda p: np.tile(Hu, (len(p), 1))), x)
turned = [Body((20e-3, 10e-3, 5e-3), orientation=G, chi=1000.0, m=3).build()]
Bt = field_B(turned, solve_system(turned, lambda p: np.tile(G.apply(Hu), (len(p), 1))), G.apply(x))
print(f"2. one body turned with its field: max rel diff {rel(Bt, G.apply(Bp)):.1e}")

# 3. the whole magnet-on-plate turned: B(G x) = G B(x)
bodies = assembly(3, G=G); c = solve_system(bodies); Bg = field_B(bodies, c, G.apply(obs))
print(f"3. whole assembly turned: max rel diff {rel(Bg, G.apply(B0)):.1e}; moment {np.linalg.norm(moment(bodies, c) - G.apply(moment(assembly(3), solve_system(assembly(3)))))/np.linalg.norm(moment(bodies, c)):.1e}")

# 4. magnet turned 90 deg about z == magnet with x and y sides swapped (6x6 is square: use a 6x4x4 magnet)
d = (6e-3, 4e-3, 4e-3)
b90 = assembly(3, mag_turn=Rot.from_euler("z", 90, degrees=True), mag_dim=d)
B90 = field_B(b90, solve_system(b90), obs)
J = MAG["J0"]
swapped = [Body(m=3, **YOKE), Body(m=3, position=MAG["position"], dimension=(d[1], d[0], d[2]), chi=MAG["chi"],
                                    J0=Rot.from_euler("z", 90, degrees=True).apply(J))]
add_contact_lines(swapped)
for b in swapped: b.build()
Bs = field_B(swapped, solve_system(swapped), obs)
print(f"4. magnet turned 90 deg vs sides swapped: max rel diff {rel(B90, Bs):.1e}")
