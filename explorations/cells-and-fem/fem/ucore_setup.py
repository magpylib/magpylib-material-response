"""The docs example's U-core, coil and probe line, verbatim geometry."""
import magpylib as magpy
import numpy as np
from magpylib_material_response.polyline import create_polyline_fillet, move_grid_along_polyline

a = 0.02
yoke = magpy.magnet.Cuboid(dimension=(0.10, a, a))
leg_left = magpy.magnet.Cuboid(dimension=(a, a, 0.06), position=(-0.04, 0, 0.04))
leg_right = magpy.magnet.Cuboid(dimension=(a, a, 0.06), position=(0.04, 0, 0.04))
core = magpy.Collection(yoke, leg_left, leg_right)
core.susceptibility = 999

h = 0.014
square = np.array([[0, h, 0], [0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h], [0, h, 0]])
turn = create_polyline_fillet(square, max_radius=0.006, N=6)
pitch = 0.0022
x_turns = (np.arange(8) - 3.5) * pitch
y_layers = h + np.arange(2) * pitch
pack = np.array([(xt, yl, 0) for xt in x_turns for yl in y_layers])
windings = move_grid_along_polyline(turn, pack)
coil = magpy.Collection(*(magpy.current.Polyline(current=25, vertices=v) for v in windings))
system = magpy.Collection(core, coil)

x = np.linspace(-0.08, 0.08, 161)
line_pts = np.stack([x, np.zeros_like(x), np.full_like(x, 0.10)], axis=-1)

u_profile = np.array([(-0.05, -0.01), (0.05, -0.01), (0.05, 0.07), (0.03, 0.07), (0.03, 0.01), (-0.03, 0.01), (-0.03, 0.07), (-0.05, 0.07)])
cap = [(0, 1, 4), (1, 3, 4), (1, 2, 3), (0, 4, 5), (0, 5, 7), (5, 6, 7)]
n = len(u_profile)
vertices = np.array([(ux, uy, uz) for uy in (-0.01, 0.01) for ux, uz in u_profile])
faces = list(cap) + [tuple(i + n for i in tri) for tri in cap]
for i in range(n):
    j = (i + 1) % n
    faces += [(i, j, j + n), (i, j + n, i + n)]
u_iron = magpy.magnet.TriangularMesh(vertices=vertices, faces=np.array(faces), reorient_faces=True)

def probe(coll):
    B = magpy.getB(coll, line_pts)
    i_pole = np.argmin(np.abs(x - 0.04))
    return float(np.abs(B[:, 2]).max()), float(B[i_pole, 2])
