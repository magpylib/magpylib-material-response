"""U-core as its three cuboids, each tetrahedralized on its own (cells do not conform at the joints)."""
import itertools, json, sys
import magpylib as magpy
import numpy as np
from adapt import BLOCKS
from adapt_tet import run
from magpylib_material_response.meshing import mesh_TriangularMesh

def box_tets(target_per_m3):
    V = []
    for d, c in BLOCKS:
        corners = np.array(list(itertools.product(*[(-s / 2, s / 2) for s in d]))) + c
        box = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
        V.append([cell.vertices for cell in mesh_TriangularMesh(box, max(1, round(target_per_m3 * np.prod(d)))).sources_all])
    return np.vstack(V)

for t in sys.argv[1:]:
    print(json.dumps(run(box_tets(float(t) / 88e-6), {"target_total": int(t)})[0]), flush=True)
