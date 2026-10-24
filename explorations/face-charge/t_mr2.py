import numpy as np, magpylib as magpy
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid
from engine import CuboidResponse, solve_in_field, MU0
from cases import SHAPES, observers, source
np.set_printoptions(precision=4, linewidth=180, suppress=True)
half = SHAPES["cube"]; obs, tags = observers(half)
r = CuboidResponse(2*half, m=6, g=2.5)
src = source(half)
c = solve_in_field(r, 1000., [src])
print("BEM moment (A m2)", r.moment(c))
Bb = r.B(obs, c)
for n in [512, 4096]:
    src = source(half); src.susceptibility = 0.0
    soft = magpy.magnet.Cuboid(dimension=2*half, polarization=(0, 0, 0)); soft.susceptibility = 1000.0
    out = apply_demag(magpy.Collection(src, mesh_Cuboid(soft, target_elems=n)))
    cells = out.children[1].sources_all
    m = sum(np.array(cl.polarization)*np.prod(cl.dimension) for cl in cells)/MU0
    print(n, "MR moment", m, " src moment-check", src.polarization)
    Bm = magpy.getB(out.children[1], obs, sumup=True)
    for k in [0, 1, 2, 9, 10, 11, 15, 16, 17]:
        print("  ", tags[k], "BEM", Bb[k]*1e3, "MR", Bm[k]*1e3, "mT")
