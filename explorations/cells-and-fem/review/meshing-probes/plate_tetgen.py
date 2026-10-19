"""B5: mesh-only TetGen counts on the 40x40x1 mm plate. No solve."""
import inspect, itertools, sys
import numpy as np, magpylib as magpy, tetgen
from magpylib_material_response.meshing import mesh_TriangularMesh

print("tetgen", tetgen.__version__ if hasattr(tetgen, "__version__") else "?")
print("tetrahedralize signature:", inspect.signature(tetgen.TetGen.tetrahedralize))
L, T = 0.04, 0.001
corners = np.array(list(itertools.product((-L / 2, L / 2), (-L / 2, L / 2), (-T / 2, T / 2))))
box = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
V = L * L * T
print(f"plate volume {V*1e9:.0f} mm^3; maxvolume at target 50/200/1000/5000: {[round(V/t*1e9,2) for t in (50,200,1000,5000)]} mm^3 "
      f"-> cube-equivalent edge {[round((V/t)**(1/3)*1e3,2) for t in (50,200,1000,5000)]} mm vs thickness 1 mm")

print("\nmesh_TriangularMesh counts: target -> [default (1.5,10), relaxed (2,5), quality=False]")
for target in (50, 200, 1000, 5000):
    row = []
    for kw in ({}, {"minratio": 2.0, "mindihedral": 5.0}, {"quality": False}):
        try:
            row.append(len(mesh_TriangularMesh(box, target, **kw).sources_all))
        except Exception as e:
            row.append(f"ERR {type(e).__name__}: {str(e)[:60]}")
    print(f"   {target:>5}: {row}")

def tg(**kw):
    t = tetgen.TetGen(box.vertices.astype(float), box.faces.astype(np.int32))
    t.tetrahedralize(order=1, quiet=True, **kw)
    return len(t.elem)

print("\ndirect tetgen, maxvolume = V/50, quality=True, varying minratio x mindihedral:")
for mr in (1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
    row = []
    for md in (10.0, 5.0, 1.0, 0.0):
        try:
            row.append(tg(quality=True, minratio=mr, mindihedral=md, fixedvolume=True, maxvolume=V / 50))
        except Exception as e:
            row.append(f"ERR {str(e)[:40]}")
    print(f"   minratio {mr:>4}: mindihedral 10/5/1/0 -> {row}")

print("\ndirect tetgen, other switches:")
for label, kw in (("quality=True only (no -a)", dict(quality=True)),
                  ("quality=True, maxvolume=V (fixedvolume)", dict(quality=True, fixedvolume=True, maxvolume=V)),
                  ("quality=True, maxvolume=V/50, fixedvolume=False", dict(quality=True, fixedvolume=False, maxvolume=V / 50)),
                  ("quality=False, maxvolume=V/50 fixedvolume", dict(quality=False, fixedvolume=True, maxvolume=V / 50)),
                  ("quality=False, maxvolume=V/1000 fixedvolume", dict(quality=False, fixedvolume=True, maxvolume=V / 1000)),
                  ("quality=False, nobisect", dict(quality=False, nobisect=True)),
                  ("quality=True, minratio=2, mindihedral=5, maxvolume=V/50", dict(quality=True, minratio=2.0, mindihedral=5.0, fixedvolume=True, maxvolume=V / 50)),
                  ("quality=True, minratio=2, mindihedral=5, maxvolume=V/5000", dict(quality=True, minratio=2.0, mindihedral=5.0, fixedvolume=True, maxvolume=V / 5000)),
                  ("quality=True, minratio=1.5, mindihedral=10, maxvolume=V/5000", dict(quality=True, minratio=1.5, mindihedral=10.0, fixedvolume=True, maxvolume=V / 5000)),
                  ("quality=True, minratio=1.5, mindihedral=10, maxvolume=V/20000", dict(quality=True, minratio=1.5, mindihedral=10.0, fixedvolume=True, maxvolume=V / 20000))):
    try:
        print(f"   {label}: {tg(**kw)} tets")
    except Exception as e:
        print(f"   {label}: ERR {type(e).__name__}: {str(e)[:80]}")

# the U-core one-surface mesh: does target_elems bite there?
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
