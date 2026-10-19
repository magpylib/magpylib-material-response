"""Attack 2, third geometry: icosphere (320 faces) with chi = 1000, exact m = 3 V_poly H0 chi/(chi+3) (+O(dr^2) ~ 3e-5).
Also the cube at TetGen target 200 for f = 0.4472, 0.5, 0.55 (does the zero crossing f* move with the mesh?)."""
import os, sys, time, itertools
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy
from loguru import logger; logger.remove()
import p1tet
from magpylib_material_response.meshing import mesh_TriangularMesh
MU0 = magpy.mu_0

def icosphere(nsub):
    t = (1 + 5**0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = list(v / np.linalg.norm(v, axis=1)[:, None])
    for _ in range(nsub):
        cache, nf = {}, []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]; v.append(m / np.linalg.norm(m)); cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v)

def tets_of(points, target):
    body = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=points)
    return np.array([s.vertices for s in mesh_TriangularMesh(body, target).sources_all])

def moment(V, basis, colloc, frac, chi=1000.0):
    os.environ["COLLOC"] = colloc; os.environ["FRAC"] = str(frac)
    t0 = time.perf_counter()
    u, _ = p1tet.assemble_solve(V, chi, lambda X: np.broadcast_to([0, 0, 0.01], X.shape), basis)
    vol = np.abs(np.linalg.det(V[:, 1:] - V[:, :1])) / 6
    return (u[:, 2] * vol).sum() / MU0, time.perf_counter() - t0

chi = 1000.0
for target in (50, 150):
    V = tets_of(icosphere(2) * 0.005, target)
    vol = (np.abs(np.linalg.det(V[:, 1:] - V[:, :1])) / 6).sum()
    m_ref = 3 * vol * (0.01 / MU0) * chi / (chi + 3)
    print(f"icosphere r=5mm, 320 faces: V_poly/V_sphere = {vol / (4 / 3 * np.pi * 0.005**3):.5f}, {len(V)} tets, m_ref = {m_ref:.6e}")
    m, t = moment(V, "p0", "child8", 0.5); print(f"   p0 centroid          err {100 * (m / m_ref - 1):+.3f} %  ({t:.2f} s)")
    m, t = moment(V, "p1", "child8", 0.5); print(f"   p1 child8 LSQ        err {100 * (m / m_ref - 1):+.3f} %  ({t:.2f} s)")
    for f in (0.3, 0.4, 0.4472, 0.48, 0.5, 0.52, 0.55, 0.6, 0.7):
        m, t = moment(V, "p1", "corner4", f); print(f"   p1 corner4 f={f:<6} err {100 * (m / m_ref - 1):+.3f} %  ({t:.2f} s)")
    sys.stdout.flush()

print("\ncube, TetGen target 200 (zero crossing vs mesh):")
corners = np.array(list(itertools.product((-0.005, 0.005), repeat=3)))
V = tets_of(corners, 200)
for f in (0.4472, 0.5, 0.55):
    m, t = moment(V, "p1", "corner4", f); print(f"   {len(V)} tets, f={f}: err vs 0.02889 {100 * (m / 0.02889 - 1):+.3f} %  ({t:.1f} s)")
    sys.stdout.flush()
