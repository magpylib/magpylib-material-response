"""B2/B3/B6 probes, mesh-only: TetGen counts and shape stats for the U-core one-surface vs per-cuboid meshes;
#52 cluster structure for the uniform cuboid grids; operator build at h=10 and 6.67 only."""
import itertools, time
import numpy as np, magpylib as magpy
from adapt import BLOCKS, uniform_mesh, cells
from adapt_tet import tetgen_mesh, volume
from magpylib_material_response.meshing import mesh_TriangularMesh
from magpylib_material_response.demag import _analyze_collection, _build_operator, _build_matvec, DENSE_BLOCK_MAX_ENTRIES

def box_tets(target_total):
    out = []
    for d, c in BLOCKS:
        corners = np.array(list(itertools.product(*[(-s / 2, s / 2) for s in d]))) + c
        box = magpy.magnet.TriangularMesh.from_ConvexHull(polarization=(0, 0, 0), points=corners)
        t = max(1, round(target_total / 88e-6 * np.prod(d)))
        out.append((t, np.array([cell.vertices for cell in mesh_TriangularMesh(box, t).sources_all])))
    return out

def radius_edge(V):
    v0 = V[:, 0]; A = V[:, 1:] - v0[:, None]
    b = 0.5 * (A ** 2).sum(-1)
    c = np.linalg.solve(A, b[..., None])[..., 0]
    R = np.linalg.norm(c, axis=1)
    e = np.array([np.linalg.norm(V[:, i] - V[:, j], axis=1) for i, j in itertools.combinations(range(4), 2)]).min(0)
    return R / e

print("B2/B3 TetGen counts (mesh only). one-surface: target -> cells; per-cuboid: total target -> per-body targets -> per-body cells")
for t in (30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000):
    V1 = tetgen_mesh(t)
    parts = box_tets(t)
    Vb = np.vstack([v for _, v in parts])
    re1, reb = radius_edge(V1), radius_edge(Vb)
    print(f"   target {t:>5}: one-surface {len(V1):>5} cells (vol max/min {volume(V1).max()/volume(V1).min():5.1f}, radius-edge median {np.median(re1):.2f} max {re1.max():.2f})"
          f" | per-cuboid targets {[p[0] for p in parts]} -> cells {[len(p[1]) for p in parts]} = {len(Vb):>5} (vol max/min {volume(Vb).max()/volume(Vb).min():5.1f}, r-e median {np.median(reb):.2f} max {reb.max():.2f})")

print("\nB6 #52 cluster structure of the uniform cuboid grids (positions only):")
for h in (10, 100 / 15, 5, 4, 100 / 30, 100 / 35):
    pos, dim = uniform_mesh(h * 1e-3)
    srcs = cells(pos, dim)
    positions, clusters = _analyze_collection(srcs, "point")
    desc = [f"{c['kind']}({len(c['indices'])}" + (f", shape {c['shape']}" if c['kind'] == 'grid' else "") + ")" for c in clusters]
    n = len(pos)
    print(f"   h {h:5.2f} mm, {n:>5} cells: {desc}; 9 n^2 = {9*n*n:.2e} {'>' if 9*n*n > DENSE_BLOCK_MAX_ENTRIES else '<='} DENSE_BLOCK_MAX_ENTRIES {DENSE_BLOCK_MAX_ENTRIES:.0e} -> {'sparsified CSR' if 9*n*n > DENSE_BLOCK_MAX_ENTRIES else 'dense block'}")

print("\nB6 operator build and matvec timing at h = 10 and 6.67 (iterative path, no GMRES):")
for h in (10, 100 / 15):
    pos, dim = uniform_mesh(h * 1e-3)
    srcs = cells(pos, dim)
    n = len(pos)
    positions, clusters = _analyze_collection(srcs, "point")
    chi = np.tile(np.eye(3) * 999.0, (n, 1, 1))
    t0 = time.perf_counter()
    ops = _build_operator(srcs, positions, clusters, chi, 1e-6, None, "point")
    t1 = time.perf_counter()
    mv = _build_matvec(n, chi, ops)
    v = np.random.default_rng(0).standard_normal(3 * n)
    mv(v); t2 = time.perf_counter(); [mv(v) for _ in range(10)]; t3 = time.perf_counter()
    kinds = [(o['kind'], type(o['M']).__name__ if 'M' in o else 'fft', o['M'].shape if 'M' in o else o['shape']) for o in ops]
    print(f"   h {h:5.2f}: {n} cells, ops {kinds}; build {t1-t0:.3f} s, matvec {(t3-t2)/10*1e3:.2f} ms")
