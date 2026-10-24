"""Why the magnet-on-plate MoM jumps from 7 s (0.5 mm) to >12 min, 10.8 GB (0.4 mm)."""
import sys, time, numpy as np, magpylib as magpy
sys.path.insert(0, sys.argv[1])  # explorations/face-charge, for yoke_case
import magpylib_material_response as mr
from magpylib_material_response import demag as D
from magpylib_material_response.meshing import mesh_Cuboid
from yoke_case import YOKE, MAG
print("package:", mr.__file__)
for h in [0.6, 0.5, 0.45, 0.4]:
    yoke = magpy.magnet.Cuboid(dimension=YOKE["dimension"], position=YOKE["position"], polarization=(0, 0, 0)); yoke.susceptibility = YOKE["chi"]
    mag = magpy.magnet.Cuboid(dimension=MAG["dimension"], position=MAG["position"], polarization=MAG["J0"]); mag.susceptibility = MAG["chi"]
    ny = int(round(np.prod(YOKE["dimension"]) / (h*1e-3)**3)); nm = int(round(np.prod(MAG["dimension"]) / (h*1e-3)**3))
    cells = list(magpy.Collection(mesh_Cuboid(yoke, ny), mesh_Cuboid(mag, nm)).sources_all)
    positions, clusters = D._analyze_collection(cells, "point")
    solver, share = D._auto_solver(clusters)
    desc = [(c["kind"], len(c["indices"]), tuple(np.round(np.array(c["spacing"]) * 1e3, 4)) if c["kind"] == "grid" else None) for c in clusters]
    a, b = clusters[0], clusters[1]
    na, nb = len(a["indices"]), len(b["indices"])
    same = D._same_lattice(a, b)
    print(f"h={h} mm: clusters {desc}; auto -> {solver} (FFT share {share:.0%}); same lattice {same}; cross 9*na*nb = {9*na*nb/1e6:.0f}M entries", flush=True)
    if same:
        continue
    forced_sparse = 9 * na * nb > D.DENSE_BLOCK_MAX_ENTRIES and not (72 * na * nb <= D.DENSE_BLOCK_MAX_BYTES)
    chi_eff = 1000.0; K = len(clusters)
    row_budget = 0.1 * 1e-6 * D.AUTO_TOL_FACTOR / (chi_eff * K)
    chunk = max(1, D.KERNEL_PAIRS_PER_CALL // nb)
    pos_a, pos_b = positions[a["indices"]], positions[b["indices"]]
    t = time.perf_counter(); blk = D._point_cuboid_block_T(pos_a[:chunk], pos_b, b["dim"], None); tk = time.perf_counter() - t
    t = time.perf_counter(); r, c, v = D._row_sparsify(blk, row_budget); tsp = time.perf_counter() - t
    kept = len(v) / blk.size
    n_chunks = int(np.ceil(na / chunk))
    print(f"   block a<-b: forced sparse path {forced_sparse}; row budget {row_budget:.1e}; first chunk keeps {kept:.1%} of entries "
          f"(droppable {D._droppable_fraction(blk, row_budget):.1%}); per chunk kernel {tk:.2f}s + sparsify {tsp:.2f}s, {n_chunks} chunks "
          f"-> ~{n_chunks*(tk+tsp)/60:.1f} min; CSR ~{9*na*nb*kept*(v.itemsize + c.itemsize + r.itemsize)/1e9:.1f} GB while built, "
          f"{9*na*nb*kept*(v.itemsize + c.itemsize)/1e9:.1f} GB as CSR, vs dense {72*na*nb/1e9:.1f} GB", flush=True)
