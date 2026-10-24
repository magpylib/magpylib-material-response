"""Bytes per kept entry of the forced CSR: as built (triplets) and as stored."""
import sys, numpy as np, scipy.sparse as sp, magpylib as magpy
sys.path.insert(0, sys.argv[1])
from magpylib_material_response import demag as D
from magpylib_material_response.meshing import mesh_Cuboid
from yoke_case import YOKE, MAG
h = 0.4
yoke = magpy.magnet.Cuboid(dimension=YOKE["dimension"], position=YOKE["position"], polarization=(0, 0, 0)); yoke.susceptibility = 1000.0
mag = magpy.magnet.Cuboid(dimension=MAG["dimension"], position=MAG["position"], polarization=MAG["J0"]); mag.susceptibility = 0.05
cells = list(magpy.Collection(mesh_Cuboid(yoke, int(round(2000 / 0.064))), mesh_Cuboid(mag, int(round(144 / 0.064)))).sources_all)
positions, clusters = D._analyze_collection(cells, "point")
a, b = clusters[0], clusters[1]
na, nb = len(a["indices"]), len(b["indices"])
# a slice of observers, run through _pair_block itself (forced sparse path), with the byte cap kept at 2 GB
sub = dict(a); sub["indices"] = a["indices"][:2000]
D.DENSE_BLOCK_MAX_ENTRIES = 1  # this slice alone is small; force the large-block path as on the full block
row_budget = 0.1 * 1e-6 * D.AUTO_TOL_FACTOR / (1000.0 * 2)
D.DENSE_BLOCK_MAX_BYTES = 1  # the full block is above 2 GB: the condition on main then forces sparse
M = D._pair_block(sub, b, cells, positions, "point", row_budget)
print("type", type(M).__name__, "index dtype", M.indices.dtype, "indptr dtype", M.indptr.dtype)
nnz = M.nnz
stored = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
print(f"slice {len(sub['indices'])} x {nb} cells: kept {nnz / (9 * len(sub['indices']) * nb):.1%}; stored {stored / nnz:.1f} B per kept entry (dense: 8)")
full = 9 * na * nb
print(f"full block {na} x {nb}: dense {8 * full / 1e9:.2f} GB, CSR stored ~{stored / nnz * full * nnz / (9 * len(sub['indices']) * nb) / 1e9:.2f} GB per direction")
