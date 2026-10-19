"""Independent FEM reference: linear soft iron in the field of a source H_s.

Two-potential formulation (Simkin & Trowbridge): total potential psi in the iron
(H = -grad psi), reduced potential phi in the air (H = H_s - grad phi), coupled
on the iron surface by psi = phi + chi_s, where H_s = -grad chi_s inside the iron.
Weak form, divided by mu0, for continuous test functions v (v = 0 far away):
    int_air grad phi . grad v + mu_r int_iron grad psi . grad v = -int_surface (H_s . n) v
H_s enters only through the iron surface (div H_s = 0), so the coil is never meshed.
P2 elements, gmsh mesh, Dirichlet phi = 0 on a far sphere.

usage: MU_R=1000 fem.py ucore|cube h_mm
"""

import json
import os
import pathlib
import sys
import time

import gmsh
import magpylib as magpy
import numpy as np
import pyamg
import scipy.sparse as sp
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP2, FacetBasis, LinearForm, MeshTet, asm
from skfem.helpers import dot, grad

case, h = sys.argv[1], float(sys.argv[2]) * 1e-3
mu_r = float(os.environ.get("MU_R", 1000))
here = pathlib.Path(__file__).parent
t_start = time.perf_counter()

if case == "ucore":
    sys.path.insert(0, str(here))
    from ucore_setup import coil, line_pts, x

    boxes = [(-0.05, -0.01, -0.01, 0.10, 0.02, 0.02), (-0.05, -0.01, 0.01, 0.02, 0.02, 0.06), (0.03, -0.01, 0.01, 0.02, 0.02, 0.06)]
    center, R = np.array([0, 0, 0.03]), float(os.environ.get("R_AIR", 1.0))
    H_s = lambda pts: np.concatenate([magpy.getH(coil, pts[k : k + 2000]) for k in range(0, len(pts), 2000)])  # A/m, chunked
else:  # 10 mm cube in a uniform field of 10 mT
    boxes = [(-0.005, -0.005, -0.005, 0.01, 0.01, 0.01)]
    center, R = np.zeros(3), 0.2
    H0 = 0.01 / magpy.mu_0
    H_s = lambda pts: np.tile([0, 0, H0], (len(pts), 1))

# ---- mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 4)
occ = gmsh.model.occ
tags = [(3, occ.addBox(*b)) for b in boxes]
iron = occ.fuse(tags[:1], tags[1:])[0] if len(tags) > 1 else tags
occ.fragment([(3, occ.addSphere(*center, R))], iron)
occ.synchronize()
vols = [t for _, t in gmsh.model.getEntities(3)]
iron_vol = min(vols, key=lambda t: occ.getMass(3, t))
air_vol = max(vols, key=lambda t: occ.getMass(3, t))
iron_surfs = [t for _, t in gmsh.model.getBoundary([(3, iron_vol)], oriented=False)]
iron_curves = sorted({t for _, t in gmsh.model.getBoundary([(2, s) for s in iron_surfs], oriented=False)})
f = gmsh.model.mesh.field
f.add("Distance", 1); f.setNumbers(1, "SurfacesList", iron_surfs)
f.add("Threshold", 2); f.setNumber(2, "InField", 1)
f.setNumber(2, "SizeMin", h); f.setNumber(2, "SizeMax", R / 8); f.setNumber(2, "DistMin", 0); f.setNumber(2, "DistMax", R / 2)
f.add("Distance", 3); f.setNumbers(3, "CurvesList", iron_curves); f.setNumber(3, "Sampling", 200)
f.add("Threshold", 4); f.setNumber(4, "InField", 3)
f.setNumber(4, "SizeMin", h / 3); f.setNumber(4, "SizeMax", h); f.setNumber(4, "DistMin", 0); f.setNumber(4, "DistMax", 4 * h)
f.setNumber(4, "StopAtDistMax", 1)  # no size constraint beyond 4 h from an edge
f.add("Constant", 5); f.setNumber(5, "VIn", h); f.setNumber(5, "VOut", R); f.setNumbers(5, "VolumesList", [iron_vol])
f.add("Min", 6); f.setNumbers(6, "FieldsList", [2, 4, 5])
f.setAsBackgroundMesh(6)
for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
    gmsh.option.setNumber(f"Mesh.{opt}", 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
MAX_TETS = int(os.environ.get("MAX_TETS", 1_500_000))
gmsh.model.mesh.generate(3)
n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
if n_tets > MAX_TETS:
    sys.exit(f"mesh has {n_tets} tets, above MAX_TETS={MAX_TETS}: refusing to assemble")
node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = coords.reshape(-1, 3)
index = np.zeros(int(node_tags.max()) + 1, dtype=int)
index[node_tags.astype(int)] = np.arange(len(node_tags))


def submesh(vol):
    types, _, enodes = gmsh.model.mesh.getElements(3, vol)
    t = index[enodes[list(types).index(4)].astype(int).reshape(-1, 4)]
    used, t = np.unique(t, return_inverse=True)
    t = t.reshape(-1, 4)
    p = coords[used]
    det = np.einsum("ij,ij->i", np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), p[t[:, 3]] - p[t[:, 0]])
    t[det < 0, :2] = t[det < 0, 1::-1]
    return MeshTet(p.T, t.T)


mesh_i, mesh_a = submesh(iron_vol), submesh(air_vol)  # interface nodes duplicated
gmsh.finalize()
print(f"mesh: {mesh_i.t.shape[1]} iron + {mesh_a.t.shape[1]} air tets", flush=True)

# ---- assemble
e = ElementTetP2()
bi, ba = Basis(mesh_i, e, intorder=4), Basis(mesh_a, e, intorder=4)
fbi = FacetBasis(mesh_i, e, intorder=4)  # iron boundary facets, outward normals
lap = BilinearForm(lambda u, v, w: dot(grad(u), grad(v)))
Ki, Ka = asm(lap, bi), asm(lap, ba)
qp = fbi.global_coordinates().value  # (3, nfacets, nqp)
Hs = H_s(qp.reshape(3, -1).T).T.reshape(qp.shape)
f_i = -asm(LinearForm(lambda v, w: dot(w.hs, w.n) * v), fbi, hs=Hs)


def amg_solve(A, b):
    res = []
    sol = pyamg.smoothed_aggregation_solver(A.tocsr(), symmetry="symmetric").solve(b, tol=1e-11, accel="cg", maxiter=2000, residuals=res)
    return sol, len(res), res[-1] / res[0]


# chi_s with -grad chi_s = H_s in the iron: Neumann problem, one dof pinned
keep = np.arange(1, bi.N)
chi_s = np.zeros(bi.N)
chi_s[keep], it_chi, _ = amg_solve(Ki[keep][:, keep], f_i[keep])

# couple: iron surface dofs are the matching air dofs, psi = phi + chi_s there
iface_i = np.unique(fbi.get_dofs().flatten())
dist, iface_a = cKDTree(ba.doflocs.T).query(bi.doflocs[:, iface_i].T)
assert dist.max() < 1e-12 * R, dist.max()
inner_i = np.setdiff1d(np.arange(bi.N), iface_i)
g = np.empty(bi.N, dtype=int)
g[iface_i] = iface_a
g[inner_i] = ba.N + np.arange(len(inner_i))
N = ba.N + len(inner_i)
Q = sp.csr_matrix((np.ones(bi.N), (np.arange(bi.N), g)), shape=(bi.N, N))
lift = np.zeros(bi.N)
lift[iface_i] = chi_s[iface_i]
A = sp.block_diag([Ka, sp.csr_matrix((len(inner_i), len(inner_i)))]).tocsr() + mu_r * (Q.T @ Ki @ Q)
b = Q.T @ (f_i - mu_r * (Ki @ lift))
far = np.unique(ba.get_dofs(lambda p: np.linalg.norm(p - center[:, None], axis=0) > R / 2).flatten())
free = np.setdiff1d(np.arange(N), far)
sol = np.zeros(N)
sol[free], it, rel = amg_solve(A[free][:, free], b[free])
psi = Q @ sol + lift

# ---- post-process: element-mean magnetization of the iron
gpsi = bi.interpolate(psi).grad  # (3, nel, nqp)
vol_e = bi.dx.sum(axis=1)
H_e = -(gpsi * bi.dx).sum(axis=2) / vol_e
J_e = magpy.mu_0 * (mu_r - 1) * H_e.T  # T, (nel, 3)
moment = (J_e * vol_e[:, None]).sum(axis=0) / magpy.mu_0
out = dict(case=case, mu_r=mu_r, h_mm=h * 1e3, iron_tets=mesh_i.t.shape[1], air_tets=mesh_a.t.shape[1], dofs=int(N),
           cg_iters=it, rel_res=float(rel), chi_iters=it_chi, moment=moment.tolist())
if case == "ucore":
    verts = mesh_i.p.T[mesh_i.t.T]  # (nel, 4, 3)
    B_ind = np.zeros((len(line_pts), 3))
    for k in range(0, len(verts), 1000):
        v, j = verts[k : k + 1000], J_e[k : k + 1000]
        obs = np.repeat(line_pts, len(v), axis=0)
        B = magpy.func.tetrahedron_field("B", obs, np.tile(v, (len(line_pts), 1, 1)), np.tile(j, (len(line_pts), 1)))
        B_ind += B.reshape(len(line_pts), len(v), 3).sum(axis=1)
    Bz = (B_ind + magpy.getB(coil, line_pts))[:, 2]
    i_pole = np.argmin(np.abs(x - 0.04))
    out.update(peak_mT=float(np.abs(Bz).max() * 1e3), pole_mT=float(Bz[i_pole] * 1e3), Bz_mT=(Bz * 1e3).round(5).tolist())
else:
    ref = 0.01**3 * H0 / (1 / 3.6442 + 1 / (mu_r - 1))
    out.update(ref=ref, error=float(moment[2] / ref - 1))
out["seconds"] = round(time.perf_counter() - t_start, 1)
print(json.dumps({k: v for k, v in out.items() if k != "Bz_mT"}), flush=True)
json.dump(out, open(here / f"fem-{case}-{sys.argv[2]}-mu{mu_r:g}.json", "w"))
