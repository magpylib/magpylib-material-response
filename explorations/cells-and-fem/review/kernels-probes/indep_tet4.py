"""Tet inside check, third attempt: surface-charge quadrature graded radially toward the closest face point AND
angularly toward the foot of that point on each face edge; plus staircase Richardson (8^3, 8^4, 8^5 sub-tets) with
magpylib's tetrahedron kernel; plus magpylib's own far-field precision for context."""
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/review/kernels")
import magpylib as magpy  # noqa: E402
from indep_tet import Pbasis, dipole_quad, inside_mask, staircase  # noqa: E402
from indep_tet2 import graded01  # noqa: E402
from indep_tet3 import closest_point_triangle  # noqa: E402
from lincell import tensor as cub_tensor  # noqa: E402
from lintet import tensor as tet_tensor  # noqa: E402

MU0 = magpy.mu_0
rng = np.random.default_rng(23)


def surface2(X, V, scale, lv=16, lw=14, ng=10):
    gv, wv = graded01(levels=lv, ng=ng)
    gw0, ww0 = graded01(levels=lw, ng=ng)  # toward w = 0
    out = np.zeros((len(X), 3, 12))
    for r in range(len(X)):
        c = V[r].mean(0)
        acc = np.zeros((3, 12))
        for o in range(4):
            f = [i for i in range(4) if i != o]
            A, B, C = V[r][f[0]], V[r][f[1]], V[r][f[2]]
            nrm = np.cross(B - A, C - A)
            nrm /= np.linalg.norm(nrm)
            if np.dot(nrm, A - V[r][o]) < 0:
                nrm = -nrm
            s = np.sign(np.dot(nrm, np.cross(B - A, C - A)))
            q = closest_point_triangle(X[r], A, B, C)
            for P1, P2 in ((A, B), (B, C), (C, A)):
                tF = np.dot(q - P1, P2 - P1) / np.dot(P2 - P1, P2 - P1)
                if 1e-12 < tF < 1 - 1e-12:
                    F = P1 + tF * (P2 - P1)
                    pieces = [((P1, F), 1), ((F, P2), 0)]
                else:
                    pieces = [((P1, P2), 0 if tF <= 0 else 1)]
                for (U1, U2), near in pieces:
                    e1, e2 = U1 - q, U2 - q
                    area2 = s * np.dot(nrm, np.cross(e1, e2))
                    if abs(area2) < 1e-14:
                        continue
                    gw = gw0 if near == 0 else 1 - gw0
                    vv, wwg = np.meshgrid(gv, gw, indexing="ij")
                    wq = (np.outer(wv, ww0) * area2).reshape(-1) * vv.reshape(-1)
                    vv, wwg = vv.reshape(-1), wwg.reshape(-1)
                    xp = q + vv[:, None] * ((1 - wwg)[:, None] * e1 + wwg[:, None] * e2)
                    rr = X[r] - xp
                    Rr = np.sqrt((rr**2).sum(1))
                    Jb = Pbasis((xp - c) / scale[r])
                    sigma = np.einsum("a,qab->qb", nrm, Jb)
                    acc += np.einsum("q,qa,qb->ab", wq / Rr**3, rr, sigma)
                    P1v = (wq / Rr).sum()
                    for kk in range(3):
                        acc[:, 3 + 3 * kk + kk] -= nrm * P1v / scale[r]
        out[r] = acc / (4 * np.pi)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=160)
    n = 24
    V = rng.uniform(-1, 1, (n, 4, 3))
    c = V.mean(1)
    scale = np.sqrt(((V - c[:, None]) ** 2).sum(-1)).max(1)
    groups = {}
    lam = rng.dirichlet(np.ones(4), n) * 0.8 + 0.05
    groups["inside random (bary >= 0.05)"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(3), n) * 0.7 + 0.1
    lam = np.concatenate([np.full((n, 1), 1e-3), lam * (1 - 1e-3)], axis=1)
    groups["inside, 1e-3 of a face"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(2), n) * 0.7 + 0.15
    lam = np.concatenate([np.full((n, 2), 1e-3), lam * (1 - 2e-3)], axis=1)
    groups["inside, 1e-3 of an edge"] = np.einsum("ni,nij->nj", lam, V)
    lam = np.concatenate([np.full((n, 3), 1e-3), np.full((n, 1), 1 - 3e-3)], axis=1)
    groups["inside, 1e-3 of a vertex"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(3), n) * 0.7 + 0.1
    lam = np.concatenate([np.full((n, 1), -0.05), lam * 1.05], axis=1)
    groups["outside, 0.05 height beyond a face"] = np.einsum("ni,nij->nj", lam, V)
    for name, X in groups.items():
        ins = inside_mask(X, V)
        T = tet_tensor(X, V, scale)
        Sref = surface2(X, V, scale)
        Sref2 = surface2(X, V, scale, lv=13, lw=11, ng=8)
        d = np.abs(T - Sref)
        msg = (f"{name:36s}: inside {ins.sum():2d}/{n}; kernel vs 2x-graded surface quad max|diff| {d.max():.1e} (scale {np.abs(Sref).max():.2f};"
               f" rule self-consistency {np.abs(Sref - Sref2).max():.1e}); uniform {d[:, :, :3].max():.1e}, linear {d[:, :, 3:].max():.1e}")
        ref = np.stack([magpy.func.tetrahedron_field("H", X, V, np.tile(e, (n, 1))) * MU0 for e in np.eye(3)], -1)
        msg += f"; magpylib uniform vs rule {np.abs(ref - Sref[:, :, :3]).max():.1e}"
        if not ins.any():
            msg += f"; rule vs 3D dipole GL(44) {np.abs(Sref - dipole_quad(X, V, scale, 44)).max():.1e}"
        print(msg, flush=True)
    # staircase Richardson on 6 random inside points
    X = groups["inside random (bary >= 0.05)"][:6]
    Vs, ss = V[:6], scale[:6]
    T = tet_tensor(X, Vs, ss)
    S3, S4, S5 = (staircase(X, Vs, ss, lev) for lev in (3, 4, 5))
    e3, e4, e5 = (np.abs(S - T).max() for S in (S3, S4, S5))
    rich = S5 + (S5 - S4) * 1  # first-order Richardson with ratio 2: 2*S5 - S4
    rich = 2 * S5 - S4
    print(f"staircase (magpylib tet kernel) vs kernel, 6 inside pts: 512 sub-tets {e3:.1e}, 4096 {e4:.1e}, 32768 {e5:.1e}; order {np.log(e4 / e5) / np.log(2):.2f}; Richardson(4096, 32768) {np.abs(rich - T).max():.1e}; linear cols only {np.abs(rich - T)[:, :, 3:].max():.1e}")
    # magpylib far-field precision for context (cuboid h=(1,0.7,1.3), direction as in indep_cuboid)
    hc = np.array([[1.0, 0.7, 1.3]])
    dirn = np.array([0.3, -0.5, 0.8])
    dirn /= np.linalg.norm(dirn)
    Vol = 8 * np.prod(hc)
    print("far-field precision, uniform cuboid columns vs point dipole (rel): r | magpylib | lincell")
    for r in (100.0, 1e3, 1e4):
        x = (r * dirn)[None]
        Kd = 3 * np.outer(x[0], x[0]) / r**5 - np.eye(3) / r**3
        dip = Kd * Vol / (4 * np.pi)
        mp = np.stack([magpy.func.cuboid_field("H", x, 2 * hc, e[None]) * MU0 for e in np.eye(3)], -1)[0]
        lc = cub_tensor(x, hc, linear=False)[0]
        print(f"    r={r:6.0f}: magpylib {np.abs(mp - dip).max() / np.abs(dip).max():.1e} | lincell {np.abs(lc - dip).max() / np.abs(dip).max():.1e}  (dipole truncation itself ~ (h/r)^2 = {(1.3 / r) ** 2:.0e})")
