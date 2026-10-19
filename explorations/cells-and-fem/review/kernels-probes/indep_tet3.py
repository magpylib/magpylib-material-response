"""Same as indep_tet2 but the collapse vertex of each face's sub-triangles is the point of the face closest to the
observer (foot point if inside the face, else the nearest edge/vertex point), so the radial grading resolves the
near-singularity whatever the shape of the tet."""
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/review/kernels")
import magpylib as magpy  # noqa: E402
from indep_tet import Pbasis, dipole_quad, inside_mask  # noqa: E402
from indep_tet2 import graded01  # noqa: E402
from lintet import tensor as tet_tensor  # noqa: E402

MU0 = magpy.mu_0
rng = np.random.default_rng(23)


def closest_point_triangle(p, a, b, c):
    """Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    denom = 1 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def adapted_surface(X, V, scale, ngw=32, levels=20):
    gv, wv = graded01(levels=levels)
    gw, ww = np.polynomial.legendre.leggauss(ngw)
    gw, ww = (gw + 1) / 2, ww / 2
    vv, wwg = np.meshgrid(gv, gw, indexing="ij")
    wq0 = np.outer(wv, ww).reshape(-1)
    vv, wwg = vv.reshape(-1), wwg.reshape(-1)
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
                e1, e2 = P1 - q, P2 - q
                area2 = s * np.dot(nrm, np.cross(e1, e2))
                if abs(area2) < 1e-14:
                    continue
                xp = q + vv[:, None] * ((1 - wwg)[:, None] * e1 + wwg[:, None] * e2)
                wq = wq0 * area2 * vv
                rr = X[r] - xp
                R = np.sqrt((rr**2).sum(1))
                Jb = Pbasis((xp - c) / scale[r])
                sigma = np.einsum("a,qab->qb", nrm, Jb)
                acc += np.einsum("q,qa,qb->ab", wq / R**3, rr, sigma)
                P1v = (wq / R).sum()
                for kk in range(3):
                    acc[:, 3 + 3 * kk + kk] -= nrm * P1v / scale[r]
        out[r] = acc / (4 * np.pi)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=160)
    n = 30
    V = rng.uniform(-1, 1, (n, 4, 3))
    c = V.mean(1)
    scale = np.sqrt(((V - c[:, None]) ** 2).sum(-1)).max(1)
    groups = {}
    lam = rng.dirichlet(np.ones(4), n) * 0.8 + 0.05
    groups["inside random (bary >= 0.05)"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(3), n) * 0.7 + 0.1
    lam = np.concatenate([np.full((n, 1), 1e-3), lam * (1 - 1e-3)], axis=1)
    groups["inside, 1e-3 of a face (bary 1e-3)"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(2), n) * 0.7 + 0.15
    lam = np.concatenate([np.full((n, 2), 1e-3), lam * (1 - 2e-3)], axis=1)
    groups["inside, 1e-3 of an edge"] = np.einsum("ni,nij->nj", lam, V)
    lam = np.concatenate([np.full((n, 3), 1e-3), np.full((n, 1), 1 - 3e-3)], axis=1)
    groups["inside, 1e-3 of a vertex"] = np.einsum("ni,nij->nj", lam, V)
    lam = rng.dirichlet(np.ones(3), n) * 0.7 + 0.1
    lam = np.concatenate([np.full((n, 1), -0.05), lam * 1.05], axis=1)
    groups["outside, 0.05 height beyond a face"] = np.einsum("ni,nij->nj", lam, V)
    groups["outside, beyond a vertex (c + 1.3 (v - c))"] = c + 1.3 * (V[:, 0] - c)
    for name, X in groups.items():
        ins = inside_mask(X, V)
        T = tet_tensor(X, V, scale)
        Sref = adapted_surface(X, V, scale)
        Sref2 = adapted_surface(X, V, scale, ngw=24, levels=16)
        d = np.abs(T - Sref)
        msg = f"{name:44s}: inside {ins.sum():2d}/{n}; kernel vs surface quad max|diff| {d.max():.1e} (scale {np.abs(Sref).max():.2f}; rule self-consistency {np.abs(Sref - Sref2).max():.1e}); uniform cols {d[:, :, :3].max():.1e}, linear cols {d[:, :, 3:].max():.1e}; finite {np.isfinite(T).all()}"
        if not ins.any():
            Q = dipole_quad(X, V, scale, 44)
            msg += f"; surface rule vs 3D dipole GL(44) {np.abs(Sref - Q).max():.1e}"
        ref = np.stack([magpy.func.tetrahedron_field("H", X, V, np.tile(e, (n, 1))) * MU0 for e in np.eye(3)], -1)
        msg += f"; magpylib uniform vs surface rule {np.abs(ref - Sref[:, :, :3]).max():.1e}"
        print(msg, flush=True)
    Vf = np.array([[[0, 0, 0], [1.0, 0, 0], [0.3, 0.9, 0], [0.4, 0.3, 1e-3]]])
    sf = np.sqrt(((Vf - Vf.mean(1)[:, None]) ** 2).sum(-1)).max(1)
    Xn = np.array([[0.4, 0.4, 0.002], [0.4, 0.4, -0.002], [0.4, 0.4, 0.0005], [0.9, 0.1, 0.0003]])
    Tn = tet_tensor(Xn, np.tile(Vf, (4, 1, 1)), np.tile(sf, 4))
    Sn = adapted_surface(Xn, np.tile(Vf, (4, 1, 1)), np.tile(sf, 4))
    print(f"flat tet (height 1e-3), observers z=+0.002, -0.002 (outside), z=0.0005 and (0.9,0.1,0.0003) (inside): max|diff| {np.abs(Tn - Sn).max():.1e} (scale {np.abs(Sn).max():.2f}); per point {np.abs(Tn - Sn).max(axis=(1, 2))}; inside flags {inside_mask(Xn, np.tile(Vf, (4, 1, 1)))}")
