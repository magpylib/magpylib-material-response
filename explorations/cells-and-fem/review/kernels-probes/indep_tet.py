"""Independent checks of lintet.tensor (review: kernels).

1. Random tets, outside points: 3D Gauss-Legendre (collapsed coordinates) of the dipole kernel. Independent of
   magpylib and of the author's derivation.
2. Random tets, inside points: surface-charge formulation (sigma = J.n per face, volume term via gradient theorem
   -> n_f int 1/R dS), 2D GL per face. Also a sub-tet staircase with magpylib's tetrahedron kernel.
3. Kuhn box, points 1e-3 h from box faces / edges / corner and from the internal diagonal faces, vs lincell
   (lincell itself validated independently in indep_cuboid.py).
4. Flat tetrahedron (height 1e-3 of base), far observers: uniform columns vs point dipole; linear columns r^-4.
5. D10 for tets: observer in a face plane / on an edge line / on a face / edge / vertex.
6. Chirality: reversed vertex order gives the same tensor.
"""

import itertools
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy  # noqa: E402
from lincell import tensor as cub_tensor  # noqa: E402
from lintet import tensor as tet_tensor  # noqa: E402

MU0 = magpy.mu_0
rng = np.random.default_rng(7)


def Pbasis(xi):
    M = np.zeros((len(xi), 3, 12))
    M[:, [0, 1, 2], [0, 1, 2]] = 1
    for r in range(3):
        M[:, r, 3 + 3 * r : 6 + 3 * r] = xi
    return M


def tet_nodes(V, ng):
    """Collapsed-coordinate product GL rule on one tet V (4,3): points (q,3), weights (q,)."""
    g, w = np.polynomial.legendre.leggauss(ng)
    g, w = (g + 1) / 2, w / 2  # [0,1]
    u, v, ww = np.meshgrid(g, g, g, indexing="ij")
    wu, wv, wwt = np.meshgrid(w, w, w, indexing="ij")
    u, v, ww = u.reshape(-1), v.reshape(-1), ww.reshape(-1)
    vol = abs(np.linalg.det(V[1:] - V[0])) / 6
    p = (1 - v)[:, None] * V[1] + v[:, None] * ((1 - ww)[:, None] * V[2] + ww[:, None] * V[3])
    x = V[0] + u[:, None] * (p - V[0])
    wq = (wu * wv * wwt).reshape(-1) * 6 * vol * u**2 * v
    return x, wq


def tri_nodes(A, B, C, ng):
    """Collapsed product GL on triangle ABC: points (q,3), weights (q,)."""
    g, w = np.polynomial.legendre.leggauss(ng)
    g, w = (g + 1) / 2, w / 2
    v, ww = np.meshgrid(g, g, indexing="ij")
    wv, wwt = np.meshgrid(w, w, indexing="ij")
    v, ww = v.reshape(-1), ww.reshape(-1)
    x = (1 - v)[:, None] * A + v[:, None] * ((1 - ww)[:, None] * B + ww[:, None] * C)
    area2 = np.linalg.norm(np.cross(B - A, C - B))
    wq = (wv * wwt).reshape(-1) * area2 * v
    return x, wq


def dipole_quad(X, V, scale, ng=40):
    out = np.empty((len(X), 3, 12))
    for r in range(len(X)):
        xp, wq = tet_nodes(V[r], ng)
        c = V[r].mean(0)
        xi = (xp - c) / scale[r]
        B = Pbasis(xi)
        rr = X[r] - xp
        R2 = (rr**2).sum(1)
        R = np.sqrt(R2)
        K = 3 * rr[:, :, None] * rr[:, None, :] / (R2 * R2 * R)[:, None, None] - np.eye(3)[None] / (R2 * R)[:, None, None]
        out[r] = np.einsum("q,qab,qbc->ac", wq, K, B) / (4 * np.pi)
    return out


def surface_quad(X, V, scale, ng=80):
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
            xp, wq = tri_nodes(A, B, C, ng)
            rr = X[r] - xp
            R = np.sqrt((rr**2).sum(1))
            Jb = Pbasis((xp - c) / scale[r])
            sigma = np.einsum("a,qab->qb", nrm, Jb)
            acc += np.einsum("q,qa,qb->ab", wq / R**3, rr, sigma)
            P1 = (wq / R).sum()
            for kk in range(3):
                acc[:, 3 + 3 * kk + kk] -= nrm * P1 / scale[r]
        out[r] = acc / (4 * np.pi)
    return out


def split(V):
    v0, v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2], V[:, 3]
    m01, m02, m03 = (v0 + v1) / 2, (v0 + v2) / 2, (v0 + v3) / 2
    m12, m13, m23 = (v1 + v2) / 2, (v1 + v3) / 2, (v2 + v3) / 2
    kids = [(v0, m01, m02, m03), (v1, m01, m12, m13), (v2, m02, m12, m23), (v3, m03, m13, m23),
            (m02, m13, m01, m03), (m02, m13, m03, m23), (m02, m13, m23, m12), (m02, m13, m12, m01)]
    return np.stack([np.stack(k, axis=1) for k in kids], axis=1)


def staircase(X, V, scale, levels):
    """(n,3,12) field of the linear basis by sub-tets carrying J at their centroids, magpylib tetrahedron kernel."""
    out = np.empty((len(X), 3, 12))
    for r in range(len(X)):
        sub = V[r][None]
        for _ in range(levels):
            sub = split(sub).reshape(-1, 4, 3)
        cen = sub.mean(1)
        B = Pbasis((cen - V[r].mean(0)) / scale[r])  # (q,3,12)
        Xo = np.tile(X[r], (len(sub), 1))
        acc = np.zeros((3, 12))
        for m in range(3):
            H = magpy.func.tetrahedron_field("H", Xo, sub, np.tile(np.eye(3)[m], (len(sub), 1))) * MU0  # (q,3)
            acc += np.einsum("qa,qb->ab", H, B[:, m, :])
        out[r] = acc
    return out


def inside_mask(X, V):
    c = V.mean(1)
    ok = np.ones(len(X), bool)
    for o in range(4):
        f = [i for i in range(4) if i != o]
        n = np.cross(V[:, f[1]] - V[:, f[0]], V[:, f[2]] - V[:, f[0]])
        n *= np.sign(np.einsum("ij,ij->i", n, V[:, f[0]] - V[:, o]))[:, None]
        ok &= np.einsum("ij,ij->i", X - V[:, f[0]], n) < 0
    return ok


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=160)
    # ---------- 1. outside points
    n = 60
    V = rng.uniform(-1, 1, (n, 4, 3))
    c = V.mean(1)
    scale = np.sqrt(((V - c[:, None]) ** 2).sum(-1)).max(1)
    X = c + rng.normal(size=(n, 3)) * 0 + (rng.uniform(-1, 1, (n, 3)) * 3.5)
    dist = np.min(np.linalg.norm(X[:, None] - V, axis=-1), axis=1)
    keep = (~inside_mask(X, V)) & (np.linalg.norm(X - c, axis=1) > 1.6 * scale)
    X, V, scale, c = X[keep], V[keep], scale[keep], c[keep]
    T = tet_tensor(X, V, scale)
    Q = dipole_quad(X, V, scale, 40)
    Q2 = dipole_quad(X, V, scale, 28)
    print(f"[1] outside points ({len(X)}) random tets: linear tet kernel vs 3D dipole GL(40): max|diff| {np.abs(T - Q).max():.1e}"
          f" (scale {np.abs(Q).max():.3f}); GL(40)-GL(28) {np.abs(Q - Q2).max():.1e}")
    print(f"    per column: {np.abs(T - Q).max(axis=(0, 1))}")
    ref = np.stack([magpy.func.tetrahedron_field("H", X, V, np.tile(e, (len(X), 1))) * MU0 for e in np.eye(3)], -1)
    print(f"    uniform cols vs magpylib {np.abs(T[:, :, :3] - ref).max():.1e}, magpylib vs GL {np.abs(ref - Q[:, :, :3]).max():.1e}")

    # ---------- 2. inside points
    n = 40
    V = rng.uniform(-1, 1, (n, 4, 3))
    c = V.mean(1)
    scale = np.sqrt(((V - c[:, None]) ** 2).sum(-1)).max(1)
    lam = rng.dirichlet(np.ones(4), n)  # barycentric
    lam = 0.5 * lam + 0.5 / 4  # pulled toward the centroid: >= 1/8 of each height from every face
    X = np.einsum("ni,nij->nj", lam, V)
    assert inside_mask(X, V).all()
    T = tet_tensor(X, V, scale)
    Sf = surface_quad(X, V, scale, 80)
    Sf2 = surface_quad(X, V, scale, 56)
    print(f"[2] inside points ({n}, >= 1/8 height from faces): tet kernel vs surface-charge 2D GL(80): max|diff| {np.abs(T - Sf).max():.1e}"
          f" (scale {np.abs(Sf).max():.3f}); GL(80)-GL(56) {np.abs(Sf - Sf2).max():.1e}")
    print(f"    per column: {np.abs(T - Sf).max(axis=(0, 1))}")
    ref = np.stack([magpy.func.tetrahedron_field("H", X, V, np.tile(e, (n, 1))) * MU0 for e in np.eye(3)], -1)
    print(f"    uniform cols vs magpylib inside {np.abs(T[:, :, :3] - ref).max():.1e}")
    for lev in (3, 4):
        St = staircase(X[:8], V[:8], scale[:8], lev)
        print(f"    staircase {8**lev} sub-tets (magpylib tetrahedron kernel), 8 pts: max|diff| {np.abs(T[:8] - St).max():.1e}, linear cols {np.abs(T[:8, :, 3:] - St[:, :, 3:]).max():.1e}")

    # ---------- 3. Kuhn box near faces / edges / corners / internal faces
    h = np.array([0.7, 1.1, 0.5])
    corner = lambda bits: (2 * np.array(bits) - 1) * h  # noqa: E731
    tets = []
    for perm in itertools.permutations(range(3)):
        bits = [0, 0, 0]
        vs = [corner(bits)]
        for ax in perm:
            bits[ax] = 1
            vs.append(corner(bits))
        tets.append(vs)
    tets = np.array(tets)
    pts = []
    sgn = lambda: np.sign(rng.uniform(-1, 1, 3))  # noqa: E731
    for _ in range(20):
        pts.append((1 - 1e-3) * h * sgn())  # near corners
    for _ in range(30):
        p = rng.uniform(-0.9, 0.9, 3) * h
        k = rng.integers(0, 3)
        p[k] = (1 - 1e-3) * h[k] * np.sign(rng.uniform(-1, 1))
        pts.append(p)  # near faces
    for _ in range(30):
        p = rng.uniform(-0.9, 0.9, 3) * h
        k = rng.integers(0, 3)
        for kk in range(3):
            if kk != k:
                p[kk] = (1 - 1e-3) * h[kk] * np.sign(rng.uniform(-1, 1))
        pts.append(p)  # near edges
    for _ in range(30):  # near internal diagonal faces: x/hx = y/hy etc. (planes through the box diagonal)
        p = rng.uniform(-0.9, 0.9, 3) * h
        i, j = rng.choice(3, 2, replace=False)
        p[j] = (p[i] / h[i]) * h[j] + 1e-3 * h[j]
        pts.append(p)
    for _ in range(10):  # near the main diagonal line (shared by all 6 tets)
        t = rng.uniform(-0.9, 0.9)
        pts.append(t * h + 1e-3 * h * np.array([1, -1, 0]))
    Xo = np.array(pts)
    assert np.all(np.abs(Xo) < h)
    m = len(Xo)
    Tc = cub_tensor(Xo, np.tile(h, (m, 1)))
    acc = np.zeros((m, 3, 12))
    for tv in tets:
        Tt = tet_tensor(Xo, np.tile(tv, (m, 1, 1)), np.ones(m))
        ct = tv.mean(0)
        conv = np.zeros((m, 3, 12))
        conv[:, :, :3] = Tt[:, :, :3]
        for mm in range(3):
            for k in range(3):
                conv[:, :, 3 + 3 * mm + k] = (Tt[:, :, 3 + 3 * mm + k] + ct[k] * Tt[:, :, mm]) / h[k]
        acc += conv
    d = np.abs(acc - Tc)
    print(f"[3] Kuhn box, {m} inside pts at 1e-3 h from box corners/faces/edges/internal faces/diagonal vs lincell: max|diff| {d.max():.1e}"
          f" (scale {np.abs(Tc).max():.2f}); corners {d[:20].max():.1e} faces {d[20:50].max():.1e} edges {d[50:80].max():.1e} internal {d[80:110].max():.1e} diagonal {d[110:].max():.1e}; finite {np.isfinite(acc).all()}")

    # ---------- 4. flat tet, far field
    Vf = np.array([[[0, 0, 0], [1.0, 0, 0], [0.3, 0.9, 0], [0.4, 0.3, 1e-3]]])
    cf = Vf.mean(1)
    sf = np.sqrt(((Vf - cf[:, None]) ** 2).sum(-1)).max(1)
    vol = abs(np.linalg.det(Vf[0, 1:] - Vf[0, 0])) / 6
    print(f"[4] flat tet: height 1e-3, volume {vol:.2e}, scale {sf[0]:.3f}")
    dirn = np.array([0.2, 0.5, 0.84])
    dirn /= np.linalg.norm(dirn)
    rows = []
    for r in (3.0, 10.0, 30.0, 100.0, 1e3):
        Xf = cf + r * dirn
        Tf = tet_tensor(Xf, Vf, sf)[0]
        Qf = dipole_quad(Xf, Vf, sf, 24)[0]
        rr = Xf[0] - cf[0]
        Rn = np.linalg.norm(rr)
        Kd = 3 * np.outer(rr, rr) / Rn**5 - np.eye(3) / Rn**3
        dip = Kd * vol / (4 * np.pi)  # uniform columns
        rows.append((r, np.abs(Tf[:, :3] - dip).max() / np.abs(dip).max(), np.abs(Tf - Qf).max() / np.abs(Qf).max(), np.abs(Tf[:, 3:] - Qf[:, 3:]).max() / np.abs(Qf[:, 3:]).max(), np.abs(Tf[:, 3:]).max()))
    for r, e_dip, e_all, e_lin, lin in rows:
        print(f"    r={r:6.0f}: uniform cols vs point dipole rel {e_dip:.1e}; vs GL(24) all cols rel {e_all:.1e}, linear cols rel {e_lin:.1e}; |linear| {lin:.2e}")
    print(f"    linear-cols decay exponent 10->30->100: {np.log(rows[1][4] / rows[2][4]) / np.log(3):.2f}, {np.log(rows[2][4] / rows[3][4]) / np.log(10 / 3):.2f} (expect 4)")
    # flat tet near field vs GL: a point 0.5 above the flat face and one just below
    Xn = np.array([[0.4, 0.4, 0.5], [0.4, 0.4, -0.5], [0.4, 0.4, 0.002], [0.4, 0.4, -0.002]])
    Tn = tet_tensor(Xn, np.tile(Vf, (4, 1, 1)), np.tile(sf, 4))
    Qn = dipole_quad(Xn, np.tile(Vf, (4, 1, 1)), np.tile(sf, 4), 60)
    print(f"    flat tet near field (z=+0.5/-0.5 above/below): max rel diff vs GL(60) {np.abs(Tn[:2] - Qn[:2]).max() / np.abs(Qn[:2]).max():.1e}; at z=+-0.002 (2x height): finite {np.isfinite(Tn[2:]).all()}, rel diff {np.abs(Tn[2:] - Qn[2:]).max() / np.abs(Qn[2:]).max():.1e} (GL(60) may not be converged here)")

    # ---------- 5. D10 for tets
    Vt = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])[None]
    st = np.ones(1)
    cases = {
        "in face plane z=0, outside the face (2, 2, 0)": [2.0, 2.0, 0.0],
        "in face plane z=0, outside (-1, 0.3, 0)": [-1.0, 0.3, 0.0],
        "edge line extension (x axis) (3, 0, 0)": [3.0, 0.0, 0.0],
        "edge line extension (-2, 0, 0)": [-2.0, 0.0, 0.0],
        "on face z=0 (0.2, 0.3, 0)": [0.2, 0.3, 0.0],
        "on edge (0.5, 0, 0)": [0.5, 0.0, 0.0],
        "vertex (0, 0, 0)": [0.0, 0.0, 0.0],
        "in plane of slanted face, outside (1, 1, -1)": [1.0, 1.0, -1.0],
    }
    print("[5] D10 tets (reference tet): non-finite count over 36 entries; vs dipole GL(48) where outside")
    with np.errstate(all="ignore"):
        for name, x in cases.items():
            Xc = np.array([x])
            Tc_ = tet_tensor(Xc, Vt, st)
            bad = (~np.isfinite(Tc_)).sum()
            msg = ""
            if not inside_mask(Xc, Vt)[0] and not (np.all(np.array(x) >= -1e-12) and sum(x) <= 1 + 1e-12):
                Qc = dipole_quad(Xc, Vt, st, 48)
                msg = f"; vs GL max|diff| {np.nan_to_num(np.abs(Tc_ - Qc), nan=np.inf).max():.1e} (scale {np.abs(Qc).max():.3f})"
                ref = np.stack([magpy.func.tetrahedron_field("H", Xc, Vt, e[None]) * MU0 for e in np.eye(3)], -1)
                msg += f"; magpylib finite {np.isfinite(ref).all()}"
            print(f"    {name:48s}: non-finite {bad:2d}{msg}")

    # ---------- 6. chirality
    Vr = V[:5]
    Vrev = Vr[:, [0, 2, 1, 3]]
    Xr = X[:5]
    print(f"[6] reversed vertex order: max|diff| {np.abs(tet_tensor(Xr, Vr, scale[:5]) - tet_tensor(Xr, Vrev, scale[:5])).max():.1e}")
