"""Independent checks of lincell.tensor (review: kernels).

1. Outside points: brute-force 3D Gauss-Legendre quadrature of the dipole field of J(x') = a + G xi over the cell.
   Uses nothing but the dipole kernel; independent of magpylib and of the author's derivation.
2. Inside points (mostly inside, near corners, within 1e-3 of faces): layer-cake slab identity evaluated with
   magpylib's uniform cuboid kernel and a geometrically graded composite GL rule in t (kink at t = X_k).
3. Surface-charge formulation (sigma = J.n on faces, -div J volume term via the gradient theorem -> n/R),
   2D GL on each face: a second independent inside check for points not too close to faces.
4. D10: points exactly in a face plane / on an edge line extension / on face / on edge / at a corner.
5. Sign checks: J = (0,0,xi_z) centre field = 0; J_x = xi_x: centre 0, just outside the +x face -> +x.
6. Far field with a = 0: must decay as r^-4 (quadrupole); compare with quadrature.
"""

import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy  # noqa: E402
from lincell import tensor  # noqa: E402

MU0 = magpy.mu_0
rng = np.random.default_rng(11)


def Pbasis(xi):
    """(q, 3, 12): J at xi per unit coefficient, u = [a(3), G row-major(9)] (same convention as p1emul.P)."""
    M = np.zeros((len(xi), 3, 12))
    M[:, [0, 1, 2], [0, 1, 2]] = 1
    for r in range(3):
        M[:, r, 3 + 3 * r : 6 + 3 * r] = xi
    return M


def dipole_quad(X, h, ng=48):
    """(n, 3, 12): mu0 H at X (cell-local) by 3D GL quadrature of the dipole kernel. Valid outside the cell."""
    g, w = np.polynomial.legendre.leggauss(ng)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    wx, wy, wz = np.meshgrid(w, w, w, indexing="ij")
    xi = np.stack([gx, gy, gz], -1).reshape(-1, 3)  # in [-1,1]^3
    wq = (wx * wy * wz).reshape(-1)
    B = Pbasis(xi)  # (q, 3, 12)
    out = np.empty((len(X), 3, 12))
    for r in range(len(X)):
        xp = xi * h[r]
        rr = X[r] - xp  # (q, 3)
        R2 = (rr**2).sum(1)
        R = np.sqrt(R2)
        K = 3 * rr[:, :, None] * rr[:, None, :] / (R2 * R2 * R)[:, None, None] - np.eye(3)[None] / (R2 * R)[:, None, None]
        out[r] = np.einsum("q,qab,qbc->ac", wq * np.prod(h[r]), K, B) / (4 * np.pi)
    return out


def unif(Xo, cen, half, m):
    pol = np.zeros((len(Xo), 3))
    pol[:, m] = 1
    return magpy.func.cuboid_field("H", Xo, 2 * half, pol, positions=cen, squeeze=False).reshape(-1, 3) * MU0


def graded_nodes(a, b, toward, levels=14, ratio=0.25, ng=16):
    """Composite GL nodes/weights on [a, b], geometrically refined toward endpoint `toward` (a or b)."""
    g, w = np.polynomial.legendre.leggauss(ng)
    cuts = [0.0] + [ratio**j for j in range(levels - 1, -1, -1)]  # fractions of the width, 0 .. 1
    cuts = np.array(cuts)
    width = b - a
    if toward == b:
        pts = b - width * cuts[::-1]  # from a (=b - width) toward b
    else:
        pts = a + width * cuts[::-1]
    pts = np.sort(pts)
    ts, ws = [], []
    for lo, hi in zip(pts[:-1], pts[1:]):
        ts.append((lo + hi) / 2 + (hi - lo) / 2 * g)
        ws.append((hi - lo) / 2 * w)
    return np.concatenate(ts), np.concatenate(ws)


def slab_ref(X, h, inside_only_kink=True):
    """(n, 3, 9): linear columns J_m = xi_k by the layer-cake identity with magpylib's uniform kernel."""
    n = len(X)
    out = np.empty((n, 3, 9))
    g64, w64 = np.polynomial.legendre.leggauss(64)
    for k in range(3):
        # build nodes per point
        T, W, idx = [], [], []
        for r in range(n):
            lo, hi = -h[r, k], h[r, k]
            if lo < X[r, k] < hi:
                t1, w1 = graded_nodes(lo, X[r, k], X[r, k])
                t2, w2 = graded_nodes(X[r, k], hi, X[r, k])
                t, w = np.concatenate([t1, t2]), np.concatenate([w1, w2])
            else:
                t, w = (lo + hi) / 2 + (hi - lo) / 2 * g64, (hi - lo) / 2 * w64
            T.append(t), W.append(w), idx.append(np.full(len(t), r))
        T, W, idx = np.concatenate(T), np.concatenate(W), np.concatenate(idx)
        hi_k = h[idx, k]
        cen = np.zeros((len(T), 3))
        cen[:, k] = (T + hi_k) / 2
        half = h[idx].copy()
        half[:, k] = (hi_k - T) / 2
        for m in range(3):
            F = unif(X[idx], cen, half, m)  # (N, 3)
            acc = np.zeros((n, 3))
            np.add.at(acc, idx, W[:, None] * F)
            out[:, :, 3 * m + k] = acc / h[:, k][:, None] - unif(X, np.zeros((n, 3)), h, m)
    return out


def surface_ref(X, h, ng=80):
    """(n, 3, 12): surface-charge formulation, 2D GL on the 6 faces. Valid strictly inside or outside (not on faces)."""
    g, w = np.polynomial.legendre.leggauss(ng)
    gu, gv = np.meshgrid(g, g, indexing="ij")
    wuv = np.outer(w, w).reshape(-1)
    gu, gv = gu.reshape(-1), gv.reshape(-1)
    out = np.zeros((len(X), 3, 12))
    for r in range(len(X)):
        hr = h[r]
        acc = np.zeros((3, 12))
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            for s in (-1.0, 1.0):
                xp = np.zeros((len(gu), 3))
                xp[:, k] = s * hr[k]
                xp[:, i] = gu * hr[i]
                xp[:, j] = gv * hr[j]
                nrm = np.zeros(3)
                nrm[k] = s
                dS = wuv * hr[i] * hr[j]
                rr = X[r] - xp
                R = np.sqrt((rr**2).sum(1))
                xi = xp / hr
                Jb = Pbasis(xi)  # (q, 3, 12)
                sigma = np.einsum("a,qab->qb", nrm, Jb)  # (q, 12)
                acc += np.einsum("q,qa,qb->ab", dS / R**3, rr, sigma)
                # volume term: - div J * sum_f n_f int 1/R dS ; div J column G_kk -> 1/h_k
                P1 = (dS / R).sum()
                for kk in range(3):
                    acc[:, 3 + 3 * kk + kk] -= nrm * P1 / hr[kk]
        out[r] = acc / (4 * np.pi)
    return out


def rel(d, ref):
    return d / max(np.abs(ref).max(), 1e-300)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=150)
    # ---------- 1. outside points vs 3D dipole quadrature
    n = 60
    h = rng.uniform(0.5, 2.0, (n, 3))
    X = rng.uniform(-3, 3, (n, 3)) * h
    far = np.max(np.abs(X) / h, axis=1) > 1.6  # at least 0.6 h outside along some axis: GL converges
    X, h = X[far], h[far]
    T = tensor(X, h)
    Q = dipole_quad(X, h, ng=48)
    Q2 = dipole_quad(X, h, ng=32)
    print(f"[1] outside points ({len(X)}): linear kernel vs 3D dipole GL(48): max|diff| {np.abs(T - Q).max():.1e}"
          f" (scale {np.abs(Q).max():.2f}); GL(48)-GL(32) {np.abs(Q - Q2).max():.1e}")
    print(f"    per column max|diff|: {np.abs(T - Q).max(axis=(0, 1))}")

    # ---------- 2. inside points, near corners and faces
    n = 240
    h = rng.uniform(0.5, 2.0, (n, 3))
    X = rng.uniform(-1, 1, (n, 3)) * h  # all inside
    X[:40] = (1 - 1e-3) * h[:40] * np.sign(rng.uniform(-1, 1, (40, 3)))  # 1e-3 h from a corner
    k = rng.integers(0, 3, 60)
    X[40:100, :] = rng.uniform(-0.9, 0.9, (60, 3)) * h[40:100]
    X[np.arange(40, 100), k] = (1 - 1e-3) * h[np.arange(40, 100), k] * np.sign(rng.uniform(-1, 1, 60))  # 1e-3 h from a face
    X[100:120] = rng.uniform(-0.05, 0.05, (20, 3)) * h[100:120]  # near centre
    assert np.all(np.abs(X) < h)
    T = tensor(X, h)
    S = slab_ref(X, h)
    d = np.abs(T[:, :, 3:] - S)
    print(f"[2] inside points ({n}; 40 near corners, 60 within 1e-3 h of a face): linear columns vs graded slab quadrature:"
          f" max|diff| {d.max():.1e} (scale {np.abs(S).max():.2f}); corners {d[:40].max():.1e}, faces {d[40:100].max():.1e}, centre {d[100:120].max():.1e}, random {d[120:].max():.1e}")
    U = np.stack([unif(X, np.zeros((n, 3)), h, m) for m in range(3)], -1)
    print(f"    uniform columns vs magpylib inside: {np.abs(T[:, :, :3] - U).max():.1e}")
    # surface quadrature on points at least 0.1 h from faces
    ok = np.all(np.abs(X) < 0.9 * h, axis=1)
    Sf = surface_ref(X[ok][:40], h[ok][:40])
    print(f"    surface-charge 2D GL(80) (points >= 0.1 h from faces, {min(40, ok.sum())} pts): max|diff| all 12 columns {np.abs(T[ok][:40] - Sf).max():.1e}")

    # ---------- 3. D10 singular locations
    hh = np.array([[1.0, 0.7, 1.3]])
    cases = {
        "face plane x=+h, outside face (X=(h,2h_y,0))": [1.0, 2 * 0.7, 0.0],
        "face plane y=-h, outside face (X=(0.3,-h_y,2h_z))": [0.3, -0.7, 2 * 1.3],
        "edge line extension (X=(h,h_y,2h_z))": [1.0, 0.7, 2 * 1.3],
        "edge line extension (X=(h,-h_y,-3h_z))": [1.0, -0.7, -3 * 1.3],
        "on face (X=(h,0.2,0.1))": [1.0, 0.2, 0.1],
        "on edge (X=(h,h_y,0.1))": [1.0, 0.7, 0.1],
        "corner (X=(h,h_y,h_z))": [1.0, 0.7, 1.3],
        "outside, plane x=+h shifted 1e-9": [1.0 + 1e-9, 2 * 0.7, 0.0],
        "edge ext shifted 1e-9": [1.0 + 1e-9, 0.7 + 1e-9, 2 * 1.3],
    }
    print("[3] D10 singular locations (nan/inf count over 36 entries; max|diff| vs 3D dipole GL(48) where the point is outside the closed cell):")
    with np.errstate(all="ignore"):
        for name, x in cases.items():
            Xc = np.array([x])
            Tc = tensor(Xc, hh)
            bad = (~np.isfinite(Tc)).sum()
            outside = np.any(np.abs(Xc[0]) > hh[0] * (1 + 1e-12))
            msg = ""
            if outside:
                Qc = dipole_quad(Xc, hh, ng=48)
                msg = f"; vs quadrature max|diff| {np.nan_to_num(np.abs(Tc - Qc), nan=np.inf).max():.1e} (scale {np.abs(Qc).max():.2f})"
                Um = np.stack([unif(Xc, np.zeros((1, 3)), hh, m) for m in range(3)], -1)
                msg += f"; uniform cols vs magpylib {np.nan_to_num(np.abs(Tc[:, :, :3] - Um), nan=np.inf).max():.1e}"
            print(f"    {name:50s}: non-finite {bad:2d}{msg}")

    # ---------- 4. sign checks on a cube
    hc = np.ones((1, 3))
    Tc = tensor(np.zeros((1, 3)), hc)[0]
    print(f"[4] cube centre: |mu0 H| for J=(0,0,xi_z) (col 11) {np.abs(Tc[:, 11]).max():.1e}; J_x=xi_x (col 3) {np.abs(Tc[:, 3]).max():.1e};"
          f" all linear cols max {np.abs(Tc[:, 3:]).max():.1e}; uniform cols diag {Tc[0, 0]:.6f} (expect -1/3)")
    xo = np.array([[1.0 + 0.05, 0, 0], [1.0 + 0.5, 0, 0], [-(1.0 + 0.05), 0, 0]])
    To = tensor(xo, np.tile(hc, (3, 1)))
    Qo = dipole_quad(xo, np.tile(hc, (3, 1)), ng=48)
    Sl = slab_ref(xo, np.tile(hc, (3, 1)))
    print(f"    J_x = xi_x (col 3) at (1.05,0,0): kernel H_x {To[0, 0, 3]:+.6f}, dipole GL {Qo[0, 0, 3]:+.6f}, slab {Sl[0, 0, 0]:+.6f} (expect > 0: + surface charge on both x faces)")
    print(f"    J_x = xi_x (col 3) at (-1.05,0,0): kernel H_x {To[2, 0, 3]:+.6f}, dipole GL {Qo[2, 0, 3]:+.6f} (expect < 0: field points away from the + sheet)")
    print(f"    J_x = xi_x at (1.5,0,0): kernel {To[1, 0, 3]:+.6f}, GL {Qo[1, 0, 3]:+.6f}")
    # inside just below +x face: H_x should be the inside value: sigma/2 jump across the face
    xi_ = np.array([[1.0 - 1e-6, 0, 0], [1.0 + 1e-6, 0, 0]])
    Ti = tensor(xi_, np.tile(hc, (2, 1)))
    print(f"    jump of H_x across +x face for col 3 (J_x = xi_x, sigma=+1 at the face): outside-inside = {Ti[1, 0, 3] - Ti[0, 0, 3]:+.6f} (expect +1 = sigma, since mu0 H_n jumps by sigma)")
    print(f"    jump of H_x across +x face for col 0 (J_x = 1): {Ti[1, 0, 0] - Ti[0, 0, 0]:+.6f} (expect +1)")

    # ---------- 5. far field, a = 0: quadrupole decay
    hc = np.array([[1.0, 0.7, 1.3]])
    dirn = np.array([0.3, -0.5, 0.8])
    dirn /= np.linalg.norm(dirn)
    rs = [20.0, 40.0, 80.0, 160.0, 1e3, 1e4]
    vals = []
    for r in rs:
        Xf = (r * dirn)[None]
        Tf = tensor(Xf, hc)[0]
        Qf = dipole_quad(Xf, hc, ng=12)[0]
        vals.append((np.abs(Tf[:, 3:]).max(), np.abs(Tf[:, :3]).max(), np.abs(Tf - Qf).max() / np.abs(Qf).max(), np.abs(Tf[:, 3:] - Qf[:, 3:]).max() / np.abs(Qf[:, 3:]).max()))
    print("[5] far field: r, max|linear cols|, max|uniform cols|, rel diff vs GL(12) (all), rel diff linear cols only")
    for r, v in zip(rs, vals):
        print(f"    r={r:7.0f}  lin {v[0]:.3e}  uni {v[1]:.3e}  reldiff {v[2]:.1e}  reldiff_lin {v[3]:.1e}")
    sl = np.log(vals[1][0] / vals[0][0]) / np.log(rs[0] / rs[1]), np.log(vals[2][0] / vals[1][0]) / np.log(rs[1] / rs[2]), np.log(vals[3][0] / vals[2][0]) / np.log(rs[2] / rs[3])
    su = np.log(vals[1][1] / vals[0][1]) / np.log(rs[0] / rs[1])
    print(f"    decay exponents linear cols: {sl[0]:.2f}, {sl[1]:.2f}, {sl[2]:.2f} (expect 4); uniform cols {su:.2f} (expect 3)")
    # theoretical: linear column G_mk far field = quadrupole with second moment int xi_k x'_l dV = delta_kl V h_k /3
    # check one entry analytically: J = e_m xi_k, x' moment M_{kl} = int xi_k x'_l dV = (V h_k/3) delta_kl
    # mu0 H_a(x) = (1/4pi) d/dx_l [ (3 r_a r_m / r^5 - delta_am / r^3) ] * (-M_kl)... compare numerically with finite differences of the dipole kernel
    r = 40.0
    Xf = (r * dirn)[None]
    Tf = tensor(Xf, hc)[0]
    V = 8 * np.prod(hc[0])
    eps = 1e-4
    Kd = lambda x: (3 * np.outer(x, x) / np.linalg.norm(x) ** 5 - np.eye(3) / np.linalg.norm(x) ** 3)  # noqa: E731
    err = 0
    for m in range(3):
        for k in range(3):
            # field = (1/4pi) int K(x - x') e_m xi_k dV' ~ -(1/4pi) sum_l dK/dx_l[:, m] * M_kl, M_kl = V h_k/3 delta_kl
            el = np.eye(3)[k]
            dK = (Kd(Xf[0] + eps * el) - Kd(Xf[0] - eps * el)) / (2 * eps)
            pred = -dK[:, m] * V * hc[0, k] / 3 / (4 * np.pi)
            err = max(err, np.abs(pred - Tf[:, 3 + 3 * m + k]).max() / np.abs(pred).max())
    print(f"    r=40: linear cols vs analytic quadrupole (M_kl = V h_k/3 delta_kl): rel diff {err:.1e} (next term O((h/r)^2) ~ {(1.3 / 40) ** 2:.0e})")
