"""Attack 4: conditioning of p1real's least-squares system (27-cell cube, lsq8) and normal equations vs lstsq."""
import sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy
from lincell import tensor
from p1emul import COLLOC, P, grid
MU0 = magpy.mu_0

def assemble(pos, dim, chi, b0, xi_p):
    n, half, nb = len(pos), dim / 2, 12
    npc = len(xi_p)
    X = (pos[:, None, :] + xi_p[None] * half[:, None, :]).reshape(-1, 3)
    rel = X[:, None, :] - pos[None]
    T = tensor(rel.reshape(-1, 3), np.broadcast_to(half, rel.shape).reshape(-1, 3), linear=True)
    A = -chi * T.reshape(len(X), n, 3, nb).transpose(0, 2, 1, 3).reshape(len(X) * 3, n * nb)
    Pp = P(xi_p, nb).reshape(-1, nb)
    for i in range(n):
        A[i * npc * 3 : (i + 1) * npc * 3, i * nb : (i + 1) * nb] += Pp
    b = np.zeros((len(X), 3)); b[:, 2] = chi * b0
    return A, b.reshape(-1)

for ncell in (2, 3, 4):
    pos, dim = grid((0.01,) * 3, (0, 0, 0), (ncell,) * 3)
    A, b = assemble(pos, dim, 1000.0, 0.01, COLLOC["lsq8"])
    sv = np.linalg.svd(A, compute_uv=False)
    condA = sv[0] / sv[-1]
    u_ne = np.linalg.solve(A.T @ A, A.T @ b)
    u_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    q, r = np.linalg.qr(A); u_qr = np.linalg.solve(r, q.T @ b)
    mom = lambda u: (u.reshape(-1, 12)[:, 2] * np.prod(dim, 1)).sum() / MU0
    res = lambda u: np.linalg.norm(A @ u - b) / np.linalg.norm(b)
    print(f"{ncell**3:3d} cells: A {A.shape}, cond(A) = {condA:.3e}, cond(A^T A) = {condA**2:.3e}; "
          f"max|u_ne - u_lstsq|/max|u| = {np.abs(u_ne - u_ls).max() / np.abs(u_ls).max():.1e}, "
          f"moment NE {mom(u_ne):.10f} lstsq {mom(u_ls):.10f} QR {mom(u_qr):.10f}; rel residual {res(u_ne):.3e} / {res(u_ls):.3e}")
    # how far is the LSQ fit from interpolation: residual per equation type
    r = (A @ u_ls - b).reshape(-1, 8, 3)
    print(f"     residual |r| rms / |b| rms = {np.sqrt((r**2).mean()) / np.sqrt((b**2).mean()):.3e}  (collocation would be 0)")
