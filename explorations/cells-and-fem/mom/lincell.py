"""Field of an axis-aligned cuboid whose polarization varies linearly inside it (#70).

J(x') = a + G xi, xi = x'/h in [-1, 1]^3 (cell-local, h = half sides). Unknowns u = [a (3), G row-major (9)].
mu0 H = 1/(4 pi) [ sum_faces int sigma r/R^3 dS  -  div(J) int r/R^3 dV ],  sigma = J.n,  div J = sum_k G_kk / h_k.
Every face and volume term is a corner sum of sqrt, ln and atan at the 8 box corners, shared by all 12 columns.
"""

import numpy as np


def _ln_plus(a, R, rest2, xp):
    """ln(a + R), stable for a < 0 where it equals ln(rest2 / (R - a))."""
    pos = a >= 0
    return xp.where(pos, xp.log(xp.where(pos, a + R, 1.0)), xp.log(xp.where(pos, 1.0, rest2) / xp.where(pos, 1.0, R - a)))


def tensor(X, h, linear=True, xp=np):
    """mu0 H (T) at X (n, 3, cell-local) from cells with half sides h (n, 3), per unit coefficient.

    Returns (n, 3, 12) for linear cells or (n, 3, 3) for uniform ones (columns = a only).
    """
    P = xp.stack([X - h, X + h], axis=-1)  # (n, 3, 2): index 0 <-> x' = +h (face s = +1)
    px = P[:, 0, :][:, :, None, None]
    py = P[:, 1, :][:, None, :, None]
    pz = P[:, 2, :][:, None, None, :]
    px, py, pz = xp.broadcast_arrays(px, py, pz)  # (n, 2, 2, 2) corners
    R = xp.sqrt(px**2 + py**2 + pz**2)
    L = [_ln_plus(p, R, R**2 - p**2, xp) for p in (px, py, pz)]
    A = [xp.arctan(py * pz / (px * R)), xp.arctan(px * pz / (py * R)), xp.arctan(px * py / (pz * R))]
    pc = (px, py, pz)
    lower = [(xp.arange(2) == 0).astype(X.dtype)] * 3  # index 0 is the lower integration limit
    sgn3 = (-1.0) ** (lower[0][:, None, None] + lower[1][None, :, None] + lower[2][None, None, :])
    ncol = 12 if linear else 3
    out = [[xp.zeros(X.shape[0], dtype=X.dtype) for _ in range(ncol)] for _ in range(3)]  # out[comp][col]
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        for idx, s in ((0, 1.0), (1, -1.0)):
            take = [slice(None)] * 4
            take[1 + k] = idx  # fix the corner index along k: this face
            sl = tuple(take)
            W = pc[k][sl]
            Ri, pi_, pj = R[sl], pc[i][sl], pc[j][sl]
            Li, Lj, Ak = L[i][sl], L[j][sl], A[k][sl]
            # 2D corner signs over the face's remaining axes (i, j), in array order (min(i,j), max(i,j))
            s2 = (-1.0) ** (lower[0][:, None] + lower[0][None, :])
            csum = lambda F: xp.sum(F * s2, axis=(1, 2))  # noqa: E731
            M0 = (csum(-Lj), csum(-Li), csum(Ak))  # components (i, j, k)
            cols = {k: [s * m for m in M0]}
            if linear:
                Mp = (csum(pj * Li - W * Ak), csum(-Ri), csum(-W * Lj))
                Mq = (csum(-Ri), csum(pi_ * Lj - W * Ak), csum(-W * Li))
                Xi, Xj = X[:, i], X[:, j]
                cols[3 + 3 * k + k] = list(M0)
                cols[3 + 3 * k + i] = [s / h[:, i] * (Xi * m0 - mp) for m0, mp in zip(M0, Mp)]
                cols[3 + 3 * k + j] = [s / h[:, j] * (Xj * m0 - mq) for m0, mq in zip(M0, Mq)]
            for col, vec in cols.items():
                for comp, v in zip((i, j, k), vec):
                    out[comp][col] = out[comp][col] + v
    if linear:
        V = [
            -(py * L[2] + pz * L[1] - px * A[0]),
            -(px * L[2] + pz * L[0] - py * A[1]),
            -(px * L[1] + py * L[0] - pz * A[2]),
        ]
        V = [xp.sum(v * sgn3, axis=(1, 2, 3)) for v in V]
        for k in range(3):
            for comp in range(3):
                out[comp][3 + 3 * k + k] = out[comp][3 + 3 * k + k] - V[comp] / h[:, k]
    return xp.stack([xp.stack(row, axis=-1) for row in out], axis=1) / (4 * np.pi)
