"""Triangle kernel against brute-force quadrature, refined toward each observer
(7-point Dunavant rule on sub-triangles, split until 8 of their sizes away)."""
import numpy as np
from tri import field, frames

a1, b1 = 0.059715871789770, 0.470142064105115
a2, b2 = 0.797426985353087, 0.101286507323456
BARY = np.array([[1/3, 1/3, 1/3], [a1, b1, b1], [b1, a1, b1], [b1, b1, a1], [a2, b2, b2], [b2, a2, b2], [b2, b2, a2]])
WTS = np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3])


def reference(x, V, levels=16):
    cen, n, e1, e2, s, _ = frames(V)
    rho = x - ((x - V[0, 0]) @ n[0]) * n[0]
    T, out = V.copy(), np.zeros((3, 3))
    for lev in range(levels):
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack(sv, 1) for sv in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        size = np.linalg.norm(T[:, 1] - T[:, 0], axis=1)
        far = (np.linalg.norm(T.mean(1) - rho, axis=1) > 8 * size) | (lev == levels - 1)
        done, T = T[far], T[~far]
        pts = np.einsum("qk,tkd->tqd", BARY, done).reshape(-1, 3)
        area = 0.5 * np.linalg.norm(np.cross(done[:, 1] - done[:, 0], done[:, 2] - done[:, 0]), axis=1)
        wq = np.tile(WTS, len(done)) * np.repeat(area, len(WTS))
        u, v = (pts - cen[0]) @ e1[0] / s[0], (pts - cen[0]) @ e2[0] / s[0]
        r = x - pts
        r3 = np.linalg.norm(r, axis=1) ** 3
        out += np.array([(dens * wq / r3) @ r for dens in (np.ones_like(u), u, v)])
    return out


if __name__ == "__main__":
    V = np.array([[[0.1, -0.2, 0.3], [1.3, 0.1, 0.0], [0.4, 0.9, 0.5]]])
    cen, n, e1, e2, s, area = frames(V)
    obs = {
        "above the centroid, 0.5": cen[0] + 0.5 * n[0],
        "below, 0.05": cen[0] - 0.05 * n[0],
        "0.02 above, near an edge": V[0, 0] + 0.02 * n[0] + 0.3 * (V[0, 1] - V[0, 0]),
        "far, below": np.array([2.0, 1.5, -0.7]),
        "in the plane, outside": V[0, 0] + 1.5 * (V[0, 1] - V[0, 0]) + 0.3 * (V[0, 2] - V[0, 0]),
        "0.003 above the inside": cen[0] + 3e-3 * n[0] + 0.2 * e1[0],
    }
    worst = 0
    for name, x in obs.items():
        R, A = reference(x, V), field(x[None], V)[:, 0, 0]
        e = np.abs(R - A).max() / np.abs(R).max()
        worst = max(worst, e)
        print(f"{name:28s} densities 1, u, v: max rel diff {e:.1e}")
    print(f"worst {worst:.1e}")

    # a rectangle cut into two triangles against the rectangle closed form (p1.lin_rect)
    import sys
    sys.path.append("..")
    from p1 import lin_rect
    a, b = np.array([0.0, 0, 0]), np.array([2.0, 1.0, 0])
    tris = np.array([[[0, 0, 0], [2, 0, 0], [2, 1, 0]], [[0, 0, 0], [2, 1, 0], [0, 1, 0]]], float)
    pts = np.array([[0.7, 0.4, 0.3], [3.0, -1.0, 0.5], [1.0, 0.5, -0.02], [2.5, 0.5, 0.0]])
    E = field(pts, tris).sum(axis=2)[0]  # uniform density over both halves
    H = np.stack(lin_rect(pts[:, 0], pts[:, 1], pts[:, 2], 0.0, 2.0, 0.0, 1.0)[0], axis=-1) * 4 * np.pi
    print(f"rectangle as two triangles, uniform charge, against lin_rect: max rel diff {np.abs(E - H).max() / np.abs(H).max():.1e}")
