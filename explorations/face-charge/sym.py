"""Mirror symmetry of a cuboid's face mesh: split the P1 operator into the
8 parity blocks of x -> -x, y -> -y, z -> -z (the group Z2^3 of D2h)."""

import numpy as np
import scipy.sparse as sp


def reflections(pan, tol=1e-9):
    """For each axis a: image panel index and the sign of each P1 dof (1, s, t)."""
    scale = np.max(pan.half)
    key = lambda c: tuple(np.round(c / (tol * scale)).astype(np.int64))
    index = {key(c): n for n, c in enumerate(pan.center)}
    imgs, sgns = [], []
    for a in range(3):
        c = pan.center.copy()
        c[:, a] *= -1
        img = np.array([index[key(x)] for x in c])
        sg = np.ones((pan.N, 3))
        sg[pan.i == a, 1] = -1  # s flips if a is the panel's first in-plane axis
        sg[pan.j == a, 2] = -1
        imgs.append(img)
        sgns.append(sg)
    return imgs, sgns


def group(pan):
    """All 8 elements g (bitmask over axes): image index and dof signs."""
    imgs, sgns = reflections(pan)
    els = []
    for g in range(8):
        img = np.arange(pan.N)
        sg = np.ones((pan.N, 3))
        for a in range(3):
            if g >> a & 1:
                # the sign depends only on the panel's in-plane axes, which
                # its images share, so the order of composition is irrelevant
                sg = sg * sgns[a]
                img = imgs[a][img]
        els.append((img, sg))
    return els


def irrep_bases(pan):
    """Orthonormal sparse bases B_eps (3N x n_eps) for the 8 parity irreps."""
    N = pan.N
    els = group(pan)
    seen = np.zeros(N, bool)
    cols = {e: ([], [], []) for e in range(8)}  # rows, vals, col id
    counts = np.zeros(8, int)
    for p in range(N):
        if seen[p]:
            continue
        orbit = {els[g][0][p] for g in range(8)}
        for q in orbit:
            seen[q] = True
        for alpha in range(3):
            for e in range(8):
                v = {}
                for g in range(8):
                    chi = (-1) ** bin(g & e).count("1")
                    q = els[g][0][p]
                    v[3 * q + alpha] = v.get(3 * q + alpha, 0.0) + chi * els[g][1][p, alpha]
                v = {k: x for k, x in v.items() if abs(x) > 1e-12}
                if not v:
                    continue
                nrm = np.sqrt(sum(x * x for x in v.values()))
                r, val, c = cols[e]
                for k, x in v.items():
                    r.append(k)
                    val.append(x / nrm)
                    c.append(counts[e])
                counts[e] += 1
    B = []
    for e in range(8):
        r, val, c = cols[e]
        B.append(sp.csc_matrix((val, (r, c)), shape=(3 * N, counts[e])))
    assert counts.sum() == 3 * N, counts
    return B
