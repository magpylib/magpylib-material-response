"""Closed triangle meshes for the tests: icosphere, and any cuboid panel mesh cut into triangles."""
import numpy as np


def icosphere(level, radius=1.0):
    t = (1 + 5**0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v = v / np.linalg.norm(v, axis=1)[:, None]
    for _ in range(level):
        cache, verts, faces = {}, list(v), []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(verts), np.array(faces)
    return outward(v * radius, f)


def outward(v, f):
    """Turn every face to point away from the body's centroid (convex bodies)."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("fd,fd->f", n, tri.mean(1) - v.mean(0)) < 0
    f = f.copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    return v, f


def panels_to_triangles(pan):
    """The rectangles of a bem.Panels mesh, each cut into two outward triangles."""
    idx = np.arange(pan.N)
    corners = np.zeros((pan.N, 4, 3))
    for r, (pp, qq) in enumerate(((pan.p1, pan.q1), (pan.p2, pan.q1), (pan.p2, pan.q2), (pan.p1, pan.q2))):
        corners[idx, r, pan.i] = pp
        corners[idx, r, pan.j] = qq
        corners[idx, r, pan.k] = pan.w0
    tri = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.concatenate([pan.normal, pan.normal])
    flip = np.einsum("fd,fd->f", n, out) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    verts = tri.reshape(-1, 3)
    return verts, np.arange(len(verts)).reshape(-1, 3)
