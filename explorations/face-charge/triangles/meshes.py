"""Closed triangle meshes for the tests: icosphere, cuboid panels cut into triangles,
extrusions, and exports of a cylinder and of a filleted plate with a hole."""
import numpy as np
from scipy.spatial import Delaunay


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


def extrude(outline, y0, y1, cap_rects):
    """Closed mesh of a planar outline (x, z), counter-clockwise, extruded along y.

    The caps are given as rectangles (x0, x1, z0, z1) tiling the outline; they
    are cut in two triangles each and merged back by the grader."""
    verts, faces = [], []

    def quad(a, b, c, d):  # outward when a, b, c, d run counter-clockwise seen from outside
        i = len(verts)
        verts.extend([a, b, c, d])
        faces.extend([[i, i + 1, i + 2], [i, i + 2, i + 3]])

    n = len(outline)
    for k in range(n):
        (xa, za), (xb, zb) = outline[k], outline[(k + 1) % n]
        quad((xa, y0, za), (xb, y0, zb), (xb, y1, zb), (xa, y1, za))
    for x0, x1, z0, z1 in cap_rects:
        quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0))  # y = y0, normal -y
        quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))  # y = y1, normal +y
    v, f = np.array(verts, float), np.array(faces)
    # weld coincident vertices so the mesh is closed and manifold
    key = np.round(v / np.max(np.ptp(v, axis=0)) * 1e9).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return _closed_outward(v[idx], inv[f])


def _closed_outward(v, f):
    """Check that every edge runs once in each direction (closed, one consistent
    orientation), and turn the faces outward."""
    directed = {(a, b) for tri in f for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    if len(directed) != 3 * len(f) or any((b, a) not in directed for a, b in directed):
        raise ValueError("caps and walls do not close consistently")
    tri = v[f]
    if np.einsum("fd,fd->f", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        f = f[:, [0, 2, 1]]
    return v, f


def ucore():
    """The U-core of the ucore_electromagnet page as one closed mesh (m)."""
    outline = [(-0.05, -0.01), (-0.03, -0.01), (0.03, -0.01), (0.05, -0.01), (0.05, 0.01),
               (0.05, 0.07), (0.03, 0.07), (0.03, 0.01), (-0.03, 0.01), (-0.03, 0.07),
               (-0.05, 0.07), (-0.05, 0.01)]  # a vertex wherever a cap rectangle meets the wall
    rects = [(-0.05, -0.03, -0.01, 0.01), (-0.03, 0.03, -0.01, 0.01), (0.03, 0.05, -0.01, 0.01),
             (-0.05, -0.03, 0.01, 0.07), (0.03, 0.05, 0.01, 0.07)]  # conforming at x = +-0.03
    return extrude(outline, -0.01, 0.01, rects)


def cylinder(n_sides, radius, height):
    """A cylinder along z exported with n_sides flat facets, its caps fanned from
    one rim vertex (no centre vertex), as STL exporters often do (m)."""
    phi = 2 * np.pi * np.arange(n_sides) / n_sides
    ring = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
    lo = np.column_stack([ring, np.full(n_sides, -height / 2)])
    hi = np.column_stack([ring, np.full(n_sides, height / 2)])
    v = np.concatenate([lo, hi])
    f = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        f += [[i, j, n_sides + j], [i, n_sides + j, n_sides + i]]  # mantle, outward
    for i in range(1, n_sides - 1):
        f += [[0, i + 1, i], [n_sides, n_sides + i, n_sides + i + 1]]  # bottom (-z), top (+z)
    f = np.array(f)
    tri = v[f]
    if np.einsum("fd,fd->f", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        f = f[:, [0, 2, 1]]
    return v, f


def _area2(P):
    """Twice the signed area of the polygon P (k, 2)."""
    return float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def plate(n_fillet, n_hole, size=(0.02, 0.012, 0.004), fillet=0.003, hole=(0.003, 0.0, 0.003)):
    """A plate along z with filleted vertical edges and a round through-hole,
    exported as a CAD tool would: n_fillet facets per 90 deg fillet, n_hole round
    the hole, each straight side one quad, and caps triangulated between outline and
    hole from their boundary vertices only, so they are long thin triangles (m).

    size: (x, y, z) extents; hole: (x, y) of its centre and its radius."""
    a, b, t = size[0] / 2, size[1] / 2, size[2] / 2
    out = []
    for k, (cx, cy) in enumerate(((a - fillet, b - fillet), (-a + fillet, b - fillet),
                                  (-a + fillet, -b + fillet), (a - fillet, -b + fillet))):
        phi = np.pi / 2 * (k + np.arange(n_fillet + 1) / n_fillet)
        out += list(np.column_stack([cx + fillet * np.cos(phi), cy + fillet * np.sin(phi)]))
    O = np.array(out)  # counter-clockwise
    hx, hy, hr = hole
    phi = 2 * np.pi * np.arange(n_hole) / n_hole
    Hh = np.column_stack([hx + hr * np.cos(phi), hy + hr * np.sin(phi)])  # counter-clockwise
    no, nh = len(O), len(Hh)
    # the cap from its boundary vertices only, as exporters do: Delaunay keeps the
    # convex outline and the hole's short chords; drop the triangles in the hole
    P2 = np.concatenate([O, Hh])
    cap = Delaunay(P2).simplices
    c = P2[cap].mean(axis=1)
    cap = cap[np.hypot(c[:, 0] - hx, c[:, 1] - hy) > hr * np.cos(np.pi / n_hole)]
    tri2 = P2[cap]
    e1, e2 = tri2[:, 1] - tri2[:, 0], tri2[:, 2] - tri2[:, 0]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    cap[area2 < 0] = cap[area2 < 0][:, ::-1]
    if not np.isclose(np.abs(area2).sum(), _area2(O) - _area2(Hh), rtol=1e-12):
        raise ValueError("the cap's triangles overlap or leave gaps")
    # vertices: top outline, top hole, bottom outline, bottom hole
    v = np.concatenate([np.column_stack([O, np.full(no, t)]), np.column_stack([Hh, np.full(nh, t)]),
                        np.column_stack([O, np.full(no, -t)]), np.column_stack([Hh, np.full(nh, -t)])])
    top = cap.tolist()
    f = top + [[x + no + nh for x in tri[::-1]] for tri in top]
    ot, ht, ob, hb = 0, no, no + nh, 2 * no + nh
    for k in range(no):
        l = (k + 1) % no
        f += [[ob + k, ob + l, ot + l], [ob + k, ot + l, ot + k]]
    for k in range(nh):
        l = (k + 1) % nh
        f += [[hb + k, ht + l, hb + l], [hb + k, ht + k, ht + l]]
    return _closed_outward(v, np.array(f))
