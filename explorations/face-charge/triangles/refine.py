"""Grade a closed triangle mesh toward its sharp edges.

1. Sharp edges: mesh edges whose two triangles' normals differ by more than
   `angle` degrees (convex or re-entrant).
2. A target size: across the nearest sharp edge at distance d, the size of the
   cuboid panels' power-g layers, h(d) = (g ell / m)(d / ell)^((g-1)/g), down to
   ell m^-g at the edge, g ell / m beyond ell.
3. Longest-edge bisection (Rivara) where a triangle is larger than that. Cutting
   the edge longest in a stretched metric instead made slivers, then collinear
   triangles, near corners. A split vertex may sit on a neighbour's edge: the
   surface stays closed (splits lie on straight edges), and the charge is
   discontinuous between triangles anyway.
"""

import numpy as np


def sharp_edges(V, F, angle=20.0):
    """Segments (S, 2, 3) of the mesh edges sharper than `angle` degrees."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    edges = {}
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(t)
    segs = []
    for (u, v), ts in edges.items():
        if len(ts) != 2:
            raise ValueError("the mesh is not closed and manifold")
        if np.degrees(np.arccos(np.clip(n[ts[0]] @ n[ts[1]], -1, 1))) > angle:
            segs.append((V[u], V[v]))
    return np.array(segs)


def _metric(P, segs, ell, m, g):
    """Target metric (k, 3, 3) at points P (k, 3)."""
    h_far = g * ell / m
    h_min = ell * m ** (-g)
    M = np.tile(np.eye(3) / h_far**2, (len(P), 1, 1))
    a, b = segs[:, 0], segs[:, 1]
    t = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
    L = np.linalg.norm(b - a, axis=1)
    s = np.clip(np.einsum("ksd,sd->ks", P[:, None] - a[None], t), 0, L)  # foot along each segment
    foot = a[None] + s[..., None] * t[None]
    r = P[:, None] - foot
    d = np.linalg.norm(r, axis=-1)  # (k, S)
    near = d < ell
    h = np.clip((g * ell / m) * (np.maximum(d, 1e-300) / ell) ** ((g - 1) / g), h_min, h_far)
    w = np.where(near, 1 / h**2 - 1 / h_far**2, 0.0)
    nhat = r / np.maximum(d, 1e-300)[..., None]
    M += np.einsum("ks,ksd,kse->kde", w, nhat, nhat)
    return M


def size_target(P, segs, ell, m, g):
    """Isotropic target size at points P: the cuboid layers' size across the
    nearest sharp edge, h(d) = (g ell / m)(d / ell)^((g-1)/g), from ell m^-g at
    the edge to g ell / m beyond ell."""
    h_far, h_min = g * ell / m, ell * m ** (-g)
    a, b = segs[:, 0], segs[:, 1]
    t = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
    s = np.clip(np.einsum("ksd,sd->ks", P[:, None] - a[None], t), 0, np.linalg.norm(b - a, axis=1))
    d = np.linalg.norm(P[:, None] - (a[None] + s[..., None] * t[None]), axis=-1).min(axis=1)
    return np.clip(h_far * (d / ell) ** ((g - 1) / g), h_min, h_far)


def refine(V, F, ell, m, g=2.5, angle=20.0, max_rounds=60):
    """Triangles (T, 3, 3) graded toward the sharp edges, and those edges.

    A triangle whose longest edge exceeds the target size at its nearest point
    to a sharp edge (sampled at its vertices and edge midpoints) is cut at the
    midpoint of that longest edge (Rivara): shapes stay sound, never degenerate."""
    segs = sharp_edges(V, F, angle)
    T = V[F]
    for _ in range(max_rounds):
        e = np.roll(T, -1, axis=1) - T
        L = np.linalg.norm(e, axis=-1)
        k = L.argmax(axis=1)
        samples = np.concatenate([T, 0.5 * (T + np.roll(T, -1, axis=1))], axis=1)  # (T, 6, 3)
        h = size_target(samples.reshape(-1, 3), segs, ell, m, g).reshape(len(T), 6).min(axis=1)
        split = L[np.arange(len(T)), k] > h
        if not split.any():
            break
        S, kk = T[split], k[split]
        i = np.arange(len(S))
        a, b, c = S[i, kk], S[i, (kk + 1) % 3], S[i, (kk + 2) % 3]
        p = 0.5 * (a + b)
        T = np.concatenate([T[~split], np.stack([a, p, c], 1), np.stack([p, b, c], 1)])
    return T, segs


def _faces_at_sharp_edges(V, F, angle):
    """Per sharp edge: its end points and, for each of its two triangles, the
    triangle's unit normal and the in-plane direction pointing away from the edge."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    edges = {}
    for t, tri in enumerate(F):
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.setdefault((min(u, v), max(u, v)), []).append(t)
    out = []
    for (u, v), ts in edges.items():
        if np.degrees(np.arccos(np.clip(n[ts[0]] @ n[ts[1]], -1, 1))) <= angle:
            continue
        a, b = V[u], V[v]
        t_ = (b - a) / np.linalg.norm(b - a)
        sides = []
        for tt in ts:
            w = np.cross(n[tt], t_)
            if (V[F[tt]].mean(0) - a) @ w < 0:
                w = -w
            sides.append((n[tt], w))
        out.append((a, b, sides))
    return out


def _cut(T, a, w, d, tol):
    """Cut triangles T (k, 3, 3) by the plane (x - a).w = d; convex pieces fanned."""
    s = (T - a) @ w - d
    cross = (s.max(1) > tol) & (s.min(1) < -tol)
    keep, out = [T[~cross]], []
    for tri, si in zip(T[cross], s[cross]):
        for sign in (1, -1):
            poly = []
            for k in range(3):
                p, q, sp, sq = tri[k], tri[(k + 1) % 3], si[k], si[(k + 1) % 3]
                if sign * sp >= -tol:
                    poly.append(p)
                if (sp > tol and sq < -tol) or (sp < -tol and sq > tol):
                    poly.append(p + (q - p) * sp / (sp - sq))
            for k in range(1, len(poly) - 1):
                out.append((poly[0], poly[k], poly[k + 1]))
    return np.concatenate(keep + ([np.array(out)] if out else []))


def refine_layers(V, F, ell, m, g=2.5, angle=20.0):
    """Triangles graded toward the sharp edges by layers, as the cuboid panels are.

    For each sharp edge and each flat face along it, the triangles in that face's
    plane, near the edge, are cut along lines parallel to it at distances
    ell (k/m)^g, k = 1..m. Where two sharp edges meet, the lines cross into the
    cuboid's grid. The rest is then bisected (longest edge) down to g ell / m."""
    T = V[F]
    span = np.max(np.ptp(V, axis=0))
    tol = 1e-9 * span
    levels = ell * (np.arange(1, m + 1) / m) ** g
    for a, b, sides in _faces_at_sharp_edges(V, F, angle):
        L = np.linalg.norm(b - a)
        t_ = (b - a) / L
        for nrm, w in sides:
            for d in levels:
                c = T.mean(1)
                in_plane = np.abs((c - a) @ nrm) < 1e-6 * span
                along = (c - a) @ t_
                band = in_plane & (along > -ell) & (along < L + ell)
                band &= (((T - a) @ w).min(1) < d) & (((T - a) @ w).max(1) > d)
                if band.any():
                    T = np.concatenate([T[~band], _cut(T[band], a, w, d, tol)])
    # the interior, down to the layers' largest size
    h_far = g * ell / m
    for _ in range(40):
        e = np.roll(T, -1, axis=1) - T
        Ln = np.linalg.norm(e, axis=-1)
        k = Ln.argmax(1)
        split = Ln[np.arange(len(T)), k] > h_far
        if not split.any():
            break
        S, kk = T[split], k[split]
        i = np.arange(len(S))
        a_, b_, c_ = S[i, kk], S[i, (kk + 1) % 3], S[i, (kk + 2) % 3]
        p = 0.5 * (a_ + b_)
        T = np.concatenate([T[~split], np.stack([a_, p, c_], 1), np.stack([p, b_, c_], 1)])
    return T


def _convex(poly, nrm, tol):
    """True if the planar polygon (k, 3), ordered around nrm, is convex."""
    k = len(poly)
    for i in range(k):
        a, b, c = poly[i], poly[(i + 1) % k], poly[(i + 2) % k]
        if np.cross(b - a, c - b) @ nrm < -tol:
            return False
    return True


def _merge_coplanar(V, F, tol):
    """The mesh's flat faces as convex polygons: neighbouring coplanar triangles
    merged while the union stays convex. Returns [(polygon (k, 3), normal)]."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    polys = [[list(f), n[t]] for t, f in enumerate(F)]  # vertex index loops
    merged = True
    while merged:
        merged = False
        edge_at = {}
        for pi, (loop, _) in enumerate(polys):
            for i in range(len(loop)):
                edge_at[(loop[i], loop[(i + 1) % len(loop)])] = pi
        for (u, v), pi in edge_at.items():
            pj = edge_at.get((v, u))
            if pj is None or pj == pi or polys[pi] is None or polys[pj] is None:
                continue
            (la, na), (lb, nb) = polys[pi], polys[pj]
            if na @ nb < 1 - 1e-9:
                continue
            # splice: la has u -> v, lb has v -> u
            ia, ib = la.index(u), lb.index(v)
            la_r = la[ia + 1 :] + la[: ia + 1]  # starts at v, ends at u
            lb_r = lb[ib + 1 :] + lb[: ib + 1]  # starts at u, ends at v
            loop = la_r[:-1] + lb_r[:-1]  # v ... (u) ... back to v
            # drop collinear vertices
            pts = V[loop]
            keep = [i for i in range(len(loop))
                    if np.linalg.norm(np.cross(pts[i] - pts[i - 1], pts[(i + 1) % len(loop)] - pts[i])) > tol]
            loop = [loop[i] for i in keep]
            if not _convex(V[loop], na, tol):
                continue
            polys[pi], polys[pj] = [loop, na], None
            merged = True
            break
        polys = [p for p in polys if p is not None]
    return [(V[loop], nrm) for loop, nrm in polys]


def _cut_poly(poly, a, w, d, tol):
    """Cut a convex polygon by the plane (x - a).w = d: one or two convex polygons."""
    s = (poly - a) @ w - d
    if s.max() <= tol or s.min() >= -tol:
        return [poly]
    out = []
    for sign in (1, -1):
        piece = []
        k = len(poly)
        for i in range(k):
            p, q, sp, sq = poly[i], poly[(i + 1) % k], s[i], s[(i + 1) % k]
            if sign * sp >= -tol:
                piece.append(p)
            if (sp > tol and sq < -tol) or (sp < -tol and sq > tol):
                piece.append(p + (q - p) * sp / (sp - sq))
        out.append(np.array(piece))
    return out


def refine_cells(V, F, ell, m, g=2.5, angle=20.0):
    """Like refine_layers, but the flat faces are first merged back into convex
    polygons, so the export's diagonals do not cross the layers: the cuts leave
    convex cells (rectangles on a box), each fanned into triangles."""
    span = np.max(np.ptp(V, axis=0))
    tol = 1e-9 * span
    polys = _merge_coplanar(V, F, tol)
    levels = ell * (np.arange(1, m + 1) / m) ** g
    for a, b, sides in _faces_at_sharp_edges(V, F, angle):
        L = np.linalg.norm(b - a)
        t_ = (b - a) / L
        for nrm, w in sides:
            for d in levels:
                new = []
                for poly, pn in polys:
                    c = poly.mean(0)
                    if (abs((c - a) @ nrm) < 1e-6 * span and pn @ nrm > 1 - 1e-9
                            and -ell < (c - a) @ t_ < L + ell):
                        new += [(p, pn) for p in _cut_poly(poly, a, w, d, tol)]
                    else:
                        new.append((poly, pn))
                polys = new
    T = np.array([(p[0], p[k], p[k + 1]) for p, _ in polys for k in range(1, len(p) - 1)])
    h_far = g * ell / m
    for _ in range(40):
        e = np.roll(T, -1, axis=1) - T
        Ln = np.linalg.norm(e, axis=-1)
        k = Ln.argmax(1)
        split = Ln[np.arange(len(T)), k] > h_far
        if not split.any():
            break
        S, kk = T[split], k[split]
        i = np.arange(len(S))
        a_, b_, c_ = S[i, kk], S[i, (kk + 1) % 3], S[i, (kk + 2) % 3]
        p = 0.5 * (a_ + b_)
        T = np.concatenate([T[~split], np.stack([a_, p, c_], 1), np.stack([p, b_, c_], 1)])
    return T
