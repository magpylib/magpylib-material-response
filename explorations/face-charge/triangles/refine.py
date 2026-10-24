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


def _merge_coplanar(V, F, tol, ids=False):
    """The mesh's flat faces as convex polygons: neighbouring coplanar triangles
    merged while the union stays convex. Returns [(polygon (k, 3), normal)], with
    one of the polygon's triangles as a third item if ids."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    polys = [[list(f), n[t], t] for t, f in enumerate(F)]  # vertex index loops
    merged = True
    while merged:
        merged = False
        edge_at = {}
        for pi, (loop, *_) in enumerate(polys):
            for i in range(len(loop)):
                edge_at[(loop[i], loop[(i + 1) % len(loop)])] = pi
        for (u, v), pi in edge_at.items():
            pj = edge_at.get((v, u))
            if pj is None or pj == pi or polys[pi] is None or polys[pj] is None:
                continue
            (la, na, ta), (lb, nb, _) = polys[pi], polys[pj]
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
            polys[pi], polys[pj] = [loop, na, ta], None
            merged = True
            break
        polys = [p for p in polys if p is not None]
    return [(V[loop], nrm, t) if ids else (V[loop], nrm) for loop, nrm, t in polys]


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
    convex cells (rectangles on a box). Cells longer than the last layer are cut
    into equal pieces, as the cuboid grid's interior is, and each cell is fanned
    into triangles. (Bisecting the fanned triangles instead gave 5 times as many.)"""
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
    # the interior: cells longer than the last layer cut across, into equal pieces
    h_far = ell * (1 - ((m - 1) / m) ** g)
    polys = _split_cells(polys, h_far, tol)
    return np.array([(p[0], p[k], p[k + 1]) for p, _ in polys for k in range(1, len(p) - 1)])


def _split_cells(polys, h, tol):
    """Cut every convex cell longer than h, across its longest side and then
    across the other direction, into equal pieces: rectangles stay rectangles."""
    out = []
    for poly, nrm, *rest in polys:
        sides = np.roll(poly, -1, axis=0) - poly
        u = sides[np.argmax(np.linalg.norm(sides, axis=1))]
        u = u / np.linalg.norm(u)
        pieces = [poly]
        for w in (u, np.cross(nrm, u)):
            new = []
            for piece in pieces:
                proj = (piece - piece[0]) @ w
                lo, hi = proj.min(), proj.max()
                k = int(np.ceil((hi - lo) / h - 1e-9))
                parts = [piece]
                for cut in lo + (hi - lo) * np.arange(1, k) / k:
                    nxt = []
                    for part in parts:
                        nxt += _cut_poly(part, piece[0], w, cut, tol)
                    parts = nxt
                new += parts
            pieces = new
        out += [(p, nrm, *rest) for p in pieces]
    return out


def _on_sharp(p, q, segs, tol):
    """True if the segment p-q lies along one of the sharp segments."""
    a, b = segs[:, 0], segs[:, 1]
    t = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
    for x in (p, q, 0.5 * (p + q)):
        s = np.clip(np.einsum("sd,sd->s", x - a, t), 0, np.linalg.norm(b - a, axis=1))
        if np.min(np.linalg.norm(x - (a + s[:, None] * t), axis=1)) > tol:
            return False
    return True


def _offset(P, n, shift, tol):
    """Polygon P (k, 3), counter-clockwise about n, with edge i (P[i] -> P[i+1])
    moved inward by shift[i]; None if an edge turns over (the offset is too deep)."""
    k = len(P)
    t = np.roll(P, -1, axis=0) - P
    t /= np.linalg.norm(t, axis=1)[:, None]
    nu = np.cross(n, t)  # inward
    base = P + shift[:, None] * nu
    out = np.empty_like(P)
    for j in range(k):
        i = j - 1  # edges i and j meet at vertex j
        # base[i] + a t[i] = base[j] + b t[j], solved in the plane (t[j], nu[j])
        A = np.array([[t[i] @ t[j], -1.0], [t[i] @ nu[j], 0.0]])
        rhs = np.array([(base[j] - base[i]) @ t[j], (base[j] - base[i]) @ nu[j]])
        if abs(np.linalg.det(A)) < 1e-12:
            out[j] = base[j]
            continue
        a, _ = np.linalg.solve(A, rhs)
        out[j] = base[i] + a * t[i]
    e = np.roll(out, -1, axis=0) - out
    if np.any(np.einsum("kd,kd->k", e, t) < tol):
        return None
    return out


def refine_rings(V, F, ell, m, g=2.5, angle=20.0, corner=30.0, groups=False):
    """Graded triangles by rings: each flat face's convex polygons are offset
    inward along their sharp edges at ell (k/m)^g, which gives straight layers on a
    box and rings on a faceted disc. Where two sharp edges turn by more than
    `corner` degrees, the cells near that corner are also cut parallel to the other
    edge, as on the cuboid grid; a curved rim turns by 360/N and gets plain rings.
    The rest is cut into equal pieces of the last layer's size, then fanned.

    With groups, also a group per triangle and its band frame (see _ring_groups):
    the ring cells of one layer along one rim, on one smooth face, joined into runs
    as long as the last layer's size, so that narrow facets can share one linear
    charge in the band's own coordinates, along the rim and away from it."""
    span = np.max(np.ptp(V, axis=0))
    tol = 1e-9 * span
    segs = sharp_edges(V, F, angle)
    levels = ell * (np.arange(1, m + 1) / m) ** g
    cells, corners = [], []
    for P, n, rep in _merge_coplanar(V, F, tol, ids=True):
        k = len(P)
        sharp = np.array([_on_sharp(P[i], P[(i + 1) % k], segs, 1e-6 * span) for i in range(k)])
        t = np.roll(P, -1, axis=0) - P
        t /= np.linalg.norm(t, axis=1)[:, None]
        nu = np.cross(n, t)
        rings, inner = [], P
        for li, d in enumerate(levels):
            Q = _offset(P, n, np.where(sharp, d, 0.0), tol)
            if Q is None:
                break
            for i in range(k):
                quad = np.array([inner[i], inner[(i + 1) % k], Q[(i + 1) % k], Q[i]])
                if sharp[i]:
                    rings.append((quad, ("ring", rep, 0.5 * (P[i] + P[(i + 1) % k]), li)))
            inner = Q
        local = rings + [(inner, ("interior", rep))]
        # real corners: cut the cells near them parallel to the other edge too
        for j in range(k):
            i = j - 1
            if not (sharp[i] and sharp[j]) or np.degrees(np.arccos(np.clip(t[i] @ t[j], -1, 1))) <= corner:
                continue
            corners.append(P[j])
            # cells along edge i are cut parallel to edge j, and the other way
            # round: a long strip is split only where it reaches the corner
            for d in levels:
                for near, w in ((nu[i], nu[j]), (nu[j], nu[i])):
                    nxt = []
                    for c, tag in local:
                        if (c.mean(0) - P[j]) @ near < ell:
                            nxt += [(p, tag) for p in _cut_poly(c, P[j], w, d, tol)]
                        else:
                            nxt.append((c, tag))
                    local = nxt
        cells += [(c, n, tag) for c, tag in local if len(c) >= 3]
    h_far = ell * (1 - ((m - 1) / m) ** g)
    cells = _split_cells(cells, h_far, tol)
    gid, band = (_ring_groups(cells, V, F, segs, angle, corner, h_far, ell, np.array(corners).reshape(-1, 3))
                 if groups else (None, None))
    T, G, B = [], [], []
    for ci, (c, _, _) in enumerate(cells):
        for q in range(1, len(c) - 1):
            tri = (c[0], c[q], c[q + 1])
            if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > tol**2:
                T.append(tri)
                if groups:
                    G.append(gid[ci])
                    B.append(band[ci])
    return (np.array(T), np.array(G), np.array(B)) if groups else np.array(T)


def _smooth_parts(V, F, angle):
    """A label per triangle: triangles joined across edges no sharper than angle."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    parent = list(range(len(F)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = {}
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(t)
    for ta, tb in edges.values():
        if np.degrees(np.arccos(np.clip(n[ta] @ n[tb], -1, 1))) <= angle:
            parent[root(ta)] = root(tb)
    return np.array([root(t) for t in range(len(F))])


def _chains(segs, corner):
    """The sharp segments joined end to end into rims, through vertices where
    exactly two meet and turn by at most `corner` degrees. Per segment: its rim,
    its start and unit direction along the rim, its length, and the rim length
    before it; per rim: its length and whether it closes."""
    S = len(segs)
    at = {}
    for k in range(S):
        for e in (0, 1):
            at.setdefault(tuple(segs[k, e]), []).append((k, e))
    cmin = np.cos(np.radians(corner))

    def step(k, fwd):
        here = at[tuple(segs[k, 1] if fwd else segs[k, 0])]
        other = [q for q in here if q[0] != k]
        if len(here) != 2 or len(other) != 1:
            return None
        k2, e2 = other[0]
        t1 = (segs[k, 1] - segs[k, 0]) * (1 if fwd else -1)
        t2 = (segs[k2, 1] - segs[k2, 0]) * (1 if e2 == 0 else -1)
        if t1 @ t2 < cmin * np.linalg.norm(t1) * np.linalg.norm(t2):
            return None
        return k2, e2 == 0

    rim = -np.ones(S, int)
    a, t, s0 = np.zeros((S, 3)), np.zeros((S, 3)), np.zeros(S)
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    lengths, closed = [], []
    for k in range(S):
        if rim[k] >= 0:
            continue
        cur, loop = (k, False), False
        while (r := step(*cur)) is not None:
            if r[0] == k:
                loop = True
                break
            cur = r
        path = [(k, True) if loop else (cur[0], not cur[1])]
        while (r := step(*path[-1])) is not None and r[0] != path[0][0]:
            path.append(r)
        acc = 0.0
        for kk, fwd in path:
            rim[kk] = len(lengths)
            a[kk] = segs[kk, 0] if fwd else segs[kk, 1]
            t[kk] = (segs[kk, 1] - segs[kk, 0]) * (1 if fwd else -1) / L[kk]
            s0[kk] = acc
            acc += L[kk]
        lengths.append(acc)
        closed.append(loop)
    return rim, a, t, L, s0, np.array(lengths), np.array(closed)


def _ring_groups(cells, V, F, segs, angle, corner, h, ell, corners):
    """A group per cell. Ring cells of one layer along one rim, on one smooth
    face, are joined in order along the rim into runs no longer than h whose
    normals stay within `corner` degrees; cells within ell of a real corner and
    interior cells keep a group of their own, so the grading toward corners and
    the interior's sizes are unchanged. Returns the groups and, per cell, its band
    frame (rim point a, unit t along the rim, unit nu away from it in the cell's
    plane, arc length s0 at a, rim length if closed else 0), NaN outside bands:
    s = s0 + (x - a).t and d = (x - a).nu are linear on the cell."""
    part = _smooth_parts(V, F, angle)
    rim, a, t, L, s0, lengths, closed = _chains(segs, corner)
    span = np.max(np.ptp(V, axis=0))
    gid = np.arange(len(cells))
    band = np.full((len(cells), 11), np.nan)

    def arc(X, sel):
        u = np.clip(np.einsum("ksd,sd->ks", X[:, None] - a[sel][None], t[sel]), 0, L[sel])
        dist = np.linalg.norm(X[:, None] - (a[sel][None] + u[..., None] * t[sel][None]), axis=2)
        j = dist.argmin(1)
        return s0[sel][j] + u[np.arange(len(X)), j]

    mids = 0.5 * (segs[:, 0] + segs[:, 1])
    bands = {}
    for ci, (c, _, tag) in enumerate(cells):
        if tag[0] != "ring" or (len(corners) and np.linalg.norm(corners - c.mean(0), axis=1).min() < ell):
            continue
        _, rep, mid, li = tag
        u = np.clip(np.einsum("sd,sd->s", mid - a, t), 0, L)
        k = np.linalg.norm(mid - (a + u[:, None] * t), axis=1).argmin()
        bands.setdefault((part[rep], rim[k], li), []).append(ci)
        nu = np.cross(cells[ci][1], t[k])
        nu *= np.sign((c.mean(0) - a[k]) @ nu) or 1.0
        band[ci] = [*a[k], *t[k], *nu, s0[k], lengths[rim[k]] if closed[rim[k]] else 0.0]
    for (_, r, _), members in bands.items():
        sel = np.flatnonzero(rim == r)
        ext = []
        for ci in members:
            c = cells[ci][0]
            sc = arc(c.mean(0)[None], sel)[0]
            sv = arc(c, sel)
            if closed[r]:
                sv = sv - lengths[r] * np.round((sv - sc) / lengths[r])
            ext.append((sc, sv.min(), sv.max()))
        order = np.argsort([e[0] for e in ext])
        start = end = None
        for o in order:
            ci, (_, lo, hi) = members[o], ext[o]
            n = cells[ci][1]
            if (start is None or hi - start > h * (1 + 1e-6) or lo - end > 1e-6 * span
                    or n @ n0 < np.cos(np.radians(corner))):
                head, start, end, n0 = ci, lo, hi, n
            gid[ci] = head
            end = max(end, hi)
    return gid, band
