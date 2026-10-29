"""Grade a closed triangle mesh toward its sharp edges, as the cuboid panels are.

Sharp edges are the mesh edges whose two triangles' normals differ by more than
an angle. The mesh's flat faces are merged back into convex polygons, so that a
CAD export's long diagonals do not cross the layers, and each polygon is offset
inward along its sharp edges at ``ell (k / m)^g``, ``k = 1 .. m``: straight
layers on a box, rings on a faceted disc. Where two sharp edges turn by more
than a corner angle, the cells near that corner are also cut parallel to the
other edge, as on the cuboid grid. The rest is cut into equal cells of the last
layer's size, and each cell is fanned into triangles.

From the study on ``explore/face-charge`` (``triangles/refine.py``,
``refine_rings``).
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np


def rounding(V):
    """The coordinates' rounding at their largest: float32's where every one is
    a float32 number, as STL and many CAD exports give, else float64's."""
    V = np.asarray(V, float)
    eps = 2.0**-24 if np.array_equal(V, V.astype(np.float32)) else 2.0**-53
    return eps * np.abs(V).max()


def drop_slivers(V, F, tol):
    """The faces without slivers, triangles thinner than ``tol``, as STL repairs
    and float32 exports leave them. A sliver lies along its longest edge, a
    T-junction: the triangle across that edge is split at the sliver's third
    corner instead. A sliver with two corners together goes with them joined."""
    F = [list(f) for f in np.asarray(F, int)]
    for _ in range(4 * len(F)):  # each round removes one sliver
        if not F:
            break
        T = V[np.array(F)]
        e = np.roll(T, -1, axis=1) - T  # edge k from corner k to k + 1
        lengths = np.linalg.norm(e, axis=2)
        height = np.linalg.norm(np.cross(e[:, 0], e[:, 2]), axis=1) / np.maximum(
            lengths.max(1), np.finfo(float).tiny
        )
        thin = np.flatnonzero(height <= tol)
        if not thin.size:
            return np.array(F, dtype=int).reshape(-1, 3)
        s = thin[0]
        if lengths[s].min() <= tol:  # two corners together: join them
            k = int(lengths[s].argmin())
            u, v = F[s][k], F[s][(k + 1) % 3]
            F = [[u if w == v else w for w in f] for f in F]
            F = [f for f in F if len(set(f)) == 3]
            continue
        k = int(lengths[s].argmax())
        a, b, c = F[s][k], F[s][(k + 1) % 3], F[s][(k + 2) % 3]
        # the triangle across the edge a -> b runs b -> a: split it at c
        y = next(
            (
                i
                for i, f in enumerate(F)
                if any(f[j] == b and f[(j + 1) % 3] == a for j in range(3))
            ),
            None,
        )
        if y is None:
            msg = "face charge needs a closed mesh, each edge shared by two triangles"
            raise ValueError(msg)
        j = next(j for j in range(3) if F[y][j] == b)
        d = F[y][(j + 2) % 3]
        F[y] = [b, c, d]
        F[s] = [c, a, d]
    if F:
        msg = "the mesh's slivers could not be removed: repair the mesh"
        raise ValueError(msg)
    return np.zeros((0, 3), dtype=int)


def sharp_edges(V, F, angle):
    """Segments (S, 2, 3) of the mesh edges sharper than ``angle`` degrees;
    none along a triangle of no area, which has no normal."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=1)[:, None]
    edges = {}
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(t)
    segs = []
    for (u, v), ts in edges.items():
        if len(ts) != 2:
            msg = "face charge needs a closed mesh, each edge shared by two triangles"
            raise ValueError(msg)
        if (
            np.degrees(np.arccos(np.clip(n[ts[0]] @ n[ts[1]], -1, 1))) > angle
        ):  # NaN: no
            segs.append((V[u], V[v]))
    return np.reshape(np.array(segs), (-1, 2, 3))


def _convex(poly, normal, tol):
    """True if the planar polygon (k, 3), ordered round ``normal``, is convex."""
    edges = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(edges, axis=1)
    if np.any(length <= tol):  # a corner repeated
        return False
    # how far each next corner bends in from the line of each edge
    bend = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal / length
    return bool(np.all(bend >= -tol))


def _merge_coplanar(V, F, tol):
    """The mesh's flat faces as convex polygons. Returns ``[(polygon (k, 3),
    normal)]``.

    Neighbouring coplanar triangles make a flat region; a region whose
    outline is one convex loop, such as a box's face however finely the export
    cut it, is that loop. A region with vertices inside it, as an FEM surface
    mesh has, is cut from its outline alone (:func:`_cut_at_reflex`), the same
    however the export cut it. Another, all its vertices on its outline, is
    merged triangle by triangle while the union stays convex
    (:func:`_merge_greedy`)."""
    area = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = area / np.linalg.norm(area, axis=1)[:, None]
    # neighbours across each edge, joined where their normals agree
    first = {}
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            first.setdefault((u, v), t)
    parent = list(range(len(F)))

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for (u, v), t in first.items():
        s_ = first.get((v, u))
        if s_ is not None and n[t] @ n[s_] >= 1 - 1e-9:
            parent[root(t)] = root(s_)
    regions = {}
    for t in range(len(F)):
        regions.setdefault(root(t), []).append(t)
    out = []
    for faces in regions.values():
        loop = _outline(F[faces]) if len(faces) > 1 else list(F[faces[0]])
        if loop is not None:
            loop = _drop_collinear(V, loop, tol)
            if len(loop) < 3:  # of no area
                continue
            # from its least corner, so that the cells follow the geometry,
            # not the vertices' numbering
            k = int(np.lexsort(V[loop][:, ::-1].T)[0])  # by x, then y, then z
            loop = loop[k:] + loop[:k]
            # the region's normal from all its triangles, the same however the
            # face is cut: float32 faces are flat to rounding only
            normal = area[faces].sum(0)
            normal /= np.linalg.norm(normal)
            if _convex(V[loop], normal, tol):
                out.append((V[loop], normal))
                continue
        loops = _loops(F[faces])
        if loops is not None and set(F[faces].ravel()) - set().union(*loops):
            # vertices inside the face, as an FEM surface mesh has, carry no
            # shape: merged, its irregular triangles left 5 times the cells
            normal = area[faces].sum(0)
            normal /= np.linalg.norm(normal)
            loops = [_drop_collinear(V, loop, tol) for loop in loops]
            out += _cut_at_reflex(V, loops, normal, tol)
            continue
        out += _merge_greedy(V, F[faces], n[faces], tol)
    return out


def _outline(F):
    """The one boundary loop of the triangles ``F``, wound as they are, or
    None if they have more than one (a hole, or two pieces)."""
    edges = {(u, v) for a, b, c in F for u, v in ((a, b), (b, c), (c, a))}
    nxt = {u: v for u, v in edges if (v, u) not in edges}
    if not nxt or len(nxt) != len(set(nxt.values())):
        return None
    start = next(iter(nxt))
    loop, u = [start], nxt[start]
    while u != start:
        loop.append(u)
        u = nxt.get(u)
        if u is None or len(loop) > len(nxt):
            return None
    return loop if len(loop) == len(nxt) else None


def _drop_collinear(V, loop, tol):
    """The loop's vertices farther than ``tol`` from the line through their
    two neighbours: where it turns."""
    pts = V[loop]
    prev, chord = (
        np.roll(pts, 1, axis=0),
        np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0),
    )
    height = np.linalg.norm(np.cross(pts - prev, chord), axis=1) / np.maximum(
        np.linalg.norm(chord, axis=1), np.finfo(float).tiny
    )
    return [w for w, h in zip(loop, height > tol, strict=True) if h]


def _loops(F):
    """Every boundary loop of the triangles ``F``, wound as they are, the
    triangles on their left; None where two loops meet at a vertex."""
    edges = {(u, v) for a, b, c in F for u, v in ((a, b), (b, c), (c, a))}
    rim = [(u, v) for u, v in edges if (v, u) not in edges]
    nxt = dict(rim)
    if len(nxt) != len(rim) or set(nxt) != set(nxt.values()):
        return None
    loops, seen = [], set()
    for start in sorted(nxt):
        if start in seen:
            continue
        loop, u = [], start
        while u not in seen:
            seen.add(u)
            loop.append(u)
            u = nxt[u]
        if u != start:
            return None
        loops.append(loop)
    return loops


def _cut_at_reflex(V, loops, normal, tol):
    """A flat region given by its outline alone, ``loops`` of vertex indices
    with the region on their left round ``normal``, cut into convex polygons:
    at a corner turning into the region, along the extension of one of its
    two edges, the shorter way, to where it meets the outline. A cut from a
    hole joins the hole to the loop it meets, so that the next cuts split it.
    Each cut takes away a corner turning in and makes none, so an L-shaped
    face becomes two rectangles however the export cut it."""
    e1 = V[loops[0][1]] - V[loops[0][0]]
    e1 /= np.linalg.norm(e1)
    axes = np.array([e1, np.cross(normal, e1)]).T
    origin = V[loops[0][0]]

    def flat(P):
        return (np.asarray(P) - origin) @ axes

    pending = [[[V[i] for i in loop] for loop in loops if len(loop) >= 3]]
    pieces = []
    for _ in range(4 * sum(len(loop) for loop in loops) + 8):
        if not pending:
            break
        region = pending.pop()
        cut = _reflex_cut(region, flat, tol)
        if cut is None:  # no corner turns in: one convex loop
            pieces.append(region[0])
        else:
            pending += _regroup(cut, flat)
    else:
        msg = "a flat face could not be cut into convex polygons: repair the mesh"
        raise ValueError(msg)
    out = []
    for piece in pieces:
        P = np.array(piece)
        keep = _drop_collinear(P, list(range(len(P))), tol)
        if len(keep) >= 3:
            P = P[keep]
            k = int(np.lexsort(P[:, ::-1].T)[0])  # from its least corner
            out.append((np.roll(P, -k, axis=0), normal))
    return out


def _reflex_cut(region, flat, tol):
    """``region``'s loops, of points, after one cut from its least corner
    that turns into it; None if none does."""
    Q = [flat(loop) for loop in region]
    corners = []  # turning in: by their point, then loop and place
    for a, q in enumerate(Q):
        e_in = q - np.roll(q, 1, axis=0)
        e_out = np.roll(q, -1, axis=0) - q
        size = np.linalg.norm(e_in, axis=1) * np.linalg.norm(e_out, axis=1)
        turn = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
        corners += [
            (tuple(region[a][i]), a, i) for i in np.flatnonzero(turn < -1e-9 * size)
        ]
    if not corners:
        return None
    _, a, i = min(corners)
    e_in = Q[a][i] - Q[a][i - 1]
    e_out = Q[a][(i + 1) % len(Q[a])] - Q[a][i]
    # every edge of every loop, from A to B, the region on its left
    A = np.concatenate(Q)
    B = np.concatenate([np.roll(q, -1, axis=0) for q in Q])
    owner = np.concatenate([np.full(len(q), b) for b, q in enumerate(Q)])
    index = np.concatenate([np.arange(len(q)) for q in Q])
    E = B - A
    r = Q[a][i]
    hits = []
    for d in (e_in / np.linalg.norm(e_in), -e_out / np.linalg.norm(e_out)):
        den = d[0] * E[:, 1] - d[1] * E[:, 0]  # > 0: the ray leaves across it
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((A - r)[:, 0] * E[:, 1] - (A - r)[:, 1] * E[:, 0]) / den
            s = ((A - r)[:, 0] * d[1] - (A - r)[:, 1] * d[0]) / den
        slack = tol / np.maximum(np.linalg.norm(E, axis=1), tol)
        ok = (den > 1e-12) & (t > tol) & (s >= -slack) & (s <= 1 + slack)
        if ok.any():
            k = np.flatnonzero(ok)[np.argmin(t[ok])]
            hits.append((t[k], k, s[k]))
    if not hits:
        return None
    _, k, s = min(hits, key=lambda h: h[0])
    b, j = owner[k], index[k]
    loop_b = list(region[b])
    length = np.linalg.norm(E[k])
    if s * length <= tol:
        x = j
    elif (1 - s) * length <= tol:
        x = (j + 1) % len(loop_b)
    else:  # a new corner on that edge
        p, q = loop_b[j], loop_b[(j + 1) % len(loop_b)]
        loop_b.insert(j + 1, p + s * (q - p))
        x = j + 1
        if a == b and x <= i:  # the corner moved along
            i += 1
    loops = [list(loop) for loop in region]
    loops[b] = loop_b
    if a == b:  # split in two, both keep the corner and the point met
        loop = loops[a]
        lo, hi = (i, x) if i < x else (x, i)
        parts = [loop[lo : hi + 1], loop[hi:] + loop[: lo + 1]]
        rest = [lp for c, lp in enumerate(loops) if c != a]
        return parts + rest
    # joined: round a from the corner, across, round b, and back
    la, lb = loops[a], loops[b]
    joined = la[i:] + la[: i + 1] + lb[x:] + lb[: x + 1]
    return [joined] + [lp for c, lp in enumerate(loops) if c not in (a, b)]


def _regroup(loops, flat):
    """Loops into regions: each loop wound round the normal with the loops
    wound the other way, the holes, that lie inside it."""
    Q = [flat(loop) for loop in loops]
    area = [
        float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
        / 2
        for q in Q
    ]
    outer = [c for c, A in enumerate(area) if A > 0]
    regions = {c: [loops[c]] for c in outer}
    for c, A in enumerate(area):
        if A >= 0:
            continue
        q = Q[c]
        e = np.argmax(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1))
        point = (q[e] + q[(e + 1) % len(q)]) / 2  # a point of the hole's rim
        # the smallest loop around it; the largest if rounding says none
        inside = [o for o in outer if _inside(point, Q[o])] or [
            max(outer, key=lambda o: area[o])
        ]
        regions[min(inside, key=lambda o: area[o])].append(loops[c])
    return list(regions.values())


def _inside(point, q):
    """True if the 2D ``point`` lies inside the loop ``q``: crossings to its
    right, odd."""
    a, b = q, np.roll(q, -1, axis=0)
    up = (a[:, 1] <= point[1]) != (b[:, 1] <= point[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(up & (x > point[0])) % 2)


def _merge_greedy(V, F, n, tol):
    """Neighbouring triangles of one flat region merged while the union stays
    convex: the first pair that merges, scanning the polygons in order and each
    one's edges round it, and again from the start after every merge, so that
    the result does not depend on the order merges are found in.

    Without starting over: a pair that does not merge is tried again only once
    one of the two has changed, so the edges to try wait in a queue in that
    scan order, and a merge queues the new polygon's edges and its neighbours'
    toward it. An L-shaped face cut into 9,600 triangles took 13.7 s (#106)."""
    loops = {p: list(f) for p, f in enumerate(F)}  # vertex loops, by scan rank
    normal = dict(enumerate(n))
    owner = {(u, f[(i + 1) % 3]): p for p, f in enumerate(F) for i, u in enumerate(f)}
    version = dict.fromkeys(loops, 0)
    queue = [(p, i, 0) for p in loops for i in range(3)]  # sorted: a heap

    def push(p):  # every edge of polygon p, as it now is
        for i in range(len(loops[p])):
            heapq.heappush(queue, (p, i, version[p]))

    while queue:
        pi, i, seen = heapq.heappop(queue)
        if pi not in loops or seen != version[pi]:
            continue  # merged away, or changed: queued anew
        la = loops[pi]
        u, v = la[i], la[(i + 1) % len(la)]
        pj = owner.get((v, u))
        if pj is None or pj == pi or normal[pi] @ normal[pj] < 1 - 1e-9:
            continue
        lb = loops[pj]
        # splice: la runs u -> v, lb runs v -> u
        ia, ib = i, lb.index(v)
        loop = (la[ia + 1 :] + la[: ia + 1])[:-1] + (lb[ib + 1 :] + lb[: ib + 1])[:-1]
        if len(set(loop)) < len(loop):  # the two share more than this edge
            continue
        loop = _drop_collinear(V, loop, tol)
        if not _convex(V[loop], normal[pi], tol):  # collinear vertices dropped above
            continue
        for w, x in itertools.pairwise([*la, la[0]]):
            del owner[w, x]
        for w, x in itertools.pairwise([*lb, lb[0]]):
            del owner[w, x]
        del loops[pj]
        loops[pi], version[pi] = loop, version[pi] + 1
        for w, x in itertools.pairwise([*loop, loop[0]]):
            owner[w, x] = pi
        push(pi)
        # the neighbours' edges toward it: their pair with it is new
        for w, x in itertools.pairwise([*loop, loop[0]]):
            q = owner.get((x, w))
            if q is not None and q != pi:
                lq = loops[q]
                heapq.heappush(queue, (q, lq.index(x), version[q]))
    return [(V[loop], normal[p]) for p, loop in sorted(loops.items())]


def _cut_poly(poly, a, w, d, tol):
    """A convex polygon cut by the plane ``(x - a).w = d``: one or two convex
    polygons."""
    s = (poly - a) @ w - d
    if s.max() <= tol or s.min() >= -tol:
        return [poly]
    out = []
    k = len(poly)
    for sign in (1, -1):
        piece = []
        for i in range(k):
            p, q, sp, sq = poly[i], poly[(i + 1) % k], s[i], s[(i + 1) % k]
            if sign * sp >= -tol:
                piece.append(p)
            if (sp > tol and sq < -tol) or (sp < -tol and sq > tol):
                piece.append(p + (q - p) * sp / (sp - sq))
        out.append(np.array(piece))
    return out


def _split_cells(cells, h, tol):
    """Every convex cell longer than ``h`` cut across its longest side, then
    across the other direction, into equal pieces: rectangles stay rectangles."""
    out = []
    for poly, normal in cells:
        sides = np.roll(poly, -1, axis=0) - poly
        # the first of the longest sides: a square's, whatever the rounding
        length = np.linalg.norm(sides, axis=1)
        u = sides[np.flatnonzero(length >= (1 - 1e-6) * length.max())[0]]
        u = u / np.linalg.norm(u)
        pieces = [poly]
        for w in (u, np.cross(normal, u)):
            cut_up = []
            for piece in pieces:
                proj = (piece - piece[0]) @ w
                lo, hi = proj.min(), proj.max()
                # a whole number of cells to well above rounding: a face as wide
                # as two cells, as between a box's layers, stays two
                k = int(np.ceil((hi - lo) / h - 1e-3))
                # the cuts rise along w: each crosses only what is left above
                # the last, not every part cut off below it (#106)
                below, rest = [], piece
                for cut in lo + (hi - lo) * np.arange(1, k) / k:
                    parts = _cut_poly(rest, piece[0], w, cut, tol)
                    if len(parts) == 2:
                        rest = parts[0]
                        below.append(parts[1])
                cut_up += [rest, *below[::-1]]
            pieces = cut_up
        out += [(p, normal) for p in pieces]
    return out


def _on_sharp(p, q, segs, tol):
    """True if the segment ``p - q`` lies along one of the sharp segments."""
    if segs.size == 0:
        return False
    a, b = segs[:, 0], segs[:, 1]
    length = np.linalg.norm(b - a, axis=1)
    t = (b - a) / length[:, None]
    for x in (p, q, 0.5 * (p + q)):
        s = np.clip(np.einsum("sd,sd->s", x - a, t), 0, length)
        if np.min(np.linalg.norm(x - (a + s[:, None] * t), axis=1)) > tol:
            return False
    return True


def _offset(P, normal, shift, tol):
    """Polygon ``P`` (k, 3), counter-clockwise round ``normal``, with edge ``i``
    (``P[i] -> P[i + 1]``) moved inward by ``shift[i]``; None where an edge turns
    over, the offset being too deep."""
    k = len(P)
    t = np.roll(P, -1, axis=0) - P
    t /= np.linalg.norm(t, axis=1)[:, None]
    nu = np.cross(normal, t)  # inward
    base = P + shift[:, None] * nu
    out = np.empty_like(P)
    for j in range(k):
        i = j - 1  # edges i and j meet at vertex j
        A = np.array([[t[i] @ t[j], -1.0], [t[i] @ nu[j], 0.0]])
        rhs = np.array([(base[j] - base[i]) @ t[j], (base[j] - base[i]) @ nu[j]])
        if abs(np.linalg.det(A)) < 1e-12:
            out[j] = base[j]
            continue
        along, _ = np.linalg.solve(A, rhs)
        out[j] = base[i] + along * t[i]
    e = np.roll(out, -1, axis=0) - out
    if np.any(np.einsum("kd,kd->k", e, t) < tol):
        return None
    return out


def _width(P, normal):
    """A convex polygon's width, as a strip's: twice its area over its
    perimeter."""
    area = np.cross(P, np.roll(P, -1, axis=0)).sum(0) @ normal / 2
    return 2 * abs(area) / np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum()


def _cut_along(polys, lines, coarse, tol):
    """The convex polygons cut along the contact ``lines``, each ``(point,
    direction, normal)``: a line in the plane of that normal, through the
    polygons lying in it, within ``coarse``."""
    for point, t, normal in lines:
        w = np.cross(normal, t)
        polys = [
            (piece, n)
            for poly, n in polys
            for piece in (
                _cut_poly(poly, point, w, 0.0, tol)
                if abs(n @ normal) > 1 - 1e-9
                and abs((poly.mean(0) - point) @ normal) < coarse
                else [poly]
            )
        ]
    return polys


def graded_triangles(V, F, ell, m, g, angle=20.0, corner=30.0, contacts=()):
    """The closed mesh ``V``, ``F`` cut into triangles (T, 3, 3) graded toward
    its sharp edges, and toward the ``contacts``, lines ``(point, direction,
    normal)`` where a touching part's faces end: ``m`` layers of power ``g``
    over ``ell`` on each side of such an edge or line, the rest in cells of the
    last layer's size."""
    return cut_cells(*graded_cells(V, F, ell, m, g, angle, corner, contacts))


def graded_cells(V, F, ell, m, g, angle=20.0, corner=30.0, contacts=()):
    """The convex cells of :func:`graded_triangles` before they are cut to the
    last layer's size, and what :func:`cut_cells` takes with them: ``(cells,
    size, tol, ell)``. Counted first (:func:`fewest_triangles`), a part that
    cannot fit is not cut: a 10 x 10 x 0.05 mm plate, its cells a fraction of
    its thickness, did not end (#106)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    span = np.max(np.ptp(V[np.unique(F)], axis=0))  # this shell's
    noise = rounding(V)
    tol = max(1e-9 * span, 4 * noise)
    # float32 coordinates, as CAD exports give, are good to about 1e-7 of
    # their size: lines this close are one, and a polygon this thin has
    # collapsed
    coarse = max(1e-6 * span, 16 * noise)
    F = drop_slivers(V, F, coarse)
    segs = sharp_edges(V, F, angle)
    if contacts:
        # a contact line is graded toward like a sharp edge, across the whole
        # part: from the point of it nearest the part's centre
        mid = V[np.unique(F)].mean(0)
        long = np.array(
            [
                (q - 2 * span * t, q + 2 * span * t)
                for q, t in ((p + ((mid - p) @ t) * t, t) for p, t, _ in contacts)
            ]
        )
        segs = np.concatenate([segs, long])
    levels = ell * (np.arange(1, m + 1) / m) ** g
    cells = []
    for P, normal in _cut_along(_merge_coplanar(V, F, tol), contacts, coarse, tol):
        k = len(P)
        sharp = np.array(
            [_on_sharp(P[i], P[(i + 1) % k], segs, coarse) for i in range(k)]
        )
        t = np.roll(P, -1, axis=0) - P
        t /= np.linalg.norm(t, axis=1)[:, None]
        nu = np.cross(normal, t)
        local, inner, depth = [], P, 0.0
        for d in levels:
            # the last layers from two sides meet at mid-face: a layer that
            # would leave less than a quarter step between them, as rounding
            # of the depth does, is not laid, as one that overlaps
            Q = _offset(P, normal, np.where(sharp, d, 0.0), coarse)
            if Q is None or _width(Q, normal) < (d - depth) / 4:
                break
            depth = d
            local += [
                np.array([inner[i], inner[(i + 1) % k], Q[(i + 1) % k], Q[i]])
                for i in range(k)
                if sharp[i]
            ]
            inner = Q
        local.append(inner)
        # at a real corner the cells are cut parallel to the other edge too:
        # a long strip is split only where it reaches the corner
        for j in range(k):
            i = j - 1
            turn = np.degrees(np.arccos(np.clip(t[i] @ t[j], -1, 1)))
            if not (sharp[i] and sharp[j]) or turn <= corner:
                continue
            for d in levels:
                for near, w in ((nu[i], nu[j]), (nu[j], nu[i])):
                    local = [
                        piece
                        for c in local
                        for piece in (
                            _cut_poly(c, P[j], w, d, tol)
                            if (c.mean(0) - P[j]) @ near < (1 - 1e-6) * ell
                            else [c]
                        )
                    ]
        cells += [(c, normal) for c in local if len(c) >= 3]
    return cells, ell * (1 - ((m - 1) / m) ** g), tol, ell


def fewest_triangles(cells, h):
    """A count :func:`cut_cells` makes at least as many triangles as, without
    cutting: one to a cell at least, and the cell's area over that of the
    largest piece it is cut into, ``h`` a side but for the rounding of the
    number of pieces."""
    total = 0.0
    for poly, normal in cells:
        area = abs(np.cross(poly, np.roll(poly, -1, axis=0)).sum(0) @ normal) / 2
        total += max(1.0, area / (1.002 * h) ** 2)
    return int(total)


def cut_cells(cells, h, tol, ell):
    """The cells cut to ``h`` and each fanned into triangles (T, 3, 3)."""
    cells = _split_cells(cells, h, tol)
    triangles = []
    for c, _ in cells:
        if len(c) <= 4:
            triangles += [(c[0], c[q], c[q + 1]) for q in range(1, len(c) - 1)]
        else:  # many corners, as along an arc: fanned from one corner they
            # would be needles, so from the cell's centroid
            fan = np.stack([np.broadcast_to(c[0], (len(c) - 2, 3)), c[1:-1], c[2:]], 1)
            area = np.linalg.norm(
                np.cross(fan[:, 1] - fan[:, 0], fan[:, 2] - fan[:, 0]), axis=1
            )
            centre = area @ fan.mean(1) / area.sum()
            triangles += [(centre, c[q], c[(q + 1) % len(c)]) for q in range(len(c))]
    T = np.array(triangles)
    # and any needle left: its smallest height, twice its area over its longest
    # side, far below the layers' depth
    double_area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    longest = np.linalg.norm(np.roll(T, -1, axis=1) - T, axis=-1).max(axis=1)
    return T[double_area > 1e-6 * ell * longest]
