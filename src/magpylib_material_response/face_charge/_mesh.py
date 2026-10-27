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

import numpy as np


def sharp_edges(V, F, angle):
    """Segments (S, 2, 3) of the mesh edges sharper than ``angle`` degrees."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
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
        if np.degrees(np.arccos(np.clip(n[ts[0]] @ n[ts[1]], -1, 1))) > angle:
            segs.append((V[u], V[v]))
    return np.reshape(np.array(segs), (-1, 2, 3))


def _convex(poly, normal, tol):
    """True if the planar polygon (k, 3), ordered round ``normal``, is convex."""
    k = len(poly)
    return all(
        np.cross(poly[(i + 1) % k] - poly[i], poly[(i + 2) % k] - poly[(i + 1) % k])
        @ normal
        >= -tol
        for i in range(k)
    )


def _merge_coplanar(V, F, tol):
    """The mesh's flat faces as convex polygons: neighbouring coplanar
    triangles merged while the union stays convex. Returns ``[(polygon (k, 3),
    normal)]``."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    polys = [[list(f), n[t]] for t, f in enumerate(F)]  # vertex index loops
    merged = True
    while merged:
        merged = False
        edge_at = {}
        for pi, (loop, _) in enumerate(polys):
            for i, u in enumerate(loop):
                edge_at[u, loop[(i + 1) % len(loop)]] = pi
        for (u, v), pi in edge_at.items():
            pj = edge_at.get((v, u))
            if pj is None or pj == pi:
                continue
            (la, na), (lb, nb) = polys[pi], polys[pj]
            if na @ nb < 1 - 1e-9:
                continue
            # splice: la runs u -> v, lb runs v -> u
            ia, ib = la.index(u), lb.index(v)
            loop = (la[ia + 1 :] + la[: ia + 1])[:-1] + (lb[ib + 1 :] + lb[: ib + 1])[
                :-1
            ]
            pts = V[loop]
            loop = [
                loop[i]
                for i in range(len(loop))
                if np.linalg.norm(
                    np.cross(pts[i] - pts[i - 1], pts[(i + 1) % len(loop)] - pts[i])
                )
                > tol
            ]  # collinear vertices dropped
            if not _convex(V[loop], na, tol):
                continue
            polys[pi] = [loop, na]
            del polys[pj]
            merged = True
            break
    return [(V[loop], normal) for loop, normal in polys]


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
        u = sides[np.argmax(np.linalg.norm(sides, axis=1))]
        u = u / np.linalg.norm(u)
        pieces = [poly]
        for w in (u, np.cross(normal, u)):
            cut_up = []
            for piece in pieces:
                proj = (piece - piece[0]) @ w
                lo, hi = proj.min(), proj.max()
                k = int(np.ceil((hi - lo) / h - 1e-9))
                parts = [piece]
                for cut in lo + (hi - lo) * np.arange(1, k) / k:
                    parts = [
                        q
                        for part in parts
                        for q in _cut_poly(part, piece[0], w, cut, tol)
                    ]
                cut_up += parts
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


def _cut_along(polys, lines, span, tol):
    """The convex polygons cut along the contact ``lines``, each ``(point,
    direction, normal)``: a line in the plane of that normal, through the
    polygons lying in it."""
    for point, t, normal in lines:
        w = np.cross(normal, t)
        polys = [
            (piece, n)
            for poly, n in polys
            for piece in (
                _cut_poly(poly, point, w, 0.0, tol)
                if abs(n @ normal) > 1 - 1e-9
                and abs((poly.mean(0) - point) @ normal) < 1e-6 * span
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
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    span = np.max(np.ptp(V, axis=0))
    tol = 1e-9 * span
    segs = sharp_edges(V, F, angle)
    if contacts:
        # a contact line is graded toward like a sharp edge, across its face
        long = np.array([(p - 2 * span * t, p + 2 * span * t) for p, t, _ in contacts])
        segs = np.concatenate([segs, long])
    levels = ell * (np.arange(1, m + 1) / m) ** g
    cells = []
    for P, normal in _cut_along(_merge_coplanar(V, F, tol), contacts, span, tol):
        k = len(P)
        sharp = np.array(
            [_on_sharp(P[i], P[(i + 1) % k], segs, 1e-6 * span) for i in range(k)]
        )
        t = np.roll(P, -1, axis=0) - P
        t /= np.linalg.norm(t, axis=1)[:, None]
        nu = np.cross(normal, t)
        local, inner = [], P
        for d in levels:
            Q = _offset(P, normal, np.where(sharp, d, 0.0), tol)
            if Q is None:
                break
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
                            if (c.mean(0) - P[j]) @ near < ell
                            else [c]
                        )
                    ]
        cells += [(c, normal) for c in local if len(c) >= 3]
    cells = _split_cells(cells, ell * (1 - ((m - 1) / m) ** g), tol)
    triangles = [(c[0], c[q], c[q + 1]) for c, _ in cells for q in range(1, len(c) - 1)]
    T = np.array(triangles)
    keep = (
        np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) > tol**2
    )
    return T[keep]
