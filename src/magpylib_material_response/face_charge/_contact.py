"""Faces of two parts that lie in one plane and overlap: the exact part of their
coupling.

Where a face of part b lies in a face plane of part a, b's charge sits in the
plane of a's test panels, and its flux through them jumps there. The one-sided
limit from a's inside gives -1/2 of the integral of the two charges over their
overlap, whatever the panels' shapes: each test panel is clipped against each
source panel in the plane, and the product of their densities integrated over
the overlap. Quadrature leaves those source points out (``test_flux``'s
``tol``): in the plane, away from the overlap, the flux is exactly zero.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree


def polygon_rule(poly):
    """Points and weights exact for quadratics over a convex polygon: the edge
    midpoints of a triangle fan, a third of each triangle's area each."""
    pts, wts = [], []
    p0 = np.array(poly[0])
    for p1, p2 in itertools.pairwise(np.array(poly[1:])):
        area = 0.5 * abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        )
        if area:
            pts += [(p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2]
            wts += [area / 3] * 3
    return np.array(pts).reshape(-1, 2), np.array(wts)


def clip_convex(subject, clip):
    """The polygon ``subject`` (k, 2) clipped to the convex polygon ``clip``
    (Sutherland-Hodgman): a list of points, empty if they do not overlap."""
    clip = np.asarray(clip, float)
    x, y = clip[:, 0], clip[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        clip = clip[::-1]  # counter-clockwise
    out = [np.asarray(p, float) for p in subject]
    for a, b in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        d = b - a
        given, out = out, []
        for k, p in enumerate(given):
            q = given[(k + 1) % len(given)]
            sp = d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])
            sq = d[0] * (q[1] - a[1]) - d[1] * (q[0] - a[0])
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + (q - p) * sp / (sp - sq))
        if not out:
            break
    return out


def coplanar_pairs(Pa, Na, Pb, Nb, tol):
    """Pairs ``(i, j)`` of a's and b's panels (corners ``P``, normals ``N``,
    global) in one plane, within ``tol``, whose boxes overlap: as arrays.

    Candidates by distance, each side searching twice its own panels' reach,
    so that a pair is found by the larger of the two; then the plane, the same
    whichever way the normals point, and the boxes."""
    ca, cb = Pa.mean(1), Pb.mean(1)
    ra = np.linalg.norm(Pa - ca[:, None], axis=-1).max(1)
    rb = np.linalg.norm(Pb - cb[:, None], axis=-1).max(1)
    ta, tb = cKDTree(ca), cKDTree(cb)
    found = [
        (np.repeat(np.arange(len(ca)), [len(x) for x in hits]), _flat(hits))
        for hits in [tb.query_ball_point(ca, 2 * ra + tol)]
    ] + [
        (_flat(hits), np.repeat(np.arange(len(cb)), [len(x) for x in hits]))
        for hits in [ta.query_ball_point(cb, 2 * rb + tol)]
    ]
    i = np.concatenate([f[0] for f in found])
    j = np.concatenate([f[1] for f in found])
    keep = (
        (np.abs(np.einsum("kd,kd->k", Na[i], Nb[j])) > 1 - 1e-9)
        & (np.abs(np.einsum("kd,kd->k", cb[j] - ca[i], Na[i])) < tol)
        & (np.abs(np.einsum("kd,kd->k", ca[i] - cb[j], Nb[j])) < tol)
        & np.all(Pa.min(1)[i] < Pb.max(1)[j] + tol, axis=1)
        & np.all(Pb.min(1)[j] < Pa.max(1)[i] + tol, axis=1)
    )
    pairs = np.unique(np.stack([i[keep], j[keep]], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _flat(hits):
    """``query_ball_point``'s lists of indices, end to end."""
    return np.fromiter(itertools.chain.from_iterable(hits), int)


def contact_block(a, b, tol):
    """The exact part of ``G_ab`` (a.n_dofs x b.n_dofs) where panels of b lie
    in the planes of a's and overlap them, and b's panel normals (global),
    which the caller needs too."""
    Pa, Na = a.panel_polygons()
    Pb, Nb = b.panel_polygons()
    G = np.zeros((a.n_dofs, b.n_dofs))
    for i, j in zip(*coplanar_pairs(Pa, Na, Pb, Nb, tol), strict=True):
        e1 = Pa[i][1] - Pa[i][0]
        e1 = e1 / np.linalg.norm(e1)
        frame = np.stack([e1, np.cross(Na[i], e1)])  # the plane's axes
        poly = clip_convex((Pb[j] - Pa[i][0]) @ frame.T, (Pa[i] - Pa[i][0]) @ frame.T)
        if len(poly) < 3:
            continue
        xy, w = polygon_rule(poly)
        if not w.size:
            continue
        X = Pa[i][0] + xy @ frame
        psi, phi = a.basis_global(i, X), b.basis_global(j, X)  # (points, 3)
        G[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= 0.5 * psi.T @ (phi * w[:, None])
    return G, Nb
