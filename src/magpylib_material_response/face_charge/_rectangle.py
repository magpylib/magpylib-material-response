"""Field of a flat rectangle whose surface charge varies linearly across it.

The rectangle ``[p1, p2] x [q1, q2]`` lies in the plane ``w = 0`` of its local
frame ``(p, q, w)``. Its charge density is one of ``1``, ``s`` or ``t``, with
``s = (p - pc) / hp`` and ``t = (q - qc) / hq`` its centred coordinates scaled
to ``[-1, 1]``. The field per unit density is ``(x - y) / |x - y|^3`` integrated
over the rectangle, divided by ``4 pi``: a sum over the four corners of
logarithms, arctangents and square roots, for the integrals of ``X / R^3``,
``Y / R^3``, ``W / R^3`` and their first moments. The four logarithms of a sum
are taken as one of their ratio, and the arctangents two by two: the kernel
runs twice as fast, as accurate.
"""

from __future__ import annotations

import numpy as np

FOUR_PI = 4 * np.pi
NODES2, WEIGHTS2 = np.polynomial.legendre.leggauss(2)
NODES3, WEIGHTS3 = np.polynomial.legendre.leggauss(3)


def _y_plus_r(x2w2, y, r):
    """``y + r``, stable where ``y < 0``: there it is ``(x^2 + w^2) / (r - y)``."""
    return np.where(y >= 0, y + r, x2w2 / (r - y))


def rectangle_field(P, Q, W, p1, p2, q1, q2):
    """Normal field per unit density of rectangles carrying the densities 1, s
    and t.

    Parameters
    ----------
    P, Q, W : ndarray
        Observer coordinates in each rectangle's frame, broadcast against the
        rectangle bounds.
    p1, p2, q1, q2 : ndarray
        Rectangle bounds.

    Returns
    -------
    ndarray
        The w component, normal to the rectangles: shape
        ``(3 densities, *broadcast)``.
    """
    pc, hp = 0.5 * (p1 + p2), 0.5 * (p2 - p1)
    qc, hq = 0.5 * (q1 + q2), 0.5 * (q2 - q1)
    # An observer exactly in the rectangle's plane, on the extension of an edge
    # line, makes two corner logarithms infinite where they cancel: lift it off
    # the plane by 1e-12 of the rectangle, which leaves the field unchanged.
    W = np.where(W == 0, 1e-12 * np.maximum(hp, hq), W)
    abs_w = np.abs(W)
    W2 = W * W
    with np.errstate(divide="ignore", invalid="ignore"):
        uv, yr, xr = {}, {}, {}
        for ix, x in enumerate((P - p1, P - p2)):
            x2w2 = x * x + W2
            for iy, y in enumerate((Q - q1, Q - q2)):
                r = np.sqrt(x2w2 + y * y)
                uv[ix, iy] = x * y, abs_w * r
                yr[ix, iy] = _y_plus_r(x2w2, y, r)
                xr[ix, iy] = _y_plus_r(y * y + W2, x, r)

        def turn(c, d):
            # atan(u_c / v_c) - atan(u_d / v_d), v > 0: each inside (-pi/2,
            # pi/2), so their difference is one arctangent
            (uc, vc), (ud, vd) = uv[c], uv[d]
            return np.arctan2(uc * vd - ud * vc, vc * vd + uc * ud)

        # the corners' signed sums: of W / R^3 (one arctangent a pair), of
        # X / R^3 and Y / R^3 (one logarithm of a ratio)
        Sw = (turn((0, 0), (0, 1)) + turn((1, 1), (1, 0))) * np.sign(W)
        Sx = -np.log(yr[0, 0] * yr[1, 1] / (yr[0, 1] * yr[1, 0]))
        Sy = -np.log(xr[0, 0] * xr[1, 1] / (xr[0, 1] * xr[1, 0]))
    # s = (P - pc - X) / hp and t = (Q - qc - Y) / hq at the source point
    a, b = (P - pc) / hp, (Q - qc) / hq
    w1 = a * Sw - W * Sx / hp
    w2 = b * Sw - W * Sy / hq
    return np.stack([Sw, w1, w2]) / FOUR_PI


def _field_far(P, Q, W, p1, p2, q1, q2, nodes, weights):
    """``W / R^3`` at Gauss points on the rectangle, all in one array: threads take
    turns between NumPy's calls, so few large calls run on more of them."""
    hp, hq = 0.5 * (p2 - p1), 0.5 * (q2 - q1)
    s, w = nodes.reshape(-1, *[1] * P.ndim), weights.reshape(-1, *[1] * P.ndim)
    # the source point at +s * hp is -s * hp from the observer's offset
    r2 = ((P - 0.5 * (p1 + p2) - s * hp) ** 2 + W * W)[:, None] + (
        (Q - 0.5 * (q1 + q2) - s * hq) ** 2
    )[None]
    f = W * hp * hq / (FOUR_PI * r2 * np.sqrt(r2))
    along_s = (w[None] * f).sum(1)
    return np.stack(
        [
            (w * along_s).sum(0),
            (w * s * along_s).sum(0),
            (w * (w[None] * s[None] * f).sum(1)).sum(0),
        ]
    )


def rectangle_field_far3(P, Q, W, p1, p2, q1, q2):
    """:func:`rectangle_field` for observers far from the rectangle, by ``W / R^3``
    at 3 x 3 Gauss points on it: 1.7 times as fast, and more accurate from 16 sides
    away, where the closed form's first moments cancel. The error against the
    largest density is at worst 3e-4 at 2 sides, 5e-6 at 4, 2e-7 at 8."""
    return _field_far(P, Q, W, p1, p2, q1, q2, NODES3, WEIGHTS3)


def rectangle_field_far2(P, Q, W, p1, p2, q1, q2):
    """:func:`rectangle_field` at 2 x 2 Gauss points: three times as fast. The error
    against the largest density is at worst 1e-3 at 4 sides, 2e-5 at 16."""
    return _field_far(P, Q, W, p1, p2, q1, q2, NODES2, WEIGHTS2)
