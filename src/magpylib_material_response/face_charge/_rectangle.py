"""Field of a flat rectangle whose surface charge varies linearly across it.

The rectangle ``[p1, p2] x [q1, q2]`` lies in the plane ``w = 0`` of its local
frame ``(p, q, w)``. Its charge density is one of ``1``, ``s`` or ``t``, with
``s = (p - pc) / hp`` and ``t = (q - qc) / hq`` its centred coordinates scaled
to ``[-1, 1]``. The field per unit density is ``(x - y) / |x - y|^3`` integrated
over the rectangle, divided by ``4 pi``: a sum over the four corners of
logarithms, arctangents and square roots, for the integrals of ``X / R^3``,
``Y / R^3``, ``W / R^3`` and their first moments.
"""

from __future__ import annotations

import numpy as np

FOUR_PI = 4 * np.pi


def _ln_y_plus_r(x2w2, y, r):
    """``ln(y + r)``, stable where ``y < 0``: there ``y + r = (x^2 + w^2) / (r - y)``."""
    return np.where(y >= 0, np.log(y + r), np.log(x2w2) - np.log(r - y))


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
    sign_w = np.where(W < 0, -1.0, 1.0)
    abs_w = np.abs(W)
    W2 = W * W
    Sx = Sy = Sw = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for ix, x in enumerate((P - p1, P - p2)):
            for iy, y in enumerate((Q - q1, Q - q2)):
                s = 1.0 if ix == iy else -1.0
                r = np.sqrt(x * x + y * y + W2)
                at = np.arctan2(x * y, abs_w * r)
                Sw = Sw + s * sign_w * at  # integral of W / R^3
                ln_yr = _ln_y_plus_r(x * x + W2, y, r)
                ln_xr = _ln_y_plus_r(y * y + W2, x, r)
                Sx = Sx - s * ln_yr  # integral of X / R^3
                Sy = Sy - s * ln_xr  # integral of Y / R^3
    # s = (P - pc - X) / hp and t = (Q - qc - Y) / hq at the source point
    a, b = (P - pc) / hp, (Q - qc) / hq
    w1 = a * Sw - W * Sx / hp
    w2 = b * Sw - W * Sy / hq
    return np.stack([Sw, w1, w2]) / FOUR_PI
