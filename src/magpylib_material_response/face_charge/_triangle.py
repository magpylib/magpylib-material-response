"""Field of a flat triangle whose surface charge varies linearly across it.

Triangle ``a, b, c``, observer ``x`` at height ``w = (x - a).n``. Per edge
``p -> q``, with unit direction ``t`` and in-plane normal ``m = t x n``
(Wilton et al. 1984; Graglia 1993):

    f_e    = ln((R+ + l+) / (R- + l-))                         l = (vertex - x).t
    beta_e = atan(P0 l+ / (R0^2 + |w| R+)) - atan(P0 l- / (R0^2 + |w| R-))
    Phi    = int 1/R dS          = sum P0 f - |w| sum beta     P0 = (p - x).m
    E0     = int (x - y)/R^3 dS  = sum m f + n sign(w) sum beta

and for a linear density ``psi(y) = psi(rho) + g.(y - rho)``, ``rho`` the
projection of ``x`` on the plane, by the surface divergence theorem:

    int psi (x - y)/R^3 dS = psi(rho) E0 - sum (g.m) L_e - g Phi,
    L_e = int_e (x - y)/R dl = d_e f_e - t (R+ - R-),

with ``d_e`` the vector from the foot of ``x`` on the edge line to ``x``. The
densities are ``1, u, v``: ``u, v`` the in-plane coordinates along ``e1, e2``
from the centroid, divided by the triangle's scale ``sqrt(area)``.

    D. R. Wilton, S. M. Rao, A. W. Glisson, D. H. Schaubert, O. M. Al-Bundak,
    C. M. Butler (1984) "Potential integrals for uniform and linear source
    distributions on polygonal and polyhedral domains", IEEE Trans. Antennas
    Propag. 32(3), 276-281.

    R. D. Graglia (1993) "On the numerical integration of the linear shape
    functions times the 3-D Green's function or its gradient on a plane
    triangle", IEEE Trans. Antennas Propag. 41(10), 1448-1455.

    A. van Oosterom, J. Strackee (1983) "The solid angle of a plane
    triangle", IEEE Trans. Biomed. Eng. 30(2), 125-126.
"""

from __future__ import annotations

import sys

import numpy as np

from magpylib_material_response._array_api import namespace, quiet

# Written against the Python array API standard, as far as the reference values
# need (#208): frames, charge_field, far_points and charge_field_far run on any
# array library and, at 50 digits, on kernels/reference/mp_namespace.py. The
# solver's own kernels (charge_terms, far_terms, normal_field) are NumPy's.


def frames(V):
    """Per triangle of ``V`` (N, 3, 3): centroid, normal, ``e1``, ``e2``, scale
    ``sqrt(area)`` and area."""
    xp = namespace(V)
    a, b, c = V[:, 0, ...], V[:, 1, ...], V[:, 2, ...]
    cross = xp.linalg.cross(b - a, c - a)
    area = 0.5 * xp.linalg.vector_norm(cross, axis=1)
    n = cross / (2 * area)[:, None]
    e1 = (b - a) / xp.linalg.vector_norm(b - a, axis=1)[:, None]
    e2 = xp.linalg.cross(n, e1)
    return xp.mean(V, axis=1), n, e1, e2, xp.sqrt(area), area


# corners in a line to within this many eps of the product of two sides
NO_AREA = 100 * sys.float_info.epsilon


def has_area(V):
    """Which triangles of ``V`` (N, 3, 3) have an area: corners not in a line
    to within ``NO_AREA`` (``|AB x AC| > NO_AREA |AB| |AC|``). A sheet drops
    the others: they carry no charge, and their frame would be NaN. A
    triangle with a non-finite corner is kept, so that its NaN reaches the
    field instead of the triangle vanishing."""
    xp = namespace(V)
    ab, ac = V[:, 1, ...] - V[:, 0, ...], V[:, 2, ...] - V[:, 0, ...]
    norm = xp.linalg.vector_norm
    with quiet("invalid", "over"):
        flat = norm(xp.linalg.cross(ab, ac), axis=1) <= NO_AREA * norm(
            ab, axis=1
        ) * norm(ac, axis=1)
    return ~flat


_EDGES = ((0, 1), (1, 2), (2, 0))
# the six points of :func:`charge_field_far`, exact to degree 4 (Dunavant 1985):
# barycentric coordinates and weights. Three points, to degree 2, are 9e-4 off at
# 128 sides on a triangle whose own charge sums to zero, the closed form 4e-6
_FAR_A, _FAR_B = 0.816847572980459, 0.091576213509771
_FAR_C, _FAR_D = 0.108103018168070, 0.445948490915965
_FAR_POINTS = np.array(
    [
        [_FAR_A, _FAR_B, _FAR_B],
        [_FAR_B, _FAR_A, _FAR_B],
        [_FAR_B, _FAR_B, _FAR_A],
        [_FAR_C, _FAR_D, _FAR_D],
        [_FAR_D, _FAR_C, _FAR_D],
        [_FAR_D, _FAR_D, _FAR_C],
    ]
)
_FAR_WEIGHTS = np.array([0.109951743655322] * 3 + [0.223381589678011] * 3)
_FAR_WEIGHTS /= _FAR_WEIGHTS.sum()
# the sixteen points of the middle rule, exact to degree 8 (Dunavant 1985):
# the centroid, three orbits (a, b, b) and one (a, b, c), each in this order,
# the weights divided by their sum in double precision
_A1, _B1 = 0.081414823414554, 0.459292588292723
_A2, _B2 = 0.658861384496480, 0.170569307751760
_A3, _B3 = 0.898905543365938, 0.050547228317031
_A4, _B4, _C4 = 0.008394777409958, 0.263112829634638, 0.728492392955404
_MID_POINTS = np.array(
    [[1 / 3, 1 / 3, 1 / 3]]
    + [p for a, b in ((_A1, _B1), (_A2, _B2), (_A3, _B3))
       for p in ([a, b, b], [b, a, b], [b, b, a])]
    + [[_A4, _B4, _C4], [_A4, _C4, _B4], [_B4, _A4, _C4],
       [_B4, _C4, _A4], [_C4, _A4, _B4], [_C4, _B4, _A4]]
)  # fmt: skip
_MID_WEIGHTS = np.array(
    [0.144315607677787] + [0.095091634267285] * 3 + [0.103217370534718] * 3
    + [0.032458497623198] * 3 + [0.027230314174435] * 6
)  # fmt: skip
_MID_WEIGHTS /= _MID_WEIGHTS.sum()


def _r_plus_l(R, l, R02):
    """``R + l``, stable where ``l < 0``: there it is ``R0^2 / (R - l)``."""
    return namespace(R).where(l >= 0, R + l, R02 / (R - l))


def _dot(a, b):
    """Dot products along the last axis, broadcast."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _in_plane(X, V, fr):
    """Each observer in each triangle's plane coordinates, what both
    :func:`charge_field` and :func:`charge_terms` start from: the corners
    ``(0, 0), (ub, 0), (uc, vc)``, the observers' ``u, v, w`` from the first
    corner, their offsets to the corners and distances, and ``sum beta``."""
    xp = namespace(X, V)
    _, n, e1, e2, s, area = fr
    a = V[:, 0, ...]
    ub = _dot(V[:, 1, ...] - a, e1)
    uc, vc = _dot(V[:, 2, ...] - a, e1), _dot(V[:, 2, ...] - a, e2)
    corners = ((0.0, 0.0), (ub, 0.0), (uc, vc))
    axes = xp.permute_dims(xp.stack([e1, e2, n]), (0, 2, 1))
    origin = xp.stack([_dot(a, e1), _dot(a, e2), _dot(a, n)])[:, None, :]
    u, v, w = xp.unstack(xp.matmul(X, axes) - origin, axis=0)
    # lifted off the plane as in `normal_field`
    w = xp.where(w == 0, 1e-12 * s, w)
    w2 = w * w
    du = [pu - u for pu, _ in corners]
    dv = [pv - v for _, pv in corners]
    R = [xp.sqrt(du[k] * du[k] + dv[k] * dv[k] + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(_EDGES, (2, 0, 1), strict=True):
        den = den + (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    omega = 2 * xp.atan2(2 * area * xp.abs(w), den)  # sum beta
    return (ub, uc, vc), corners, u, v, w, w2, du, dv, R, omega


def _edge_terms(corners, du, dv, R, w2):
    """Per edge: its direction ``tu, tv``, and ``P0 f``, ``f`` and ``R+ - R-``
    at each observer. Called within :func:`quiet`."""
    xp = namespace(R[0])
    for i, j in _EDGES:
        (pu, pv), (qu, qv) = corners[i], corners[j]
        L = xp.sqrt((qu - pu) ** 2 + (qv - pv) ** 2)
        tu, tv = (qu - pu) / L, (qv - pv) / L
        l_minus = du[i] * tu + dv[i] * tv
        l_plus = du[j] * tu + dv[j] * tv
        P0 = du[i] * tv - dv[i] * tu
        R02 = P0 * P0 + w2
        f = xp.log(_r_plus_l(R[j], l_plus, R02) / _r_plus_l(R[i], l_minus, R02))
        yield tu, tv, P0 * f, f, R[j] - R[i]


def charge_field(X, V, fr, c, keep=None):
    """``sum_n sum_k c[n, k] int phi_k (x - y) / R^3 dS`` over the triangles:
    the field, times ``4 pi``, of charge ``c0 + c1 u + c2 v`` on each, (M, 3).

    In each triangle's plane coordinates ``u, v, w``, with ``m = t x n = (tv,
    -tu)`` and ``d_e = -P0 m + w n``, every term is a product of scalars, the
    coefficients taken in per triangle; only the sum leaves the plane, by
    three matrix products. ``sign(w) sum beta`` is the solid angle, taken as in
    :func:`normal_field`.

    Parameters
    ----------
    X : ndarray, shape (M, 3)
        Observers.
    V : ndarray, shape (N, 3, 3)
        Triangles.
    fr : tuple
        Their :func:`frames`.
    c : ndarray, shape (N, 3)
        Each triangle's coefficients of ``1, u, v``.
    keep : ndarray of bool, shape (M, N), optional
        The pairs to sum; all where None.
    """
    xp = namespace(X, V, c)
    _, n, e1, e2, s, _ = fr
    (ub, uc, vc), corners, u, v, w, w2, du, dv, R, omega = _in_plane(X, V, fr)
    c0, c1, c2 = c[:, 0], c[:, 1], c[:, 2]
    # E0 = sum m f + n sign(w) sum beta, Phi = sum P0 f - |w| sum beta, and
    # sum (g.m) L_e with g = c1 e1 + c2 e2, L_e = d_e f - t (R+ - R-)
    E0u = E0v = Phi = SLu = SLv = SLw = 0.0
    with quiet("divide", "invalid"):
        for tu, tv, P0f, f, dR in _edge_terms(corners, du, dv, R, w2):
            gm = c1 * tv - c2 * tu
            E0u = E0u + tv * f
            E0v = E0v - tu * f
            Phi = Phi + P0f
            SLu = SLu - gm * (P0f * tv + dR * tu)
            SLv = SLv + gm * (P0f * tu - dR * tv)
            SLw = SLw + gm * f
    Phi = Phi - xp.abs(w) * omega
    # the density at the projection, from the centroid (ub + uc, vc) / 3
    psi = c0 + (c1 * (u - (ub + uc) / 3) + c2 * (v - vc / 3)) / s
    Fu = psi * E0u - (SLu + c1 * Phi) / s
    Fv = psi * E0v - (SLv + c2 * Phi) / s
    Fw = psi * xp.sign(w) * omega - w * SLw / s
    if keep is not None:
        Fu, Fv, Fw = (xp.where(keep, F, 0.0) for F in (Fu, Fv, Fw))
    return Fu @ e1 + Fv @ e2 + Fw @ n


def charge_terms(X, V, fr):
    """:func:`charge_field`'s terms before the sum: the field, times ``4 pi``,
    of each triangle's ``1, u, v`` alone, (M, 3, 3, N) by observer, component,
    coefficient and triangle. So that ``charge_field(X, V, fr, c)`` is
    ``terms.reshape(3 M, 3 N) @ c.T.ravel()``, to rounding: the coefficients
    enter :func:`charge_field` linearly, by ``psi`` and ``g``."""
    _, n, e1, e2, s, _ = fr
    (ub, uc, vc), corners, u, v, w, w2, du, dv, R, omega = _in_plane(X, V, fr)
    # E0 and Phi as in `charge_field`, and sum (g.m) L_e split by g's two
    # coefficients: ``g.m = c1 tv - c2 tu``
    E0u = E0v = Phi = 0.0
    Lu1 = Lu2 = Lv1 = Lv2 = Lw1 = Lw2 = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for tu, tv, P0f, f, dR in _edge_terms(corners, du, dv, R, w2):
            E0u = E0u + tv * f
            E0v = E0v - tu * f
            Phi = Phi + P0f
            Lu, Lv = P0f * tv + dR * tu, P0f * tu - dR * tv
            Lu1, Lu2 = Lu1 - tv * Lu, Lu2 + tu * Lu
            Lv1, Lv2 = Lv1 + tv * Lv, Lv2 - tu * Lv
            Lw1, Lw2 = Lw1 + tv * f, Lw2 - tu * f
    Phi = Phi - np.abs(w) * omega
    solid = np.sign(w) * omega
    psi1, psi2 = (u - (ub + uc) / 3) / s, (v - vc / 3) / s
    out = np.empty((len(X), 3, 3, len(V)))
    for k, (Fu, Fv, Fw) in enumerate(
        (
            (E0u, E0v, solid),
            (
                psi1 * E0u - (Lu1 + Phi) / s,
                psi1 * E0v - Lv1 / s,
                psi1 * solid - w * Lw1 / s,
            ),
            (
                psi2 * E0u - Lu2 / s,
                psi2 * E0v - (Lv2 + Phi) / s,
                psi2 * solid - w * Lw2 / s,
            ),
        )
    ):
        for d in range(3):
            np.multiply(Fu, e1[:, d], out=out[:, d, k])
            out[:, d, k] += Fv * e2[:, d]
            out[:, d, k] += Fw * n[:, d]
    return out


def far_points(V, fr, c, middle=False):
    """The points of :func:`charge_field_far` on each triangle (6, N, 3) and the
    charge each stands for (6, N); with ``middle``, the sixteen points of the
    degree-8 rule (16, N, 3) and (16, N)."""
    xp = namespace(V, c)
    centroid, _, e1, e2, s, area = fr
    rule, rule_weights = (
        (_MID_POINTS, _MID_WEIGHTS) if middle else (_FAR_POINTS, _FAR_WEIGHTS)
    )
    if xp is np:  # the solver's path, unchanged to the last bit
        y, weights = np.einsum("gk,nkd->gnd", rule, V), rule_weights
    else:
        points = xp.asarray(rule.tolist(), dtype=V.dtype)
        y = xp.tensordot(points, V, axes=([1], [1]))
        weights = xp.asarray(rule_weights.tolist(), dtype=V.dtype)
    u, v = _dot(y - centroid, e1), _dot(y - centroid, e2)
    density = c[:, 0] + (c[:, 1] * u + c[:, 2] * v) / s
    return y, weights[:, None] * density * area


def charge_field_far(X, y, charge, keep=None):
    """:func:`charge_field` for observers far from the triangles, at the
    :func:`far_points` on them. Far away the closed form's terms cancel, and summed
    over triangles whose net charge is zero its error grows with distance: 14 % at
    100 times a model's size, against 1e-10 here. From 64 sides, a triangle whose
    own charge sums to zero is 6e-8 off; beyond about 1e102 the field is 0.
    ``keep`` (M, N), where given: the pairs of observer and triangle to sum."""
    xp = namespace(X, y, charge)
    d = [X[:, k, None, None] - y[..., k] for k in range(3)]
    with quiet("over", "invalid"):
        r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        scale = charge / (r2 * xp.sqrt(r2))
        if keep is not None:
            scale = xp.where(keep[:, None, :], scale, 0.0)
        return xp.stack([xp.sum(scale * dk, axis=(1, 2)) for dk in d], axis=1)


def far_terms(X, V, fr):
    """:func:`charge_field_far`'s terms before the sum, as :func:`charge_terms`
    gives :func:`charge_field`'s: (M, 3, 3, N)."""
    centroid, _, e1, e2, s, area = fr
    y = np.einsum("gk,nkd->gnd", _FAR_POINTS, V)
    u, v = _dot(y - centroid, e1), _dot(y - centroid, e2)
    weight = _FAR_WEIGHTS[:, None] * area  # (6, N)
    basis = np.stack([weight, weight * u / s, weight * v / s], axis=1)  # (6, 3, N)
    out = np.zeros((len(X), 3, 3, len(V)))
    with np.errstate(over="ignore", invalid="ignore"):
        for g in range(len(_FAR_WEIGHTS)):
            d = X[:, None, :] - y[g][None]  # (M, N, 3)
            r2 = _dot(d, d)
            scale = 1 / (r2 * np.sqrt(r2))
            for k in range(3):
                weighted = scale * basis[g, k]
                for j in range(3):
                    out[:, j, k] += weighted * d[..., j]
    return out


def normal_field(X, V, fr, pairs=False):
    """``n . int phi (x - y) / R^3 dS``: only the component along each
    triangle's normal, which is all a Galerkin flux needs. With ``n . L_e =
    w f_e`` and ``g`` in the plane it is ``psi(rho) sign(w) sum beta - w sum
    (g.m) f``, ``sign(w) sum beta`` the solid angle, taken as one arctangent
    (van Oosterom and Strackee 1983, see :func:`normal_field_far`).

    ``X`` (M, 3) observers, ``V`` (N, 3, 3) triangles and ``fr`` their
    :func:`frames`: returns (3 densities, M, N). With ``pairs``, observers and
    triangles broadcast against each other instead, ``X`` (..., 3) against
    ``V`` (..., 3, 3) and ``fr`` likewise: ``X`` (K, Q, 3) against ``V`` (K, 1,
    3, 3) gives (3, K, Q), each triangle's own terms once for its Q observers.

    In each triangle's plane coordinates ``u, v`` along ``e1, e2`` from its
    first corner, and ``w``, every term is a product of scalars.
    """
    _, n, e1, e2, s, area = fr
    if not pairs:
        X = X[:, None]
        V, n, e1, e2, s, area = (a[None] for a in (V, n, e1, e2, s, area))
    a = V[..., 0, :]
    # the corners: (0, 0), (ub, 0), (uc, vc)
    ub = _dot(V[..., 1, :] - a, e1)
    uc, vc = _dot(V[..., 2, :] - a, e1), _dot(V[..., 2, :] - a, e2)
    corners = ((0.0, 0.0), (ub, 0.0), (uc, vc))
    d = X - a
    u, v, w = _dot(d, e1), _dot(d, e2), _dot(d, n)
    # lifted off the plane as in `field`
    w = np.where(w == 0, 1e-12 * s, w)
    w2 = w * w
    du = [pu - u for pu, _ in corners]  # from the observer to each corner
    dv = [pv - v for _, pv in corners]
    R = [np.sqrt(du[k] * du[k] + dv[k] * dv[k] + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(_EDGES, (2, 0, 1), strict=True):
        den = den + (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    En0 = 2 * np.sign(w) * np.arctan2(2 * area * np.abs(w), den)
    sum_mu = sum_mv = 0.0  # sum (e1.m) f and sum (e2.m) f, m = t x n = (tv, -tu)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in _EDGES:
            (pu, pv), (qu, qv) = corners[i], corners[j]
            L = np.sqrt((qu - pu) ** 2 + (qv - pv) ** 2)
            tu, tv = (qu - pu) / L, (qv - pv) / L
            l_minus = du[i] * tu + dv[i] * tv
            l_plus = du[j] * tu + dv[j] * tv
            P0 = du[i] * tv - dv[i] * tu
            R02 = P0 * P0 + w2
            f = np.log(_r_plus_l(R[j], l_plus, R02) / _r_plus_l(R[i], l_minus, R02))
            sum_mu = sum_mu + tv * f
            sum_mv = sum_mv - tu * f
    # the densities at the projection, from the centroid (ub + uc, vc) / 3
    psi_u, psi_v = (u - (ub + uc) / 3) / s, (v - vc / 3) / s
    return np.stack([En0, psi_u * En0 - w * sum_mu / s, psi_v * En0 - w * sum_mv / s])


def normal_field_far(X, V, fr):
    """:func:`normal_field` for all observers against all triangles, (3, M, N),
    with every dot product one matrix product: ``|x - p|^2 = |x|^2 - 2 x.p +
    |p|^2``. For pairs farther apart than a triangle's size: the cancellation in
    ``|x - p|^2`` grows as ``(|x| / distance)^2`` times the rounding, ``|x|``
    from the frame's origin: the callers centre it on the part.

    Two exact forms halve the work. ``sign(w) sum beta`` is the solid angle,
    one arctangent from the corners (van Oosterom and Strackee 1983), and
    accurate far away where the edge terms cancel. ``f_e = ln((S + L) / (S -
    L))``, ``S = R+ + R-``, ``L`` the edge's length, stable off the edge itself;
    near it, the few pairs closer than ``S - L < L / 100`` are taken by the
    stable branch.
    """
    cen, n, e1, e2, s, area = fr
    # every dot product with an observer: -2 x.p for the corners, x.n, and the
    # densities' x.e / s
    XD = np.matmul(
        X,
        np.stack(
            [
                -2 * V[:, 0],
                -2 * V[:, 1],
                -2 * V[:, 2],
                n,
                e1 / s[:, None],
                e2 / s[:, None],
            ]
        ).transpose(0, 2, 1),
    )
    xx = np.sum(X * X, axis=1)[:, None]
    R2 = [np.maximum(XD[k] + np.sum(V[:, k] ** 2, axis=1) + xx, 0.0) for k in range(3)]
    R = [np.sqrt(r2) for r2 in R2]
    w = XD[3] - np.sum(V[:, 0] * n, axis=1)
    w = np.where(w == 0, 1e-12 * s, w)
    edge = [V[:, j] - V[:, i] for i, j in _EDGES]
    L2 = [np.sum(d * d, axis=1) for d in edge]
    # the corners' r_i.r_j = (R_i^2 + R_j^2 - L_ij^2) / 2, each times the third R
    den = R[0] * R[1] * R[2]
    for (i, j), l2, k in zip(_EDGES, L2, (2, 0, 1), strict=True):
        den = den + 0.5 * (R2[i] + R2[j] - l2) * R[k]
    out = np.empty((3, len(X), len(V)))
    out[0] = 2 * np.sign(w) * np.arctan2(2 * area * np.abs(w), den)
    sum_mf = [0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for (i, j), d, l2 in zip(_EDGES, edge, L2, strict=True):
            L = np.sqrt(l2)
            t = d / L[:, None]
            S = R[i] + R[j]
            gap = S - L
            f = np.log((S + L) / gap)
            a, b = np.nonzero(gap < 1e-2 * L)
            if len(a):
                x, p, q, tb = X[a], V[b, i], V[b, j], t[b]
                l_minus = np.sum((p - x) * tb, axis=1)
                l_plus = np.sum((q - x) * tb, axis=1)
                foot = (x - p) + l_minus[:, None] * tb  # from the edge line
                # at least w², lifted off the plane: finite on the edge line
                R02 = np.maximum(np.sum(foot * foot, axis=1), w[a, b] ** 2)
                R_minus = np.linalg.norm(x - p, axis=1)
                R_plus = np.linalg.norm(x - q, axis=1)
                f[a, b] = np.log(
                    _r_plus_l(R_plus, l_plus, R02) / _r_plus_l(R_minus, l_minus, R02)
                )
            m = np.cross(t, n)
            for k, e in enumerate((e1, e2)):
                sum_mf[k] = sum_mf[k] + (np.sum(e * m, axis=1) / s) * f
    for k, e in enumerate((e1, e2)):
        psi_rho = XD[4 + k] - np.sum(cen * e, axis=1) / s  # density at the projection
        out[1 + k] = psi_rho * out[0] - w * sum_mf[k]
    return out


# The reference kinds of kernels/reference/face_charge.json (#208): H in A/m of
# charge in A/m, as the array API functions of a source's own parameters.


def _check_H(field):
    if field != "H":
        msg = f"field must be 'H'; instead received {field!r}."
        raise ValueError(msg)


def beyond_far_from(observers, center, far_from):
    """Which observers (M, 3) are at or beyond ``far_from`` from ``center``,
    where a sheet takes the far rule: the one predicate of the sheet and of the
    reference."""
    return (
        namespace(observers, center).linalg.vector_norm(observers - center, axis=1)
        >= far_from
    )


# an observer this many of a sheet's longest triangle sides past all of it takes
# `charge_field_far`. Each observer chooses for the whole sheet, so a near one costs
# nothing: choosing per pair made fields 2 mm above six bars 1.3 times as slow
FAR_SIDES = 64.0


def sheet_parameters(vertices, charge_densities, fr=None):
    """A sheet of triangles ``vertices`` (N, 3, 3) with the charge density
    ``charge_densities`` (N, 3) at their corners, in A/m, as
    :func:`charge_sheet_field` takes it: the coefficients (N, 3) of ``1, u, v``
    in each triangle's frame (:func:`frames`, or ``fr``), the center, the mean
    of the centroids (3,), and ``far_from``, the farthest corner from it plus
    ``FAR_SIDES`` longest edges (0-d; infinite for no triangle)."""
    xp = namespace(vertices, charge_densities)
    cen, _, e1, e2, s, _ = frames(vertices) if fr is None else fr
    d = vertices - cen[:, None, :]
    u = xp.sum(d * e1[:, None, :], axis=2) / s[:, None]
    v = xp.sum(d * e2[:, None, :], axis=2) / s[:, None]
    q = charge_densities
    # the plane through the corners' densities, from corner 0
    du1, du2 = u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]
    dv1, dv2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    dq1, dq2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    det = du1 * dv2 - du2 * dv1
    c1 = (dq1 * dv2 - dq2 * dv1) / det
    c2 = (du1 * dq2 - du2 * dq1) / det
    c = xp.stack([q[:, 0] - c1 * u[:, 0] - c2 * v[:, 0], c1, c2], axis=1)
    if vertices.shape[0] == 0:
        return c, xp.zeros(3, dtype=vertices.dtype), xp.asarray(float("inf"))
    center = xp.mean(cen, axis=0)
    a, b, cc = vertices[:, 0, ...], vertices[:, 1, ...], vertices[:, 2, ...]
    side = xp.max(xp.linalg.vector_norm(xp.stack([b - a, cc - b, a - cc]), axis=2))
    reach = xp.max(xp.linalg.vector_norm(vertices - center, axis=2))
    return c, center, reach + FAR_SIDES * side


def triangle_linear_charge_field(field, observers, vertices, charge):
    """H in A/m (``field`` 'H') at ``observers`` (M, 3) of the charge ``charge``
    (N, 3), coefficients of ``1, u, v`` in A/m, on the triangles ``vertices``
    (N, 3, 3), summed, by the closed form."""
    _check_H(field)
    xp = namespace(observers, vertices, charge)
    return charge_field(observers, vertices, frames(vertices), charge) / (4 * xp.pi)


# The far rule for one pair of observer and triangle, by a tolerance: from k
# longest edges of the triangle from its centroid, every pair taken by it is
# off by at most ``bound`` times the field its charge would make all of one
# sign (area times the largest corner density, over distance squared). The
# bounds are 6.5e-3 / k^4, rounded up: twice the worst case an adversarial
# search found (a needle with a linearly falling charge, seen along its axis).
# Rows: (bound, k^2); a tolerance takes the first row it reaches, below the
# last none (the closed form for every near pair).
PAIR_RULE = (
    (1.3e-3, 2.25),
    (4.1e-4, 4.0),
    (8.1e-5, 9.0),
    (2.6e-5, 16.0),
    (5.1e-6, 36.0),
    (1.6e-6, 64.0),
    (3.2e-7, 144.0),
    (1.0e-7, 256.0),
    (2.0e-8, 576.0),
    (6.2e-9, 1024.0),
    (1.3e-9, 2304.0),
    (3.9e-10, 4096.0),
)


# The middle rule's rows, as PAIR_RULE's: from k longest edges its pairs are
# off by at most ``bound`` times the one-sign field; the bounds are
# 1.7e-4 / k^8, rounded up: twice the worst case an adversarial search found.
PAIR_RULE_MIDDLE = (
    (6.7e-6, 2.25), (6.7e-7, 4.0), (2.6e-8, 9.0),
    (2.6e-9, 16.0), (1.1e-10, 36.0), (1.1e-11, 64.0),
)  # fmt: skip


def pair_k2(tolerance, table=PAIR_RULE):
    """The squared distance, in squared longest edges, from which a pair takes
    the rule of ``table`` at ``tolerance``; None for never."""
    return next((k2 for bound, k2 in table if bound <= tolerance), None)


def far_pairs(X, V, fr, k2):
    """Which pairs of observers ``X`` (M, 3) and triangles take the far rule:
    ``|X - centroid|^2 >= k2 L^2``, ``L`` the longest edge, every square summed
    x, y, z in that order (M, N)."""
    xp = namespace(X, V)
    cen = fr[0]
    dx, dy, dz = (X[:, k, None] - cen[None, :, k] for k in range(3))
    d2 = dx * dx + dy * dy + dz * dz
    a, b, c = V[:, 0, ...], V[:, 1, ...], V[:, 2, ...]
    edges = []
    for p, q in ((a, b), (b, c), (c, a)):
        e = q - p
        edges.append(e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1] + e[:, 2] * e[:, 2])
    longest2 = xp.maximum(xp.maximum(edges[0], edges[1]), edges[2])
    return d2 >= k2 * longest2[None, :]


def charge_sheet_field(
    field, observers, vertices, charge, center, far_from, tolerance=0.0
):
    """:func:`triangle_linear_charge_field`, with the far rule's six point charges
    per triangle for the observers at or beyond ``far_from`` from ``center``
    (``far_from`` 0: everywhere), and, at a ``tolerance`` above 0, for every
    pair of observer and triangle :func:`far_pairs` takes: the six-point rule
    from its ``PAIR_RULE`` distance, else the sixteen-point rule from its
    ``PAIR_RULE_MIDDLE`` distance, else the closed form. ``center`` (3,) and
    ``far_from`` (a 0-d array) are arrays of the observers' namespace;
    ``tolerance`` a number or a 0-d array."""
    _check_H(field)
    xp = namespace(observers, vertices, charge, center, far_from)
    fr = frames(vertices)
    is_far = beyond_far_from(observers, center, far_from)
    rule = far_points(vertices, fr, charge)
    k2_far, k2_middle = (
        pair_k2(float(tolerance), t) for t in (PAIR_RULE, PAIR_RULE_MIDDLE)
    )
    if k2_far is None and k2_middle is None:
        near = charge_field(observers, vertices, fr, charge)
        far = charge_field_far(observers, *rule)
        return xp.where(is_far[:, None], far, near) / (4 * xp.pi)
    never = float("inf")  # a distance no pair reaches
    far = far_pairs(observers, vertices, fr, never if k2_far is None else k2_far)
    far = far | is_far[:, None]
    middle = far_pairs(
        observers, vertices, fr, never if k2_middle is None else k2_middle
    )
    middle = middle & ~far
    H = charge_field(observers, vertices, fr, charge, keep=~(far | middle))
    H = H + charge_field_far(observers, *rule, keep=far)
    H = H + charge_field_far(
        observers, *far_points(vertices, fr, charge, True), keep=middle
    )
    return H / (4 * xp.pi)


def charge_sheet_corners_field(
    field, observers, vertices, charge_densities, tolerance=0.0
):
    """:func:`charge_sheet_field` of a sheet as ``ChargeSheet`` holds it: the
    triangles ``vertices`` (N, 3, 3) and the charge density at their corners
    ``charge_densities`` (N, 3) in A/m, its coefficients, center and
    ``far_from`` by :func:`sheet_parameters`; a triangle without area
    (:func:`has_area`) dropped first."""
    keep = has_area(vertices)
    vertices, charge_densities = vertices[keep], charge_densities[keep]
    c, center, far_from = sheet_parameters(vertices, charge_densities)
    return charge_sheet_field(
        field, observers, vertices, c, center, far_from, tolerance
    )
