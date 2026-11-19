"""Write ``kernels.json``: the field kernels' inputs and expected outputs, from
the numpy reference, for every other implementation to reproduce (issue #208).

Each case states its own tolerance, measured: the numpy result's error
against the same formulas at 50 digits (mpmath), times a margin. A port
passes a case when every output row ``r`` (a field vector) satisfies

    |got_r - expected_r| <= rtol |expected_r| + atol        (Euclidean norms)

and, where ``expected_r`` is not finite, has the same non-finite components.
Floats are JSON numbers written to round-trip exactly; ``nan``, ``inf`` and
``-inf`` are strings.

Run from the repository root, with mpmath and magpylib's main branch
importable (the cuboid is #1066's, not yet released)::

    MAGPYLIB_COMMIT=<sha> PYTHONPATH=src:<magpylib main>/src:<mpmath> \\
        python kernels/reference/make_reference.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import magpylib as magpy
import mpmath as mp
import numpy as np
from magpylib._src.fields.field_BH_cuboid import _BHJM_magnet_cuboid

import magpylib_material_response as mmr
from magpylib_material_response import solve_face_charge
from magpylib_material_response.face_charge import ChargeSheet
from magpylib_material_response.face_charge import _triangle as T

mp.mp.dps = 50
OUT = Path(__file__).with_name("kernels.json")
RTOL = 1e-12  # the floor of every case's tolerance
MARGIN = 16  # tolerance over the numpy reference's own measured error
MU0 = magpy.mu_0


# ── 50-digit versions of the formulas ────────────────────────────────────


def _mp_vec(a):
    return [mp.mpf(float(x)) for x in a]


def _mp_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mp_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _mp_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _mp_frame(tri):
    a, b, c = (_mp_vec(p) for p in tri)
    cross = _mp_cross(_mp_sub(b, a), _mp_sub(c, a))
    norm = mp.sqrt(_mp_dot(cross, cross))
    area = norm / 2
    n = [x / norm for x in cross]
    ab = _mp_sub(b, a)
    lab = mp.sqrt(_mp_dot(ab, ab))
    e1 = [x / lab for x in ab]
    e2 = _mp_cross(n, e1)
    cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
    return (a, b, c), cen, n, e1, e2, mp.sqrt(area), area


def _mp_r_plus_l(R, l, R02):
    return R + l if l >= 0 else R02 / (R - l)


def _mp_charge_pair(x, frame, coef, w_numpy):
    """One triangle's ``charge_field`` at one observer, at 50 digits. The lift
    off the plane is taken where numpy took it (``w`` exactly 0 there)."""
    (a, b, c), _, n, e1, e2, s, area = frame
    c0, c1, c2 = (mp.mpf(float(q)) for q in coef)
    ub = _mp_dot(_mp_sub(b, a), e1)
    uc, vc = _mp_dot(_mp_sub(c, a), e1), _mp_dot(_mp_sub(c, a), e2)
    corners = ((0, 0), (ub, 0), (uc, vc))
    d = _mp_sub(x, a)
    u, v = _mp_dot(d, e1), _mp_dot(d, e2)
    w = 1e-12 * s if w_numpy == 0 else _mp_dot(d, n)
    w2 = w * w
    du = [pu - u for pu, _ in corners]
    dv = [pv - v for _, pv in corners]
    R = [mp.sqrt(du[k] ** 2 + dv[k] ** 2 + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(T._EDGES, (2, 0, 1), strict=True):
        den += (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    omega = 2 * mp.atan2(2 * area * abs(w), den)
    E0u = E0v = Phi = SLu = SLv = SLw = mp.mpf(0)
    for i, j in T._EDGES:
        (pu, pv), (qu, qv) = corners[i], corners[j]
        L = mp.sqrt((qu - pu) ** 2 + (qv - pv) ** 2)
        tu, tv = (qu - pu) / L, (qv - pv) / L
        lm = du[i] * tu + dv[i] * tv
        lp = du[j] * tu + dv[j] * tv
        P0 = du[i] * tv - dv[i] * tu
        R02 = P0 * P0 + w2
        f = mp.log(_mp_r_plus_l(R[j], lp, R02) / _mp_r_plus_l(R[i], lm, R02))
        dR = R[j] - R[i]
        gm = c1 * tv - c2 * tu
        E0u += tv * f
        E0v -= tu * f
        Phi += P0 * f
        SLu -= gm * (P0 * f * tv + dR * tu)
        SLv += gm * (P0 * f * tu - dR * tv)
        SLw += gm * f
    Phi -= abs(w) * omega
    psi = c0 + (c1 * (u - (ub + uc) / 3) + c2 * (v - vc / 3)) / s
    Fu = psi * E0u - (SLu + c1 * Phi) / s
    Fv = psi * E0v - (SLv + c2 * Phi) / s
    Fw = psi * mp.sign(w) * omega - w * SLw / s
    return [Fu * e1[k] + Fv * e2[k] + Fw * n[k] for k in range(3)]


def _numpy_w(X, V):
    """numpy's ``w`` per observer and triangle, (M, N), to find where it lifts."""
    fr = T.frames(V)
    n, e1, e2 = fr[1], fr[2], fr[3]
    a = V[:, 0]
    return (
        np.matmul(X, np.stack([e1, e2, n]).transpose(0, 2, 1))
        - np.stack([T._dot(a, e1), T._dot(a, e2), T._dot(a, n)])[:, None]
    )[2]


def mp_charge_field(X, V, c):
    frames = [_mp_frame(tri) for tri in V]
    w = _numpy_w(X, V)
    out = []
    for m, x in enumerate(X):
        xm = _mp_vec(x)
        acc = [mp.mpf(0)] * 3
        for t, (fr, q) in enumerate(zip(frames, c, strict=True)):
            F = _mp_charge_pair(xm, fr, q, w[m, t])
            acc = [acc[k] + F[k] for k in range(3)]
        out.append(acc)
    return out


def mp_charge_field_far(X, y, charge):
    pts = [[_mp_vec(p) for p in row] for row in y]
    q = [[mp.mpf(float(v)) for v in row] for row in charge]
    out = []
    for x in X:
        xm = _mp_vec(x)
        acc = [mp.mpf(0)] * 3
        for g in range(len(pts)):
            for t in range(len(pts[g])):
                d = _mp_sub(xm, pts[g][t])
                r2 = _mp_dot(d, d)
                sc = q[g][t] / (r2 * mp.sqrt(r2))
                acc = [acc[k] + sc * d[k] for k in range(3)]
        out.append(acc)
    return out


def _mp_cuboid_B(x, dim, pol):
    """magpylib's ``magnet_cuboid_Bfield`` at one observer, at 50 digits, with
    its mapping to the bottom quadrant and its stable sums."""
    px, py, pz = (mp.mpf(float(p)) for p in pol)
    a, b, c = (mp.mpf(float(d)) / 2 for d in dim)
    x, y, z = _mp_vec(x)
    qs = [[1] * 3 for _ in range(3)]

    def flip(f):
        for i in range(3):
            for j in range(3):
                qs[i][j] *= f[i][j]

    if x < 0:
        x = -x
        flip([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
    if y > 0:
        y = -y
        flip([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    if z > 0:
        z = -z
        flip([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    xma, xpa, ymb, ypb, zmc, zpc = x - a, x + a, y - b, y + b, z - c, z + c

    def r(p, q, s):
        return mp.sqrt(p * p + q * q + s * s)

    mmm, pmp, pmm, mmp = (
        r(xma, ymb, zmc),
        r(xpa, ymb, zpc),
        r(xpa, ymb, zmc),
        r(xma, ymb, zpc),
    )
    mpm, ppp, ppm, mpp = (
        r(xma, ypb, zmc),
        r(xpa, ypb, zpc),
        r(xpa, ypb, zmc),
        r(xma, ypb, zpc),
    )

    def sp(q, rad, tr2):
        return tr2 / (rad - q) if q < 0 else rad + q

    ff2x = mp.log(
        sp(xma, mmm, ymb**2 + zmc**2)
        * (xpa + ppm)
        * (xpa + pmp)
        * sp(xma, mpp, ypb**2 + zpc**2)
    ) - mp.log(
        (xpa + pmm)
        * sp(xma, mpm, ypb**2 + zmc**2)
        * sp(xma, mmp, ymb**2 + zpc**2)
        * (xpa + ppp)
    )
    ff2y = mp.log(
        (-ymb + mmm)
        * sp(-ypb, ppm, xpa**2 + zmc**2)
        * (-ymb + pmp)
        * sp(-ypb, mpp, xma**2 + zpc**2)
    ) - mp.log(
        (-ymb + pmm)
        * sp(-ypb, mpm, xma**2 + zmc**2)
        * (-ymb + mmp)
        * sp(-ypb, ppp, xpa**2 + zpc**2)
    )
    ff2z = mp.log(
        (-zmc + mmm)
        * (-zmc + ppm)
        * sp(-zpc, pmp, xpa**2 + ymb**2)
        * sp(-zpc, mpp, xma**2 + ypb**2)
    ) - mp.log(
        (-zmc + pmm)
        * (-zmc + mpm)
        * sp(-zpc, mmp, xma**2 + ymb**2)
        * sp(-zpc, ppp, xpa**2 + ypb**2)
    )
    at = mp.atan2
    ff1x = (
        at(ymb * zmc, xma * mmm) - at(ymb * zmc, xpa * pmm) - at(ypb * zmc, xma * mpm)
        + at(ypb * zmc, xpa * ppm) - at(ymb * zpc, xma * mmp) + at(ymb * zpc, xpa * pmp)
        + at(ypb * zpc, xma * mpp) - at(ypb * zpc, xpa * ppp)
    )  # fmt: skip
    ff1y = (
        at(xma * zmc, ymb * mmm) - at(xpa * zmc, ymb * pmm) - at(xma * zmc, ypb * mpm)
        + at(xpa * zmc, ypb * ppm) - at(xma * zpc, ymb * mmp) + at(xpa * zpc, ymb * pmp)
        + at(xma * zpc, ypb * mpp) - at(xpa * zpc, ypb * ppp)
    )  # fmt: skip
    ff1z = (
        at(xma * ymb, zmc * mmm) - at(xpa * ymb, zmc * pmm) - at(xma * ypb, zmc * mpm)
        + at(xpa * ypb, zmc * ppm) - at(xma * ymb, zpc * mmp) + at(xpa * ymb, zpc * pmp)
        + at(xma * ypb, zpc * mpp) - at(xpa * ypb, zpc * ppp)
    )  # fmt: skip
    B = [
        px * ff1x * qs[0][0] + py * ff2z * qs[1][0] + pz * ff2y * qs[2][0],
        px * ff2z * qs[0][1] + py * ff1y * qs[1][1] - pz * ff2x * qs[2][1],
        px * ff2y * qs[0][2] - py * ff2x * qs[1][2] + pz * ff1z * qs[2][2],
    ]
    return [v / (4 * mp.pi) for v in B]


def mp_cuboid(field, X, dims, pols):
    """``_BHJM_magnet_cuboid`` at 50 digits: the special cases (edges, zero
    dimension or polarization) as numpy decides them, the rest exact."""
    zero = _BHJM_magnet_cuboid("B", X, dims, pols * 0 + 1) == 0  # numpy's special cases
    half = np.abs(dims) / 2
    face = np.any(np.abs(np.abs(X) - half) < 1e-15 * half, axis=1)
    out = []
    for x, d, p, z, f in zip(X, dims, pols, zero.all(1), face, strict=True):
        if f and not z:
            out.append(
                None
            )  # on a face B jumps by J: the side is numpy's signed zeros'
            continue
        B = [mp.mpf(0)] * 3 if z or not np.any(p) else _mp_cuboid_B(x, d, p)
        if field == "H":
            inside = np.all(np.abs(x) - np.abs(d) / 2 < 1e-15 * np.abs(d) / 2)
            if inside:
                B = [B[k] - mp.mpf(float(p[k])) for k in range(3)]
            B = [v / mp.mpf(MU0) for v in B]
        out.append(B)
    return out


# ── cases ────────────────────────────────────────────────────────────────


def _row_errors(expected, exact):
    """Per row: numpy's error against the 50-digit value, and that value's norm."""
    err, size = [], []
    for e, x in zip(expected, exact, strict=True):
        if x is None or not np.all(np.isfinite(e)):
            err.append(0.0)
            size.append(float(np.linalg.norm(e)) if np.all(np.isfinite(e)) else 0.0)
            continue
        d = [mp.mpf(float(e[k])) - x[k] for k in range(3)]
        err.append(float(mp.sqrt(_mp_dot(d, d))))
        size.append(float(mp.sqrt(_mp_dot(x, x))))
    return np.array(err), np.array(size)


def _tolerance(expected, exact):
    """Per row ``rtol`` and per case ``atol`` that numpy passes with ``MARGIN``
    to spare. ``atol`` is relative to the case's largest field, as a field
    near a part is only as accurate as the part's field; ``rtol`` covers the
    rest, row by row, so that one ill-conditioned row loosens only itself.
    Rows without an exact value (``None``) take the floor."""
    err, size = _row_errors(expected, exact)
    scale = size.max() if size.size and size.max() > 0 else 1.0
    atol = RTOL * scale
    ok = size > 0
    need = np.where(ok, (MARGIN * err - atol) / np.where(ok, size, 1), 0.0)
    rtol = np.maximum(RTOL, need)
    return [float(f"{r:.2g}") if r > RTOL else RTOL for r in rtol], atol


def _encode(a):
    a = np.asarray(a, float)
    if np.all(np.isfinite(a)):
        return a.tolist()
    flat = [x if np.isfinite(x) else ("nan" if np.isnan(x) else ("inf" if x > 0 else "-inf"))
            for x in a.ravel().tolist()]  # fmt: skip
    return np.array(flat, object).reshape(a.shape).tolist()


CASES = []


def case(name, kernel, why, inputs, expected, exact=None, rtol=None, atol=None):
    expected = np.asarray(expected, float)
    entry = {"name": name, "kernel": kernel, "why": why}
    if exact is not None:
        rtol, atol = _tolerance(expected, exact)
    else:
        rtol = [rtol] * len(expected)
    entry |= {"rtol": rtol, "atol": atol}
    entry["inputs"] = {k: _encode(v) for k, v in inputs.items()}
    entry["expected"] = _encode(expected)
    CASES.append(entry)
    r = np.array(rtol)
    print(
        f"{name:26s} {kernel:17s} rows {len(r):4d}  rtol median {np.median(r):.0e} "
        f"max {r.max():.0e}, {np.sum(r > RTOL):3d} rows above 1e-12"
    )


def _charge_case(name, why, X, V, c):
    X, V, c = (np.asarray(a, float) for a in (X, V, c))
    F = T.charge_field(X, V, T.frames(V), c)
    case(
        name, "charge_field", why, {"X": X, "V": V, "c": c}, F, mp_charge_field(X, V, c)
    )


def _one_triangle():
    """A plain triangle in the plane z = 0, 1 mm across."""
    return np.array([[[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.3e-3, 0.8e-3, 0.0]]])


def charge_cases(rng):
    for n_tri, n_obs in ((1, 200), (40, 250)):
        c0 = rng.uniform(-0.01, 0.01, (n_tri, 3))
        V = c0[:, None, :] + rng.uniform(-0.001, 0.001, (n_tri, 3, 3))
        q = rng.normal(size=(n_tri, 3))
        X = rng.uniform(-0.02, 0.02, (n_obs, 3))
        _charge_case(
            f"charge_random_{n_tri}x{n_obs}",
            "random triangles and observers, as the issue's baseline",
            X, V, q,
        )  # fmt: skip
    V = _one_triangle()
    q = np.array([[0.7, -0.4, 1.3]])
    a, b, c = V[0]
    s = 1e-3
    inside = [(0.4e-3, 0.3e-3, 0.0), (0.2e-3, 0.1e-3, 0.0), a / 3 + b / 3 + c / 3]
    outside = [(2e-3, 1e-3, 0.0), (-1e-3, -0.5e-3, 0.0), (0.5e-3, -0.2e-3, 0.0)]
    edges = [a + t * (b - a) for t in (0.5, 0.123)] + [
        b + 0.37 * (c - b),
        c + 0.81 * (a - c),
    ]
    lines = [a + t * (b - a) for t in (-0.6, 1.7)] + [c + 1.4 * (a - c)]
    _charge_case(
        "charge_in_plane",
        "observers in the triangle's plane (w exactly 0, lifted): inside, outside, "
        "on edges, on edge lines beyond the corners, on corners",
        np.array(inside + outside + edges + lines + [a, b, c]), V, q,
    )  # fmt: skip
    base = np.array(inside[:1] + outside[:1] + edges[:1] + [b])
    above = [
        p + np.array((0.0, 0.0, h))
        for h in (1e-12 * s, 1e-9 * s, -1e-6 * s, 1e-3 * s)
        for p in base
    ]
    _charge_case(
        "charge_just_off_plane",
        "observers a hair above or below the plane over the inside, outside, an edge, a corner",
        np.array(above), V, q,
    )  # fmt: skip
    near_edge = [
        edges[0] + (0, sign * 10.0**-k, 0) for k in (7, 10, 13) for sign in (-1, 1)
    ] + [b + np.full(3, 10.0**-k) for k in (7, 10, 13)]
    _charge_case(
        "charge_near_edge_corner",
        "observers in the plane within 1e-4 to 1e-10 sides of an edge, and near a corner",
        np.array(near_edge), V, q,
    )  # fmt: skip
    far = [
        d * np.array(u)
        for d in (7e-2, 2e-1)
        for u in ((1, 0, 0), (0.3, 0.4, 0.866), (0, 0, -1))
    ]
    _charge_case(
        "charge_near_kernel_far",
        "the closed form 70 and 200 sides away, just past where a sheet switches to the far rule",
        np.array(far), V, q,
    )  # fmt: skip
    sliver = np.array([[[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.5e-3, 1e-9, 1e-10]]])
    _charge_case(
        "charge_sliver",
        "a triangle 1e6 times longer than wide, observers around it",
        rng.uniform(-2e-3, 2e-3, (20, 3)), sliver, q,
    )  # fmt: skip
    offset = V + np.array([3.0, -2.0, 1.5])
    _charge_case(
        "charge_far_from_origin",
        "a 1 mm triangle 4 m from the origin, observers near it: rounding in the frame",
        offset[0, 0] + rng.uniform(-2e-3, 2e-3, (20, 3)), offset, q,
    )  # fmt: skip
    # non-finite: numpy's result, kept as the contract (no exact value)
    X = np.array([[np.nan, 0, 0], [np.inf, 0, 0], [0, 0, 1e200], [0, 0, 1e-300]])
    with np.errstate(all="ignore"):
        F = T.charge_field(X, V, T.frames(V), q)
    case(
        "charge_non_finite",
        "charge_field",
        "NaN and infinite observers, an overflowing distance, a denormal height: "
        "NaN must stay NaN, the rest as numpy gives", {"X": X, "V": V, "c": q}, F, rtol=1e-12, atol=1e-12 * np.nanmax(np.abs(F[np.isfinite(F)]), initial=1.0),
    )  # fmt: skip


def far_cases(rng):
    n_tri = 30
    c0 = rng.uniform(-0.01, 0.01, (n_tri, 3))
    V = c0[:, None, :] + rng.uniform(-0.001, 0.001, (n_tri, 3, 3))
    q = rng.normal(size=(n_tri, 3))
    fr = T.frames(V)
    y, charge = T.far_points(V, fr, q)
    # the far points from V and c: a port computes them, or ships them
    exact_y = []
    for tri, row in zip(V, q, strict=True):
        f = _mp_frame(tri)
        (a, b, c), cen, _, e1, e2, s, area = f
        pts = []
        for (wa, wb, wc), wgt in zip(T._FAR_POINTS, T._FAR_WEIGHTS, strict=True):
            p = [wa * a[k] + wb * b[k] + wc * c[k] for k in range(3)]
            d = _mp_sub(p, cen)
            dens = row[0] + (row[1] * _mp_dot(d, e1) + row[2] * _mp_dot(d, e2)) / s
            pts.append([*p, wgt * dens * area])
        exact_y.append(pts)
    rows_np = np.concatenate(
        [y.transpose(1, 0, 2), charge.T[..., None]], axis=2
    ).reshape(-1, 4)
    exact_rows = [p for tri in exact_y for p in tri]
    # compare point and charge separately: rows of four
    err_p = max(float(mp.sqrt(sum((mp.mpf(float(rn[k])) - re[k]) ** 2 for k in range(3))))
                for rn, re in zip(rows_np, exact_rows, strict=True))  # fmt: skip
    err_q = max(
        abs(float(mp.mpf(float(rn[3])) - re[3]))
        for rn, re in zip(rows_np, exact_rows, strict=True)
    )
    case(
        "far_points_random",
        "far_points",
        "far_points: per triangle (rows n*6 + g) the six points (x, y, z) and the charge each carries", {"V": V, "c": q}, rows_np,
        rtol=1e-12, atol=MARGIN * max(err_p, err_q, 1e-17),
    )  # fmt: skip
    X = rng.normal(size=(100, 3))
    X *= (np.logspace(-1, 3, 100) / np.linalg.norm(X, axis=1))[:, None]
    F = T.charge_field_far(X, y, charge)
    case(
        "charge_far_random",
        "charge_field_far",
        "the far rule from 0.1 m to 1 km, the triangles within 1 cm of the origin", {"X": X, "y": y, "charge": charge}, F,
        mp_charge_field_far(X, y, charge),
    )  # fmt: skip
    X = np.array([[0, 0, 1e101], [0, 0, 1e103], [1e160, 0, 0], [np.nan, 0, 0], y[0, 0]])
    with np.errstate(all="ignore"):
        F = T.charge_field_far(X, y, charge)
    case(
        "charge_far_extremes",
        "charge_field_far",
        "the far rule at 1e101 m, past 1e102 m (0), at 1e160 m (overflow: 0 or NaN as numpy), "
        "a NaN observer, an observer on a point (inf or NaN)", {"X": X, "y": y, "charge": charge}, F,
        rtol=1e-12, atol=0.0,
    )  # fmt: skip


def cuboid_cases(rng):
    n = 300
    dims = rng.uniform(1e-3, 2e-2, (n, 3))
    pols = rng.normal(size=(n, 3))
    X = rng.uniform(-3e-2, 3e-2, (n, 3))
    for field in ("B", "H"):
        F = _BHJM_magnet_cuboid(field, X, dims, pols)
        case(
            f"cuboid_{field}_random",
            f"cuboid_{field}",
            f"{field} of random cuboids at random observers, inside and out (one cuboid per row)", {"X": X, "dimension": dims, "polarization": pols}, F,
            mp_cuboid(field, X, dims, pols),
        )  # fmt: skip
    d = np.array([1e-2, 2e-2, 3e-2])
    a, b, c = d / 2
    pts = [
        (a, 0.1 * b, 0.2 * c), (-a, 0.3 * b, -0.4 * c), (0.2 * a, b, 0.1 * c),  # on faces
        (a, b, 0.3 * c), (0.4 * a, -b, c), (a, b, c), (-a, -b, -c),  # on edges, corners
        (2 * a, b, c), (a, 3 * b, c), (a, b, -1.5 * c), (-2 * a, -b, 0.5 * c),  # edge lines
        (a * (1 + 1e-12), b * (1 + 1e-9), 0.0), (a * (1 - 1e-12), b, 0.0),  # beside an edge
        (0.0, 0.0, 0.0), (1e3, 2e3, -1e3), (1e-1, 0, 0),
    ]  # fmt: skip
    X = np.array(pts, float)
    dims = np.tile(d, (len(X), 1))
    pols = np.tile((0.3, -0.5, 1.2), (len(X), 1))
    for field in ("B", "H"):
        F = _BHJM_magnet_cuboid(field, X, dims, pols)
        case(
            f"cuboid_{field}_special",
            f"cuboid_{field}",
            "on faces (a field), on edges and corners (0 by convention), on edge lines "
            "beyond the corners, beside an edge, the center, far away", {"X": X, "dimension": dims, "polarization": pols}, F,
            mp_cuboid(field, X, dims, pols),
        )  # fmt: skip
    X = np.array([[1e-3, 0, 0], [1e-3, 0, 0], [np.nan, 0, 0]])
    dims = np.array([[1e-2, 0.0, 1e-2], [1e-2, 1e-2, 1e-2], [1e-2, 1e-2, 1e-2]])
    pols = np.array([[0, 0, 1.0], [0, 0, 0.0], [0, 0, 1.0]])
    with np.errstate(all="ignore"):
        F = _BHJM_magnet_cuboid("B", X, dims, pols)
    case(
        "cuboid_B_degenerate",
        "cuboid_B",
        "a zero dimension and a zero polarization (0 by convention), a NaN observer", {"X": X, "dimension": dims, "polarization": pols}, F, rtol=1e-12, atol=0.0,
    )  # fmt: skip


def sheet_case(rng):
    """End to end: studio's solved plate, the sheet's H at observers on both
    sides of its switch to the far rule, as ``ChargeSheet._field`` gives it."""
    start = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.2), dimension=(0.01, 0.01, 0.005), position=(0, 0, 0.0085)
    )
    iron = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.06, 0.03, 0.004))
    iron.susceptibility = 1000
    solved = solve_face_charge(magpy.Collection(start, iron), level=1)
    (sheet,) = [s for s in solved.sources_all if isinstance(s, ChargeSheet)]
    near = rng.uniform(-0.05, 0.05, (150, 3))
    near = near[
        np.any(np.abs(near) > (0.03, 0.015, 0.002), axis=1)
    ]  # outside the plate
    far = rng.normal(size=(50, 3))
    far *= (sheet._far_from * np.linspace(0.9, 3, 50) / np.linalg.norm(far, axis=1))[
        :, None
    ]
    X = np.concatenate([near, far + sheet._center])
    H = sheet._field("H", X)
    y, charge = T.far_points(sheet._V, sheet._frames, sheet._c)
    near_rows = np.linalg.norm(X - sheet._center, axis=1) < sheet._far_from
    exact = [None] * len(X)
    for rows, values in (
        (np.flatnonzero(near_rows), mp_charge_field(X[near_rows], sheet._V, sheet._c)),
        (np.flatnonzero(~near_rows), mp_charge_field_far(X[~near_rows], y, charge)),
    ):
        for r, v in zip(rows, values, strict=True):
            exact[r] = [x / (4 * mp.pi) for x in v]
    case(
        "sheet_solved_plate",
        "sheet_H",
        "studio's solved plate (60 x 30 x 4 mm iron under a magnet, level 1): the sheet's H "
        "in A/m, near (charge_field) or far (charge_field_far, from |X - center| >= far_from), "
        "divided by 4 pi",
        {"X": X, "V": sheet._V, "c": sheet._c, "center": sheet._center,
         "far_from": np.array([sheet._far_from])},
        H, exact,
    )  # fmt: skip


def _commit(path, env=None):
    """The checkout's commit, or ``env``'s value when it is not a checkout."""
    if env and os.environ.get(env):
        return os.environ[env]
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    rng = np.random.default_rng(208)
    charge_cases(rng)
    far_cases(rng)
    cuboid_cases(rng)
    sheet_case(rng)
    doc = {
        "format": "magpylib-material-response field kernel reference, version 1",
        "issue": "https://github.com/magpylib/magpylib-material-response/issues/208",
        "generated_by": {
            "script": "kernels/reference/make_reference.py",
            "magpylib_material_response": [
                mmr.__version__,
                _commit(Path(mmr.__file__).parent),
            ],
            "magpylib": [
                magpy.__version__,
                _commit(Path(magpy.__file__).parent, "MAGPYLIB_COMMIT"),
            ],
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "comparison": (
            "Per output row r: |got_r - expected_r| <= rtol |expected_r| + atol (Euclidean "
            "norms); a non-finite expected component must be matched exactly (NaN by NaN, "
            "inf by inf of the same sign). Each case's tolerance is MARGIN times numpy's own "
            "measured error against 50-digit mpmath, floored at rtol 1e-12 and atol 1e-12 "
            "times the case's largest field."
        ),
        "constants": {"mu_0": MU0},
        "encoding": "floats as JSON numbers, exact on round trip; 'nan', 'inf', '-inf' as strings",
        "kernels": {
            "charge_field": "X (M, 3), V (N, 3, 3) triangles, c (N, 3) coefficients of 1, u, v "
            "-> (M, 3): 4 pi times the field of all triangles, summed (_triangle.charge_field; "
            "frames computed from V)",
            "far_points": "V, c -> (6 N, 4): point and charge, row n * 6 + g (_triangle.far_points)",
            "charge_field_far": "X (M, 3), y (6, N, 3), charge (6, N) -> (M, 3), summed",
            "cuboid_B": "X, dimension, polarization, each (M, 3), one cuboid per row in its own "
            "frame -> B (M, 3) in T (magpylib _BHJM_magnet_cuboid)",
            "cuboid_H": "as cuboid_B -> H (M, 3) in A/m",
            "sheet_H": "X (M, 3), V, c, center (3,), far_from (1,) -> H (M, 3) in A/m: "
            "charge_field below far_from from center, charge_field_far (from far_points) at or "
            "beyond, divided by 4 pi (ChargeSheet._field)",
        },
        "cases": CASES,
    }
    OUT.write_text(json.dumps(doc, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(CASES)} cases)")


if __name__ == "__main__":
    main()
