"""A kept factorization for face charge, as a prototype outside the package:
solve one configuration with apply_face_charge, keep its bodies and system
matrix, then solve any number of configurations in which only the fixed
sources move, each a new load and a back-substitution, and read the field of
each at its own observer.

The load and the read are written here for many configurations at once,
batched along magpylib paths. Cuboid parts only, no parts that touch.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg

from magpylib_material_response.face_charge import _apply, _bodies
from magpylib_material_response.face_charge._bodies import Body
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._triangle import (
    _EDGES,
    _dot,
    _r_plus_l,
    frames,
)

STEPS_PER_CALL = (
    100  # magpylib main tiles every row of a path call: 200 steps peaked at 3.5 GB
)
ROWS_PER_READ = 500  # observers per call of the read, each with its own charge


class Kept:
    """apply_face_charge's solve of ``collection``, with its factorization kept."""

    def __init__(self, collection, **kwargs):
        seen = {}
        solve_system, solve = _apply.solve_system, _bodies._solve

        def capture_system(bodies, external, joined=None):
            seen.update(bodies=bodies, joined=joined)
            return solve_system(bodies, external, joined)

        def capture_solve(A, b):
            seen.update(A=A.copy(order="F"), b=b.copy())
            return solve(A, b)

        _apply.solve_system, _bodies._solve = capture_system, capture_solve
        try:
            t = time.perf_counter()
            self.solved = _apply.apply_face_charge(collection, **kwargs)
            self.first_s = time.perf_counter() - t
        finally:
            _apply.solve_system, _bodies._solve = solve_system, solve

        self.bodies = bodies = seen["bodies"]
        if not all(isinstance(b, Body) for b in bodies):
            msg = "the prototype keeps cuboid parts only"
            raise NotImplementedError(msg)
        if seen["joined"] is not None and any(s.any() for s in seen["joined"][0]):
            msg = "the prototype keeps parts that do not touch only"
            raise NotImplementedError(msg)
        self.offsets = np.cumsum([0] + [b.n_dofs for b in bodies])
        self.size = len(seen["b"])  # the unknowns and the net-charge rows
        t = time.perf_counter()
        self.lu = scipy.linalg.lu_factor(seen["A"], check_finite=False)
        self.factor_s = time.perf_counter() - t
        self.nominal_rhs = seen["b"]

        # what does not change: the remanence's load
        self.constant = np.zeros(self.size)
        for a, lo, hi in zip(bodies, self.offsets[:-1], self.offsets[1:], strict=True):
            if a.J0 is not None:
                self.constant[lo:hi] += a.load_remanence() / a.chi

        # the load's points, all bodies in one array: each body's panels by its
        # own rule, those near a magnet by the fine one, as Galerkin.load does
        self.segments, points, first = [], [], 0
        for k, a in enumerate(bodies):
            fine = np.zeros(a.panels.N, bool) if a.fine is None else np.asarray(a.fine)
            for nq, idx in ((4, np.flatnonzero(~fine)), (16, np.flatnonzero(fine))):
                if not idx.size:
                    continue
                pts, wts, S, T = a.galerkin.gauss_points(nq)
                P = a.to_global(pts[:, idx].reshape(-1, 3))
                self.segments.append(
                    (k, idx, wts[:, idx], S, T, slice(first, first + len(P)))
                )
                points.append(P)
                first += len(P)
        self.points = np.concatenate(points)

        # each body's sheet: two triangles a panel, as ChargeSheet._from_body
        # makes them, and the map from corner charges to their coefficients
        self.sheets = []
        for a in bodies:
            p = a.panels
            corners, normal = p.corners(), p.normal
            V = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
            n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
            inward = np.einsum("fd,fd->f", n, np.concatenate([normal, normal])) < 0
            V[inward] = V[inward][:, [0, 2, 1]]
            fr = frames(V)
            cen, _, e1, e2, s, _ = fr
            d = V - cen[:, None]
            A = np.stack(
                [
                    np.ones((len(V), 3)),
                    np.einsum("nkd,nd->nk", d, e1) / s[:, None],
                    np.einsum("nkd,nd->nk", d, e2) / s[:, None],
                ],
                axis=-1,
            )
            self.sheets.append((V, fr, inward, np.linalg.inv(A)))

    def loads(self, field, steps):
        """The right-hand sides (steps, size) of the fixed sources' field:
        ``field(points, sl)`` gives H (len(sl), len(points), 3), A/m, global,
        for the steps ``sl``."""
        R = np.tile(self.constant, (steps, 1))
        for a0 in range(0, steps, STEPS_PER_CALL):
            sl = slice(a0, min(a0 + STEPS_PER_CALL, steps))
            H = field(self.points, sl)
            if not np.isfinite(H).all():
                msg = "the prototype does not take the field off a magnet's edge"
                raise NotImplementedError(msg)
            for k, idx, wts, S, T, pts in self.segments:
                a = self.bodies[k]
                h = ((H[:, pts] + a.applied) @ a.R).reshape(len(H), len(S), len(idx), 3)
                wHn = wts * np.einsum("sgnd,nd->sgn", h, a.panels.normal[idx])
                F = np.zeros((len(H), a.panels.N, 3))
                F[:, idx] = np.stack(
                    [wHn.sum(1), (wHn * S[:, None]).sum(1), (wHn * T[:, None]).sum(1)],
                    -1,
                )
                R[sl, self.offsets[k] : self.offsets[k + 1]] += F.reshape(len(H), -1)
        return R

    def solve(self, R):
        """The unknowns (steps, size) of the right-hand sides R (steps, size)."""
        return scipy.linalg.lu_solve(self.lu, R.T, check_finite=False).T

    def field(self, X, x):
        """H (M, 3), A/m, global, of the solved parts at the observers X (M, 3),
        each observer with its own configuration's unknowns x (M, size)."""
        out = np.zeros((len(X), 3))
        for a, (V, fr, inward, Ainv), lo, hi in zip(
            self.bodies, self.sheets, self.offsets[:-1], self.offsets[1:], strict=True
        ):
            c = x[:, lo:hi].reshape(len(X), -1, 3)
            s, t = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
            qc = c[..., :1] + c[..., 1:2] * s + c[..., 2:3] * t
            q = np.concatenate([qc[..., [0, 1, 2]], qc[..., [0, 2, 3]]], axis=1)
            q[:, inward] = q[:, inward][..., [0, 2, 1]]
            C = np.einsum("fbk,mfk->mfb", Ainv, q)
            local = (X - a.position) @ a.R
            for r in range(0, len(X), ROWS_PER_READ):
                rows = slice(r, r + ROWS_PER_READ)
                out[rows] += charge_field_rows(local[rows], V, fr, C[rows]) @ a.R.T
        return out / FOUR_PI


def charge_field_rows(X, V, fr, c):
    """``_triangle.charge_field`` with each observer's own coefficients ``c``
    (M, N, 3) instead of one set (N, 3) for all."""
    _, n, e1, e2, s, area = fr
    a = V[:, 0]
    ub = _dot(V[:, 1] - a, e1)
    uc, vc = _dot(V[:, 2] - a, e1), _dot(V[:, 2] - a, e2)
    corners = ((0.0, 0.0), (ub, 0.0), (uc, vc))
    u, v, w = (
        np.matmul(X, np.stack([e1, e2, n]).transpose(0, 2, 1))
        - np.stack([_dot(a, e1), _dot(a, e2), _dot(a, n)])[:, None]
    )
    w = np.where(w == 0, 1e-12 * s, w)
    w2, abs_w = w * w, np.abs(w)
    du = [pu - u for pu, _ in corners]
    dv = [pv - v for _, pv in corners]
    R = [np.sqrt(du[k] * du[k] + dv[k] * dv[k] + w2) for k in range(3)]
    den = R[0] * R[1] * R[2]
    for (i, j), k in zip(_EDGES, (2, 0, 1), strict=True):
        den = den + (du[i] * du[j] + dv[i] * dv[j] + w2) * R[k]
    omega = 2 * np.arctan2(2 * area * abs_w, den)
    c0, c1, c2 = c[..., 0], c[..., 1], c[..., 2]
    E0u = E0v = Phi = SLu = SLv = SLw = 0.0
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
            P0f, dR, gm = P0 * f, R[j] - R[i], c1 * tv - c2 * tu
            E0u = E0u + tv * f
            E0v = E0v - tu * f
            Phi = Phi + P0f
            SLu = SLu - gm * (P0f * tv + dR * tu)
            SLv = SLv + gm * (P0f * tu - dR * tv)
            SLw = SLw + gm * f
    Phi = Phi - abs_w * omega
    psi = c0 + (c1 * (u - (ub + uc) / 3) + c2 * (v - vc / 3)) / s
    Fu = psi * E0u - (SLu + c1 * Phi) / s
    Fv = psi * E0v - (SLv + c2 * Phi) / s
    Fw = psi * np.sign(w) * omega - w * SLw / s
    return Fu @ e1 + Fv @ e2 + Fw @ n
