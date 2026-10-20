"""Evaluation side: field of a solved U-core on the probe line and the 161x81 plane, three ways.

usage: solved_body.py <tetgen_target>
  exact      : every cell through pymagba's tetrahedron kernel (validated 1e-14 vs magpylib), plus the coil
  fmm        : cells as point dipoles through fmm3dpy (targets path), plus the coil
  multipole  : one spherical-harmonic expansion of the body about its centre, orders 4..16
"""

import json
import sys
import time

import fmm3dpy
import magpylib as magpy
import numpy as np
import pymagba.fields as pf
from adapt import CHI, COIL, LINE
from adapt_tet import split, tetgen_mesh
from magpylib_material_response.demag import apply_demag
from scipy.special import sph_harm_y

MU0 = magpy.mu_0
target = int(sys.argv[1])
V = tetgen_mesh(target)
n = len(V)
core = magpy.Collection(*(magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V))
core.susceptibility = CHI
t0 = time.perf_counter()
solved = apply_demag(magpy.Collection(core, COIL, override_parent=True))
t_solve = time.perf_counter() - t0
J = np.array([s.polarization for s in solved.sources_all if isinstance(s, magpy.magnet.Tetrahedron)])
vol = np.abs(np.linalg.det(V[:, 1:] - V[:, :1])) / 6
cent = V.mean(1)

x = np.linspace(-0.08, 0.08, 161)
y = np.linspace(-0.04, 0.04, 81)
X, Y = np.meshgrid(x, y, indexing="ij")
PLANE = np.stack([X, Y, np.full_like(X, 0.10)], -1).reshape(-1, 3)
sets = {"line": LINE, "plane": PLANE}
out = {"cells": n, "solve_s": round(t_solve, 2)}


def exact_B(P):
    """Sum of every cell's exact field (pymagba, Rayon over the points) plus the coil."""
    B = np.zeros_like(P)
    for j in range(n):
        B += pf.tetrahedron_B(P, polarization=J[j], vertices=V[j].tolist())
    return B + COIL.getB(P)


def fmm_B(P):
    o = fmm3dpy.lfmm3d(eps=1e-10, sources=cent.T.copy(), dipvec=(J * vol[:, None] / MU0).T.copy(), targets=P.T.copy(), pgt=2)
    return -MU0 * o.gradtarg.T + COIL.getB(P)


# ---- multipole expansion about the body centre, from sub-dipoles (each cell split 3x -> 512 sub-tets)
c0 = np.array([0.0, 0.0, 0.03])
R = np.linalg.norm(V.reshape(-1, 3) - c0, axis=1).max()
sub = V
for _ in range(3):
    sub = split(sub).reshape(-1, 4, 3)
m_sub = np.repeat(J * vol[:, None] / MU0 / 512, 512, axis=0)  # moment per sub-tet (A m^2)
p_sub = sub.mean(1) - c0


def solid(l, m, P):
    """Regular solid harmonic r^l Y_lm at points P (complex)."""
    r = np.linalg.norm(P, axis=1)
    th = np.arccos(np.clip(P[:, 2] / np.where(r > 0, r, 1), -1, 1))
    ph = np.arctan2(P[:, 1], P[:, 0])
    return r**l * sph_harm_y(l, m, th, ph)


def coefficients(L):
    q = {}
    h = 1e-6 * R
    for l in range(L + 1):
        for m in range(-l, l + 1):
            grad = np.zeros((len(p_sub), 3), dtype=complex)
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                grad[:, k] = (np.conj(solid(l, m, p_sub + e)) - np.conj(solid(l, m, p_sub - e))) / (2 * h)
            q[(l, m)] = np.sum(m_sub * grad.real, axis=None) + 1j * np.sum(m_sub * grad.imag, axis=None)
    return q


def potential(q, P, L):
    Pr = P - c0
    r = np.linalg.norm(Pr, axis=1)
    th = np.arccos(np.clip(Pr[:, 2] / r, -1, 1))
    ph = np.arctan2(Pr[:, 1], Pr[:, 0])
    phi = np.zeros(len(P), dtype=complex)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            phi += q[(l, m)] * sph_harm_y(l, m, th, ph) / ((2 * l + 1) * r ** (l + 1))
    return phi.real


def multipole_B(q, P, L):
    h = 1e-6 * R
    H = np.zeros_like(P)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        H[:, k] = -(potential(q, P + e, L) - potential(q, P - e, L)) / (2 * h)
    return MU0 * H + COIL.getB(P)


t0 = time.perf_counter()
Q16 = coefficients(16)
out["coefficients_s"] = round(time.perf_counter() - t0, 1)
out["R_mm"] = round(R * 1e3, 1)
print(json.dumps(out), flush=True)
res = {}
for name, P in sets.items():
    t0 = time.perf_counter()
    Bex = exact_B(P)
    t_ex = time.perf_counter() - t0
    scale = np.abs(Bex).max()
    dmin = np.min(np.linalg.norm(P[:, None, :] - cent[None], axis=-1)) if len(P) < 5000 else np.min(np.linalg.norm(P[::7, None, :] - cent[None], axis=-1))
    t0 = time.perf_counter()
    Bf = fmm_B(P)
    t_f = time.perf_counter() - t0
    r_ratio = np.linalg.norm(P - c0, axis=1).min() / R
    res[name] = {"points": len(P), "exact_s": round(t_ex, 3), "fmm_s": round(t_f, 3), "fmm_err": float(np.abs(Bf - Bex).max() / scale),
                 "nearest_cell_mm": round(dmin * 1e3, 1), "r_min_over_R": round(r_ratio, 3), "multipole": {}}
    for L in (4, 8, 12, 16):
        t0 = time.perf_counter()
        Bm = multipole_B(Q16, P, L)
        t_m = time.perf_counter() - t0
        res[name]["multipole"][L] = {"err": float(np.abs(Bm - Bex).max() / scale), "s": round(t_m, 3)}
    print(json.dumps({name: res[name]}), flush=True)
# today's cost: magpylib getB of the solved collection on the plane
t0 = time.perf_counter()
Bmp = np.vstack([solved.getB(PLANE[k : k + 1000]) for k in range(0, len(PLANE), 1000)])  # chunked: the whole plane in one call needs > 5 GB
t_mp = time.perf_counter() - t0
res["plane"]["magpylib_getB_s"] = round(t_mp, 2)
res["plane"]["magpylib_vs_exact"] = float(np.abs(Bmp - exact_B(PLANE)).max() / np.abs(Bmp).max())
out.update(res)
print(json.dumps(out), flush=True)
