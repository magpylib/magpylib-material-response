"""The U-core in saturation, by two methods.

The documented U-core (case F: three cuboids, chi 999, a 16-winding coil) with
the coil current scaled until the core saturates, on the B-H curve of issue
#81: J = chi0 mu0 H / (1 + chi0 mu0 H / Js), chi0 999, Js 1.6 T, so the
secant susceptibility is chi(H) = chi0 / (1 + chi0 mu0 H / Js).

- MoM (material-response): each cell's chi from the curve at the field at its
  centre, `apply_demag` solved again until no chi moves (issue #81's scheme).
- Face charge on sub-bodies: each core cuboid cut into touching sub-cuboids,
  each with its own uniform chi from the curve at its average field (its
  moment / volume / chi, exact for a uniform chi), the coupled system solved
  again with the operator built once.

Both iterate on log chi with Anderson acceleration (depth 5). Run with
material-response's environment (case F builds its coil with it).
"""

import itertools
import os
import sys
import time

import magpylib as magpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..")
sys.path[:0] = [STUDY, os.path.join(STUDY, "validation")]
import examples  # noqa: E402
from multibody import MU0, Body, add_contact_lines, cross_block, field_B, solve_system  # noqa: E402

CHI0, JS = 999.0, 1.6
LINE = examples.F_LINE  # Bz along x, 30 mm above the pole faces


def chi_of(H):
    """Secant susceptibility at field magnitude H (A/m)."""
    return CHI0 / (1 + CHI0 * MU0 * np.asarray(H) / JS)


def build(current):
    """(core collection, coil collection) with every winding at `current` (A)."""
    core, coil = examples.build_F().children
    for w in coil.children:
        w.current = current
    return core, coil


def anderson(g, x0, depth=5, tol=1e-4, maxit=150):
    """Solve x = g(x), Anderson-accelerated: (x, iterations)."""
    x, xs, gs = np.asarray(x0, float), [], []
    for k in range(1, maxit + 1):
        gx = g(x)
        if np.abs(gx - x).max() < tol:
            return gx, k
        xs.append(x)
        gs.append(gx)
        xs, gs = xs[-depth - 1 :], gs[-depth - 1 :]
        if len(xs) == 1:
            x = gx
            continue
        F = np.array(gs) - np.array(xs)
        dF, dG = np.diff(F, axis=0), np.diff(np.array(gs), axis=0)
        gamma = np.linalg.lstsq(dF.T, F[-1], rcond=None)[0]
        x = gx - gamma @ dG
    return x, maxit


def mom(currents, cells_across):
    """MoM, meshed once: per current, (Bz along LINE in mT, linear and
    saturated), and the cells, solves and seconds."""
    from magpylib_material_response.demag import apply_demag
    from magpylib_material_response.meshing import mesh_uniform

    meshed = mesh_uniform(build(1.0)[0].parent, cells_across=cells_across)
    cells = [s for s in meshed.sources_all if isinstance(s, magpy.magnet.Cuboid)]
    windings = [s for s in meshed.sources_all if isinstance(s, magpy.current.Polyline)]
    centres = np.array([c.position for c in cells])
    state = {}

    def g(logchi):
        for c, x in zip(cells, np.exp(logchi), strict=True):
            c.susceptibility = float(x)
        state["solved"] = apply_demag(meshed)
        return np.log(chi_of(np.linalg.norm(state["solved"].getH(centres), axis=1)))

    out = []
    for current in currents:
        t = time.perf_counter()
        for w in windings:
            w.current = current
        start = np.full(len(cells), np.log(CHI0))
        g(start)
        linear = state["solved"].getB(LINE)[:, 2] * 1e3
        logchi, solves = anderson(g, start)
        g(logchi)
        out.append({"current": current, "linear": linear.tolist(), "saturated": (state["solved"].getB(LINE)[:, 2] * 1e3).tolist(),
                    "cells": len(cells), "solves": solves + 2, "seconds": time.perf_counter() - t,
                    "chi_min": float(np.exp(logchi).min())})
    return out


def sub_bodies(core, size, m, across=1):
    """Each core cuboid cut along its length into touching slabs about
    `size` (m) long, each cut `across` times along both other axes."""
    bodies = []
    for cub in core.children:
        dim = np.asarray(cub.dimension)
        n = np.full(3, across)
        k = int(np.argmax(dim))
        n[k] = max(1, round(dim[k] / size))
        d = dim / n
        for idx in itertools.product(*(range(j) for j in n)):
            pos = np.asarray(cub.position) - dim / 2 + (np.array(idx) + 0.5) * d
            bodies.append(Body(d, position=pos, chi=CHI0, m=m))
    return bodies


def face_charge(currents, size, m, across=1):
    """Face charge on slabs, the operator built once: per current, (Bz along
    LINE in mT, linear and saturated), and the bodies, unknowns, solves,
    seconds and per-body chi."""
    t0 = time.perf_counter()
    core, coil = build(1.0)
    bodies = sub_bodies(core, size, m, across)
    add_contact_lines(bodies)
    for b in bodies:
        b.build()
    cross = {(i, j): cross_block(a, b) for i, a in enumerate(bodies) for j, b in enumerate(bodies) if i != j}
    build_s = time.perf_counter() - t0
    out = []
    for current in currents:
        t = time.perf_counter()
        cache = {}

        def H_ext(p, current=current):
            key = p.tobytes()
            if key not in cache:
                cache[key] = current * magpy.getH(coil, p)  # the coil at 1 A, scaled
            return cache[key]

        state = {}

        def g(logchi, H_ext=H_ext, state=state):
            for b, x in zip(bodies, np.exp(logchi), strict=True):
                b.chi = float(x)
            state["coefs"] = solve_system(bodies, H_ext, cross=cross)
            H = [np.linalg.norm(b.s.moment(c)) / np.prod(b.dimension) / b.chi for b, c in zip(bodies, state["coefs"], strict=True)]
            return np.log(chi_of(H))

        def Bz(state=state):
            return (field_B(bodies, state["coefs"], LINE)[:, 2] + current * magpy.getB(coil, LINE)[:, 2]) * 1e3

        start = np.full(len(bodies), np.log(CHI0))
        g(start)
        linear = Bz()
        logchi, solves = anderson(g, start)
        g(logchi)
        out.append({"current": current, "linear": linear.tolist(), "saturated": Bz().tolist(), "bodies": len(bodies),
                    "unknowns": int(sum(b.Nd for b in bodies)), "solves": solves + 2, "build_s": build_s,
                    "seconds": time.perf_counter() - t, "chi": np.exp(logchi).round(1).tolist()})
    return out
