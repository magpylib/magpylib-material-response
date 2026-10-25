"""Face-charge solve of a magpylib collection.

Cuboids with an isotropic susceptibility (on themselves or a parent collection)
become bodies; every other source (magnets with susceptibility 0, currents)
stays a fixed source. An ``H_ext`` attribute (mu0 H in T, as material-response
reads it) is a uniform applied field. The returned B, like magpylib's getB of a
solved collection, excludes that applied field.
"""

import os
import sys
import time

import magpylib as magpy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the study: multibody, bem, p1
from multibody import MU0, Body, add_contact_lines, field_B, solve_system  # noqa: E402
from source import ChargeSheet  # noqa: E402


def _inherited(obj, name):
    while obj is not None:
        value = getattr(obj, name, None)
        if value is not None:
            return value
        obj = obj.parent
    return None


def split(coll):
    """(bodies as (object, chi), fixed sources, uniform applied H in A/m)."""
    bodies, fixed, applied = [], [], np.zeros(3)
    for src in coll.sources_all:
        chi = _inherited(src, "susceptibility")
        if isinstance(src, magpy.magnet.Cuboid) and chi is not None and np.any(chi):
            if np.ndim(chi):
                raise NotImplementedError("anisotropic susceptibility needs volume charges")
            bodies.append((src, float(chi)))
            h = _inherited(src, "H_ext")
            if h is not None:
                applied = np.asarray(h, float) / MU0
        else:
            fixed.append(src)
    return bodies, fixed, applied


def solve(coll, m, observers, g=2.5, refine=0):
    """B (T) at observers (n, 3), and timings, sizes, the largest size-weighted
    panel jump over the bodies (ChargeSheet.jump_indicator) after each round of
    refinement (see `apply_face_charge`), and each body's average polarization
    |J| in T (ChargeSheet.mean_polarization): while that stays below the knee of
    the material's B-H curve, the linear solve holds."""
    observers = np.asarray(observers, float).reshape(-1, 3)
    objs, fixed, bodies, coefs, (t0, t1, t2), rounds = _solve_bodies(coll, m, g, refine)
    B = field_B(bodies, coefs, observers)
    if fixed:
        B = B + magpy.getB(fixed, observers, sumup=True)
    t3 = time.perf_counter()
    info = {"m": m, "refine": refine, "bodies": len(bodies), "unknowns": rounds[-1]["unknowns"],
            "build_s": t1 - t0, "solve_s": t2 - t1, "eval_s": t3 - t2,
            "jump_indicator": rounds[-1]["jump_indicator"], "rounds": rounds,
            "polarization": [float(np.linalg.norm(MU0 * b.s.moment(c) / np.prod(b.dimension))) for b, c in zip(bodies, coefs, strict=True)]}
    return B, info


def apply_face_charge(coll, m, g=2.5, refine=0):
    """A copy of the collection in which every solved body is replaced by its
    surface charge, a ChargeSheet (source.py) at the same place in the
    hierarchy; the other sources are copied as they are. Like a collection solved
    by material-response, getB of the copy excludes the applied field H_ext.

    `refine` rounds each split the panels whose size-weighted jump is above
    half the largest over all bodies, and solve again: the panels go where the
    mesh level `m` is too coarse for the field around the bodies, e.g. under a
    magnet close to a soft face. A round roughly doubles the unknowns."""
    objs, _, bodies, coefs, _, _ = _solve_bodies(coll, m, g, refine)
    out = coll.copy()
    twin = dict(zip((id(s) for s in coll.sources_all), out.sources_all, strict=True))
    for (obj, _), body, c in zip(objs, bodies, coefs, strict=True):
        old = twin[id(obj)]
        label = obj.style.label or f"{type(obj).__name__}, face charge"
        new = ChargeSheet.from_cuboid(body, c, style_label=label)
        parent = old.parent
        parent.remove(old)
        parent.add(new)
    return out


def _solve_bodies(coll, m, g, refine=0, mark=0.5):
    """Split, mesh and solve, then refine: (bodies' objects, fixed sources,
    bodies, coefficients, (start, built, solved) times, and per solve the
    unknowns and the largest jump indicator)."""
    t0 = time.perf_counter()
    objs, fixed, applied = split(coll)
    bodies = []
    for obj, chi in objs:
        J0 = None if obj.polarization is None else np.asarray(obj.polarization, float)
        bodies.append(Body(obj.dimension, position=obj.position, orientation=obj.orientation,
                           chi=chi, J0=J0 if J0 is not None and np.any(J0) else None, m=m, g=g))
    add_contact_lines(bodies)
    for b in bodies:
        b.build()
    t1 = time.perf_counter()

    def H_ext(p):
        H = np.tile(applied, (len(p), 1))
        if fixed:
            H = H + magpy.getH(fixed, p, sumup=True)
        return H

    rounds = []
    for r in range(refine + 1):
        coefs = solve_system(bodies, H_ext)
        eta = [ChargeSheet.from_cuboid(b, c).panel_indicator() for b, c in zip(bodies, coefs, strict=True)]
        top = max((e.max(initial=0.0) for e in eta), default=0.0)
        rounds.append({"unknowns": int(sum(b.Nd for b in bodies)), "jump_indicator": float(top)})
        if r == refine or not top:
            break
        for b, e in zip(bodies, eta, strict=True):
            marked = np.flatnonzero(e > mark * top)
            if marked.size:
                b.bisect(marked)
                b.build()
    t2 = time.perf_counter()
    return objs, fixed, bodies, coefs, (t0, t1, t2), rounds
