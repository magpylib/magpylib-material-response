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


def solve(coll, m, observers, g=2.5):
    """B (T) at observers (n, 3), and timings and sizes."""
    observers = np.asarray(observers, float).reshape(-1, 3)
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

    coefs = solve_system(bodies, H_ext)
    t2 = time.perf_counter()
    B = field_B(bodies, coefs, observers)
    if fixed:
        B = B + magpy.getB(fixed, observers, sumup=True)
    t3 = time.perf_counter()
    info = {"m": m, "bodies": len(bodies), "unknowns": int(sum(b.Nd for b in bodies)),
            "build_s": t1 - t0, "solve_s": t2 - t1, "eval_s": t3 - t2}
    return B, info
