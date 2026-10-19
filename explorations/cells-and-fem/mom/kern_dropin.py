"""Drop another library's tetrahedron kernel into #52's point-matched assembly and solve the U-core.

usage: kern_dropin.py <stock|pymagba|jax> <tetgen_target>
The point rule needs mu0*H of source cell j at observer i: B_ij/… minus J for the self term (i inside j
only when i == j: the cells tile the body without overlap).
"""

import json
import os
import sys
import time

import magpylib as magpy
import numpy as np
from adapt import CHI, COIL, LINE
from adapt_tet import tetgen_mesh
from magpylib_material_response import demag
from magpylib_material_response.demag import apply_demag

MU0 = magpy.mu_0
UNIT = np.eye(3)


def world_vertices(srcs):
    return np.array([s.orientation.apply(s.vertices) + s.position for s in srcs])


def inside(V, pos_a):
    """(n_b, n_a) mask: observer i is the centroid of source j. The point rule matches at centroids and the
    cells tile the body, so that is the only observer inside source j (an exact float match: #52 builds
    pos_a from the same centroids)."""
    return (pos_a[None] == V.mean(1)[:, None]).all(-1)


def block_from_B(B, V, pos_a):
    """B (3 unit pols, n_b, n_a, 3) in tesla -> #52's T block layout; mu0 H = B - J inside the source."""
    n_a, n_b = len(pos_a), len(V)
    m = inside(V, pos_a)
    for k in range(3):
        B[k][m] -= UNIT[k]
    return B.swapaxes(2, 3).reshape(3 * n_b, 3 * n_a).T


def point_block_pymagba(srcs_b, pos_a):
    import pymagba.fields as pf

    V = world_vertices(srcs_b)
    B = np.empty((3, len(srcs_b), len(pos_a), 3))

    def one(j):
        v = V[j].tolist()
        for k in range(3):
            B[k, j] = pf.tetrahedron_B(pos_a, polarization=UNIT[k], vertices=v)

    if os.environ.get("POOL"):  # parallel over sources; the Rust kernel releases the GIL
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(int(os.environ["POOL"])) as ex:
            list(ex.map(one, range(len(V))))
    else:
        for j in range(len(V)):
            one(j)
    return block_from_B(B, V, pos_a)


_jit = {}


def point_block_jax(srcs_b, pos_a):
    import jax
    import jax.numpy as jnp
    import magpylib_jax.core as jc

    if "f" not in _jit:
        _jit["f"] = jax.jit(jc.tetrahedron_hfield)
    V = world_vertices(srcs_b)
    n_a, n_b = len(pos_a), len(srcs_b)
    obs = np.tile(pos_a, (n_b, 1))
    verts = np.repeat(V, n_a, axis=0)
    H = np.stack([np.asarray(_jit["f"](jnp.asarray(obs), jnp.asarray(verts), jnp.broadcast_to(UNIT[k], (n_a * n_b, 3)))) for k in range(3)])
    T_raw = H.reshape(3, n_b, n_a, 3) * MU0  # magpylib_jax returns H inside and out, like magpylib
    return T_raw.swapaxes(2, 3).reshape(3 * n_b, 3 * n_a).T


if __name__ == "__main__":
    kind, target = sys.argv[1], int(sys.argv[2])
    if kind == "pymagba":
        demag._point_block_T = point_block_pymagba
    elif kind == "jax":
        demag._point_block_T = point_block_jax
    V = tetgen_mesh(target)
    core = magpy.Collection(*(magpy.magnet.Tetrahedron(vertices=v, polarization=(0, 0, 0)) for v in V))
    core.susceptibility = CHI
    t0 = time.perf_counter(); c0 = time.process_time()
    solved = apply_demag(magpy.Collection(core, COIL, override_parent=True))
    t = time.perf_counter() - t0; cw = (time.process_time() - c0) / t
    Bz = solved.getB(LINE)[:, 2]
    print(json.dumps({"kernel": kind, "rayon": os.environ.get("RAYON_NUM_THREADS", "all"), "pool": os.environ.get("POOL", "-"), "cells": len(V), "peak_mT": float(np.abs(Bz).max() * 1e3), "solve_s": round(t, 2), "cpu_over_wall": round(cw, 1)}), flush=True)
