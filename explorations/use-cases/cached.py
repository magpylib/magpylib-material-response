"""A kept factorization behind apply_face_charge, as a prototype outside the
package: no new API, and any call whose solved parts make the same matrix as the
last call's takes only a new load and a back-substitution.

The key is what the matrix is assembled from, read off the meshed bodies, not
the user's input, so a magnet that comes to touch a part and changes its panels
misses by itself. Poses are relative to the first body, as each block is in its
body's own frame: solved parts that move as one hit. What only enters the load
stays out of the key: the fixed sources, H_ext, the remanence, the finer rule
near a magnet.

A matrix seen once may never come back, and a double LU takes longer than the
mixed-precision solve and stays in memory, so it is factored only when its key
comes back: KEEP_AFTER calls in all, then kept. Each lookup's hit costs a load
and a back-substitution.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import scipy.linalg

from magpylib_material_response.face_charge import _apply, _bodies
from magpylib_material_response.face_charge._tribody import TriBody

MAX_BYTES = 2 * 10**9  # LU kept in all
ROUND = 1e-12  # of the parts' size: a rigid motion's rounding must still hit
KEEP_AFTER = 2  # the call on which a matrix's LU is kept
SEEN = 256  # matrices remembered by their key alone

_kept = OrderedDict()
_seen = OrderedDict()  # key: calls
_solve_system = _bodies.solve_system
stats = {"hits": 0, "misses": 0, "factored": 0, "too_big": 0}


def key_of(bodies, joined):
    size = max(float(np.max(b.half)) for b in bodies)
    h = hashlib.blake2b(digest_size=16)

    def add(a):
        h.update(np.rint(np.asarray(a, float) / (ROUND * size)).astype(np.int64).data)

    first = bodies[0]
    for b in bodies:
        h.update(type(b).__name__.encode())
        h.update(np.float64(b.chi).data)
        # its own frame's panels, as the self block and its mass see them
        add(b.T if isinstance(b, TriBody) else b.panels.corners())
        if isinstance(b, TriBody):
            h.update(np.float64(b.noise).data)
            h.update(np.asarray(b.shell, np.int64).data)
        # its place relative to the first, as the coupling blocks see it
        add((b.position - first.position) @ first.R)
        add((first.R.T @ b.R) * size)
    if joined is not None:
        for shared in joined[0]:
            h.update(np.asarray(shared, bool).data)
        h.update(np.asarray(joined[1], np.int64).data)
    return h.digest()


def solve_system(bodies, external, joined=None):
    """As _bodies.solve_system, keeping the factorization between calls."""
    key = key_of(bodies, joined)
    lu = _kept.get(key)
    offsets = np.cumsum([0] + [b.n_dofs for b in bodies])

    if lu is None:
        stats["misses"] += 1
        calls = _seen[key] = _seen.pop(key, 0) + 1
        while len(_seen) > SEEN:
            _seen.popitem(last=False)
        if calls < KEEP_AFTER:
            return _solve_system(bodies, external, joined)
        stats["factored"] += 1
        seen = {}

        def factor(A, b):
            seen["lu"] = scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)
            return scipy.linalg.lu_solve(seen["lu"], b, check_finite=False)

        solve, _bodies._solve = _bodies._solve, factor
        try:
            coefs = _solve_system(bodies, external, joined)
        finally:
            _bodies._solve = solve
        bytes_ = seen["lu"][0].nbytes
        if bytes_ > MAX_BYTES:
            stats["too_big"] += 1
            return coefs
        _kept[key] = seen["lu"]
        while sum(v[0].nbytes for v in _kept.values()) > MAX_BYTES:
            _kept.popitem(last=False)
        return coefs

    stats["hits"] += 1
    _kept.move_to_end(key)
    rhs = np.zeros(len(lu[0]))
    for ia, a in enumerate(bodies):
        rows = slice(offsets[ia], offsets[ia + 1])

        def field(x, a=a):
            return (external(a.to_global(x)) + a.applied) @ a.R

        rhs[rows] += a.load(field)
        if a.J0 is not None:
            rhs[rows] += a.load_remanence() / a.chi
    if joined is not None:
        for ia, shared in enumerate(joined[0]):
            dofs = (
                offsets[ia]
                + (3 * np.flatnonzero(shared)[:, None] + np.arange(3)).ravel()
            )
            rhs[dofs] = 0.0
    x = scipy.linalg.lu_solve(lu, rhs, check_finite=False)
    return [x[offsets[k] : offsets[k + 1]] for k in range(len(bodies))]


def install():
    _apply.solve_system = solve_system


def uninstall():
    _apply.solve_system = _solve_system


def clear():
    _kept.clear()
    _seen.clear()
    stats.update(hits=0, misses=0, factored=0, too_big=0)
