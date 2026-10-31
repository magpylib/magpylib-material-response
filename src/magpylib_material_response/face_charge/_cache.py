"""The factorizations kept between solves.

The matrix of a solve depends only on the solved parts, so a call whose parts
make the matrix of an earlier call takes only a new load and a
back-substitution. The key is what the matrix is assembled from, read off the
meshed bodies rather than the user's input: a magnet that comes close enough to
change a part's panels misses by itself. Poses are relative to the first body,
as each block is in its body's own frame, so parts that move as one hit. What
enters only the load stays out of the key: the fixed sources, ``H_ext``, the
remanence, the finer rule near a magnet.

A matrix seen once may never come back, and a factorization in double takes
longer than the mixed-precision solve and stays in memory, so it is made only
when a key comes back.
"""

from __future__ import annotations

import hashlib
import os
import threading
import warnings
from collections import OrderedDict

import numpy as np
import scipy.linalg

from magpylib_material_response.face_charge import _bodies
from magpylib_material_response.face_charge._tribody import TriBody

ROUND = 1e-12  # of the parts' size: a rigid motion's rounding must still hit
SEEN = 256  # matrices remembered by their key alone

_limit = 1e9  # bytes, all that is kept together
_kept = OrderedDict()  # key: (lu, piv), the oldest first
_seen = OrderedDict()  # keys that came once
_lock = threading.Lock()


def _reset_after_fork():
    # a forked child inherits the lock, maybe held by a thread it does not have
    global _lock  # noqa: PLW0603  # pylint: disable=global-statement
    _lock = threading.Lock()
    _kept.clear()
    _seen.clear()


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(after_in_child=_reset_after_fork)


def clear_cache():
    """Forget the factorizations kept by
    :func:`~magpylib_material_response.face_charge.apply_face_charge`, freeing
    their memory."""
    with _lock:
        _kept.clear()
        _seen.clear()


def set_cache_limit(gigabytes):
    """Set how much memory the kept factorizations may take together, 1 GB by
    default. A factorization takes ``8 n**2`` bytes for ``n`` unknowns: 18 MB at
    1,500, 1.7 GB at 14,500. One that does not fit is not made, and the solve
    costs what it would without a cache, after freeing what is kept for the
    memory it needs. 0 turns the cache off, ``math.inf`` removes the limit."""
    global _limit  # noqa: PLW0603  # pylint: disable=global-statement
    if not gigabytes >= 0:
        msg = f"The cache limit must be 0 or more gigabytes; got {gigabytes!r}."
        raise ValueError(msg)
    with _lock:
        _limit = gigabytes * 1e9
        _evict()


def _held():
    return sum(lu.nbytes + piv.nbytes for lu, piv in _kept.values())


def _evict(room=0):
    """Drop the oldest until the rest and ``room`` more bytes fit the limit."""
    while _kept and _held() + room > _limit:
        _kept.popitem(last=False)


def _bytes(bodies):
    """What a factorization of these bodies takes, at most: each closed shell
    adds a net-charge row, fewer where shells join."""
    shells = sum(b.shell.max() + 1 if isinstance(b, TriBody) else 1 for b in bodies)
    n = sum(b.n_dofs for b in bodies) + shells
    return 8 * n * n + 8 * n


def _key(bodies, joined):
    size = max(float(np.max(b.half)) for b in bodies)
    digest = hashlib.blake2b(digest_size=16)
    # lengths below are relative to the parts' size, but the matrix is not
    # scale-free: its mass and net-charge entries go with the area
    digest.update(np.float64(size).data)

    def add(values, unit):
        # + 0.0 makes a rounded -0.0 the 0.0 it is
        digest.update((np.round(np.asarray(values, float) / (ROUND * unit)) + 0.0).data)

    first = bodies[0]
    for b in bodies:
        digest.update(type(b).__name__.encode())
        digest.update(np.float64(b.chi).data)
        digest.update(np.float64(b.noise).data)
        # its panels in its own frame, as its self block and mass see them
        if isinstance(b, TriBody):
            add(b.T, size)
            digest.update(np.asarray(b.shell, np.int64).data)
        else:
            add(b.panels.corners(), size)
        # its place against the first, as the coupling blocks see it
        add((b.position - first.position) @ first.R, size)
        add(first.R.T @ b.R, 1.0)
    if joined is not None:
        for shared in joined[0]:
            digest.update(np.asarray(shared, bool).data)
        digest.update(np.asarray(joined[1], np.int64).data)
    return digest.digest()


def _factor(A):
    """The LU of ``A``, overwriting it, and whether it is worth keeping: as
    ``scipy.linalg.solve``, a singular matrix raises and an ill-conditioned one
    warns, and it is then not kept, for later calls not to lose that warning."""
    anorm = np.linalg.norm(A, 1)
    (getrf, gecon) = scipy.linalg.get_lapack_funcs(("getrf", "gecon"), (A,))
    lu, piv, info = getrf(A, overwrite_a=True)
    if info:
        msg = "The face-charge system is singular."
        raise scipy.linalg.LinAlgError(msg)
    rcond, _ = gecon(lu, anorm)
    if rcond < np.spacing(1.0):
        msg = f"Ill-conditioned face-charge system (rcond={rcond:.3g}): results may be inaccurate."
        warnings.warn(msg, scipy.linalg.LinAlgWarning, stacklevel=4)
        return lu, piv, False
    return lu, piv, True


def solve_kept(bodies, external, joined=None):
    """As :func:`~._bodies.solve_system`, keeping the factorization between
    calls. Returns the charge coefficients of each body, and whether a kept
    factorization solved them."""
    if not _limit:
        return _bodies.solve_system(bodies, external, joined), False
    key = _key(bodies, joined)
    needed = _bytes(bodies)
    with _lock:
        kept = _kept.get(key)
        if kept is not None:
            _kept.move_to_end(key)
        else:
            factor = key in _seen and needed <= _limit
            _seen[key] = True
            _seen.move_to_end(key)
            while len(_seen) > SEEN:
                _seen.popitem(last=False)
            # before the matrix is made, for it to fit beside what stays
            if needed > _limit:
                _kept.clear()
            elif factor:
                _evict(needed)
    if kept is not None:
        lu, piv = kept
        rhs = _bodies.load_system(bodies, external, len(lu), joined)
        return _bodies.split_unknowns(_back_substitute(lu, piv, rhs), bodies), True
    if not factor:
        return _bodies.solve_system(bodies, external, joined), False
    A = _bodies.assemble_system(bodies, joined)
    rhs = _bodies.load_system(bodies, external, len(A), joined)
    lu, piv, keep = _factor(A)
    if keep:
        with _lock:
            _kept[key] = lu, piv
            # evicted, it starts again from one sighting, so a cycle of more
            # than fit does not factor on every call
            _seen.pop(key, None)
            _evict()
    return _bodies.split_unknowns(_back_substitute(lu, piv, rhs), bodies), False


def _back_substitute(lu, piv, rhs):
    # its own pivots: scipy's lu_solve shifts them in place during the call, so
    # threads sharing them solve with wrong ones and can leave them wrong (#131)
    return scipy.linalg.lu_solve((lu, piv.copy()), rhs, check_finite=False)
