"""A factorization kept for ``solve_face_charge(..., reuse=...)``.

The matrix of a solve depends only on the solved parts, so a call whose parts
make the matrix of an earlier call takes only a new load and a
back-substitution. The fingerprint is what the matrix is assembled from, read off
the meshed bodies rather than the user's input: a magnet that comes close enough
to change a part's panels changes it by itself. Poses are relative to the first
body, as each block is in its body's own frame, so parts that move as one keep
it. What enters only the load stays out: the fixed sources, ``H_ext``, the
remanence, the finer rule near a magnet.

The caller holds the factorization, on the result: it goes when the result goes,
and nothing is kept between calls that did not ask.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import scipy.linalg

from magpylib_material_response.face_charge._bodies import (
    ALIGNED,
    PARALLEL,
    assemble_system,
    load_system,
    split_unknowns,
)
from magpylib_material_response.face_charge._tribody import TriBody

ROUND = 1e-12  # of the parts' size: a rigid motion's rounding must still match


class KeptMatrix:
    """The LU of one matrix and the fingerprint of what made it. Read-only once
    made, so threads and copies of a result share it."""

    def __init__(self, fingerprint, lu, piv):
        self.fingerprint = fingerprint
        self.lu = lu
        self.piv = piv

    def __deepcopy__(self, memo):  # a result's copy would double the matrix
        return self

    def solve(self, rhs):
        # its own pivots: scipy's lu_solve shifts them in place during the call, so
        # threads sharing them solve with wrong ones and can leave them wrong (#131)
        return scipy.linalg.lu_solve(
            (self.lu, self.piv.copy()), rhs, check_finite=False
        )


def _fingerprint(bodies, joined):
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
        add(b.half, size)  # a mesh's comes from all its vertices, used or not
        # its place against the first, as the coupling blocks see it
        add((b.position - first.position) @ first.R, size)
        add(first.R.T @ b.R, 1.0)
    # rounded poses can fall on both sides of these choices, and the matrix differs
    frames = np.stack([b.R for b in bodies])
    turn = np.abs(frames.transpose(0, 2, 1)[:, None] @ frames[None])
    digest.update((np.abs(turn - np.round(turn)).max(axis=(2, 3)) <= ALIGNED).tobytes())
    digest.update((turn > 1 - PARALLEL).tobytes())
    if joined is not None:
        for shared in joined[0]:
            digest.update(np.asarray(shared, bool).data)
        digest.update(np.asarray(joined[1], np.int64).data)
    return digest.digest()


def _factor(A):
    """The LU of ``A``, overwriting it, and whether it is worth keeping: as
    ``scipy.linalg.solve``, a singular matrix raises and an ill-conditioned one
    warns, and it is then not kept, for later calls not to lose that warning."""
    (lange, getrf, gecon) = scipy.linalg.get_lapack_funcs(
        ("lange", "getrf", "gecon"), (A,)
    )
    anorm = lange("1", A)  # no copy of A, as np.linalg.norm makes
    lu, piv, info = getrf(A, overwrite_a=True)
    if info:
        msg = "The face-charge system is singular."
        raise scipy.linalg.LinAlgError(msg)
    rcond, _ = gecon(lu, anorm)
    if not rcond >= np.spacing(1.0):  # NaN too
        msg = f"Ill-conditioned face-charge system (rcond={rcond:.3g}): results may be inaccurate."
        warnings.warn(msg, scipy.linalg.LinAlgWarning, stacklevel=4)
        return lu, piv, False
    return lu, piv, True


def solve_kept(bodies, external, joined, previous):
    """As :func:`~._bodies.solve_system`, on ``previous``'s factorization when these
    bodies make its matrix. Returns the charge coefficients of each body, the
    :class:`KeptMatrix` to hand on, or None if this matrix is not worth keeping,
    and whether ``previous`` solved them."""
    fingerprint = _fingerprint(bodies, joined)
    if previous is not None and previous.fingerprint == fingerprint:
        rhs = load_system(bodies, external, len(previous.lu), joined)
        return split_unknowns(previous.solve(rhs), bodies), previous, True
    A = assemble_system(bodies, joined)
    rhs = load_system(bodies, external, len(A), joined)
    lu, piv, sound = _factor(A)
    kept = KeptMatrix(fingerprint, lu, piv)
    return split_unknowns(kept.solve(rhs), bodies), kept if sound else None, False
