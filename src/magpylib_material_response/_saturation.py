"""A material's B-H curve, and the iteration that puts each cell on it.

A ``susceptibility`` can be a :class:`BHCurve` in place of a number, on a magnet
or a parent collection. A cell on a curve takes the secant
susceptibility ``chi = J / (mu0 H)`` at the field inside it, the way Radia
treats non-linear iron: repeat the linear solve, read each cell's field, move
its chi onto the curve, until no chi changes by more than ``TOL``. Anderson
acceleration (depth ``DEPTH``) of that fixed point cuts the solves 3 to 4 times.
"""

from __future__ import annotations

import warnings

import magpylib as magpy
import numpy as np
from loguru import logger
from scipy.interpolate import PchipInterpolator

MU0 = magpy.mu_0
TOL = 1e-4  # relative change of any cell's chi at which the iteration stops
DEPTH = 6  # Anderson's memory
DAMPING = 0.7
MAX_SOLVES = 200
J_MAX = 5.0  # T: past any material, so a curve above it has its columns or units wrong
CHI_MIN = 1e-6  # the least susceptibility a cell on a curve takes: log chi must exist


class BHCurve:
    """A material's B-H curve, to set as a ``susceptibility``, on a magnet or a
    parent collection.

    Parameters
    ----------
    H : array-like, shape (n,)
        Field strength in A/m, from 0, increasing.
    B : array-like, shape (n,)
        Flux density in T, from 0, not decreasing.

    Between the points B follows a monotone cubic (PCHIP). Past the last, the
    polarization ``J = B - mu0 H`` keeps its last value, so B rises with the
    slope of vacuum. A curve does not change, so copies of a model share it.

    Examples
    --------
    >>> import magpylib_material_response as mmr
    >>> curve = mmr.BHCurve(H=[0, 100, 1000, 1e5], B=[0, 0.5, 1.4, 1.8])
    >>> round(float(curve.polarization(1000)), 3)
    1.399
    """

    def __init__(self, *, H, B):
        H, B = np.array(H, float), np.array(B, float)
        if H.ndim != 1 or H.shape != B.shape or len(H) < 2:
            msg = (
                "BHCurve needs H and B of one length, at least two points; got "
                f"shapes {H.shape} and {B.shape}."
            )
            raise ValueError(msg)
        if not (np.isfinite(H).all() and np.isfinite(B).all()) or H[0] or B[0]:
            msg = "A B-H curve must start at (0, 0) and hold finite numbers."
            raise ValueError(msg)
        if (np.diff(H) <= 0).any() or (np.diff(B) < 0).any():
            msg = "A B-H curve needs H increasing and B not decreasing."
            raise ValueError(msg)
        J = B - MU0 * H
        # a point on vacuum, written with another value of mu0, is rounding
        if (-1e-9 * np.maximum(MU0 * H, 1e-300) > J).any():
            msg = "The B-H curve lies below vacuum (B < mu0 H) somewhere: the polarization would be negative."
            raise ValueError(msg)
        J = np.maximum(J, 0.0)
        if not (J > 0).any():
            msg = "The B-H curve is vacuum everywhere (B = mu0 H): the part would not respond."
            raise ValueError(msg)
        if J.max() > J_MAX:
            warnings.warn(
                f"This B-H curve reaches a polarization of {J.max():.3g} T, past any "
                "material: H must be in A/m and B in T. Check that they are not "
                "swapped, and the units.",
                stacklevel=2,
            )
        H.flags.writeable = B.flags.writeable = False
        self.H, self.B = H, B
        self._J = PchipInterpolator(H, J, extrapolate=False)
        self.J_last = float(J[-1])
        self.H_last = float(H[-1])
        # the initial susceptibility, from the curve's slope at the origin; a
        # curve that starts flat, as a steel's can, starts from CHI_MIN instead
        self.chi0 = float(max(self._J(0.0, 1) / MU0, CHI_MIN))
        # the largest secant the curve can give, for what is built once
        with np.errstate(divide="ignore", invalid="ignore"):
            secants = np.where(H > 0, J / (MU0 * H), 0.0)
        self.chi_max = float(max(self.chi0, secants.max()))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):  # scipy's interpolator does not pickle
        return _restore, (self.H, self.B)

    def __repr__(self):
        return f"BHCurve({len(self.H)} points, J up to {self.J_last:.3g} T)"

    def polarization(self, H):
        """``J`` in T at the field strengths ``H`` in A/m."""
        H = np.asarray(H, float)
        J = self._J(np.minimum(H, self.H_last))
        return np.where(self.H_last <= H, self.J_last, J)

    def chi(self, mu0H):
        """The secant susceptibility ``J / (mu0 H)`` at ``mu0H`` in T, ``chi0``
        at zero field."""
        mu0H = np.asarray(mu0H, float)
        small = mu0H <= 0
        safe = np.where(small, 1.0, mu0H)
        chi = np.where(small, self.chi0, self.polarization(safe / MU0) / safe)
        return np.maximum(chi, CHI_MIN)


def _restore(H, B):
    return BHCurve(H=H, B=B)


def is_table(value):
    """Whether a susceptibility has the shape of a table of (H, B) rows."""
    try:
        shape = np.shape(value)
    except ValueError:  # entries of different shapes, as one per cell may be
        return False
    return len(shape) == 2 and shape[1] == 2


def refuse_table(value):
    """Raise if a susceptibility is a table: columns are easily swapped."""
    if is_table(value):
        msg = (
            "A susceptibility cannot be a table: give the B-H curve as "
            "BHCurve(H=H, B=B), with H in A/m and B in T."
        )
        raise TypeError(msg)


def refuse_bh_curve(source):
    """Raise if `source` or a parent has ``bh_curve``, which nothing reads any
    more: a magnet takes any attribute, so the part would silently be linear."""
    while source is not None:
        if getattr(source, "bh_curve", None) is not None:
            msg = (
                f"{source} has bh_curve: set its susceptibility to BHCurve(H=H, B=B) "
                "instead, with H in A/m and B in T."
            )
            raise ValueError(msg)
        source = source.parent


def curve_of(source):
    """The :class:`BHCurve` of `source`, if the nearest susceptibility on it or its
    parents is one, or None."""
    refuse_bh_curve(source)
    while source is not None:
        value = getattr(source, "susceptibility", None)
        if value is not None:
            refuse_table(value)
            return value if isinstance(value, BHCurve) else None
        source = source.parent
    return None


def iterate(solve, field, on_curve, curves, chi):
    """Solve until every cell on a curve has its chi at its own field.

    ``solve(chi)`` returns the polarizations ``(n, 3)`` for the (n, 3, 3) cell
    susceptibilities ``chi``, and ``field(J)`` the mu0 H (n, 3) in T they give at
    each cell; ``on_curve`` indexes the cells with a curve, in ``curves``. The
    field is read from the operator, not as J / chi: at a small chi, as on a curve
    that starts flat, J lies below what an iterative solve resolves beside a magnet
    of 1 T, and a part stayed at J = 0, its field read as 0, converged. Returns the
    polarizations, and the number of solves."""
    x = np.log(np.array([c.chi0 for c in curves]))
    # within what the curves can give: Anderson's steps can overshoot, and on a
    # steep curve chi then leaves what floating point can solve
    low, high = np.log(CHI_MIN), np.log(2 * np.array([c.chi_max for c in curves]))
    X, F = [], []
    for solves in range(1, MAX_SOLVES + 1):
        chi[on_curve] = np.exp(x)[:, None, None] * np.eye(3)
        J = solve(chi)
        mu0H = np.linalg.norm(field(J)[on_curve], axis=1)
        f = np.log(np.array([c.chi(h) for c, h in zip(curves, mu0H, strict=True)])) - x
        if np.max(np.abs(np.expm1(f))) < TOL:
            logger.info(
                "Saturation: {k} cells on B-H curves, converged in {solves} solves",
                k=len(x),
                solves=solves,
            )
            return J, solves
        X, F = [*X, x.copy()][-DEPTH:], [*F, f.copy()][-DEPTH:]
        if len(F) == 1:
            x = np.clip(x + DAMPING * f, low, high)
            continue
        dF, dX = np.diff(F, axis=0).T, np.diff(X, axis=0).T
        gamma = np.linalg.lstsq(dF, f, rcond=None)[0]
        x = np.clip(x + DAMPING * f - (dX + DAMPING * dF) @ gamma, low, high)
    msg = (
        f"The B-H iteration did not converge in {MAX_SOLVES} solves: the largest "
        f"change of a cell's susceptibility is still {np.max(np.abs(np.expm1(f))):.1e}. "
        "A sharp knee can do this: give it more points, or a gentler rise."
    )
    raise RuntimeError(msg)
