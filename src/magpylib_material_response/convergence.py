"""Mesh-convergence studies.

The method of moments gives no error estimate of its own: a result is as good as
its mesh. :func:`mesh_study` solves one model on meshes of increasing resolution
and estimates the error of the finest from how the results change between them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import magpylib as magpy
import numpy as np
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_all

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


#: Orders of convergence in the cell count that the estimate accepts: piecewise
#: uniform cells converge between about h**1 and h**2 (0.33 to 0.67). An order
#: outside this band means the three finest results are not yet in that regime,
#: or that the finest happened to pass through the limit.
PLAUSIBLE_ORDERS = (0.1, 1.0)


@dataclass(frozen=True)
class MeshStudy:
    """Result of :func:`mesh_study`.

    Attributes
    ----------
    cells : ndarray, shape (k,)
        Number of magnet cells of each mesh, increasing.
    values : ndarray, shape (k, ...)
        The quantity of interest on each mesh.
    seconds : ndarray, shape (k,)
        Wall time of each solve.
    order : float or None
        Observed convergence rate: the error falls as ``cells**-order``. None when
        the three finest results do not change steadily, component by component,
        at an order within ``PLAUSIBLE_ORDERS``; there is then no estimate.
    estimate : ndarray
        The value extrapolated to an infinitely fine mesh, or the finest value
        when ``order`` is None.
    error : float
        Estimated error of the finest mesh, ``|estimate - values[-1]|`` relative
        to the largest ``|value|`` of the study (Euclidean norms); NaN when
        ``order`` is None.
    last_change : float
        Change between the two finest meshes, relative in the same way. It is no
        error bound: meshes that differ little change little without being
        converged.
    """

    cells: np.ndarray
    values: np.ndarray
    seconds: np.ndarray
    order: float | None
    estimate: np.ndarray
    error: float
    last_change: float


def _richardson(cells, values):
    """Fit ``f = f_inf - C * N**-p`` through the last three (cells, value) pairs.

    Returns (p, f_inf, error) with the error of the last value relative to the
    largest value, or (None, last value, NaN) when the changes do not shrink
    steadily, in every component, at a plausible order.
    """
    n1, n2, n3 = (float(c) for c in cells[-3:])
    f1, f2, f3 = (np.asarray(v, dtype=float) for v in values[-3:])
    d1, d2 = f2 - f1, f3 - f2
    scale = max(np.linalg.norm(np.asarray(v, dtype=float)) for v in values)
    step = np.linalg.norm(d2)
    declined = (None, f3, np.nan)
    if scale == 0 or step == 0:
        return declined
    # Every component that still moves must move the same way twice.
    moving = np.abs(d2) > 1e-12 * scale
    if not np.all(d1[moving] * d2[moving] > 0):
        return declined
    s = np.linalg.norm(d1) / step

    def ratio(p):
        return (n1**-p - n2**-p) / (n2**-p - n3**-p)

    p_min, p_max = PLAUSIBLE_ORDERS
    if not ratio(p_min) <= s <= ratio(p_max):
        return declined
    lo, hi = p_min, p_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(mid) < s else (lo, mid)
    p = 0.5 * (lo + hi)
    f_inf = f3 + d2 * n3**-p / (n2**-p - n3**-p)
    return p, f_inf, float(np.linalg.norm(f_inf - f3) / scale)


def mesh_study(
    collection,
    quantity,
    target_elems: Sequence[int],
    mesher: Callable | None = None,
    **demag_kwargs,
) -> MeshStudy:
    """Solve `collection` on meshes of increasing resolution and estimate the
    error of the finest.

    Parameters
    ----------
    collection : magpylib.Collection
        The model before meshing, with its susceptibilities, applied fields and
        current sources.
    quantity : array_like or callable
        Observer positions, shape (m, 3), at which the flux density B is compared;
        or a function of the solved collection returning the quantity of interest,
        such as a moment or a force.
    target_elems : sequence of int
        Increasing cell targets, one mesh each. Three or more give an error
        estimate; refine by a factor of 2 or more between meshes.
    mesher : callable, optional
        ``mesher(collection, target_elems)`` returning the meshed collection;
        :func:`~magpylib_material_response.meshing.mesh_all` by default.
    **demag_kwargs
        Passed on to :func:`~magpylib_material_response.demag.apply_demag`.

    Returns
    -------
    MeshStudy
        Cell counts, values and solve times per mesh, the observed order, the
        extrapolated value and the estimated error of the finest mesh.

    Notes
    -----
    The estimate assumes the error falls as a power of the cell count, which
    holds once the meshes resolve the body's features. It is given only when
    every component of the three finest results changes steadily in one direction
    at a plausible order (``PLAUSIBLE_ORDERS``); otherwise ``order`` is None and
    ``error`` NaN. Meshes that are too similar (for example 640 and 768 cells)
    change little without being converged, which is why the targets should grow
    by a factor of 2 or more. One error figure covers all components, weighted by
    their size: the relative error of a weak component can be larger.
    """
    targets = [int(t) for t in target_elems]
    if len(targets) < 2 or any(b <= a for a, b in pairwise(targets)):
        msg = "target_elems must hold at least two increasing values."
        raise ValueError(msg)
    if mesher is None:
        mesher = mesh_all
    if callable(quantity):
        evaluate = quantity
    else:
        observers = np.asarray(quantity, dtype=float)

        def evaluate(solved):
            return magpy.getB(solved, observers)

    cells, values, seconds = [], [], []
    for target in targets:
        meshed = mesher(collection, target)
        n = sum(isinstance(src, BaseMagnet) for src in meshed.sources_all)
        if cells and n <= cells[-1]:
            msg = (
                f"target_elems={target} gave {n} cells, not more than the previous "
                f"mesh's {cells[-1]}; use targets further apart."
            )
            raise ValueError(msg)
        start = time.perf_counter()
        solved = apply_demag(meshed, **demag_kwargs)
        seconds.append(time.perf_counter() - start)
        cells.append(n)
        values.append(np.asarray(evaluate(solved), dtype=float))

    cells, values, seconds = np.array(cells), np.array(values), np.array(seconds)
    scale = max(np.linalg.norm(v) for v in values)
    change = np.linalg.norm(values[-1] - values[-2]) / scale if scale else 0.0
    if len(cells) < 3:
        return MeshStudy(
            cells, values, seconds, None, values[-1], np.nan, float(change)
        )
    order, estimate, error = _richardson(cells, values)
    return MeshStudy(cells, values, seconds, order, estimate, error, float(change))
