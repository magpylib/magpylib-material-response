"""Raise the order on one mesh and estimate the error of the highest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magpylib_material_response.convergence import (
    SAFETY_FACTOR,
    _relative,
    _sizes,
    _warn_no_estimate,
)
from magpylib_material_response.fem._solve import Model


@dataclass(frozen=True)
class FemStudy:
    """Result of :func:`fem_study`: one model solved at increasing orders.

    Attributes
    ----------
    orders : ndarray, shape (k,)
        Polynomial order of each solve, increasing.
    ndofs : ndarray, shape (k,)
        Number of unknowns of each solve.
    values : ndarray, shape (k, ...)
        The quantity of interest at each order.
    seconds : ndarray, shape (k,)
        Wall time of each solve, assembly included, mesh excluded.
    mesh_seconds : float
        Wall time of the geometry and the mesh, made once for all orders.
    ratio : float or None
        How much smaller the last change is than the one before, in Euclidean
        norm over the whole quantity; None with two orders.
    estimate : ndarray
        The value at the highest order.
    error : float
        Estimated relative error of the highest order: the last change
        relative to the highest-order value, times ``SAFETY_FACTOR``, with
        Euclidean norms over the whole quantity. NaN unless the changes shrink
        (``ratio < 1``), with a warning.
    errors : ndarray
        The same per entry of the quantity: per observer for fields at
        observers, per element for anything else.
    last_change : float
        Change between the two highest orders, relative to the highest.
    solution : FemSolution
        The solve at the highest order. Its moment is as good as the study's;
        fields at points that were not observers come from a mesh not refined
        there and carry no estimate, so pass every point of interest as an
        observer.
    """

    orders: np.ndarray
    ndofs: np.ndarray
    values: np.ndarray
    seconds: np.ndarray
    mesh_seconds: float
    ratio: float | None
    estimate: np.ndarray
    error: float
    errors: np.ndarray
    last_change: float
    solution: object


def fem_study(
    collection,
    quantity,
    orders=(2, 3, 4),
    *,
    maxh=None,
    edge_layers=4,
    observer_maxh=None,
    solver_tol=1e-10,
    max_iter=5000,
    min_log_time=None,
) -> FemStudy:
    """Solve `collection` by finite elements at increasing orders on one mesh
    and estimate the error of the highest.

    Parameters
    ----------
    collection : magpylib.Collection
        The model, as :func:`fem_solve` takes it.
    quantity : array_like or callable
        Positions (..., 3) in m at which the flux density B is compared; the
        mesh is refined there. Or a function of a :class:`FemSolution`
        returning the quantity of interest, such as ``lambda s: s.moment``.
    orders : sequence of int, default (2, 3, 4)
        Polynomial orders, increasing. Three or more give an error estimate,
        two only a warning.
    maxh, edge_layers, observer_maxh, solver_tol, max_iter, min_log_time
        As for :func:`fem_solve`.

    Returns
    -------
    FemStudy
        Values, unknowns and solve times per order, and the estimated error of
        the highest order, as a whole and per entry.

    Notes
    -----
    The error of an order is about its difference to the next, as long as the
    error falls fast with the order: geometrically on a mesh refined
    geometrically toward the edges (Babuška and Guo 1996), until the finest
    layer limits it. The estimate takes that difference, the last change, as
    the error of the highest order too, times the factor of safety
    ``SAFETY_FACTOR`` of ``magpylib_material_response.convergence``: an
    overestimate while the error still falls. It extrapolates nothing, because
    the rate slows as the layers start to limit: on a cube at orders 3, 4, 5
    the last ratio of changes, 0.09, would put the remaining error of the
    moment at 5e-6, where it is 2e-5. Fields at single observers do not
    converge monotonically, so the per-observer errors are indicative. Close
    to an edge the error falls slowly with the order, so a small change can
    hide a mesh too coarse there: 0.1 mm from a cube's edge, orders 4 and 5
    agreed to well under 1 mT on elements of 0.25 mm, which were 1 mT off.
    Check a reference once more with half the ``observer_maxh``.

    References
    ----------
    - Babuška, I., & Guo, B. Q. (1996). Approximation properties of the h-p
      version of the finite element method. Comput. Methods Appl. Mech. Eng.
      133(3-4), 319-346.
    """
    if len(orders) < 2 or any(int(p) != p or p < 1 for p in orders):
        msg = f"orders must be at least two positive integers; got {orders}."
        raise ValueError(msg)
    if np.any(np.diff(orders) <= 0):
        msg = f"orders must increase; got {orders}."
        raise ValueError(msg)
    orders = [int(p) for p in orders]
    if len(orders) < 3:  # said before the solves, which may take minutes
        _warn_no_estimate("orders", 2, "a higher order", stacklevel=2)
    if callable(quantity):
        evaluate, observers = quantity, None
    else:
        # one row per observer, so that errors are per observer, a lone one too
        observers = np.asarray(quantity, dtype=float).reshape(-1, 3)

        def evaluate(solution):
            return solution.getB(observers)

    model = Model(
        collection,
        observers=observers,
        observer_maxh=observer_maxh,
        maxh=maxh,
        edge_layers=edge_layers,
        min_log_time=min_log_time,
    )
    values, ndofs, seconds = [], [], []
    for order in orders:
        solution = model.solve(order, solver_tol=solver_tol, max_iter=max_iter)
        values.append(np.asarray(evaluate(solution), dtype=float))
        ndofs.append(solution.ndof)
        seconds.append(solution.seconds["assemble"] + solution.seconds["solve"])
    values = np.asarray(values)
    last = values[-1]
    ratio, change, error, errors = _estimate(values)
    if ratio is not None and np.isnan(error):
        _warn_no_estimate("orders", len(orders), "a higher order", stacklevel=2)
    return FemStudy(
        orders=np.asarray(orders),
        ndofs=np.asarray(ndofs),
        values=values,
        seconds=np.asarray(seconds),
        mesh_seconds=model.mesh_seconds,
        ratio=ratio,
        estimate=last,
        error=error,
        errors=errors,
        last_change=change,
        solution=solution,
    )


def _estimate(values):
    """(ratio, last change, error, errors) from the two last changes: the last
    change, relative to the last value, as the error when it is smaller than the
    one before; NaN errors otherwise, and with two values a ratio of None."""
    last, step = values[-1], values[-1] - values[-2]
    change = float(_relative(np.linalg.norm(step), np.linalg.norm(last)))
    errors = SAFETY_FACTOR * _relative(_sizes(step), _sizes(last))
    nan = np.full(np.shape(errors), np.nan)
    if len(values) < 3:
        return None, change, np.nan, nan
    before = np.linalg.norm(values[-2] - values[-3])
    if before == 0:  # no change at all: exact, or exactly converged
        ratio = 0.0 if np.linalg.norm(step) == 0 else np.inf
    else:
        ratio = float(np.linalg.norm(step) / before)
    if ratio >= 1:
        return ratio, change, np.nan, nan
    return ratio, change, SAFETY_FACTOR * change, errors
