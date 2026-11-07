"""Raise the panel level and estimate the error of the highest."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import magpylib as magpy
import numpy as np

from magpylib_material_response.convergence import (
    _evaluator,
    _last_change_estimate,
    _warn_no_estimate,
)
from magpylib_material_response.face_charge._apply import solve_face_charge


@dataclass(frozen=True)
class FaceChargeStudy:
    """Result of :func:`study_face_charge`: one model solved at increasing
    panel levels.

    Attributes
    ----------
    levels : ndarray, shape (k,)
        Panel level of each solve, increasing.
    unknowns : ndarray, shape (k,)
        Number of unknowns of each solve.
    values : ndarray, shape (k, ...)
        The quantity of interest at each level.
    seconds : ndarray, shape (k,)
        Wall time of each solve and evaluation.
    estimate : ndarray
        The value at the highest level.
    error : float
        Estimated relative error of the highest level: the last change relative
        to the highest-level value, times ``SAFETY_FACTOR`` of
        ``magpylib_material_response.convergence``, with Euclidean norms over
        the whole quantity. NaN unless the changes shrink (``ratio < 1``), with
        a warning.
    errors : ndarray
        The same per entry of the quantity: per observer for fields at
        observers, per element for anything else.
    ratio : float or None
        How much smaller the last change is than the one before, in Euclidean
        norm over the whole quantity; None with two levels.
    last_change : float
        Change between the two highest levels, relative to the highest.
    solution : magpylib.Collection
        The solve at the highest level.
    """

    levels: np.ndarray
    unknowns: np.ndarray
    values: np.ndarray
    seconds: np.ndarray
    estimate: np.ndarray
    error: float
    errors: np.ndarray
    ratio: float | None
    last_change: float
    solution: object


def study_face_charge(
    collection,
    quantity,
    levels=(2, 3, 4),
    *,
    refine=0,
    max_unknowns=20_000,
    min_log_time=None,
) -> FaceChargeStudy:
    """Solve `collection` by face charge at increasing panel levels and
    estimate the error of the highest.

    Parameters
    ----------
    collection : magpylib.Collection
        The model, as :func:`solve_face_charge` takes it.
    quantity : array_like, Sensor or callable
        Observers at which the flux density B is compared: positions of shape
        (..., 3) or magpylib sensors, as :func:`magpylib.getB` takes them. Or a
        function of the solved collection returning the quantity of interest,
        such as a force.
    levels : sequence of int, default (2, 3, 4)
        Panel levels, increasing. Three or more give an error estimate, two
        only a warning.
    refine, max_unknowns, min_log_time
        As for :func:`solve_face_charge`, at every level.

    Returns
    -------
    FaceChargeStudy
        Values, unknowns and solve times per level, and the estimated error of
        the highest level, as a whole and per entry. :func:`~magpylib_material_response.convergence.compare`
        puts it beside a study of another solver.

    Notes
    -----
    The panels are graded toward the faces' edges and each level adds a layer,
    so the error falls faster with the level than a power of the panel size
    that a Richardson extrapolation could fit (:func:`~magpylib_material_response.convergence.study_cells`
    declines every such series). The estimate is therefore the one of
    :func:`~magpylib_material_response.fem.study_fem`: the last change, times
    the factor of safety, as the error of the highest level, an overestimate
    while the error still falls. On the 18 cases of the repository's validation
    suite that face charge solves, levels 1-3, 2-4 and 3-5 gave an estimate
    above the error against the finite-element reference wherever that
    reference was accurate enough to tell, and on a soft cylinder at 200
    observers around it. A sphere is the exception: its facets are cut anew at
    each level, its error falls unevenly, and on a soft sphere in a uniform
    field the estimate fell short of the exact error by up to 2.6 times at
    levels 3-5; a warning says so. Fields at single observers do not converge
    monotonically, so the per-observer errors are indicative.
    """
    levels = list(levels)
    if len(levels) < 2 or any(int(m) != m or m < 1 for m in levels):
        msg = f"levels must be at least two positive integers; got {levels}."
        raise ValueError(msg)
    if np.any(np.diff(levels) <= 0):
        msg = f"levels must increase; got {levels}."
        raise ValueError(msg)
    levels = [int(m) for m in levels]
    if len(levels) < 3:  # said before the solves
        _warn_no_estimate("levels", 2, "a higher level", stacklevel=2)
    if any(isinstance(src, magpy.magnet.Sphere) for src in collection.sources_all):
        warnings.warn(
            "A sphere's facets are cut anew at each level and its error falls "
            "unevenly: the estimate can fall short, by up to 2.6 times on a soft "
            "sphere. Check it against study_fem.",
            stacklevel=2,
        )
    evaluate = _evaluator(quantity)
    values, unknowns, seconds = [], [], []
    for level in levels:
        start = time.perf_counter()
        solved = solve_face_charge(
            collection,
            level=level,
            refine=refine,
            max_unknowns=max_unknowns,
            min_log_time=min_log_time,
        )
        values.append(np.asarray(evaluate(solved), dtype=float))
        seconds.append(time.perf_counter() - start)
        unknowns.append(solved.face_charge_info["unknowns"])
    values = np.asarray(values)
    ratio, change, error, errors = _last_change_estimate(values)
    if ratio is not None and np.isnan(error):
        _warn_no_estimate("levels", len(levels), "a higher level", stacklevel=2)
    return FaceChargeStudy(
        levels=np.asarray(levels),
        unknowns=np.asarray(unknowns),
        values=values,
        seconds=np.asarray(seconds),
        ratio=ratio,
        estimate=values[-1],
        error=error,
        errors=errors,
        last_change=change,
        solution=solved,
    )
