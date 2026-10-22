"""Mesh-convergence studies.

The method of moments gives no error estimate of its own: a result is as good as
its mesh. :func:`mesh_study` solves one model on meshes of increasing resolution
and estimates the error of the finest from how the results change between them.
"""

from __future__ import annotations

import time
import warnings
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


#: Orders of convergence in the cell count that the estimate accepts, this
#: package's choice: Celik et al. take any order, and the studies of uniform
#: cuboids in the docs' examples measure 0.43 to 0.63. An order outside this band
#: means the three finest results are not yet in that regime, or that the finest
#: happened to pass through the limit.
PLAUSIBLE_ORDERS = (0.1, 1.0)
#: Roache's factor of safety for an error estimated from three meshes (Roache
#: 1994; Celik et al. 2008, Eq. 7).
SAFETY_FACTOR = 1.25
#: Share of the last change that entries moving against the trend may carry
#: before the study counts as unsteady, this package's choice: Celik et al. report
#: such oscillating points and still estimate, but set no limit. With many
#: sensors some weak ones always turn: in a study of three magnets on 301
#: sensors, 48 of 903 components did, carrying 1 to 2 % of the change, while the
#: whole converged steadily.
UNSTEADY_SHARE = 0.1
#: Least ratio of cell counts between two meshes. Celik et al. ask for a cell
#: size at least 1.3 times smaller, "based on experience", with the size taken
#: as (volume / cells)**(1/3); that is 1.3**3 times the cells.
MIN_GROWTH = 1.3**3


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
        the three finest results do not change steadily at an order within
        ``PLAUSIBLE_ORDERS``; there is then no estimate.
    estimate : ndarray
        The value extrapolated to an infinitely fine mesh, or the finest value
        when ``order`` is None.
    error : float
        Estimated relative error of the finest mesh, ``|estimate - finest| /
        |finest|`` times ``SAFETY_FACTOR``: the grid convergence index of the
        finest mesh (see :func:`mesh_study`), with
        Euclidean norms over the whole quantity; NaN when ``order`` is None.
    errors : ndarray
        The same, one per entry of the quantity: per observer for fields at
        observers (an (m, 3) array), per element for anything else, so that a
        function returning a single vector gets one error per component.
    last_change : float
        Change between the two finest meshes, relative to the finest. It is no
        error bound: meshes that differ little change little without being
        converged.
    """

    cells: np.ndarray
    values: np.ndarray
    seconds: np.ndarray
    order: float | None
    estimate: np.ndarray
    error: float
    errors: np.ndarray
    last_change: float


def _sizes(value):
    """The size of each entry of a value: the norm of each row of an (m, 3)
    array, such as fields at observers, and the magnitude of each element of
    anything else."""
    value = np.asarray(value, dtype=float)
    if value.ndim >= 2 and value.shape[-1] == 3:
        return np.linalg.norm(value, axis=-1)
    return np.abs(value)


def _relative(change, reference):
    """``change / reference`` where the reference is not zero, NaN where it is."""
    return np.divide(
        change, reference, out=np.full(np.shape(change), np.nan), where=reference > 0
    )


def _gci(cells, values):
    """Fit ``f = f_inf - C * N**-p`` through the last three (cells, value) pairs.

    One order for the whole quantity, from the sizes of its two last changes;
    entries moving against the trend may carry ``UNSTEADY_SHARE`` of the last
    change. Returns (p, f_inf, error, errors), the errors relative to the finest
    value and times ``SAFETY_FACTOR``, or (None, last value, NaN, NaN) when the
    results do not change steadily at a plausible order.
    """
    n1, n2, n3 = (float(c) for c in cells[-3:])
    f1, f2, f3 = (np.asarray(v, dtype=float) for v in values[-3:])
    d1, d2 = f2 - f1, f3 - f2
    step = np.linalg.norm(d2)
    declined = (None, f3, np.nan, np.full(np.shape(_sizes(f3)), np.nan))
    if step == 0 or np.linalg.norm(d2[d1 * d2 < 0]) > UNSTEADY_SHARE * step:
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
    beyond = d2 * n3**-p / (n2**-p - n3**-p)  # the finest value's distance to the limit
    errors = SAFETY_FACTOR * _relative(_sizes(beyond), _sizes(f3))
    error = SAFETY_FACTOR * float(_relative(np.linalg.norm(beyond), np.linalg.norm(f3)))
    return p, f3 + beyond, error, errors


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
        Increasing cell targets, one mesh each, or whatever ``mesher`` takes as
        its second argument. Three or more give an error estimate; refine by a
        factor of 4 to 8 between meshes.
    mesher : callable, optional
        ``mesher(collection, target_elems)`` returning the meshed collection;
        :func:`~magpylib_material_response.meshing.mesh_all` by default.
    **demag_kwargs
        Passed on to :func:`~magpylib_material_response.demag.apply_demag`.

    Returns
    -------
    MeshStudy
        Cell counts, values and solve times per mesh, the observed order, the
        extrapolated value and the estimated error of the finest mesh, as a whole
        and per entry.

    Notes
    -----
    The estimate is the grid convergence index (Roache 1994) as Celik et al.
    (2008) set it out: Richardson extrapolation (Richardson 1911) over the three
    finest meshes, the error of the finest relative to its own value, times a
    factor of safety of 1.25. Where Celik et al. average the orders of the points
    into one global order, this takes one order from the sizes of the two last
    changes of the whole quantity.
    The estimate assumes the error falls as a power of the cell count, which
    holds once the meshes resolve the body's features. It is given only when the
    three finest results change steadily at a plausible order
    (``PLAUSIBLE_ORDERS``), entries moving against the trend carrying at most
    ``UNSTEADY_SHARE`` of the change; otherwise ``order`` is None and the errors
    NaN. Meshes that are too similar change little without being converged: grow
    the cell count by a factor of 4 to 8 between meshes. A warning says when it
    grew by less than ``MIN_GROWTH``, which a tetrahedral mesher may cause by
    ignoring targets below what a detailed surface needs. A function quantity
    should return one physical quantity, as its entries share one norm.

    References
    ----------
    - Richardson, L. F. (1911). The approximate arithmetical solution by finite
      differences of physical problems involving differential equations, with an
      application to the stresses in a masonry dam. Philos. Trans. R. Soc. A
      210, 307-357. doi:10.1098/rsta.1911.0009
    - Roache, P. J. (1994). Perspective: a method for uniform reporting of grid
      refinement studies. J. Fluids Eng. 116(3), 405-413. doi:10.1115/1.2910291
    - Celik, I. B., Ghia, U., Roache, P. J., & Freitas, C. J. (2008). Procedure
      for estimation and reporting of uncertainty due to discretization in CFD
      applications. J. Fluids Eng. 130(7), 078001. doi:10.1115/1.2960953
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
            return np.reshape(magpy.getB(solved, observers), (-1, 3))

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
    alike = [f"{a} to {b}" for a, b in pairwise(cells) if b < MIN_GROWTH * a]
    if alike:
        warnings.warn(
            f"The cell count grew only from {', '.join(alike)}: meshes this alike change "
            "little without being converged. A tetrahedral mesher may ignore targets "
            "below what a detailed surface needs.",
            stacklevel=2,
        )
    change = float(
        _relative(np.linalg.norm(values[-1] - values[-2]), np.linalg.norm(values[-1]))
    )
    if len(cells) < 3:
        nan = np.full(np.shape(_sizes(values[-1])), np.nan)
        return MeshStudy(cells, values, seconds, None, values[-1], np.nan, nan, change)
    order, estimate, error, errors = _gci(cells, values)
    return MeshStudy(cells, values, seconds, order, estimate, error, errors, change)
