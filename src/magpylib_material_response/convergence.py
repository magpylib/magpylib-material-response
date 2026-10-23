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
from magpylib_material_response.meshing import _responds, mesh_uniform

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


#: Orders of convergence that the estimate accepts, in the cell size
#: h = (volume / cells)**(1/3) as Celik et al. (2008, Eq. 1) define it. This
#: package's choice: Celik et al. take any order, and the studies of uniform
#: cuboids in the docs' examples measure 1.3 to 1.9. An order outside this band
#: means the three finest results are not yet in that regime, or that the finest
#: happened to pass through the limit.
PLAUSIBLE_ORDERS = (0.3, 3.0)
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
    """Result of a mesh study: of :func:`mesh_study` for this package's solver,
    or of :meth:`from_values` for results from another, such as FEM.

    Attributes
    ----------
    cells : ndarray, shape (k,)
        Number of cells (or elements) of each mesh that respond to fields,
        increasing.
    values : ndarray, shape (k, ...)
        The quantity of interest on each mesh.
    seconds : ndarray, shape (k,)
        Wall time of each solve, NaN where not given.
    order : float or None
        Observed order of convergence in the cell size h = (volume /
        cells)**(1/3): the error falls as ``h**order``, that is as
        ``cells**(-order / 3)``. None when the three finest results do not change
        steadily at an order within ``PLAUSIBLE_ORDERS``; there is then no
        estimate.
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

    @classmethod
    def from_values(cls, cells, values, seconds=None) -> MeshStudy:
        """Estimate the error of the finest of a series of results.

        Parameters
        ----------
        cells : sequence of int
            Number of cells (or elements) of each mesh, increasing. For a 3D
            mesh refined throughout, so that its cell size is (volume /
            cells)**(1/3).
        values : sequence of array_like
            The quantity on each mesh, one per cell count: B at observers as an
            (m, 3) array, or any array or number.
        seconds : sequence of float, optional
            Wall time of each solve.

        Returns
        -------
        MeshStudy
            The estimate as :func:`mesh_study` gives it; see its notes. Three or
            more results give an estimate; with two, ``order`` is None.
        """
        return _study(cells, values, seconds, stacklevel=3)


def _study(cells, values, seconds, stacklevel):
    """The checks, the warning and the estimate behind :meth:`MeshStudy.from_values`
    and :func:`mesh_study`; `stacklevel` points the warning at their caller."""
    cells, values = np.asarray(cells), np.asarray(values, dtype=float)
    if cells.ndim != 1 or len(cells) < 2 or np.any(np.diff(cells) <= 0):
        msg = "cells must hold at least two increasing counts."
        raise ValueError(msg)
    seconds = np.full(len(cells), np.nan) if seconds is None else np.asarray(seconds)
    if len(values) != len(cells) or len(seconds) != len(cells):
        msg = (
            f"Got {len(cells)} cell counts, {len(values)} values and {len(seconds)} "
            "times; give one of each per mesh."
        )
        raise ValueError(msg)
    alike = [f"{a} to {b}" for a, b in pairwise(cells) if b < MIN_GROWTH * a]
    if alike:
        warnings.warn(
            f"The cell count grew only from {', '.join(alike)}: meshes this alike "
            "change little without being converged. A tetrahedral mesher may ignore "
            "targets below what a detailed surface needs.",
            stacklevel=stacklevel,
        )
    change = float(
        _relative(np.linalg.norm(values[-1] - values[-2]), np.linalg.norm(values[-1]))
    )
    if len(cells) < 3:
        nan = np.full(np.shape(_sizes(values[-1])), np.nan)
        return MeshStudy(cells, values, seconds, None, values[-1], np.nan, nan, change)
    order, estimate, error, errors = _gci(cells, values)
    return MeshStudy(cells, values, seconds, order, estimate, error, errors, change)


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
    change. Returns (3 p, f_inf, error, errors), the order in the cell size, the
    errors relative to the finest value and times ``SAFETY_FACTOR``, or (None,
    last value, NaN, NaN) when the results do not change steadily at a
    plausible order.
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

    p_min, p_max = (p / 3 for p in PLAUSIBLE_ORDERS)  # in the cell count
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
    return 3 * p, f3 + beyond, error, errors


def mesh_study(
    collection,
    quantity,
    levels: Sequence,
    mesher: Callable | None = None,
    demag_kwargs: dict | None = None,
) -> MeshStudy:
    """Solve `collection` on meshes of increasing resolution and estimate the
    error of the finest.

    Parameters
    ----------
    collection : magpylib.Collection
        The model before meshing, with its susceptibilities, applied fields and
        current sources.
    quantity : array_like, Sensor or callable
        Observers at which the flux density B is compared: positions of shape
        (..., 3) or magpylib sensors, as :func:`magpylib.getB` takes them. Or a
        function of the solved collection returning the quantity of interest,
        such as a moment or a force.
    levels : sequence
        Refinement levels from coarse to fine, one mesh each, passed to
        ``mesher`` as its second argument. With the default mesher, cells across
        the thinnest part that responds: ``(2, 4, 8)`` gives meshes 8 times
        apart. Three or more give an error estimate; refine by a factor of 4 to 8
        in cells between meshes.
    mesher : callable, optional
        ``mesher(collection, level)`` returning the meshed collection;
        :func:`~magpylib_material_response.meshing.mesh_uniform` by default,
        which gives every part cells of one size.
    demag_kwargs : dict, optional
        Passed on to :func:`~magpylib_material_response.demag.apply_demag`, for
        example ``{"solver": "direct"}``. Not ``inplace`` or ``susceptibility``:
        every level solves a fresh mesh, whose cells carry the model's
        susceptibilities.

    Returns
    -------
    MeshStudy
        Counts of the cells that respond, values and solve times per mesh, the
        observed order, the extrapolated value and the estimated error of the
        finest mesh, as a whole and per entry.

    Notes
    -----
    The estimate is the grid convergence index (Roache 1994) as Celik et al.
    (2008) set it out: Richardson extrapolation (Richardson 1911) over the three
    finest meshes, the error of the finest relative to its own value, times a
    factor of safety of 1.25. Where Celik et al. average the orders of the points
    into one global order, this takes one order from the sizes of the two last
    changes of the whole quantity.
    The estimate assumes the error falls as a power of the cell size, which
    holds once the meshes resolve the body's features. Cells that do not respond
    to fields, such as a whole permanent magnet, are not counted. It is given only when the
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
    levels = list(levels)
    if len(levels) < 2:
        msg = "levels must hold at least two values."
        raise ValueError(msg)
    demag_kwargs = dict(demag_kwargs or {})
    clash = sorted({"inplace", "susceptibility"} & set(demag_kwargs))
    if clash:
        msg = (
            f"mesh_study cannot pass {clash} to apply_demag: every level solves a "
            "fresh mesh, whose cells carry the model's susceptibilities."
        )
        raise ValueError(msg)
    if mesher is None:
        mesher = mesh_uniform
    if callable(quantity):
        evaluate = quantity
    else:

        def evaluate(solved):
            field = np.asarray(magpy.getB(solved, quantity), dtype=float)
            return field.reshape(1, 3) if field.ndim == 1 else field

    cells, values, seconds = [], [], []
    for level in levels:
        meshed = mesher(collection, level)
        n = sum(
            isinstance(src, BaseMagnet) and _responds(src) for src in meshed.sources_all
        )
        if cells and n <= cells[-1]:
            msg = (
                f"Level {level} gave {n} cells, not more than the previous mesh's "
                f"{cells[-1]}; order the levels from coarse to fine, further apart."
            )
            raise ValueError(msg)
        start = time.perf_counter()
        solved = apply_demag(meshed, **demag_kwargs)
        seconds.append(time.perf_counter() - start)
        cells.append(n)
        values.append(np.asarray(evaluate(solved), dtype=float))

    return _study(cells, values, seconds, stacklevel=3)
