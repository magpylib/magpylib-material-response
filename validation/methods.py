"""The methods, each an adapter: its settings from coarse to fine, which cases
it can solve, and one solve that times its phases apart.

A new method is one more adapter in ``METHODS``; from then on the runner
solves every case it can, against the same stored references.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import magpylib as magpy
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_uniform

if TYPE_CHECKING:
    import numpy as np


class TooLarge(Exception):
    """A setting asks for more than the method allows."""


@dataclass
class Solved:
    """B at the sensors, the time of each phase in s, and the size of the
    problem."""

    B: np.ndarray
    seconds: dict[str, float]
    counts: dict[str, int]


class MoM:
    """Method of moments, ``apply_demag`` on uniform cells. A setting is the
    number of cells across the thinnest part that responds
    (:func:`~magpylib_material_response.meshing.mesh_uniform`); each level has
    8 times the cells of the one before."""

    name = "MoM"
    settings = (1, 2, 4, 8)
    # time per level, at most: 8 times the cells, solved densely when FFT
    # does not apply (curved cells)
    growth = 64
    max_cells = 20_000

    def settings_for(self, case):  # noqa: ARG002
        return self.settings

    @staticmethod
    def can_solve(case):
        return case.waits is None

    def solve(self, collection, sensors, setting, **options):
        """`options` go to ``apply_demag``."""
        start = time.perf_counter()
        try:
            meshed = mesh_uniform(collection, setting, max_cells=self.max_cells)
        except ValueError as error:
            if "max_cells" not in str(error):
                raise
            raise TooLarge(str(error)) from error
        meshed_at = time.perf_counter()
        solved = apply_demag(meshed, **options)
        solved_at = time.perf_counter()
        B = magpy.getB(solved, sensors)
        cells = sum(isinstance(s, BaseMagnet) for s in meshed.sources_all)
        return Solved(
            B,
            {
                "prepare": meshed_at - start,
                "solve": solved_at - meshed_at,
                "getB": time.perf_counter() - solved_at,
            },
            {"cells": cells, "unknowns": 3 * cells},
        )


class FEM:
    """Finite elements, ``fem_solve`` with the ``fem`` extra. A setting is the
    polynomial order on one mesh refined toward the edges and the sensors."""

    name = "FEM"
    settings = (2, 3, 4, 5)
    # time per order, measured: 1.3 to 3.4
    growth = 3

    def settings_for(self, case):
        """Orders up to the reference's highest: past it, FEM would be checked
        against a coarser solution of its own, and may not fit in memory."""
        return tuple(s for s in self.settings if s <= max(case.reference_orders))

    @staticmethod
    def can_solve(case):
        return case.waits is None and importlib.util.find_spec("ngsolve") is not None

    @staticmethod
    def solve(collection, sensors, setting, **options):
        """`options` go to ``fem_solve``."""
        from magpylib_material_response.fem import fem_solve  # noqa: PLC0415

        solution = fem_solve(collection, order=setting, observers=sensors, **options)
        start = time.perf_counter()
        B = solution.getB(sensors)
        return Solved(
            B,
            {
                "prepare": solution.seconds["mesh"],
                "solve": solution.seconds["assemble"] + solution.seconds["solve"],
                "getB": time.perf_counter() - start,
            },
            {"unknowns": int(solution.ndof)},
        )


METHODS = {method.name: method for method in (MoM(), FEM())}
