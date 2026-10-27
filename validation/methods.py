"""The methods, each an adapter: its settings from coarse to fine, which cases
it can solve, and one solve that times its phases apart.

A new method is one more adapter in ``METHODS``; from then on the runner
solves every case it can, against the same stored references.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

import magpylib as magpy
import numpy as np
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.face_charge import ChargeSheet, apply_face_charge
from magpylib_material_response.face_charge._apply import _split
from magpylib_material_response.meshing import mesh_uniform

from .cases import CASES


class TooLarge(Exception):
    """A setting asks for more than the method allows."""


def _same_solve(case, method):
    """Whether `case` shares another case's reference and `method`'s options,
    so that `method` would solve the same model as that case."""
    return case.reference is not None and case.options.get(method) == CASES[
        case.reference
    ].options.get(method)


@dataclass
class Solved:
    """B at the sensors, the time of each phase in s, and the size of the
    problem."""

    B: np.ndarray
    seconds: dict[str, float]
    counts: dict[str, int]


class Magpylib:
    """magpylib alone: the model as given, without its material response. The
    floor every method must beat. Soft parts carry no polarization until
    solved, so on a model of soft parts only it is 100 % off. One setting, one
    getB."""

    name = "magpylib"
    settings = (0,)
    growth = 1
    baseline = True

    def settings_for(self, case):  # noqa: ARG002
        return self.settings

    @staticmethod
    def can_solve(case):
        # it solves anything magpylib takes; refusing is the other methods' part
        return case.waits is None and case.refused is None

    @staticmethod
    def solve(collection, sensors, setting, **options):  # noqa: ARG004
        start = time.perf_counter()
        B = magpy.getB(collection, sensors)
        seconds = {"prepare": 0.0, "solve": 0.0, "getB": time.perf_counter() - start}
        return Solved(B, seconds, {"unknowns": 0})


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
        owner = CASES[case.reference] if case.reference else case
        return tuple(s for s in self.settings if s <= max(owner.reference_orders))

    @staticmethod
    def can_solve(case):
        return (
            case.waits is None
            and not _same_solve(case, "FEM")
            and importlib.util.find_spec("ngsolve") is not None
        )

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


class FaceCharge:
    """Face charge, ``apply_face_charge``: the charge on the cuboids' faces, no
    volume mesh. A setting is the panel level, the panels in each graded layer
    toward a face's edges. It solves cuboids with a uniform, isotropic
    susceptibility, beside any other source, coils included."""

    name = "FaceCharge"
    settings = (1, 2, 3, 4, 5)
    # time per level, measured on the three cuboids: 3 to 6
    growth = 5

    def settings_for(self, case):  # noqa: ARG002
        return self.settings

    @staticmethod
    def can_solve(case):
        if case.waits is not None or _same_solve(case, "FaceCharge"):
            return False
        if case.refused:
            return True  # its refusal is checked
        try:
            _split(case.model()[0])
        except NotImplementedError:
            return False
        return True

    @staticmethod
    def solve(collection, sensors, setting, **options):
        """`options` go to ``apply_face_charge``."""
        start = time.perf_counter()
        try:
            solved = apply_face_charge(collection, level=setting, **options)
        except ValueError as error:
            if "max_unknowns" not in str(error):
                raise
            raise TooLarge(str(error)) from error
        solved_at = time.perf_counter()
        B = magpy.getB(solved, sensors)
        sheets = [s for s in solved.sources_all if isinstance(s, ChargeSheet)]
        return Solved(
            B,
            {
                "prepare": 0.0,
                "solve": solved_at - start,
                "getB": time.perf_counter() - solved_at,
            },
            {"unknowns": sum(_unknowns(s) for s in sheets)},
        )


def _unknowns(sheet):
    """Three per panel: the charge is linear across each."""
    panels = sheet._panels
    return 3 * (len(sheet.faces) if panels is None else len(np.unique(panels)))


METHODS = {method.name: method for method in (Magpylib(), MoM(), FaceCharge(), FEM())}
