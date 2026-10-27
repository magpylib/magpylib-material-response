"""Finite-element references, one file per case, made once and stored.

    python -m validation.references                  # the cases without a file
    python -m validation.references plate            # these, even if they exist
    python -m validation.references --check          # make them all again, compare
    python -m validation.references --check plate    # these only
    python -m validation.references --check --having CylinderSegment

Needs the fem extra. Each case is solved by ``fem_study`` at its reference
orders (3, 4 and 5 unless the case says otherwise) on one mesh whose elements at
the sensors are half the default size, and once more at the highest order on
the default mesh, the mesh the FEM method runs on. Each
group's stored error is the larger of the change between the two highest
orders and the change between the two meshes, so it covers a mesh too coarse
near an edge, where raising the order hides it. A reference depends on the
case, not on this package's code, so it stays valid until the case changes;
checks compare against the file and need no NGSolve.

A case may keep a reference made elsewhere, with the components it gives and
how it was made, which this module never overwrites: ``ucore.json``, Bz on the
docs' line from the two-potential FEM of issue #76 (scikit-fem), an independent
check of fem's currents.

``--check`` makes every reference of this module again, without writing it, and
fails when one moved by more than its stored error and its new one together: a
change in fem or in NGSolve, which the nightly run catches.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import magpylib as magpy
import numpy as np

from magpylib_material_response.convergence import SAFETY_FACTOR

from .cases import CASES, select

FOLDER = Path(__file__).parent / "references"
#: element size at the sensors against the default, for the reference mesh
FINER = 0.5


def path(case):
    """The reference file of `case`, or of the case it shares one with."""
    return FOLDER / f"{case.reference or case.name}.json"


@contextlib.contextmanager
def _finer_at_sensors(factor):
    """Elements at the observers `factor` times their default size."""
    from magpylib_material_response.fem import _solve  # noqa: PLC0415

    share, floor = _solve.OBSERVER_SHARE, _solve.OBSERVER_FLOOR
    _solve.OBSERVER_SHARE, _solve.OBSERVER_FLOOR = share * factor, floor * factor
    try:
        yield
    finally:
        _solve.OBSERVER_SHARE, _solve.OBSERVER_FLOOR = share, floor


def make(case):
    """Solve `case` by finite elements and write its reference."""
    data = _solve(case)
    FOLDER.mkdir(exist_ok=True)
    temporary = path(case).with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    temporary.replace(path(case))
    return data


def check(case):
    """Solve `case` again and compare with its stored reference: per group, how
    far B moved, ‖ΔB‖ / ‖B‖, and how far it may move, the stored error and
    the new one together."""
    stored, new = load(case), _solve(case)
    moved = {}
    for group, old in stored["groups"].items():
        B = np.array(new["groups"][group]["B"])
        offset = float(np.linalg.norm(B - old["B"]) / np.linalg.norm(old["B"]))
        moved[group] = (offset, old["error"] + new["groups"][group]["error"])
    return moved


def _solve(case):
    """The reference of `case` by finite elements, as ``make`` writes it."""
    import ngsolve  # noqa: PLC0415

    from magpylib_material_response.fem import fem_solve, fem_study  # noqa: PLC0415

    if case.reference or case.refused or case.waits:
        msg = f"{case.name!r} has no reference of its own to make."
        raise ValueError(msg)
    if path(case).exists() and "made_by" in json.loads(path(case).read_text()):
        msg = (
            f"The reference of {case.name!r} was made elsewhere and is kept: "
            "remove its file first to make one by fem."
        )
        raise ValueError(msg)
    collection, groups = case.model()
    sensors = np.concatenate(list(groups.values()))
    orders, options = case.reference_orders, case.options.get("FEM", {})
    start = time.perf_counter()
    with _finer_at_sensors(FINER):
        study = fem_study(collection, sensors, orders=orders, **options)
    if not np.isfinite(study.error):
        msg = f"{case.name}: orders {orders} do not converge on the reference mesh."
        raise RuntimeError(msg)
    coarse = fem_solve(collection, order=orders[-1], observers=sensors, **options).getB(
        sensors
    )
    seconds = time.perf_counter() - start
    change = study.values[-1] - study.values[-2]
    stored, first = {}, 0
    for group, points in groups.items():
        rows = slice(first, first + len(points))
        first += len(points)
        B = study.estimate[rows]
        size = np.linalg.norm(B)
        if size == 0:
            msg = f"{case.name}, {group}: B is zero there; nothing to compare."
            raise ValueError(msg)
        by_order = SAFETY_FACTOR * np.linalg.norm(change[rows]) / size
        by_mesh = np.linalg.norm(coarse[rows] - B) / size
        stored[group] = {
            "B": B.tolist(),
            "error": float(max(by_order, by_mesh)),
            "order_change": float(by_order),
            "mesh_change": float(by_mesh),
        }
    return {
        "case": case.name,
        "hash": case.hash(),
        "orders": list(orders),
        "options": options,
        "sensor_elements": FINER,
        "unknowns": study.ndofs.tolist(),
        "seconds": round(seconds, 1),
        "versions": {"ngsolve": ngsolve.__version__, "magpylib": magpy.__version__},
        "groups": stored,
    }


def load(case):
    """The stored reference of `case`: B by group as arrays, and its errors.

    Raises
    ------
    FileNotFoundError
        If the case has no reference yet.
    ValueError
        If the case changed since its reference was made.
    """
    data = json.loads(path(case).read_text())
    if data["hash"] != case.hash():
        owner = case.reference or case.name
        msg = (
            f"The reference of {owner!r} was made for another version of the "
            f"case; run python -m validation.references {owner}."
        )
        raise ValueError(msg)
    for group in data["groups"].values():
        group["B"] = np.array(group["B"])
        group.setdefault("components", [0, 1, 2])
    return data


def _own(case):
    """Whether this module makes the reference of `case`: one of its own, not
    shared, refused, waiting or made elsewhere."""
    if case.reference or case.refused or case.waits:
        return False
    return not (path(case).exists() and "made_by" in json.loads(path(case).read_text()))


def _check_all(cases):
    """Make the references of `cases` again, those this module made; 1 if one
    moved too far or could not be made. A case that fails does not stop the
    others."""
    failed = 0
    mine = [case for case in cases if _own(case) and path(case).exists()]
    if not mine:
        names = ", ".join(case.name for case in cases)
        print(f"None of {names} has a reference of this module's to check.")
    for case in mine:
        try:
            moved = check(case)
        except Exception as error:  # noqa: BLE001  # report it, check the rest
            failed += 1
            print(
                f"{case.name:<14} FAILED: {type(error).__name__}: {error}", flush=True
            )
            continue
        for group, (offset, allowed) in moved.items():
            verdict = "ok" if offset <= allowed else "MOVED"
            failed += not offset <= allowed
            print(
                f"{case.name:<14} {group:<26} moved {offset:.1e}, "
                f"may move {allowed:.1e}  {verdict}",
                flush=True,
            )
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--having", nargs="+", default=[])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if importlib.util.find_spec("ngsolve") is None:
        msg = "Finite elements need the fem extra: pip install -e '.[fem]'."
        raise ValueError(msg)
    if args.check:
        return _check_all(select(args.cases, args.having))
    if args.cases:
        cases = select(args.cases, args.having)
    elif args.having:
        cases = [c for c in select(features=args.having) if _own(c)]
    else:
        cases = [
            c
            for c in CASES.values()
            if _own(c) and "FEM" not in c.known_failures and not path(c).exists()
        ]
    for case in cases:
        data = make(case)
        errors = ", ".join(f"{g} {v['error']:.1e}" for g, v in data["groups"].items())
        print(f"{case.name}: {data['seconds']} s; {errors}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
