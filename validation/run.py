"""Solve cases with methods, compare B with the stored references and time it.

    python -m validation.run                        # every case and method
    python -m validation.run --cases plate --methods MoM --settings 1 2
    python -m validation.run --pr                   # the settings of pr_check.json
    python -m validation.run --pr --write           # store their results there
    python -m validation.run --out records.jsonl    # append one record per solve

A method climbs its settings from coarse to fine and stops before a setting
expected to take longer than ``--budget`` seconds: the last time times the
method's ``growth``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import magpylib as magpy
import numpy as np

import magpylib_material_response

from . import references
from .cases import CASES
from .methods import METHODS, TooLarge

PR_CHECK = Path(__file__).parent / "pr_check.json"


def run(case, method, setting):
    """Solve `case` with `method` at `setting`: a record of the error of B in
    each sensor group against the reference, ‖ΔB‖ / ‖B‖, the time of each
    phase and the size of the problem."""
    reference = references.load(case)
    collection, groups = case.model()
    sensors = np.concatenate(list(groups.values()))
    solved = method.solve(
        collection, sensors, setting, **case.options.get(method.name, {})
    )
    errors, first = {}, 0
    for group, points in groups.items():
        stored = reference["groups"][group]
        B = solved.B[first : first + len(points), stored["components"]]
        first += len(points)
        if B.shape != stored["B"].shape:
            msg = f"{case.name}, {group}: B is {B.shape}, its reference {stored['B'].shape}."
            raise ValueError(msg)
        errors[group] = float(
            np.linalg.norm(B - stored["B"]) / np.linalg.norm(stored["B"])
        )
    return {
        "case": case.name,
        "method": method.name,
        "setting": setting,
        "error": max(errors.values()),
        "errors": errors,
        "reference_errors": {g: v["error"] for g, v in reference["groups"].items()},
        "known_failure": case.known_failures.get(method.name),
        "seconds": solved.seconds,
        "counts": solved.counts,
        "versions": {
            "magpylib_material_response": magpylib_material_response.__version__,
            "magpylib": magpy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}",
        },
    }


def refusal(case, method):
    """The error `method` raises on `case`, or None if it solves it."""
    collection, groups = case.model()
    try:
        method.solve(
            collection,
            np.concatenate(list(groups.values())),
            method.settings[0],
            **case.options.get(method.name, {}),
        )
    except (ValueError, TypeError, NotImplementedError) as error:
        return error
    return None


def _percent(value):
    return f"{100 * value:.2f} %" if value >= 1e-3 else f"{100 * value:.0e} %"


def _row(record):
    seconds = record["seconds"]
    note = " known failure" if record["known_failure"] else ""
    # a group's error below its reference's own error says nothing; the worst
    # group whose error is measured stands for the case
    limits = record["reference_errors"]
    measured = {g: e for g, e in record["errors"].items() if e >= limits[g]}
    if measured:
        worst = max(measured, key=measured.get)
        error = _percent(measured[worst])
    else:
        worst = max(limits, key=limits.get)
        error = "< " + _percent(limits[worst])
    return (
        f"{record['case']:<20} {record['method']:<4} {record['setting']!s:>3} "
        f"{error:>11}  {worst:<20} {sum(seconds.values()):>8.2f} s "
        f"({seconds['prepare']:.2f} / {seconds['solve']:.2f} / "
        f"{seconds['getB']:.3f})  {record['counts']['unknowns']:>8}{note}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--settings", nargs="+", type=int)
    parser.add_argument("--pr", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.pr:
        stored = json.loads(PR_CHECK.read_text())
        plan = [
            (CASES[c], METHODS[m], [entry["setting"]])
            for c, by_method in stored.items()
            for m, entry in by_method.items()
        ]
    else:
        for case in (CASES[c] for c in args.cases):
            if case.waits:
                print(f"{case.name}: waits for {case.waits}")
            for method in (METHODS[m] for m in args.methods):
                if case.refused and method.can_solve(case):
                    error = refusal(case, method)
                    said = "NOT REFUSED" if error is None else f"refused: {error}"
                    print(f"{case.name}, {method.name}: {said}"[:120])
        plan = [
            (CASES[c], METHODS[m], args.settings or METHODS[m].settings_for(CASES[c]))
            for c in args.cases
            for m in args.methods
            if METHODS[m].can_solve(CASES[c]) and not CASES[c].refused
        ]
    print(
        f"{'case':<20} {'':<4} {'set':>3} {'error':>11}  {'worst group':<20} "
        f"{'time':>10} (prepare / solve / getB)  unknowns"
    )
    records = []
    for case, method, settings in plan:
        if not references.path(case).exists():
            print(f"{case.name:<20} {method.name:<4}     no reference yet")
            continue
        for setting in settings:
            try:
                record = run(case, method, setting)
            except TooLarge as error:
                print(
                    f"{case.name:<20} {method.name:<4} {setting!s:>3} too large: {error}"[
                        :120
                    ]
                )
                break
            records.append(record)
            print(_row(record), flush=True)
            if args.out:
                with args.out.open("a") as out:
                    out.write(json.dumps(record) + "\n")
            if method.growth * sum(record["seconds"].values()) > args.budget:
                break

    if args.pr and args.write:
        for record in records:
            stored[record["case"]][record["method"]] = {
                "setting": record["setting"],
                "errors": {g: float(f"{e:.4g}") for g, e in record["errors"].items()},
                "unknowns": record["counts"]["unknowns"],
            }
        PR_CHECK.write_text(json.dumps(stored, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
