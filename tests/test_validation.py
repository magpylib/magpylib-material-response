"""The validation cases (validation/): each method at its PR setting is as
accurate as stored, against the stored references."""

from __future__ import annotations

import json

import pytest

from validation import references, report
from validation.cases import CASES, select
from validation.methods import METHODS
from validation.run import PR_CHECK, refusal, run

STORED = json.loads(PR_CHECK.read_text())


@pytest.mark.parametrize(
    "name", [n for n, c in CASES.items() if not (c.waits or c.refused)]
)
def test_reference_is_current(name):
    case = CASES[name]
    if "FEM" in case.known_failures and not references.path(case).exists():
        pytest.skip(f"no FEM reference until {case.known_failures['FEM']}")
    references.load(case)


@pytest.mark.parametrize(
    ("name", "method"),
    [
        (n, m)
        for n, c in CASES.items()
        if c.refused
        for m, method in METHODS.items()
        if not getattr(method, "baseline", False)
    ],
)
def test_refused(name, method):
    if not METHODS[method].can_solve(CASES[name]):
        pytest.skip(f"{method} does not solve this case here")
    error = refusal(CASES[name], METHODS[method])
    assert error is not None
    assert CASES[name].refused in str(error)


@pytest.mark.parametrize(
    ("name", "method"), [(c, m) for c, by_method in STORED.items() for m in by_method]
)
def test_pr_setting_holds(name, method):
    stored = STORED[name][method]
    record = run(CASES[name], METHODS[method], stored["setting"])
    for group, error in record["errors"].items():
        assert error <= stored["errors"][group] * 1.001, group
    assert record["counts"]["unknowns"] <= stored["unknowns"]


def _record(error, case="plate", method="MoM", unknowns=10, reference_error=1e-4):
    return {
        "case": case,
        "method": method,
        "setting": 1,
        "errors": {"g": error},
        "reference_errors": {"g": reference_error},
        "counts": {"unknowns": unknowns},
        "seconds": {"prepare": 0.0, "solve": 0.0, "getB": 0.0},
        "versions": {},
    }


@pytest.mark.parametrize(
    ("tonight", "fails"),
    [
        (_record(0.011), True),  # grew past the slack
        (_record(0.01001), False),
        (_record(float("nan")), True),
        (_record(0.01, unknowns=11), True),
        (_record(2.1e-6, reference_error=1e-5), False),  # the reference cannot tell
    ],
)
def test_report_against_the_baseline(tonight, fails):
    baseline = [_record(0.01 if tonight["errors"]["g"] > 1e-4 else 2e-6)]
    worse, _, _ = report.compare([tonight], baseline)
    assert bool(worse) == fails
    # the baseline keeps the least error any night reached, so a drift adds up
    assert report.merge(baseline, [tonight])[0]["errors"]["g"] <= 0.01


@pytest.mark.parametrize(
    ("case", "error", "fails"),
    [
        ("plate", 0.6, True),
        ("plate", float("nan"), True),
        ("plate", 0.1, False),
        ("thin_ring", 0.6, False),  # a known failure
    ],
)
def test_report_floor(case, error, fails):
    records = [_record(0.5, case=case, method="magpylib"), _record(error, case=case)]
    assert bool(report.floor(records)) == fails


def test_select_by_feature():
    curved = {case.name for case in select(features=["CylinderSegment"])}
    assert {"thin_ring", "shield_tube"} <= curved
    assert [case.name for case in select(features=["newell"])] == [
        "concentrator_newell"
    ]
    with pytest.raises(ValueError, match="No case has"):
        select(features=["Tetrahedra"])
