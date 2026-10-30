"""The validation cases (validation/): each method at its PR setting is as
accurate as stored, against the stored references."""

from __future__ import annotations

import json

import pytest

from validation import references, report, timing
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


def _record(error, case="plate", method="MoM", unknowns=1000, reference_error=1e-4):
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
        (_record(0.01, unknowns=1001), True),
        (_record(2.1e-6, reference_error=1e-5), False),  # the reference cannot tell
        # netgen's mesh varies with the runner, and FEM's error with it
        (_record(0.013, method="FEM", unknowns=1010), False),
        (_record(0.016, method="FEM"), True),
    ],
)
def test_report_against_the_baseline(tonight, fails):
    error = 0.01 if tonight["errors"]["g"] > 1e-4 else 2e-6
    baseline = [_record(error, method=tonight["method"])]
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


@pytest.mark.parametrize(
    ("then", "now", "cell", "flagged"),
    [
        (200e6, 290e6, "⚪ 200 MB → 290 MB", False),  # within the noise
        (1158e6, 1380e6, "🔴 1.16 GB → **1.38 GB**", True),
        (3.99e9, 280e6, "🟢 3.99 GB → 280 MB", False),
        (None, 280e6, "280 MB", False),  # not on the base
        (200e6, None, "🔴 **failed**", True),
    ],
)
def test_timing_flags_memory(then, now, cell, flagged):
    assert timing._memory_cell(then, now) == (cell, flagged)


def test_timing_comment_shows_memory(tmp_path, monkeypatch):
    pytest.importorskip("resource")  # not on Windows
    monkeypatch.setattr(timing, "MORE_MEMORY", 1e6)
    small, broken = "gmres_stops, FaceCharge", "no_such_case, MoM"
    monkeypatch.setattr(
        timing,
        "_pr_check",
        lambda: [
            (small, "gmres_stops", "FaceCharge", 2),
            (broken, "no_such_case", "MoM", 1),
        ],
    )
    timing.memory(tmp_path / "memory-head.json")
    peaks = json.loads((tmp_path / "memory-head.json").read_text())
    assert 1e7 < peaks[small] < 1e10  # in bytes
    assert peaks[broken] is None

    (tmp_path / "memory-base.json").write_text(json.dumps({small: peaks[small] // 2}))
    round_ = {"version": "1", "times": {key: {"solve": 1.0} for key in peaks}}
    for name in ("base-1.json", "head-1.json"):
        (tmp_path / name).write_text(json.dumps(round_))
    comment = timing.compare(tmp_path)
    assert f"Over 1 MB more memory, or failed: {small}; {broken}." in comment
    assert "**failed**" in comment
