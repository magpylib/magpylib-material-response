"""The validation cases (validation/): each method at its PR setting is as
accurate as stored, against the stored references."""

from __future__ import annotations

import json

import pytest

from validation import references
from validation.cases import CASES
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
    [(n, m) for n, c in CASES.items() if c.refused for m in METHODS],
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
