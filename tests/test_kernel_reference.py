"""The numpy kernels against ``kernels/reference/kernels.json``, the file every
other implementation of them is checked against (#208): if a kernel changes,
the file is written again by ``kernels/reference/make_reference.py``. The
cuboid's cases are magpylib's main branch, not the release installed here."""

import json
from pathlib import Path

import numpy as np
import pytest

from magpylib_material_response.face_charge import _triangle as T

REFERENCE = Path(__file__).parents[1] / "kernels" / "reference" / "kernels.json"
CASES = [
    c
    for c in json.loads(REFERENCE.read_text())["cases"]
    if not c["kernel"].startswith("cuboid")
]


def _decode(a):
    # the strings 'nan', 'inf' and '-inf' parse as floats
    return np.array(a, dtype=object).astype(float)


def _sheet_H(X, V, c, center, far_from):
    out = T.charge_field(X, V, T.frames(V), c)
    far = np.linalg.norm(X - center, axis=1) >= far_from[0]
    out[far] = T.charge_field_far(X[far], *T.far_points(V, T.frames(V), c))
    return out / (4 * np.pi)


def _far_points(V, c):
    y, charge = T.far_points(V, T.frames(V), c)
    return np.concatenate([y.transpose(1, 0, 2), charge.T[..., None]], axis=2).reshape(
        -1, 4
    )


KERNELS = {
    "charge_field": lambda X, V, c: T.charge_field(X, V, T.frames(V), c),
    "charge_field_far": T.charge_field_far,
    "far_points": _far_points,
    "sheet_H": _sheet_H,
}


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_kernel_reference(case):
    inputs = {k: _decode(v) for k, v in case["inputs"].items()}
    expected = _decode(case["expected"])
    with np.errstate(all="ignore"):
        got = KERNELS[case["kernel"]](**inputs)
    finite = np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(got[~finite], expected[~finite])
    err = np.linalg.norm(got[finite] - expected[finite], axis=1)
    bound = (
        np.array(case["rtol"])[finite] * np.linalg.norm(expected[finite], axis=1)
        + case["atol"]
    )
    assert np.all(err <= bound), np.max(err / bound)
