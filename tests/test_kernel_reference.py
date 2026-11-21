"""The field kernels against ``kernels/reference/*.json``, the files every
implementation of them is checked against (#208), by the rules of
``kernels/reference/SPEC.md``.

Each kernel runs on NumPy and on ``array_api_strict``, which allows nothing
beyond the array API standard; the installed magpylib's cuboid is checked as
a port."""

import json
from pathlib import Path

import array_api_strict
import magpylib as magpy
import numpy as np
import pytest
from packaging.version import Version

from magpylib_material_response._array_api import cuboid

REFERENCE = Path(__file__).parents[1] / "kernels" / "reference"
CUBOID = json.loads((REFERENCE / "cuboid.json").read_text())
# the points magpylib gets wrong (NaN, or off) before its stable edge
# logarithms (#1066, after 6.3.0)
BEFORE_1066 = {"beside-edge-1e-12"} | {
    f"near-edge-{e}-{d}" for e in "xyz" for d in ("1e-6", "1e-9", "1e-12")
}
OLD_MAGPYLIB = Version(magpy.__version__) <= Version("6.3.0")


def _decode(a):
    # the strings 'nan', 'inf' and '-inf' parse as floats
    return np.array(a, dtype=object).astype(float)


def _within(got, expected, rtol, atol, why):
    """The files' rule, per row: non-finite components exactly, the rest within
    ``rtol |expected| + atol``."""
    got, expected = np.atleast_2d(got), np.atleast_2d(expected)
    finite = np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(got[~finite], expected[~finite])
    err = np.linalg.norm(got - expected, axis=1)[finite]
    size = np.linalg.norm(expected[finite], axis=1)
    bound = rtol * size + np.asarray(atol)[finite]
    assert np.all(err <= bound), (why, np.max(err / np.maximum(bound, 1e-300)))


def _magpylib_cuboid(field, X, dimension, polarization):
    return magpy.func.cuboid_field(field, X, dimension, polarization, squeeze=False)


# per implementation: the function, its array constructor, its fields
CUBOID_KERNELS = {
    "array-api-numpy": (cuboid.cuboid_field, np.asarray, "BHJM"),
    "array-api-strict": (cuboid.cuboid_field, array_api_strict.asarray, "BHJM"),
    "magpylib": (_magpylib_cuboid, np.asarray, "BH"),
}


def _cuboid(impl, field, source, observer):
    kernel, asarray, _ = CUBOID_KERNELS[impl]
    source = CUBOID["sources"][source]
    args = (
        _decode([v]) for v in (observer, source["dimension"], source["polarization"])
    )
    with np.errstate(all="ignore"):
        return np.asarray(kernel(field, *map(asarray, args)))[0]


@pytest.mark.parametrize("impl", CUBOID_KERNELS)
@pytest.mark.parametrize(
    "point", CUBOID["points"], ids=[p["id"] for p in CUBOID["points"]]
)
def test_cuboid_point(point, impl):
    if impl == "magpylib" and OLD_MAGPYLIB and point["id"] in BEFORE_1066:
        pytest.skip("magpylib <= 6.3.0 lacks the stable edge logarithms (#1066)")
    for field in CUBOID_KERNELS[impl][2]:
        got = _cuboid(impl, field, point["source"], point["observer"])
        atol = [point.get(f"atol_{field}", 0.0)]
        _within(got, _decode([point[field]]), CUBOID["rtol"], atol, point["why"])


@pytest.mark.parametrize("impl", ["array-api-numpy", "array-api-strict"])
@pytest.mark.parametrize(
    "relation", CUBOID["relations"], ids=[r["id"] for r in CUBOID["relations"]]
)
def test_cuboid_relation(relation, impl):
    total = 0
    for t in relation["terms"]:
        w = np.asarray(t["weight"], float)
        f = _cuboid(impl, t["field"], t["source"], t["observer"])
        total = total + (w @ f if w.ndim else w * f)
    _within(
        total,
        _decode(relation["target"]),
        CUBOID["rtol"],
        [relation["atol"]],
        relation["why"],
    )
