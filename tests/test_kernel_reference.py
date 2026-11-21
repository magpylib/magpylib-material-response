"""The kernels against ``kernels/reference/*.json``, the files every
implementation of them is checked against (#208): if a kernel changes, its
file is written again by ``kernels/reference/make_<family>.py``.

The cuboid is checked three ways: its array API implementation
(``_array_api.cuboid``) on NumPy and on ``array_api_strict``, which allows
nothing beyond the standard, and the installed magpylib's, as a port; points
marked ``needs`` are skipped against a magpylib released without that fix."""

import json
from pathlib import Path

import array_api_strict
import magpylib as magpy
import numpy as np
import pytest
from packaging.version import Version

from magpylib_material_response._array_api import cuboid

FILES = sorted((Path(__file__).parents[1] / "kernels" / "reference").glob("*.json"))
DOCS = {f.stem: json.loads(f.read_text()) for f in FILES}
CUBOID = DOCS["cuboid"]
# 6.3.0 is the last release without the stable edge logarithms (magpylib #1066)
STABLE_CUBOID = Version(magpy.__version__) > Version("6.3.0")


def _magpylib_cuboid(field, X, dimension, polarization):
    return magpy.func.cuboid_field(field, X, dimension, polarization, squeeze=False)


CUBOID_IMPLEMENTATIONS = {
    "array-api-numpy": (cuboid.cuboid_field, np.asarray),
    "array-api-strict": (cuboid.cuboid_field, array_api_strict.asarray),
    "magpylib": (_magpylib_cuboid, np.asarray),
}


def _decode(a):
    # the strings 'nan', 'inf' and '-inf' parse as floats
    return np.array(a, dtype=object).astype(float)


def _within(got, expected, rtol, atol):
    """The files' rule, per row: non-finite components exactly, the rest within
    ``rtol |expected| + atol``."""
    got, expected = np.atleast_2d(got), np.atleast_2d(expected)
    finite = np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(got[~finite], expected[~finite])
    err = np.linalg.norm(got - expected, axis=1)[finite]
    bound = rtol * np.linalg.norm(expected[finite], axis=1) + np.asarray(atol)[finite]
    assert np.all(err <= bound), np.max(err / np.maximum(bound, 1e-300))


@pytest.mark.parametrize("impl", CUBOID_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "point", CUBOID["points"], ids=[p["id"] for p in CUBOID["points"]]
)
def test_cuboid_point(point, impl):
    if impl == "magpylib" and "needs" in point and not STABLE_CUBOID:
        pytest.skip(f"the installed magpylib lacks {point['needs']}")
    kernel, asarray = CUBOID_IMPLEMENTATIONS[impl]
    source = CUBOID["sources"][point["source"]]
    args = [
        asarray(_decode([v]))
        for v in (point["observer"], source["dimension"], source["polarization"])
    ]
    for field in ("B", "H"):
        with np.errstate(all="ignore"):
            got = np.asarray(kernel(field, *args))
        _within(
            got,
            _decode([point[field]]),
            CUBOID["rtol"],
            [point.get(f"atol_{field}", 0.0)],
        )
