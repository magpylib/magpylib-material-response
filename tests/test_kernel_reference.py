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
from magpylib_material_response.face_charge import _triangle as T

REFERENCE = Path(__file__).parents[1] / "kernels" / "reference"
FILES = {p.stem: json.loads(p.read_text()) for p in sorted(REFERENCE.glob("*.json"))}
FIELDS = "BHJM"
# the points magpylib gets wrong (NaN, or off) before its stable edge
# logarithms (#1066, after 6.3.0)
BEFORE_1066 = {"beside-edge-1e-12"} | {
    f"near-edge-{e}-{d}" for e in "xyz" for d in ("1e-6", "1e-9", "1e-12")
}
OLD_MAGPYLIB = Version(magpy.__version__) <= Version("6.3.0")
# the kinds whose parameters are shared by all observers; the others take one
# row per observer
SHARED = {"triangle_charge", "charge_sheet"}


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


# per kind: the implementations, each a function, its array constructor and the
# fields it has
IMPLS = {
    "cuboid": {
        "array-api-numpy": (cuboid.cuboid_field, np.asarray, "BHJM"),
        "array-api-strict": (cuboid.cuboid_field, array_api_strict.asarray, "BHJM"),
        "magpylib": (_magpylib_cuboid, np.asarray, "BH"),
    },
    "triangle_charge": {
        "array-api-numpy": (T.triangle_charge_field, np.asarray, "H"),
        "array-api-strict": (T.triangle_charge_field, array_api_strict.asarray, "H"),
    },
    "charge_sheet": {
        "array-api-numpy": (T.charge_sheet_field, np.asarray, "H"),
        "array-api-strict": (T.charge_sheet_field, array_api_strict.asarray, "H"),
    },
}
PORTS = {"magpylib"}  # not in the relations: they cross kinds


def _evaluate(doc, impl, field, source, observer):
    """One field of one source at one observer, by one implementation."""
    src = doc["sources"][source]
    func, asarray, _ = IMPLS[src["kind"]][impl]
    params = {k: _decode(v) for k, v in src.items() if k not in ("kind", "why")}
    if src["kind"] not in SHARED:
        params = {k: v[None] for k, v in params.items()}
    X = asarray(_decode([observer]))
    with np.errstate(all="ignore"):
        return np.asarray(func(field, X, **{k: asarray(v) for k, v in params.items()}))[
            0
        ]


def _cases():
    """Every (file, point, implementation) the files and IMPLS give."""
    for name, doc in FILES.items():
        for point in doc["points"]:
            for impl in IMPLS[doc["sources"][point["source"]]["kind"]]:
                yield pytest.param(name, point, impl, id=f"{name}-{point['id']}-{impl}")


@pytest.mark.parametrize(("name", "point", "impl"), list(_cases()))
def test_point(name, point, impl):
    doc = FILES[name]
    kind = doc["sources"][point["source"]]["kind"]
    if impl in PORTS and OLD_MAGPYLIB and point["id"] in BEFORE_1066:
        pytest.skip("magpylib <= 6.3.0 lacks the stable edge logarithms (#1066)")
    for field in (f for f in FIELDS if f in point and f in IMPLS[kind][impl][2]):
        got = _evaluate(doc, impl, field, point["source"], point["observer"])
        atol = [point.get(f"atol_{field}", 0.0)]
        _within(got, _decode([point[field]]), doc["rtol"], atol, point["why"])


def _relations():
    for name, doc in FILES.items():
        for relation in doc.get("relations", []):
            for impl in ("array-api-numpy", "array-api-strict"):
                yield pytest.param(
                    name, relation, impl, id=f"{name}-{relation['id']}-{impl}"
                )


@pytest.mark.parametrize(("name", "relation", "impl"), list(_relations()))
def test_relation(name, relation, impl):
    doc = FILES[name]
    total = 0
    for t in relation["terms"]:
        w = np.asarray(t["weight"], float)
        f = _evaluate(doc, impl, t["field"], t["source"], t["observer"])
        total = total + (w @ f if w.ndim else w * f)
    atol = [relation["atol"]]
    _within(total, _decode(relation["target"]), doc["rtol"], atol, relation["why"])
