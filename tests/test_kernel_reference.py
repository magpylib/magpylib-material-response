"""The field kernels against ``kernels/reference/*.json``, the files every
implementation of them is checked against (#208), by the rules of
``kernels/reference/SPEC.md``.

Each kernel runs on NumPy and, where it is installed, on ``array_api_strict``,
which allows nothing beyond the array API standard; the installed magpylib's
cuboid, straight current segment, circle, cylinder, sphere and dipole are checked as
ports.
Every point is checked alone and, per file and kind, all in one call, so that
batches are covered too."""

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest
from packaging.version import Version

from magpylib_material_response._array_api import (
    circle,
    cuboid,
    cylinder,
    polyline,
    sphere,
)
from magpylib_material_response.face_charge import _triangle as T

try:
    import array_api_strict
except ImportError:  # a CI job without the test group: NumPy and the port only
    array_api_strict = None

REFERENCE = Path(__file__).parents[1] / "kernels" / "reference"
FILES = {p.stem: json.loads(p.read_text()) for p in sorted(REFERENCE.glob("*.json"))}
FIELDS = "BHJM"
# the points magpylib gets wrong (NaN, or off) before its stable edge
# logarithms (#1066, after 6.3.0)
BEFORE_1066 = {"cuboid/beside-edge-1e-12"} | {
    f"cuboid/near-edge-{e}-{d}" for e in "xyz" for d in ("1e-6", "1e-9", "1e-12")
}
OLD_MAGPYLIB = Version(magpy.__version__) <= Version("6.3.0")
# the points magpylib gets wrong in every release: beyond a segment's end it
# subtracts two sines that nearly cancel
MAGPYLIB_OFF = {"current/axis-far-1e3", "current/axis-far-1e3-before"}
# and its Cylinder (SPEC: cylinder)
_AXIS = "magpylib's Cylinder near or far along the axis: a three-term series, or elliptic integrals that cancel"
MAGPYLIB_CYLINDER_OFF = {
    "cylinder/cyl-near-axis-0.049": _AXIS,
    "cylinder/far-cyl-axis-1e3": _AXIS,
    "cylinder/tall-axis-middle": _AXIS,
    "cylinder/tall-axis-low": _AXIS,
    "cylinder/tall-far-axis": _AXIS,
    "cylinder/cyl-magpylib-small-r-0.0499": _AXIS,
    "cylinder/cyl-magpylib-small-r-0.0501": _AXIS,
}
# and in a batch only: a cylinder without height as any other, rounding not 0
MAGPYLIB_BATCH_OFF = {"cylinder/cyl-no-height"}
# the kinds whose implementations take one row of parameters per observer
PER_OBSERVER = {"cuboid", "polyline", "circle", "cylinder", "sphere", "dipole"}


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


def _magpylib_polyline(field, X, segment_start, segment_end, current):
    return magpy.func.polyline_field(
        field, X, segment_start, segment_end, current, squeeze=False
    ).reshape(-1, 3)


def _magpylib_circle(field, X, diameter, current):
    return magpy.func.circle_field(field, X, diameter, current, squeeze=False).reshape(
        -1, 3
    )


def _magpylib_cylinder(field, X, dimension, polarization):
    return magpy.func.cylinder_field(
        field, X, dimension, polarization, squeeze=False
    ).reshape(-1, 3)


def _magpylib_sphere(field, X, diameter, polarization):
    return magpy.func.sphere_field(
        field, X, diameter, polarization, squeeze=False
    ).reshape(-1, 3)


def _magpylib_dipole(field, X, moment):
    return magpy.func.dipole_field(field, X, moment, squeeze=False).reshape(-1, 3)


def _impls(func, fields):
    """NumPy, and array_api_strict where installed."""
    out = {"array-api-numpy": (func, np.asarray, fields)}
    if array_api_strict is not None:
        out["array-api-strict"] = (func, array_api_strict.asarray, fields)
    return out


# per kind: the implementations, each a function, its array constructor and the
# fields it has
IMPLS = {
    "cuboid": _impls(cuboid.cuboid_field, "BHJM")
    | {"magpylib": (_magpylib_cuboid, np.asarray, "BH")},
    "triangle_linear_charge": _impls(T.triangle_linear_charge_field, "H"),
    "charge_sheet": _impls(T.charge_sheet_field, "H"),
    "charge_sheet_corners": _impls(T.charge_sheet_corners_field, "H"),
    "polyline": _impls(polyline.polyline_field, "BH")
    | {"magpylib": (_magpylib_polyline, np.asarray, "BH")},
    "circle": _impls(circle.circle_field, "BH")
    | {"magpylib": (_magpylib_circle, np.asarray, "BH")},
    "cylinder": _impls(cylinder.cylinder_field, "BHJM")
    | {"magpylib": (_magpylib_cylinder, np.asarray, "BH")},
    "sphere": _impls(sphere.sphere_field, "BHJM")
    | {"magpylib": (_magpylib_sphere, np.asarray, "BH")},
    "dipole": _impls(sphere.dipole_field, "BH")
    | {"magpylib": (_magpylib_dipole, np.asarray, "BH")},
}
PORTS = {"magpylib"}  # not in the relations: they cross kinds


def _params(src):
    return {k: _decode(v) for k, v in src.items() if k not in ("kind", "why")}


def _evaluate(doc, impl, field, source, observers):
    """One field of one source at observers (M, 3), by one implementation."""
    src = doc["sources"][source]
    func, asarray, _ = IMPLS[src["kind"]][impl]
    params = _params(src)
    X = _decode(observers)
    if src["kind"] in PER_OBSERVER:
        params = {k: np.broadcast_to(v, (len(X), *v.shape)) for k, v in params.items()}
    with np.errstate(all="ignore"):
        return np.asarray(
            func(field, asarray(X), **{k: asarray(v) for k, v in params.items()})
        )


def _cases():
    """Every (file, point, implementation) the files and IMPLS give."""
    for name, doc in FILES.items():
        for point in doc["points"]:
            for impl in IMPLS[doc["sources"][point["source"]]["kind"]]:
                yield pytest.param(name, point, impl, id=f"{name}-{point['id']}-{impl}")


def _skipped(name, impl, point, batch=False):
    """Why a port skips a point of file ``name``, alone or in a batch, or
    None."""
    if impl not in PORTS:
        return None
    key = f"{name}/{point['id']}"
    if OLD_MAGPYLIB and key in BEFORE_1066:
        return "magpylib <= 6.3.0 lacks the stable edge logarithms (#1066)"
    if key in MAGPYLIB_OFF:
        return "magpylib subtracts two sines that cancel beyond a segment's end (#1097)"
    if key in MAGPYLIB_CYLINDER_OFF:
        return MAGPYLIB_CYLINDER_OFF[key]
    if batch and key in MAGPYLIB_BATCH_OFF:
        return "magpylib's batch computes a cylinder without height: rounding, not 0"
    return None


def test_files_found():
    """Every family's file is read: an empty folder would pass silently."""
    assert set(FILES) >= {"cuboid", "current", "cylinder", "face_charge", "sphere"}


@pytest.mark.parametrize(("name", "point", "impl"), list(_cases()))
def test_point(name, point, impl):
    doc = FILES[name]
    kind = doc["sources"][point["source"]]["kind"]
    if reason := _skipped(name, impl, point):
        pytest.skip(reason)
    fields = [f for f in FIELDS if f in point and f in IMPLS[kind][impl][2]]
    assert fields, f"{point['id']}: no field to check"
    for field in fields:
        got = _evaluate(doc, impl, field, point["source"], [point["observer"]])
        atol = [point.get(f"atol_{field}", 0.0)]
        _within(got, _decode([point[field]]), doc["rtol"], atol, point["why"])


def _batches():
    """Per file, kind and implementation: all that kind's points at once."""
    for name, doc in FILES.items():
        kinds = {doc["sources"][p["source"]]["kind"] for p in doc["points"]}
        for kind in sorted(kinds):
            for impl in IMPLS[kind]:
                yield pytest.param(name, kind, impl, id=f"{name}-{kind}-{impl}")


@pytest.mark.parametrize(("name", "kind", "impl"), list(_batches()))
def test_batch(name, kind, impl):
    """One call per field: the cuboid's points with one row of parameters each
    (their sources mixed), a shared-parameter kind's points per source."""
    doc = FILES[name]
    points = [
        p
        for p in doc["points"]
        if doc["sources"][p["source"]]["kind"] == kind
        and not _skipped(name, impl, p, batch=True)
    ]
    groups = (
        [points]
        if kind in PER_OBSERVER
        else [[p for p in points if p["source"] == s] for s in doc["sources"]]
    )
    func, asarray, fields = IMPLS[kind][impl]
    for group in filter(None, groups):
        X = _decode([p["observer"] for p in group])
        if kind in PER_OBSERVER:
            rows = [_params(doc["sources"][p["source"]]) for p in group]
            params = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        else:
            params = _params(doc["sources"][group[0]["source"]])
        for field in fields:
            with np.errstate(all="ignore"):
                got = np.asarray(
                    func(
                        field, asarray(X), **{k: asarray(v) for k, v in params.items()}
                    )
                )
            expected = _decode([p[field] for p in group])
            atol = [p.get(f"atol_{field}", 0.0) for p in group]
            _within(got, expected, doc["rtol"], atol, f"{kind} {field} batch")


@pytest.mark.parametrize("kind", ["circle", "cylinder"])
def test_float32(kind):
    """In float32 the kernels that iterate ``cel`` stop at its precision: every
    point of the kind finite where its value is (1e-8 would never be met)."""
    func = IMPLS[kind]["array-api-numpy"][0]
    for doc in FILES.values():
        points = [
            p for p in doc["points"] if doc["sources"][p["source"]]["kind"] == kind
        ]
        for p in points:
            params = _params(doc["sources"][p["source"]])
            got = func(
                "B",
                _decode([p["observer"]]).astype(np.float32),
                **{k: v[None].astype(np.float32) for k, v in params.items()},
            )
            assert np.isfinite(got).all() == np.isfinite(_decode(p["B"])).all(), p["id"]


def _relations():
    for name, doc in FILES.items():
        for relation in doc.get("relations", []):
            for impl in ("array-api-numpy", "array-api-strict"):
                if impl in IMPLS["cuboid"]:
                    yield pytest.param(
                        name, relation, impl, id=f"{name}-{relation['id']}-{impl}"
                    )


@pytest.mark.parametrize(("name", "relation", "impl"), list(_relations()))
def test_relation(name, relation, impl):
    doc = FILES[name]
    total = 0
    for t in relation["terms"]:
        w = np.asarray(t["weight"], float)
        f = _evaluate(doc, impl, t["field"], t["source"], [t["observer"]])[0]
        total = total + (w @ f if w.ndim else w * f)
    atol = [relation["atol"]]
    _within(total, _decode(relation["target"]), doc["rtol"], atol, relation["why"])
