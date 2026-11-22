"""Fill the missing expected values of a reference file (#208); the rules are
in ``SPEC.md``. Run from the repository root, with the ``reference``
dependency group::

    python kernels/reference/fill.py kernels/reference/cuboid.json [--redo]

A point's values already in the file stay, unless ``--redo``. It takes the one
implementation at 50 digits (on ``mp_namespace``), rounded to double, with an
``atol_<field>`` where numpy's own error needs one. A point outside the files'
scope (SPEC) is refused. NumPy never sets a value: where it is off by more
than ``LOOSE`` the point is printed, without an ``atol``, for the checkers to
show, and the run exits with status 1. A relation that holds at 50 digits (an exact law) gets its ``atol`` from
numpy's residual; an approximate law carries its ``atol`` in the file, and
``fill.py`` checks it against the rule (SPEC).
"""

import argparse
import inspect
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import mp_namespace as M
import mpmath as mp
import numpy as np

from magpylib_material_response._array_api.cuboid import cuboid_field
from magpylib_material_response._array_api.polyline import polyline_field
from magpylib_material_response.face_charge import _triangle as T

mp.mp.dps = 50
MARGIN = 16  # atol over numpy's own error
LOOSE = 1e-6  # numpy off by more than this, relative: printed, no atol
EXACT = 1e-30  # an exact law's 50-digit residual, relative to its terms
EPS = np.finfo(float).eps


def cuboid_scope(x, p):
    """Why an observer is out of scope for a cuboid, or None: on a face, edge
    or corner (within today's 1e-15 h, or magpylib #1081's 100 eps max(size,
    |p|), of it), on a zero-thickness cuboid, of a cuboid with a negative
    dimension (magpylib's formula then flips B), or not finite."""
    if not np.isfinite(x).all():
        return "a non-finite observer"
    if (p["dimension"] < 0).any():
        return "a negative dimension"
    h = np.abs(p["dimension"]) / 2
    band = np.maximum(1e-15 * h, 100 * EPS * max(2 * h.max(), np.linalg.norm(x)))
    dist = np.abs(x) - h  # to each axis's pair of faces
    if (h == 0).any():
        flat = h == 0
        if (np.abs(x[flat]) <= band[flat]).all() and (dist[~flat] <= band[~flat]).all():
            return "on a zero-thickness cuboid"
        return None
    if (dist <= band).all() and (np.abs(dist) <= band).any():
        return "on a face, edge or corner"
    return None


def polyline_scope(x, p):
    """Why an observer is out of scope for a straight current segment, or
    None: on the segment's line, its extensions included (within today's
    1e-15 L, or #1081's 100 eps max(L, |x|, |start|, |end|), of it: there magpylib gives 0,
    and the distance is rounding), a segment without length, or not
    finite."""
    if not np.isfinite(x).all():
        return "a non-finite observer"
    a, b = p["segment_start"], p["segment_end"]
    L = np.linalg.norm(b - a)
    if L == 0:
        return "a segment without length"
    e = (b - a) / L
    t = e @ (x - a)
    d = np.linalg.norm(x - a - t * e)
    size = max(L, *(np.linalg.norm(v) for v in (x, a, b)))
    band = max(1e-15 * L, 100 * EPS * size)
    if d <= band:
        return "on the segment's line"
    return None


def triangles_scope(x, p):
    """Why an observer is out of scope for charged triangles, or None: not
    finite, a triangle without area (its corners in a line to within rounding),
    or in a triangle's plane or within magpylib #1081's band of it (100 eps
    max(its size, |p|)). Distances are exact, at 50 digits: in doubles a
    sliver's normal is too rough."""
    if not np.isfinite(x).all():
        return "a non-finite observer"
    X = [mp.mpf(float(v)) for v in x]
    for tri in p["vertices"]:
        a, b, c = ([mp.mpf(float(v)) for v in corner] for corner in tri)
        ab, ac = (
            [q - o for q, o in zip(b, a, strict=True)],
            [q - o for q, o in zip(c, a, strict=True)],
        )
        n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ]
        sides = [norm(ab), norm(ac), norm([q - o for q, o in zip(c, b, strict=True)])]
        if norm(n) <= 100 * EPS * sides[0] * sides[1]:
            return "a triangle without area"
        w = abs(
            mp.fsum((xi - ai) * ni for xi, ai, ni in zip(X, a, n, strict=True))
        ) / norm(n)
        if w <= 100 * EPS * max(*sides, norm(X)):
            return "in or by a triangle's plane"
    return None


def sheet_scope(x, p):
    """:func:`triangles_scope`, and an observer whose side of ``far_from``
    differs between the double and the exact distance from the center: there
    rounding picks the branch."""
    if reason := triangles_scope(x, p):
        return reason
    exact = norm(
        [mp.mpf(float(a)) - float(b) for a, b in zip(x, p["center"], strict=True)]
    )
    rounded = T.beyond_far_from(x[None], p["center"], p["far_from"])[0]
    if bool(rounded) != bool(exact >= float(p["far_from"])):
        return "rounding picks the branch at far_from"
    if rounded:  # the whole sheet takes the far rule
        return None
    V = p["vertices"]
    Vm = M.from_doubles(V)
    for table in (T.PAIR_RULE, T.PAIR_RULE_MIDDLE):  # the per-pair rules, pair by pair
        k2 = T.pair_k2(float(p["tolerance"]), table)
        if k2 is None:
            continue
        doubles = T.far_pairs(x[None], V, T.frames(V), k2)[0]
        exact = T.far_pairs(M.from_doubles(x[None]), Vm, T.frames(Vm), k2)
        if [bool(v) for v in doubles] != [bool(v) for v in exact.tolist()[0]]:
            return "rounding picks a pair's rule"
    return None


def corners_scope(x, p):
    """:func:`sheet_scope` at the derived coefficients, center and
    ``far_from``, and an observer so near ``far_from`` that the 50-digit
    center and distance could fall on its other side."""
    c, center, far_from = T.sheet_parameters(p["vertices"], p["charge_densities"])
    derived = {**p, "charge": c, "center": center, "far_from": far_from}
    if reason := sheet_scope(x, derived):
        return reason
    if abs(np.linalg.norm(x - center) - far_from) <= 1e-9 * far_from:
        return "the derived far_from rounds either way"
    return None


class Kind(NamedTuple):
    """A source kind: its implementation, called as ``f(field, observers,
    **parameters)``; its fields; whether its parameters are per observer
    (arrays with a leading axis of observers) or shared; its scope check."""

    func: Callable
    fields: str
    per_observer: bool
    scope: Callable


KINDS = {
    "cuboid": Kind(cuboid_field, "BHJM", True, cuboid_scope),
    "polyline": Kind(polyline_field, "BH", True, polyline_scope),
    "triangle_linear_charge": Kind(
        T.triangle_linear_charge_field, "H", False, triangles_scope
    ),
    "charge_sheet": Kind(T.charge_sheet_field, "H", False, sheet_scope),
    "charge_sheet_corners": Kind(
        T.charge_sheet_corners_field, "H", False, corners_scope
    ),
}


def decode(v):
    """A JSON number or nested list as floats; "nan", "inf", "-inf" too."""
    return np.asarray([decode(x) for x in v] if isinstance(v, list) else float(v))


def encode(a):
    a = np.asarray(a, float)
    out = [x if math.isfinite(x) else repr(x) for x in a.ravel().tolist()]
    return np.array(out, object).reshape(a.shape).tolist() if a.ndim else out[0]


def decade(x):
    """``x`` rounded up to a power of ten: last-bit differences between
    platforms' libm leave the file unchanged."""
    if x == 0:
        return 0.0
    k = math.ceil(math.log10(x))
    return float(f"1e{k}") if float(f"1e{k}") >= x else float(f"1e{k + 1}")


def norm(v):
    return mp.sqrt(sum(mp.mpf(x) ** 2 for x in v))


def parameters(doc, source):
    """A source's kind and its parameters as floats."""
    src = doc["sources"][source]
    return KINDS[src["kind"]], {
        k: decode(v) for k, v in src.items() if k not in ("kind", "why")
    }


def check_scope(doc, source, observer, what):
    kind, params = parameters(doc, source)
    if reason := kind.scope(decode(observer), params):
        msg = f"{what}: {reason}: not in the files' scope (SPEC)"
        raise AssertionError(msg)


def field(doc, source, name, observer, exact):
    """One field of one source at one observer: doubles, or mpf at 50
    digits."""
    kind, params = parameters(doc, source)
    if kind.per_observer:
        params = {k: v[None] for k, v in params.items()}
    x = decode(observer)[None]
    if not exact:
        with np.errstate(all="ignore"):
            return kind.func(name, x, **params)[0]
    out = kind.func(
        name, M.from_doubles(x), **{k: M.from_doubles(v) for k, v in params.items()}
    )
    return out.tolist()[0]


def fill_point(doc, p, redo):
    """Fill one point's missing fields; the fields numpy disagrees with."""
    names = KINDS[doc["sources"][p["source"]]["kind"]].fields
    check_scope(doc, p["source"], p["observer"], p["id"])
    off = []
    for name in names:
        if name in p and not redo:
            continue
        n = field(doc, p["source"], name, p["observer"], exact=False)
        e = field(doc, p["source"], name, p["observer"], exact=True)
        if not all(mp.isfinite(v) for v in e):
            msg = f"{p['id']} {name}: no finite value, not in the files' scope (SPEC)"
            raise AssertionError(msg)
        p.pop(f"atol_{name}", None)
        p[name] = encode([float(v) for v in e])  # mpmath rounds to nearest
        err = norm(mp.mpf(float(a)) - b for a, b in zip(n, e, strict=True))
        size = norm(e)
        if not np.all(np.isfinite(n)) or err > LOOSE * size or (size == 0 and err > 0):
            off.append(name)
        elif MARGIN * err > doc["rtol"] * size:
            p[f"atol_{name}"] = decade(float(MARGIN * err))
    # key order: id, why, inputs, fields, then their atols
    order = ["id", "why", "source", "observer", *names, *(f"atol_{f}" for f in names)]
    rest = {k: p.pop(k) for k in list(p) if k not in order}
    p.update({k: p.pop(k) for k in order if k in p} | rest)
    return off


def residual(doc, r, exact):
    """Sum of weight @ field over the terms, minus the target; and the sum of
    the terms' sizes."""
    total, scale = None, mp.mpf(0)
    for t in r["terms"]:
        check_scope(doc, t["source"], t["observer"], r["id"])
        f = field(doc, t["source"], t["field"], t["observer"], exact)
        f = [mp.mpf(float(v)) if not exact else v for v in f]
        w = decode(t["weight"])
        rows = (
            [[w if i == j else 0 for j in range(3)] for i in range(3)]
            if w.ndim == 0
            else w
        )
        part = [
            mp.fsum(mp.mpf(float(a)) * b for a, b in zip(row, f, strict=True))
            for row in rows
        ]
        scale += norm(part)
        total = (
            part if total is None else [a + b for a, b in zip(total, part, strict=True)]
        )
    return [
        a - mp.mpf(float(b)) for a, b in zip(total, decode(r["target"]), strict=True)
    ], scale


def fill_relation(doc, r):
    """Check one relation and its ``atol`` (SPEC: where the values come from),
    and write it for an exact law that has none; True where numpy is off by
    more than ``LOOSE`` of the terms' size."""
    exact, scale = residual(doc, r, exact=True)
    rounded, _ = residual(doc, r, exact=False)
    if not all(mp.isfinite(v) for v in exact + rounded):
        msg = f"{r['id']}: a residual is not finite"
        raise AssertionError(msg)
    diff = norm([a - b for a, b in zip(rounded, exact, strict=True)])
    if diff > LOOSE * scale:
        return True
    if norm(exact) <= EXACT * scale:  # an exact law
        atol = decade(float(max(MARGIN * norm(rounded), doc["rtol"] * scale)))
        r.setdefault("atol", atol)
    else:  # approximate: the law's own error, and numpy's rounding
        atol = decade(float(norm(exact) + MARGIN * diff))
        if "atol" not in r:
            msg = f"{r['id']}: an approximate law (off by {float(norm(exact)):.3g} at 50 digits): write its atol, {atol:g} by the rule"
            raise AssertionError(msg)
    if r["atol"] != atol:
        msg = f"{r['id']}: atol {r['atol']:g}, {atol:g} by the rule"
        raise AssertionError(msg)
    return False


def dump(obj, depth=0):
    """JSON, one key per line and every list of numbers on one line, so that a
    diff names the value that moved."""
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(obj, dict):
        items = [
            f"{inner}{json.dumps(k)}: {dump(v, depth + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list):
        if all(isinstance(v, int | float | str) for v in obj):
            return json.dumps(obj, separators=(", ", ": "))
        return "[\n" + ",\n".join(inner + dump(v, depth + 1) for v in obj) + f"\n{pad}]"
    return json.dumps(obj)


def check_keys(doc):
    """Every key known: a source's kind and its function's parameters, a
    point's fields of its kind and their atols, a relation's and a term's keys;
    every source and field named exists."""
    bad = []
    for name, src in doc["sources"].items():
        kind = KINDS.get(src.get("kind"))
        if kind is None:
            bad.append(f"source {name}: kind {src.get('kind')!r}")
            continue
        allowed = {"kind", "why", *inspect.signature(kind.func).parameters} - {
            "field",
            "observers",
        }
        bad += [f"source {name}: {k}" for k in sorted(set(src) - allowed)]
    for p in doc["points"]:
        if p.get("source") not in doc["sources"]:
            bad.append(f"{p.get('id')}: source {p.get('source')!r}")
            continue
        fields = KINDS[doc["sources"][p["source"]]["kind"]].fields
        allowed = {
            "id",
            "why",
            "source",
            "observer",
            *fields,
            *(f"atol_{f}" for f in fields),
        }
        bad += [f"{p['id']}: {k}" for k in sorted(set(p) - allowed)]
    for r in doc.get("relations", []):
        bad += [
            f"{r.get('id')}: {k}"
            for k in sorted(set(r) - {"id", "why", "terms", "target", "atol"})
        ]
        for t in r["terms"]:
            if set(t) != {"source", "field", "observer", "weight"}:
                bad.append(f"{r['id']}: term keys {sorted(t)}")
            elif t["source"] not in doc["sources"]:
                bad.append(f"{r['id']}: source {t['source']!r}")
            elif t["field"] not in KINDS[doc["sources"][t["source"]]["kind"]].fields:
                bad.append(f"{r['id']}: field {t['field']!r}")
    assert not bad, f"unknown keys, kinds, sources or fields: {bad}"


def check_names(doc):
    """Unique slug ids, never renamed once committed; a one-line ``why`` on
    every source, point and relation."""
    entries = doc["points"] + doc.get("relations", [])
    ids = [x["id"] for x in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"ids must be unique: {dup}"
    bad = [i for i in ids if not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", i)]
    assert not bad, f"ids must be lowercase slugs: {bad}"
    named = [*doc["sources"].items(), *((x["id"], x) for x in entries)]
    bad = [
        k
        for k, x in named
        if not isinstance(x.get("why"), str) or not x["why"].strip() or "\n" in x["why"]
    ]
    assert not bad, f"a one-line why is missing: {bad}"


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("file", type=Path)
    cli.add_argument("--redo", action="store_true", help="recompute existing values")
    args = cli.parse_args()
    doc = json.loads(args.file.read_text())
    check_keys(doc)
    check_names(doc)
    disagree = [
        f"{p['id']} {''.join(off)}"
        for p in doc["points"]
        if (off := fill_point(doc, p, args.redo))
    ]
    disagree += [
        f"{r['id']} (no atol written)"
        for r in doc.get("relations", [])
        if fill_relation(doc, r)
    ]
    args.file.write_text(dump(doc) + "\n")
    print(f"{args.file.name}: {len(doc['points'])} points, "
          f"{len(doc.get('relations', []))} relations")  # fmt: skip
    if disagree:
        print("NUMPY DISAGREES:", ", ".join(disagree))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
