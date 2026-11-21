"""Fill the missing expected values of a reference file (#208); the rules are
in ``SPEC.md``. Run from the repository root, with the ``reference``
dependency group::

    python kernels/reference/fill.py kernels/reference/cuboid.json [--redo]

A point's values already in the file stay, unless ``--redo``. It takes the one
implementation at 50 digits (on ``mp_namespace``), rounded to double, with an
``atol_<field>`` where numpy's own error needs one; numpy's value where the
50-digit one is not finite or numpy is off by more than ``LOOSE``, and those
points are printed for review. A relation without ``atol`` must hold at 50
digits; it then gets one from numpy's residual. A relation with ``atol`` is an
approximate law, or one filled before: that ``atol`` is checked, never
changed (delete it to have it set again).
"""

import argparse
import json
import math
import re
from pathlib import Path

import mp_namespace as M
import mpmath as mp
import numpy as np

from magpylib_material_response._array_api.cuboid import cuboid_field

mp.mp.dps = 50
MARGIN = 16  # atol over numpy's own error
LOOSE = 1e-6  # numpy off by more than this, relative: numpy's value, printed
EXACT = 1e-30  # an exact law's 50-digit residual, relative to its terms

# per kind: the implementation, called as f(field, observers, **parameters)
# with (1, ...) arrays, and its fields
KINDS = {"cuboid": (cuboid_field, "BHJM")}


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
    return 0.0 if x == 0 else float(f"1e{math.ceil(math.log10(x))}")


def norm(v):
    return mp.sqrt(sum(mp.mpf(x) ** 2 for x in v))


def field(doc, source, name, observer, exact):
    """One field of one source at one observer: doubles, or mpf at 50 digits
    (None where the 50-digit run fails)."""
    src = doc["sources"][source]
    func, _ = KINDS[src["kind"]]
    params = {k: decode(v)[None] for k, v in src.items() if k not in ("kind", "why")}
    x = decode(observer)[None]
    if not exact:
        with np.errstate(all="ignore"):
            return func(name, x, **params)[0]
    try:
        out = func(
            name, M.from_doubles(x), **{k: M.from_doubles(v) for k, v in params.items()}
        )
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return out.tolist()[0]


def fill_point(doc, p, redo):
    _, names = KINDS[doc["sources"][p["source"]]["kind"]]
    took_numpy = []
    for name in names:
        if name in p and not redo:
            continue
        n = field(doc, p["source"], name, p["observer"], exact=False)
        e = field(doc, p["source"], name, p["observer"], exact=True)
        p.pop(f"atol_{name}", None)
        finite = e is not None and all(mp.isfinite(v) for v in e)
        err = (
            norm(mp.mpf(float(a)) - b for a, b in zip(n, e, strict=True))
            if finite
            else None
        )
        size = norm(e) if finite else None
        if not finite or not np.all(np.isfinite(n)) or err > LOOSE * size:
            p[name] = encode(n)
            took_numpy.append(name)
            continue
        p[name] = encode([float(v) for v in e])  # mpmath rounds to nearest
        if MARGIN * err > doc["rtol"] * size:
            p[f"atol_{name}"] = decade(float(MARGIN * err))
    # key order: id, why, inputs, fields, then their atols
    order = ["id", "why", "source", "observer", *names, *(f"atol_{f}" for f in names)]
    rest = {k: p.pop(k) for k in list(p) if k not in order}
    p.update({k: p.pop(k) for k in order if k in p} | rest)
    return took_numpy


def residual(doc, r, exact):
    """Sum of weight @ field over the terms, minus the target; and the sum of
    the terms' sizes."""
    total, scale = None, mp.mpf(0)
    for t in r["terms"]:
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
    exact, scale = residual(doc, r, exact=True)
    rounded, _ = residual(doc, r, exact=False)
    if "atol" in r:
        for what, res in (("50 digits", exact), ("numpy", rounded)):
            if norm(res) > r["atol"]:
                msg = f"{r['id']}: off by {float(norm(res)):.3g} at {what}, atol {r['atol']}"
                raise AssertionError(msg)
        return
    if norm(exact) > EXACT * scale:
        msg = (f"{r['id']}: does not hold at 50 digits (off by {float(norm(exact)):.3g}); "
               "an approximate law carries its own atol in the file")  # fmt: skip
        raise AssertionError(msg)
    r["atol"] = decade(float(max(MARGIN * norm(rounded), doc["rtol"] * scale)))


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
    bad = [k for k, x in named if not isinstance(x.get("why"), str) or "\n" in x["why"]]
    assert not bad, f"a one-line why is missing: {bad}"


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("file", type=Path)
    cli.add_argument("--redo", action="store_true", help="recompute existing values")
    args = cli.parse_args()
    doc = json.loads(args.file.read_text())
    check_names(doc)
    for p in doc["points"]:
        if took := fill_point(doc, p, args.redo):
            print(f"numpy's value: {p['id']} {''.join(took)}")
    for r in doc.get("relations", []):
        fill_relation(doc, r)
    args.file.write_text(dump(doc) + "\n")
    print(f"{args.file.name}: {len(doc['points'])} points, "
          f"{len(doc.get('relations', []))} relations")  # fmt: skip


if __name__ == "__main__":
    main()
