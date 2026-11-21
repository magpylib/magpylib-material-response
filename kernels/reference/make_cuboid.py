"""Write ``cuboid.json``: named cuboids, and points by them, each with its
own id, reason, and expected B and H, for every implementation of the
cuboid's field to reproduce (#208).

The cuboid has one implementation, ``magpylib_material_response._array_api.
cuboid``. Its double result, run on NumPy, gives the points where no exact
value is defined (the conventions) and its own error, which sets a point's
``atol``. Its exact values come from the same code run at 50 digits on
``mp_namespace``: the accuracy guardrail, used only here, where speed does not
matter. It catches rounding, not a wrong formula; the relations between
fields do that. Run from the repository root, with the ``reference``
dependency group::

    python kernels/reference/make_cuboid.py

and look at the points with ``python kernels/reference/show.py``.
"""

from pathlib import Path

import mp_namespace as M
import mpmath as mp
import numpy as np
import reference_format as F

from magpylib_material_response._array_api import cuboid

mp.mp.dps = 50


def exact_cuboid(field, x, d, p):
    """The cuboid's field at 50 digits, from the one implementation, run on
    the mpmath namespace; None on a face, where B jumps by J and the side is
    the one IEEE signed zeros pick, which mpmath does not have."""
    half = np.abs(d) / 2
    plane = np.abs(np.abs(x) - half) < 1e-15 * half
    within = np.abs(x) - half < 1e-15 * half
    if (
        np.any(plane & np.roll(within, 1) & np.roll(within, 2))
        or not np.isfinite(x).all()
    ):
        return None
    args = (M.from_doubles(v[None]) for v in (x, d, p))
    return cuboid.cuboid_field(field, *args).tolist()[0]


CUBOID_D = np.array([1e-2, 2e-2, 3e-2])
CUBOID_J = np.array([0.3, -0.5, 1.2])  # every component, so every term counts

CUBOID_SOURCES = {
    "cube": (CUBOID_D, CUBOID_J, "1 x 2 x 3 cm, J with every component nonzero"),
    "flat": ((1e-2, 1e-2, 1e-4), CUBOID_J, "a plate 100 times wider than thick"),
    "long": ((1e-4, 1e-4, 1e-2), CUBOID_J, "a rod 100 times longer than wide"),
    "negative": ((-1e-2, 2e-2, 3e-2), CUBOID_J, ("a negative dimension: magpylib's masks "
                 "take abs(), its formula the raw value")),
    "nano": (CUBOID_D * 1e-7, CUBOID_J, ("cube scaled to 3 nm: B does not depend on the "
             "length unit")),
    "huge": (CUBOID_D * 1e8, CUBOID_J, "cube scaled to 3000 km"),
    "zero_dim": ((1e-2, 0.0, 1e-2), (0.0, 0.0, 1.0), "a zero dimension: field 0 by convention"),
    "zero_J": ((1e-2, 1e-2, 1e-2), (0.0, 0.0, 0.0), "zero polarization: field 0"),
}  # fmt: skip


# what a point needs beyond the formula itself: a checker against a library
# without it skips the point and says why
STABLE_EDGES = "stable edge logarithms (magpylib #1066, after 6.3.0)"


def _cuboid_points():
    """Every point of ``cuboid.json``: id, group, source, observer, why."""
    a, b, c = CUBOID_D / 2
    P = []

    def add(id_, group, source, x, why, raw=False, needs=None):
        # clean decimals, except where the last bits are the point
        x = np.array(x, float) if raw else F.clean(x)
        P.append((id_, group, source, x, why, needs))

    names = {1: "p", -1: "m"}
    octants = [1 - 2 * np.array(s) for s in np.ndindex(2, 2, 2)]
    for sx, sy, sz in octants:
        o = f"{names[sx]}{names[sy]}{names[sz]}"
        add(
            f"inside-{o}",
            "inside",
            "cube",
            (0.3 * a * sx, 0.45 * b * sy, 0.6 * c * sz),
            f"inside, octant {o} (p: +, m: -): the octant mapping flips signs per axis",
        )
    for sx, sy, sz in octants:
        o = f"{names[sx]}{names[sy]}{names[sz]}"
        add(
            f"outside-{o}",
            "outside",
            "cube",
            (1.7 * a * sx, 1.3 * b * sy, 1.9 * c * sz),
            f"outside on all three axes, octant {o}",
        )
    for ax, sign, x in (
        ("x", "+", (1.4 * a, 0.2 * b, -0.3 * c)), ("x", "-", (-1.4 * a, 0.2 * b, -0.3 * c)),
        ("y", "+", (0.25 * a, 1.6 * b, 0.1 * c)), ("y", "-", (0.25 * a, -1.6 * b, 0.1 * c)),
        ("z", "+", (-0.35 * a, 0.15 * b, 1.2 * c)), ("z", "-", (-0.35 * a, 0.15 * b, -1.2 * c)),
    ):  # fmt: skip
        add(
            f"beside-face-{ax}{sign}",
            "beside face",
            "cube",
            x,
            f"outside only along {ax}, facing the {sign}{ax} face",
        )
    for ax, x in (("z", (1.3 * a, -1.2 * b, 0.4 * c)), ("y", (-1.5 * a, 0.3 * b, 1.4 * c)),
                  ("x", (0.2 * a, 1.25 * b, -1.6 * c))):  # fmt: skip
        add(
            f"beside-edge-{ax}",
            "beside edge",
            "cube",
            x,
            f"outside along two axes, facing an edge parallel to {ax}",
        )
    add(
        "symmetry-xy",
        "symmetry plane",
        "cube",
        (0.0, -0.0, 0.5 * c),
        "inside on the planes x = 0 and y = -0.0: the octant mapping tests x < 0, y > 0, "
        "so signed zeros must not flip it",
    )
    add(
        "symmetry-xz",
        "symmetry plane",
        "cube",
        (-0.0, 0.3 * b, 0.0),
        "inside on the planes x = -0.0 and z = 0",
    )
    add(
        "center",
        "inside",
        "cube",
        (0.0, 0.0, 0.0),
        "the center: every logarithm pairs with its mirror image",
    )
    for id_, x, why in (
        ("face-x+", (a, 0.1 * b, 0.2 * c), "on the +x face"),
        ("face-x-", (-a, 0.3 * b, -0.4 * c), "on the -x face"),
        ("face-y+", (0.2 * a, b, 0.1 * c), "on the +y face"),
        ("face-y+-by-edge", (a * (1 - 1e-12), b, 0.0), "on the +y face, 1e-12 a from its edge, where a radius rounds to |q|: magpylib before #1066 lost the field here by cancellation"),
        ("face-x+-tolerance", (a * (1 + 5e-16), 0.2 * b, 0.1 * c),
         "5e-16 a outside the +x face: within magpylib's 1e-15 surface tolerance, so on it"),
    ):  # fmt: skip
        add(
            id_,
            "on face",
            "cube",
            x,
            f"{why}. B jumps by J across a face: the side taken is the one IEEE signed "
            "zeros pick in magpylib's formulas",
            raw=True,
            needs=STABLE_EDGES if id_ == "face-y+-by-edge" else None,
        )
    add(
        "beyond-tolerance-x+",
        "beside face",
        "cube",
        (a * (1 + 4e-15), 0.2 * b, 0.1 * c),
        "4e-15 a outside the +x face, beyond the surface tolerance: outside, exact",
        raw=True,
    )
    for id_, x, why in (
        ("edge-z", (a, b, 0.3 * c), "on the +x+y edge, parallel to z"),
        ("edge-x", (0.4 * a, -b, c), "on the -y+z edge, parallel to x"),
        ("corner-ppp", (a, b, c), "on the +x+y+z corner"),
        ("corner-mmm", (-a, -b, -c), "on the -x-y-z corner"),
    ):  # fmt: skip
        add(id_, "on edge", "cube", x, f"{why}: the field is 0 there by convention")
    for id_, x, why in (
        ("edge-line-x", (2 * a, b, c), "on the line of the +y+z edge, beyond its corner along x"),
        ("edge-line-y", (a, 3 * b, c), "on the line of the +x+z edge, beyond its corner along y"),
        ("edge-line-z", (a, b, -1.5 * c), "on the line of the +x+y edge, beyond its corner along z"),
    ):  # fmt: skip
        add(
            id_,
            "edge line",
            "cube",
            x,
            f"{why}: outside, exact; ln(0) is avoided by mapping to one octant (Cichon 2019)",
        )
    add(
        "face-plane-y-",
        "edge line",
        "cube",
        (-2 * a, -b, 0.5 * c),
        "in the plane of the -y face, beyond the cuboid along x: outside, exact",
    )
    add(
        "beside-edge-1e-12",
        "beside edge",
        "cube",
        (a * (1 + 1e-12), b * (1 + 1e-9), 0.0),
        "1e-12 a and 1e-9 b off the +x+y edge, outside: a radius rounds to |q|, and magpylib before #1066 lost the field here by cancellation",
        raw=True,
        needs=STABLE_EDGES,
    )
    add("far-10", "far", "cube", (0.1, 0.0, 0.0), "10 sizes away along x")
    add(
        "far-50",
        "far",
        "cube",
        (0.3, -0.2, 0.4),
        "50 sizes away: the closed form cancels, numpy's error rises above the floor",
    )
    for src, d in (("flat", (1e-2, 1e-2, 1e-4)), ("long", (1e-4, 1e-4, 1e-2))):
        h = np.array(d) / 2
        thin = int(np.argmin(d))
        add(
            f"{src}-inside",
            "shape",
            src,
            h * (0.4, -0.3, 0.2),
            f"inside the {src} cuboid",
        )
        side = h * (0.2, 0.3, 0.1)
        side[thin] = 3 * h[thin]
        add(
            f"{src}-beside",
            "shape",
            src,
            side,
            f"beside the {src} cuboid's large face, 3 half-thicknesses out",
        )
        add(
            f"{src}-outside",
            "shape",
            src,
            h * (2.5, -1.8, 1.6) if src == "long" else (2e-2, 1e-2, 1e-2),
            f"outside the {src} cuboid on all axes",
        )
    add(
        "negative-inside",
        "shape",
        "negative",
        (0.1e-2, 0.2e-2, 0.3e-2),
        "inside the cuboid with a negative dimension",
    )
    add(
        "negative-outside",
        "shape",
        "negative",
        (2e-2, -0.5e-2, 1e-2),
        "outside the cuboid with a negative dimension",
    )
    for src, k in (("nano", 1e-7), ("huge", 1e8)):
        add(
            f"{src}-inside",
            "shape",
            src,
            CUBOID_D * k * (0.15, 0.3, -0.2),
            f"inside the {src} cuboid, where cube's inside points sit, scaled",
        )
        add(
            f"{src}-outside",
            "shape",
            src,
            CUBOID_D * k * (1.2, -0.7, 0.9),
            f"outside the {src} cuboid",
        )
    add(
        "zero-dimension",
        "degenerate",
        "zero_dim",
        (1e-3, 0.0, 0.0),
        "a cuboid with a zero dimension: 0 by convention",
    )
    add(
        "zero-polarization",
        "degenerate",
        "zero_J",
        (1e-3, 0.0, 0.0),
        "a cuboid without polarization: 0",
    )
    add("observer-nan", "degenerate", "cube", (np.nan, 0.0, 0.0), "a NaN observer: NaN")
    add(
        "observer-inf",
        "degenerate",
        "cube",
        (np.inf, 0.0, 0.0),
        "an infinite observer: as IEEE arithmetic in magpylib's formulas gives it",
    )
    return P


def _cuboid_expect(field, x, d, p, exact_row):
    """numpy's value, the exact one (None by convention) and the atol it needs."""
    with np.errstate(all="ignore"):
        n = cuboid.cuboid_field(field, x[None], d[None], p[None])[0]
    if exact_row is None or not np.all(np.isfinite(n)):
        return n, None, 0.0
    e = np.array([float(v) for v in exact_row])
    err = float(
        mp.sqrt(sum((mp.mpf(float(n[k])) - exact_row[k]) ** 2 for k in range(3)))
    )
    size = float(np.linalg.norm(e))
    atol = F.decade(F.MARGIN * err) if F.MARGIN * err > F.RTOL * size else 0.0
    bound = F.RTOL * size + atol
    assert np.linalg.norm(n - e) <= bound
    if (size == 0 and atol > 0) or (size > 0 and bound > F.LOOSE * size):
        msg = f"cuboid {field} at {x}: looser than {F.LOOSE}"
        raise AssertionError(msg)
    return n, e, atol


def cuboid_file():
    """``cuboid.json``: named sources, then the points."""
    sources = {
        k: {"kind": "cuboid", "dimension": F.clean(d).tolist(),
            "polarization": F.clean(p).tolist(), "why": why}
        for k, (d, p, why) in CUBOID_SOURCES.items()
    }  # fmt: skip
    points = []
    for id_, group, src, x, why, needs in _cuboid_points():
        d, p = (F.clean(v) for v in CUBOID_SOURCES[src][:2])
        # id first, then inputs and expected values, then flags and reasons: a
        # changed value's diff has the point's id within its context lines
        point = {"id": id_, "source": src, "observer": F.encode(x)}
        conv, atols = False, {}
        for field in ("B", "H"):
            exact = exact_cuboid(field, x, d, p)
            n, e, atol = _cuboid_expect(field, x, d, p, exact)
            conv |= e is None
            point[field] = F.encode(n if e is None else e)
            if atol:
                atols[f"atol_{field}"] = atol
        point |= atols
        if conv:
            point["convention"] = True
        if needs:
            point["needs"] = needs
        point |= {"group": group, "why": why}
        points.append(point)
    ids = [p["id"] for p in points]
    assert len(set(ids)) == len(ids), "point ids must be unique"
    print(f"cuboid: {len(sources)} sources, {len(points)} points, "
          f"{sum('convention' in p for p in points)} by convention, "
          f"{sum(any(k.startswith('atol') for k in p) for p in points)} with an atol")  # fmt: skip
    return sources, points


CUBOID_HEADER = {
    "kernel": "the field of a homogeneously polarized cuboid in its own frame (centered, "
    "sides along the axes): source {dimension (3,) in m, polarization J (3,) in T}, "
    "observer (3,) in m -> B (3,) in T and H (3,) in A/m (src/magpylib_material_response/_array_api/cuboid.py, "
    "magpylib's _BHJM_magnet_cuboid written against the array API)",
    "comparison": "Per point and field F (B, H): |got - expected| <= rtol |expected| + "
    "atol_F (Euclidean norms), atol_F 0 where absent. A non-finite expected component must "
    "be matched exactly: NaN by NaN, inf by inf of the same sign. Expected values are "
    "exact (the implementation run at 50 digits on mpmath, rounded to double), except at "
    "points marked convention, where no "
    "exact value is defined and the conventions below say what decides it; there they are "
    "numpy's. atol_F appears where 16 times numpy's own error passes the rtol floor, "
    "rounded up to a power of ten. "
    "A point with needs depends on that fix: a checker against a library without it "
    "skips the point and reports it.",
    "rtol": F.RTOL,
    "conventions": [
        (
            "On a face (| |x_k| - d_k/2 | < 1e-15 d_k/2 on one axis k, and |x_j| - d_j/2 < "
            "1e-15 d_j/2 on the other two) B jumps by J: the side taken is the one IEEE signed "
            "zeros give in magpylib's formulas, which use atan2. Outside the cuboid, in a "
            "face's plane, the field is continuous and exact."
        ),
        (
            "On edges and corners, and for a zero dimension or polarization, the field is "
            "(0, 0, 0), signed as polarization * 0 (NaN stays NaN)."
        ),
        "Inside (|x| - d/2 < 1e-15 d/2 on every axis, faces included) H = (B - J) / mu_0.",
    ],
}


def main():
    sources, points = cuboid_file()
    doc = {
        "format": "magpylib-material-response field kernel reference",
        "format_version": 3,
        "family": "cuboid",
        "issue": F.ISSUE,
        # what decides the values, nothing else: the file's own commit is in git
        "generated_by": {
            "script": "kernels/reference/make_cuboid.py",
            "implementation": "src/magpylib_material_response/_array_api/cuboid.py",
            "numpy": np.__version__,
            "mpmath": mp.__version__,
            "platform": F.platform_name(),
        },
        "encoding": F.ENCODING,
        "constants": {"mu_0": cuboid.MU0},
        **CUBOID_HEADER,
        "sources": sources,
        "points": points,
    }
    out = Path(__file__).with_name("cuboid.json")
    out.write_text(F.dump(doc) + "\n")
    print(f"wrote {out.name}: {out.stat().st_size / 1e3:.0f} kB, {len(points)} points")


if __name__ == "__main__":
    main()
