"""Measure again the numbers the docs state in prose: one probe per claim.

    python -m validation.docs_numbers --list
    python -m validation.docs_numbers fc_array     # stores its result in docs_numbers.json

Each probe builds the model its page describes, or runs the page's own code
cells, measures what the sentence claims and stores the numbers with the
versions in ``docs_numbers.json``, which the docs' figures cite. Run each in
its own process, one at a time, on a quiet machine: the times and the peak
memory are its own. ``python -m validation.rect_vs_tri`` measures the figure of
rectangle panels against triangle meshes.
"""

from __future__ import annotations

import itertools
import json
import platform
import resource
import sys
import time
from pathlib import Path

import magpylib as magpy
import numpy as np

import magpylib_material_response as mr
from magpylib_material_response import (
    UniformField,
    solve_cells,
    solve_face_charge,
    solve_fem,
)
from magpylib_material_response.face_charge import ChargeSheet, field_matrix
from magpylib_material_response.meshing import mesh_all, mesh_uniform

MM = 1e-3
REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix(".json")
PROBES = {}


def probe(f):
    PROBES[f.__name__] = f
    return f


def timed(f, *args, **kwargs):
    t = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def sheets(solved):
    return [s for s in solved.sources_all if isinstance(s, ChargeSheet)]


def unknowns(solved):
    return int(solved.face_charge_info["unknowns"])


def soft(dim, chi=1000, **kw):
    part = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=dim, **kw)
    part.susceptibility = chi
    return part


def page_namespace(page):
    """Run a docs page's code cells, as its build does, and return their names."""
    text = (REPO / "docs" / page).read_text()
    code, inside = [], False
    for line in text.splitlines():
        if line.startswith("```{code-cell}"):
            inside = True
            continue
        if inside and line.startswith("```"):
            inside = False
            continue
        if inside and not line.startswith(":"):
            code.append(line)
    ns = {"__name__": "page"}
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    magpy.show = lambda *_a, **_k: None  # the pages draw with studio's widget
    exec(compile("\n".join(code), page, "exec"), ns)  # noqa: S102
    return ns


# ── accuracy.md ─────────────────────────────────────────────────────────────
def _plate_case():
    from . import references  # noqa: PLC0415
    from .cases import CASES  # noqa: PLC0415

    case = CASES["plate"]
    ref = references.load(case)
    collection, groups = case.model()
    return collection, groups, ref


def _group_errors(B, groups, ref):
    """Per sensor |dB|/|B| by group, against the stored reference."""
    out, first = {}, 0
    for g, pts in groups.items():
        stored = ref["groups"][g]
        got = B[first : first + len(pts)][:, stored["components"]]
        first += len(pts)
        want = np.asarray(stored["B"])
        out[g] = (
            np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
        ).tolist()
    return out


@probe
def accuracy_plate():
    """jump_indicator against the worst error, refine against uniform levels,
    and the 1.25 x last-change bar per sensor, on the validation plate."""
    collection, groups, ref = _plate_case()
    sensors = np.concatenate(list(groups.values()))
    limits = {g: ref["groups"][g].get("error") for g in groups}
    rows = []
    for level, refine in [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (3, 1), (3, 2)]:
        solved, t = timed(solve_face_charge, collection, level=level, refine=refine)
        B = magpy.getB(solved, sensors)
        errs = _group_errors(B, groups, ref)
        jump = max(s.jump_indicator for s in sheets(solved))
        rows.append(
            {
                "level": level,
                "refine": refine,
                "unknowns": unknowns(solved),
                "seconds": t,
                "jump_indicator": float(jump),
                "worst": {g: max(e) for g, e in errs.items()},
                "per_sensor": errs,
                "B": B.tolist(),
            }
        )
    # the bar 1.25 |B4 - B3| against the level-4 error, per sensor
    B3 = np.asarray(rows[2]["B"])
    B4 = np.asarray(rows[3]["B"])
    bar = 1.25 * np.linalg.norm(B4 - B3, axis=1) / np.linalg.norm(B4, axis=1)
    err4 = np.concatenate([rows[3]["per_sensor"][g] for g in groups])
    return {
        "rows": [{k: v for k, v in r.items() if k != "B"} for r in rows],
        "reference_errors": limits,
        "bar_L3_L4": bar.tolist(),
        "error_L4": err4.tolist(),
        "short": int((bar < err4).sum()),
        "sensors": len(err4),
        "short_factor": float((err4 / bar).max()),
    }


@probe
def accuracy_quarter():
    """The validation plate, B 0.25 mm above and below it: the worst error against
    the jump indicator, refine from level 3 against uniform levels, against
    finite elements at order 5 (order 4 to check the reference)."""
    collection, _, _ = _plate_case()
    x = np.linspace(-9.5, 9.5, 39) * MM
    top = 2.5 * MM + 0.25 * MM  # the plate's top face is at 2.5 mm
    pts = np.array(
        [(a, -6 * MM, top) for a in x]  # beside the magnet's footprint, on the plate
        + [(a, 0, -2.75 * MM) for a in x]  # under the plate
    )
    f5, t5 = timed(solve_fem, collection, order=5, observers=pts)
    Bref = magpy.getB(f5, pts)
    B4 = magpy.getB(solve_fem(collection, order=4, observers=pts), pts)
    ref_err = float(
        (np.linalg.norm(B4 - Bref, axis=1) / np.linalg.norm(Bref, axis=1)).max()
    )
    rows = []
    for level, refine in [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (3, 1),
        (3, 2),
    ]:
        try:
            s, t = timed(solve_face_charge, collection, level=level, refine=refine)
        except ValueError as e:
            rows.append({"level": level, "refine": refine, "refused": str(e)[:120]})
            continue
        B = magpy.getB(s, pts)
        e = np.linalg.norm(B - Bref, axis=1) / np.linalg.norm(Bref, axis=1)
        jump = max(sh.jump_indicator for sh in sheets(s))
        k = len(x)
        rows.append(
            {
                "level": level,
                "refine": refine,
                "unknowns": unknowns(s),
                "seconds": t,
                "jump_indicator": float(jump),
                "worst": float(e.max()),
                "worst_top": float(e[:k].max()),
                "worst_under": float(e[k:].max()),
                "worst_at_mm": (pts[int(e.argmax())] / MM).round(2).tolist(),
                "worst_over_jump": float(e.max() / jump),
            }
        )
    return {"rows": rows, "fem5_s": t5, "fem4_vs_5_worst": ref_err}


# ── loops.md ────────────────────────────────────────────────────────────────
def _plate_magnet(shift=0.0):
    plate = soft((20 * MM, 20 * MM, 2 * MM))
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.2), dimension=(5 * MM,) * 3, position=(shift, 0, 4.5 * MM)
    )
    return magpy.Collection(plate, magnet), plate, magnet


@probe
def loops_snapshot():
    """dBz/dx at a sensor beside the magnet, solved afresh against moving the
    magnet in the solved model."""
    sensor = (5 * MM, 0, 1.5 * MM)  # 0.5 mm above the plate, 2.5 mm off the magnet
    h = 0.1 * MM
    out = {}
    fresh = []
    for s in (-h, h):
        model, _, _ = _plate_magnet(s)
        fresh.append(magpy.getB(solve_face_charge(model), sensor)[2])
    out["face_charge_mT_per_mm"] = (fresh[1] - fresh[0]) / (2 * h) * 1e3 * MM
    model, _, _ = _plate_magnet(0)
    solved = solve_face_charge(model)
    moved = []
    for s in (-h, h):
        magnet = solved.children[1]
        magnet.position = (s, 0, 4.5 * MM)
        moved.append(magpy.getB(solved, sensor)[2])
    out["moved_in_solved_mT_per_mm"] = (moved[1] - moved[0]) / (2 * h) * 1e3 * MM
    alone = [magpy.getB(_plate_magnet(s)[2], sensor)[2] for s in (-h, h)]
    out["magnet_alone_mT_per_mm"] = (alone[1] - alone[0]) / (2 * h) * 1e3 * MM
    H = 0.25 * MM
    fem = []
    for s in (-H, H):
        model, _, _ = _plate_magnet(s)
        fem.append(magpy.getB(solve_fem(model, order=3), sensor)[2])
    out["fem_order3_mT_per_mm_step_0.25"] = (fem[1] - fem[0]) / (2 * H) * 1e3 * MM
    return out


@probe
def loops_reuse():
    """Time per call: plain solve, first call with reuse, later calls."""
    rows = []
    for level in (2, 3, 4, 5):
        model, _, magnet = _plate_magnet(0)
        plain = []
        for _ in range(3):
            s, t = timed(solve_face_charge, model, level=level)
            plain.append(t)
        n = unknowns(s)
        first, t_first = timed(solve_face_charge, model, level=level, reuse=True)
        later = []
        solved = first
        for k in range(5):
            magnet.polarization = (0, 0, 1.0 + 0.05 * k)
            solved, t = timed(solve_face_charge, model, level=level, reuse=solved)
            assert solved.face_charge_info["reused"]
            later.append(t)
        rows.append(
            {
                "level": level,
                "unknowns": n,
                "plain_s": min(plain),
                "first_reuse_s": t_first,
                "later_s": float(np.median(later)),
            }
        )
    return {"rows": rows}


@probe
def loops_field_matrix():
    model, _, _ = _plate_magnet(0)
    solved = solve_face_charge(model, level=1)
    sh = sheets(solved)
    x = np.linspace(-15, 15, 51) * MM
    pts = np.array([(a, b, -3 * MM) for a in x for b in x])
    fm, t_build = timed(field_matrix, sh, pts)
    B_get, t_get = timed(magpy.getB, sh, pts, sumup=True)
    B_fm, _ = timed(fm, sh)
    t_fm = min(timed(fm, sh)[1] for _ in range(20))
    t_get = min(timed(magpy.getB, sh, pts, sumup=True)[1] for _ in range(5))
    fm32 = field_matrix(sh, pts, dtype=np.float32)
    far = pts * 2000 * 0.02 / np.abs(pts).max()  # 2,000 plate sizes out
    fm_far = field_matrix(sh, far)
    fm_far32 = field_matrix(sh, far, dtype=np.float32)
    B_far = fm_far(sh)
    return {
        "triangles": int(sum(len(s.faces) for s in sh)),
        "unknowns": unknowns(solved),
        "points": len(pts),
        "getB_s": t_get,
        "matrix_product_s": t_fm,
        "build_s": t_build,
        "agree_rel_max": float(np.abs(B_fm - B_get).max() / np.abs(B_get).max()),
        "matrix_MB": fm.matrix.nbytes / 1e6,
        "matrix32_MB": fm32.matrix.nbytes / 1e6,
        "f32_near_rel": float(np.abs(fm32(sh) - B_get).max() / np.abs(B_get).max()),
        "f32_far_rel": float(np.abs(fm_far32(sh) - B_far).max() / np.abs(B_far).max()),
    }


# ── reading.md ──────────────────────────────────────────────────────────────
@probe
def reading_force():
    ns = page_namespace("guide/reading.md")
    model, magnet, sheet = ns["model"], ns["magnet"], ns["sheet"]
    out = {"face_charge": {}, "cells": {}}
    for n in (1000, 2000, 4000, 8000, 16000):
        out["face_charge"][n] = float(-magpy.getFT(sheet, magnet, meshing=n)[0][2])
    for across in (3, 6):
        solved, t = timed(solve_cells, mesh_uniform(model, across))
        force = magpy.getFT(solved.children[1], solved.children[0], meshing=2000)[0]
        out["cells"][across] = {"pull_N": float(-force[2]), "seconds": t}
    return out


# ── saturation.md ───────────────────────────────────────────────────────────
@probe
def saturation_pin():
    ns = page_namespace("guide/saturation.md")
    model, tip = ns["model"], ns["tip"]
    out = {}
    s8 = ns["saturated"]
    out["cells_8_mT"] = float(magpy.getB(s8, tip)[2] * 1e3)
    s16, t16 = timed(solve_cells, mesh_uniform(model, 16))
    out["cells_16"] = {
        "mT": float(magpy.getB(s16, tip)[2] * 1e3),
        "seconds": t16,
        "cells": len(s16.children[1].sources_all),
    }
    for order in (3, 4):
        try:
            sf, t = timed(solve_fem, model, order=order, observers=np.array([tip]))
            out[f"fem_order{order}"] = {
                "mT": float(magpy.getB(sf, tip)[2] * 1e3),
                "seconds": t,
                "info": {k: str(v) for k, v in getattr(sf, "fem_info", {}).items()},
            }
        except Exception as e:  # noqa: BLE001
            out[f"fem_order{order}"] = repr(e)
    return out


# ── face_charge.md ──────────────────────────────────────────────────────────
@probe
def fc_memory():
    """Peak RSS of one large part solved just under max_unknowns."""
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 16  # 18,432 unknowns
    part = soft((10 * MM, 10 * MM, 10 * MM))
    model = magpy.Collection(part, UniformField(B=(0, 0, 0.01)))
    solved, t = timed(solve_face_charge, model, level=level)
    return {
        "level": level,
        "unknowns": unknowns(solved),
        "seconds": t,
        "peak_GB": rss_gb(),
    }


@probe
def fc_disc():
    """A disc magnet on a soft cube: face charge levels against finite elements."""
    cube = soft((10 * MM,) * 3)
    disc = magpy.magnet.Cylinder(
        polarization=(0, 0, 1.2), dimension=(6 * MM, 2 * MM), position=(0, 0, 6 * MM)
    )
    model = magpy.Collection(cube, disc)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    near = np.array(
        [(3.5 * MM * np.cos(a), 3.5 * MM * np.sin(a), 5.5 * MM) for a in ang]
    )
    below = np.array([(0, 0, -6 * MM), (2 * MM, 0, -5.5 * MM)])
    pts = np.vstack([near, below])
    ref, t_ref = timed(solve_fem, model, order=5, observers=pts)
    Bref = magpy.getB(ref, pts)
    ref4 = magpy.getB(solve_fem(model, order=4, observers=pts), pts)
    rows = []
    for level in (2, 3, 4, 5):
        s, t = timed(solve_face_charge, model, level=level)
        B = magpy.getB(s, pts)
        e = np.linalg.norm(B - Bref, axis=1) / np.linalg.norm(Bref, axis=1)
        rows.append(
            {
                "level": level,
                "unknowns": unknowns(s),
                "seconds": t,
                "near_worst": float(e[: len(near)].max()),
                "near_rms": float(np.sqrt((e[: len(near)] ** 2).mean())),
                "below_worst": float(e[len(near) :].max()),
            }
        )
    return {
        "rows": rows,
        "fem5_seconds": t_ref,
        "fem4_vs_5_near": float(
            (np.linalg.norm(ref4 - Bref, axis=1) / np.linalg.norm(Bref, axis=1))[
                : len(near)
            ].max()
        ),
    }


@probe
def fc_array():
    """25 cube magnets on a soft yoke: unknowns per level, level 2 against 3 and 4."""
    yoke = soft((50 * MM, 50 * MM, 5 * MM))
    mags = []
    for i in range(5):
        for j in range(5):
            m = magpy.magnet.Cuboid(
                polarization=(0, 0, 1.2 * (-1) ** (i + j)),
                dimension=(5 * MM,) * 3,
                position=((i - 2) * 10 * MM, (j - 2) * 10 * MM, 5 * MM),
            )
            mags.append(m)
    model = magpy.Collection(yoke, *mags)
    x = np.linspace(-22, 22, 23) * MM
    between = (np.arange(4) - 1.5) * 10 * MM
    near = np.array(
        [(a, 0, 8.5 * MM) for a in x]  # 1 mm above the magnets
        + [
            (a, b, 3 * MM) for a in between for b in between
        ]  # 0.5 mm over the yoke, between
    )
    rows, Bs = [], {}
    for level in (2, 3, 4):
        try:
            s, t = timed(solve_face_charge, model, level=level, max_unknowns=30_000)
        except Exception as e:  # noqa: BLE001
            rows.append({"level": level, "refused": repr(e)[:200]})
            continue
        Bs[level] = magpy.getB(s, near)
        rows.append({"level": level, "unknowns": unknowns(s), "seconds": t})
    diffs = {}
    for hi in (3, 4):
        if hi in Bs:
            d = np.linalg.norm(Bs[2] - Bs[hi], axis=1) / np.linalg.norm(Bs[hi], axis=1)
            diffs[f"L2_vs_L{hi}_above_worst"] = float(d[: len(x)].max())
            diffs[f"L2_vs_L{hi}_between_worst"] = float(d[len(x) :].max())
            diffs[f"L2_vs_L{hi}_between_rms"] = float(
                np.sqrt((d[len(x) :] ** 2).mean())
            )
    return {"rows": rows, **diffs}


@probe
def fc_thin_sheet():
    """A 0.5 mm plate under a magnet: level 2 against level 3."""
    plate = soft((10 * MM, 10 * MM, 0.5 * MM))
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.2), dimension=(5 * MM,) * 3, position=(0, 0, 3.75 * MM)
    )
    model = magpy.Collection(plate, magnet)
    x = np.linspace(-4, 4, 17) * MM
    pts = np.vstack([[(a, 0, -1 * MM) for a in x], [(a, 0, 1 * MM) for a in x]])
    out = {}
    for level in (2, 3):
        s, t = timed(solve_face_charge, model, level=level)
        t = min(t, *(timed(solve_face_charge, model, level=level)[1] for _ in range(2)))
        out[level] = {"unknowns": unknowns(s), "seconds": t, "B": magpy.getB(s, pts)}
    d = np.linalg.norm(out[2]["B"] - out[3]["B"], axis=1) / np.linalg.norm(
        out[3]["B"], axis=1
    )
    return {
        "L2": {k: v for k, v in out[2].items() if k != "B"},
        "L3": {k: v for k, v in out[3].items() if k != "B"},
        "L2_vs_L3_worst": float(d.max()),
        "L2_vs_L3_group_norm": rel(out[2]["B"], out[3]["B"]),
    }


@probe
def fc_sphere():
    """A soft sphere in a uniform field: face charge levels 1 to 4 against the
    exact field, 1 mm off its surface."""
    R, chi, B0 = 5 * MM, 1000.0, 0.01
    ball = magpy.magnet.Sphere(polarization=(0, 0, 0), diameter=2 * R)
    ball.susceptibility = chi
    model = magpy.Collection(ball, UniformField(B=(0, 0, B0)))
    n = 2000
    i = np.arange(n) + 0.5
    phi, th = np.pi * (1 + 5**0.5) * i, np.arccos(1 - 2 * i / n)
    dirs = np.c_[np.cos(phi) * np.sin(th), np.sin(phi) * np.sin(th), np.cos(th)]
    axes = np.array([(0, 0, 1), (1, 0, 0)], float)  # along the field, across it
    r = R + 1 * MM
    pts = r * np.vstack([dirs, axes])
    J = 3 * chi / (chi + 3) * B0  # the sphere's uniform polarization, T
    exact = magpy.getB(magpy.magnet.Sphere(polarization=(0, 0, J), diameter=2 * R), pts)
    exact = exact + np.array([0, 0, B0])
    rows = []
    for level in (1, 2, 3, 4):
        s, t = timed(solve_face_charge, model, level=level)
        B = magpy.getB(s, pts)
        e = np.linalg.norm(B - exact, axis=1) / np.linalg.norm(exact, axis=1)
        rows.append(
            {
                "level": level,
                "unknowns": unknowns(s),
                "seconds": t,
                "worst": float(e[:n].max()),
                "along_axis": float(e[n]),
                "across_axis": float(e[n + 1]),
            }
        )
    return {"rows": rows}


@probe
def hundred_cubes_l2():
    """The validation's 100 small bodies at face-charge level 2, max_unknowns raised."""
    from . import references  # noqa: PLC0415
    from .cases import CASES  # noqa: PLC0415

    case = CASES["small_bodies"]
    ref = references.load(case)
    collection, groups = case.model()
    sensors = np.concatenate(list(groups.values()))
    s, t = timed(solve_face_charge, collection, level=2, max_unknowns=40_000)
    B = magpy.getB(s, sensors)
    group = {}
    first = 0
    for g, pts in groups.items():
        stored = ref["groups"][g]
        b = B[first : first + len(pts)][:, stored["components"]]
        first += len(pts)
        group[g] = rel(b, np.asarray(stored["B"]))
    return {
        "unknowns": unknowns(s),
        "seconds": t,
        "peak_GB": rss_gb(),
        "group_errors": group,
        "reference_errors": {g: ref["groups"][g].get("error") for g in groups},
    }


def _u_mesh():
    """The U-core's iron as one closed triangle mesh: the U outline in xz, 20 mm deep."""
    xz = (
        np.array(
            [
                (-50, -10),
                (50, -10),
                (50, 10),
                (50, 70),
                (30, 70),
                (30, 10),
                (-30, 10),
                (-30, 70),
                (-50, 70),
                (-50, 10),
            ],
            float,
        )
        * MM
    )
    caps = [
        (0, 1, 5),
        (1, 2, 5),
        (0, 5, 6),
        (0, 6, 9),
        (9, 6, 7),
        (9, 7, 8),
        (5, 2, 3),
        (5, 3, 4),
    ]
    n = len(xz)
    V = np.vstack(
        [
            np.c_[xz[:, 0], np.full(n, -10 * MM), xz[:, 1]],
            np.c_[xz[:, 0], np.full(n, 10 * MM), xz[:, 1]],
        ]
    )
    F = [*caps, *((a + n, c + n, b + n) for a, b, c in caps)]
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, j + n), (i, j + n, i + n)]
    return V, np.array(F)


@probe
def ucore_closed_mesh():
    """The U-core's iron as one closed TriangularMesh with the case's coil, face
    charge levels 1 to 3 against the validation reference."""
    from . import references  # noqa: PLC0415
    from .cases import CASES  # noqa: PLC0415

    case = CASES["ucore"]
    ref = references.load(case)
    collection, groups = case.model()
    coil = [c for c in collection.sources_all if not isinstance(c, magpy.magnet.Cuboid)]
    V, F = _u_mesh()
    core = magpy.magnet.TriangularMesh(
        vertices=V, faces=F, polarization=(0, 0, 0), check_open="raise"
    )
    core.susceptibility = 999
    model = magpy.Collection(core, *coil, override_parent=True)
    sensors = np.concatenate(list(groups.values()))
    rows = []
    for level in (1, 2, 3):
        s, t = timed(solve_face_charge, model, level=level)
        B = magpy.getB(s, sensors)
        group, first = {}, 0
        for g, pts in groups.items():
            stored = ref["groups"][g]
            b = B[first : first + len(pts)][:, stored["components"]]
            first += len(pts)
            group[g] = rel(b, np.asarray(stored["B"]))
        rows.append(
            {"level": level, "unknowns": unknowns(s), "seconds": t, "errors": group}
        )
    return {
        "rows": rows,
        "reference_errors": {g: ref["groups"][g].get("error") for g in groups},
        "coil_sources": len(coil),
    }


# ── fem.md ──────────────────────────────────────────────────────────────────
@probe
def fem_shield_tube():
    """The shield tube at order 4: unknowns, seconds and the peak, in its own process."""
    from .cases import CASES  # noqa: PLC0415
    from .methods import METHODS  # noqa: PLC0415

    case = CASES["shield_tube"]
    collection, groups = case.model()
    sensors = np.concatenate(list(groups.values()))
    solved, t = timed(
        METHODS["FEM"].solve, collection, sensors, 4, **case.options.get("FEM", {})
    )
    return {
        "seconds": t,
        "peak_GB": rss_gb(),
        "counts": {k: str(v) for k, v in vars(solved).items() if k != "B"},
    }


# ── inside/cells_equations.md ───────────────────────────────────────────────
def _cube_in_field():
    cube = soft((10 * MM,) * 3)
    return magpy.Collection(cube, UniformField(B=(0, 0, 0.01)))


def _moment(solved):
    cells = [
        c
        for c in solved.sources_all
        if hasattr(c, "volume") and not isinstance(c, UniformField)
    ]
    return float(sum(c.polarization[2] * c.volume for c in cells) / magpy.mu_0)


@probe
def cells_rules():
    """Point matching against volume averaging (newell): the soft cube's moment
    and the field above it at 1,728 cells; a thin plate's moment with refinement."""
    REF_MOMENT = 0.028880
    model = _cube_in_field()
    above = np.array([(0, 0, 6 * MM), (3 * MM, 0, 6 * MM), (0, 0, 8 * MM)])
    fem = solve_fem(model, order=5, observers=above)
    Bref = magpy.getB(fem, above)
    out = {"cube_1728": {}}
    for rule in ("point", "newell"):
        s = solve_cells(mesh_uniform(model, 12), interaction=rule)
        out["cube_1728"][rule] = {
            "moment_err": _moment(s) / REF_MOMENT - 1,
            "field_above_err": rel(magpy.getB(s, above), Bref),
        }
    # 10 mm wide and 1 mm thick: k cells across the thickness are 100 k^3 cells
    plate = magpy.Collection(
        soft((10 * MM, 10 * MM, 1 * MM)), UniformField(B=(0.01, 0, 0))
    )

    def mx(s):
        cells = [c for c in s.sources_all if hasattr(c, "volume")]
        return float(sum(c.polarization[0] * c.volume for c in cells) / magpy.mu_0)

    plate_rows = []
    for across in (1, 2, 3, 4):
        row = {"across_thickness": across}
        for rule in ("point", "newell"):
            s = solve_cells(mesh_uniform(plate, across), interaction=rule)
            row[rule] = mx(s)
            row["cells"] = len([c for c in s.sources_all if hasattr(c, "volume")])
        plate_rows.append(row)
    fc = solve_face_charge(plate, level=5)
    fc_sheet = sheets(fc)[0]
    out["plate_fc_level5_moment"] = float(
        fc_sheet.mean_polarization[0] * 10 * 10 * 1 * MM**3 / magpy.mu_0
    )
    out["plate"] = plate_rows
    return out


@probe
def cells_rule_jump():
    """A magnet on an iron block: B under each rule, the jump a mixed rule would make."""
    block = soft((10 * MM, 10 * MM, 5 * MM))
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1.2), dimension=(5 * MM,) * 3, position=(0, 0, 5 * MM)
    )
    magnet.susceptibility = 0.05
    model = magpy.Collection(block, magnet)
    pts = np.array([(0, 0, 8 * MM), (4 * MM, 0, 3 * MM), (0, 0, -4 * MM)])
    B = {}
    for rule in ("point", "newell"):
        B[rule] = magpy.getB(solve_cells(mesh_uniform(model, 4), interaction=rule), pts)
    return {
        "point_vs_newell": rel(B["point"], B["newell"]),
        "per_point": (
            np.linalg.norm(B["point"] - B["newell"], axis=1)
            / np.linalg.norm(B["newell"], axis=1)
        ).tolist(),
    }


def _ball(diameter, position=(0, 0, 0), n=200):
    """A ball as a closed TriangularMesh, which mesh_all cuts into tetrahedra."""
    i = np.arange(n) + 0.5
    phi, th = np.pi * (1 + 5**0.5) * i, np.arccos(1 - 2 * i / n)
    pts = (
        diameter
        / 2
        * np.c_[np.cos(phi) * np.sin(th), np.sin(phi) * np.sin(th), np.cos(th)]
    )
    ball = magpy.magnet.TriangularMesh.from_ConvexHull(
        points=pts, polarization=(0, 0, 0), position=position
    )
    ball.susceptibility = 1000
    return ball


@probe
def tube_92():
    """The thin ring of #92, 16 cells around and 16 along: its moment against
    finite elements with 1 to 3 cells across the wall, with the averaging of
    close cylinder-segment pairs and with point matching alone."""
    import magpylib_material_response.cells as C  # noqa: PLC0415

    fem = 1.23753404e-07  # T m^3, the same on two FEM meshes to 0.002 %
    r1, r2, h = 4 * MM, 5 * MM, 30 * MM

    def model(nr, nphi=16, nz=16):
        src = magpy.magnet.Cuboid(
            polarization=(1, 0, 0), dimension=(10 * MM,) * 3, position=(15 * MM, 0, 0)
        )
        src.susceptibility = 0
        radii = np.linspace(r1, r2, nr + 1)
        angles = np.linspace(0, 360, nphi + 1)
        zs = np.linspace(-h / 2, h / 2, nz + 1)
        ring = magpy.Collection()
        for a, b in itertools.pairwise(radii):
            for p, q in itertools.pairwise(angles):
                for z0, z1 in itertools.pairwise(zs):
                    ring.add(
                        magpy.magnet.CylinderSegment(
                            polarization=(0, 0, 0),
                            dimension=(a, b, z1 - z0, p, q),
                            position=(0, 0, (z0 + z1) / 2),
                        )
                    )
        ring.susceptibility = 1000
        return magpy.Collection(src, ring)

    def moment(solved):
        return sum(c.polarization[0] * c.volume for c in solved.children[1].sources_all)

    out = {}
    for nr in (1, 2, 3):
        averaged = moment(solve_cells(model(nr)))
        reach, C.SEGMENT_AVERAGE_REACH = C.SEGMENT_AVERAGE_REACH, 0.0
        try:
            point = moment(solve_cells(model(nr)))
        finally:
            C.SEGMENT_AVERAGE_REACH = reach
        out[nr] = {"averaged": averaged / fem - 1, "point": point / fem - 1}
    return out


@probe
def cells_tetra_gmres():
    """GMRES iterations on a grid and tetrahedra at chi 1000, with the tr/3
    blocks of the tetrahedra and with their blocks left out."""
    import magpylib_material_response.cells as C  # noqa: PLC0415

    counts = []
    real = C.gmres

    def counting(*a, **k):
        n = [0]
        k["callback"] = lambda _r: n.__setitem__(0, n[0] + 1)
        k["callback_type"] = "pr_norm"
        out = real(*a, **k)
        counts.append(n[0])
        return out

    C.gmres = counting
    grid = soft((10 * MM, 10 * MM, 10 * MM))
    ball = _ball(8 * MM, position=(12 * MM, 0, 0))
    model = magpy.Collection(grid, ball, UniformField(B=(0.01, 0, 0)))
    meshed = mesh_all(model, 1650)
    kinds = {}
    for c in meshed.sources_all:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    out = {"cells": kinds}
    solve_cells(meshed, solver="iterative", max_iter=3000)
    out["with_tr3"] = counts[-1]
    real_self = C._self_blocks

    def no_generic(n, clusters, interaction, ops):
        return real_self(
            n, clusters, interaction, [o for o in ops if o.get("self") != "generic"]
        )

    C._self_blocks = no_generic
    try:
        solve_cells(meshed, solver="iterative", max_iter=1000)
        out["without"] = counts[-1]
    except RuntimeError as e:
        out["without"] = f"no convergence in 1000: {e}"[:150]
    finally:
        C._self_blocks = real_self
        C.gmres = real
    return out


@probe
def cells_threads():
    """A tetrahedra-only solve with magpylib's threads against one thread."""
    ball = _ball(10 * MM)
    model = magpy.Collection(ball, UniformField(B=(0.01, 0, 0)))
    meshed = mesh_all(model, 1000)  # about 1,900 tetrahedra, as the table row
    n = len([c for c in meshed.sources_all if hasattr(c, "volume")])
    default = magpy.defaults.threads
    t_many = min(timed(solve_cells, meshed, solver="direct")[1] for _ in range(2))
    magpy.defaults.threads = 1
    try:
        t_one = min(timed(solve_cells, meshed, solver="direct")[1] for _ in range(2))
    finally:
        magpy.defaults.threads = default
    return {
        "cells": n,
        "threads": str(default),
        "seconds_threads": t_many,
        "seconds_one": t_one,
        "ratio": t_one / t_many,
    }


# ── examples/fem_check.md ───────────────────────────────────────────────────
@probe
def fem_check_16():
    ns = page_namespace("examples/fem_check.md")
    model, line, fem_study = ns["model"], ns["line"], ns["fem_study"]
    s, t = timed(solve_cells, mesh_uniform(model, 16))
    i = int(np.argmin(np.abs(line[:, 0] - 1.5 * MM)))
    Bc = magpy.getB(s, line[i])
    Bf = np.asarray(fem_study.values[-1])[i] if hasattr(fem_study, "values") else None
    return {
        "cells": len([c for c in s.sources_all if hasattr(c, "volume")]),
        "seconds": t,
        "x_mm": float(line[i, 0] / MM),
        "cells_B": Bc.tolist(),
        "fem_value": None if Bf is None else np.asarray(Bf).tolist(),
        "fem_error_bar": float(np.asarray(fem_study.errors)[i])
        if hasattr(fem_study, "errors")
        else None,
    }


# ── map: the U-core iron alone ──────────────────────────────────────────────
@probe
def ucore_iron_alone():
    """The U-core's iron alone in a uniform field along the core: cells at 2, 4 and
    8 across against finite elements, 30 mm above the poles."""
    sys.path.insert(0, str(REPO / "docs" / "examples"))
    import ucore_model  # noqa: PLC0415

    line = ucore_model.line_above_poles()
    out = {}
    for name, B in (("x", (0.01, 0, 0)), ("z", (0, 0, 0.01))):
        model = magpy.Collection(ucore_model.core(), UniformField(B=B))
        fem4 = magpy.getB(solve_fem(model, order=4, observers=line), line)
        fem5, t5 = timed(solve_fem, model, order=5, observers=line)
        Bref = magpy.getB(fem5, line)
        rows = []
        for across in (2, 4, 8):
            s, t = timed(solve_cells, mesh_uniform(model, across))
            rows.append(
                {
                    "across": across,
                    "cells": len([c for c in s.sources_all if hasattr(c, "volume")]),
                    "seconds": t,
                    "err": rel(magpy.getB(s, line), Bref),
                }
            )
        out[name] = {"rows": rows, "fem4_vs_5": rel(fem4, Bref), "fem5_s": t5}
    return out


# ── guide/cells.md and saturation.md: solves of the B-H iteration ───────────
@probe
def bh_solves():
    """How many linear solves the B-H iteration takes: the pin of the saturation
    guide and the holding magnet's plate, with cells and finite elements."""
    from loguru import logger  # noqa: PLC0415

    import magpylib_material_response.fem._solve as F  # noqa: PLC0415

    ns = page_namespace("guide/saturation.md")
    steel = ns["steel"]
    sys.path.insert(0, str(REPO / "docs" / "examples"))
    from holding_magnet_study import holding_magnet  # noqa: PLC0415

    def pin():
        return ns["model"]

    def holder(thickness):
        def build():
            _, model = holding_magnet(gap=0.0, thickness=thickness)
            model.children[1].susceptibility = steel
            return model

        return build

    lines = []
    logger.enable("magpylib_material_response")
    logger.add(lambda m: lines.append(str(m)), level="INFO")
    calls = [0]
    real = F.ng.Integrate

    def counting(*a, **k):
        calls[0] += bool(k.get("element_wise"))  # the B-H loop's, one per solve
        return real(*a, **k)

    F.ng.Integrate = counting
    out = {}
    for name, build in (("pin", pin), ("holder_2mm", holder(2.0))):
        for across in (4, 8, 16) if name == "pin" else (2, 3):
            lines.clear()
            try:
                solve_cells(mesh_uniform(build(), across))
                out[f"{name}_cells_{across}"] = [l for l in lines if "Saturation" in l][
                    -1:
                ]
            except Exception as e:  # noqa: BLE001
                out[f"{name}_cells_{across}"] = repr(e)[:150]
        for order in (3, 4) if name == "pin" else (3,):
            calls[0] = 0
            try:
                solve_fem(build(), order=order)
                out[f"{name}_fem_{order}"] = (
                    calls[0] - 1
                )  # one Integrate before the loop
            except Exception as e:  # noqa: BLE001
                out[f"{name}_fem_{order}"] = repr(e)[:150]
    F.ng.Integrate = real
    return out


def main():
    if sys.argv[1] == "--list":
        print("\n".join(PROBES))
        return
    name = sys.argv[1]
    result, seconds = timed(PROBES[name])
    stored = json.loads(OUT.read_text()) if OUT.exists() else {}
    stored[name] = {
        "doc": PROBES[name].__doc__,
        "result": result,
        "wall_s": seconds,
        "peak_GB": rss_gb(),
        "versions": {
            "magpylib_material_response": mr.__version__,
            "magpylib": magpy.__version__,
            "threads": str(magpy.defaults.threads),
            "machine": platform.platform(),
        },
    }
    OUT.write_text(
        json.dumps(dict(sorted(stored.items())), indent=1, default=str) + "\n"
    )
    print(json.dumps(stored[name]["result"], default=str)[:2000])


if __name__ == "__main__":
    main()
