"""Rectangle panels against triangle meshes: every validation case face charge
solves with cuboids, solved as it is and with each solved cuboid given as a
TriangularMesh of 12 triangles, at levels 1 to 3. Errors against the stored
references as validation.run measures them; one JSON line per solve.

    python -m validation.rect_vs_tri measure [case ...]   # appends to rect_vs_tri.jsonl
    python -m validation.rect_vs_tri figure OUT.png       # docs/_static/images/face_charge_rect_vs_tri.png
"""

from __future__ import annotations

import collections
import json
import sys
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
from scipy.spatial import ConvexHull

from magpylib_material_response import solve_face_charge as apply_face_charge

from . import references
from .cases import CASES
from .methods import METHODS

RECORDS = Path(__file__).with_suffix(".jsonl")
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def as_triangles(collection):
    """The collection with each solved cuboid a TriangularMesh of 12 triangles."""
    for src in list(collection.sources_all):
        if not isinstance(src, magpy.magnet.Cuboid):
            continue
        chi, owner = None, src
        while owner is not None and chi is None:
            chi = getattr(owner, "susceptibility", None)
            owner = owner.parent
        if chi is None or not np.any(chi):
            continue
        V = SIGNS * np.asarray(src.dimension) / 2
        F = ConvexHull(V).simplices
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        inward = np.einsum("fd,fd->f", n, V[F].mean(1)) < 0
        F[inward] = F[inward][:, [0, 2, 1]]
        mesh = magpy.magnet.TriangularMesh(
            vertices=V,
            faces=F,
            position=src.position,
            orientation=src.orientation,
            polarization=src.polarization
            if src.polarization is not None
            else (0, 0, 0),
        )
        mesh.susceptibility = float(np.ravel(chi)[0])
        if getattr(src, "H_ext", None) is not None:
            mesh.H_ext = src.H_ext
        parent = src.parent
        parent.remove(src)
        parent.add(mesh)
    return collection


def _names(given):
    """The cases asked for, or every case face charge solves."""
    return given or [
        n
        for n, c in CASES.items()
        if METHODS["FaceCharge"].can_solve(c) and not c.refused
    ]


def measure(path, names=()):
    """One JSON line per solve in `path`: each case as it is and with its solved
    cuboids as triangle meshes, levels 1 to 3."""
    out = path.open("a")
    for name in _names(list(names)):
        case = CASES[name]
        ref = references.load(case)
        for kind in ("rectangles", "triangles"):
            for level in (1, 2, 3):
                collection, groups = case.model()
                if kind == "triangles":
                    collection = as_triangles(collection)
                sensors = np.concatenate(list(groups.values()))
                t0 = time.perf_counter()
                try:
                    solved = apply_face_charge(
                        collection, level=level, **case.options.get("FaceCharge", {})
                    )
                except (NotImplementedError, ValueError) as error:
                    rec = {
                        "case": name,
                        "kind": kind,
                        "level": level,
                        "refused": str(error)[:120],
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(rec, flush=True)
                    break
                seconds = time.perf_counter() - t0
                B = magpy.getB(solved, sensors)
                errors, first = {}, 0
                for group, points in groups.items():
                    stored = ref["groups"][group]
                    b = B[first : first + len(points), stored["components"]]
                    first += len(points)
                    errors[group] = float(
                        np.linalg.norm(b - stored["B"]) / np.linalg.norm(stored["B"])
                    )
                unknowns = int(solved.face_charge_info["unknowns"])
                rec = {
                    "case": name,
                    "kind": kind,
                    "level": level,
                    "seconds": seconds,
                    "unknowns": unknowns,
                    "error": max(errors.values()),
                    "reference_error": max(v["error"] for v in ref["groups"].values()),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(
                    f"{name:14s} {kind:10s} level {level}: {seconds:6.2f} s {unknowns:7,} unknowns worst {rec['error']:.3%}",
                    flush=True,
                )
                if seconds > 30:
                    break


FIGURE_CASES = [
    ("soft_magnets", "Magnet + soft cube"),
    ("cuboids", "Three cuboids, turned"),
    ("concentrator", "Flux concentrator"),
    ("gmres_stops", "Iron cube in 10 mT"),
    ("ucore", "U-core with coil"),
    ("plate", "Magnet on soft plate"),
    ("sides_off", "Plate, sides off"),
    ("halbach", "Halbach array"),
    ("sheet", "Thin shielding sheet"),
    ("many_sizes", "Cuboids of many sizes"),
]
KINDS = [
    ("rectangles", "rectangle panels", "#2a78d6"),
    ("triangles", "the same cuboids as triangle meshes", "#9a5bd6"),
]


def _reach(rows, case, kind, target=0.01):
    for level in sorted(rows[case, kind]):
        if rows[case, kind][level]["error"] <= target:
            return rows[case, kind][level]["seconds"]
    return None


def _why_not(rows, refused, case, kind):
    reason = refused.get((case, kind), "")
    if "touch" in reason:
        return "refused: parts touch"
    if "max_unknowns" in reason and not rows[case, kind]:
        return "refused: too many unknowns at level 1"
    return "not within 1 % before max_unknowns"


def figure(path, out):
    """The figure of the docs' face-charge page from the records in `path`."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = collections.defaultdict(dict)
    refused = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "refused" in r:
            refused[r["case"], r["kind"]] = r["refused"]
        else:
            rows[r["case"], r["kind"]][r["level"]] = r
    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout="constrained")
    y = np.arange(len(FIGURE_CASES))[::-1]
    h = 0.36
    for k, (kind, label, colour) in enumerate(KINDS):
        for yi, (case, _) in zip(y, FIGURE_CASES, strict=True):
            t = _reach(rows, case, kind)
            yy = yi + (0.5 - k) * h
            if t is None:
                ax.text(
                    0.012,
                    yy,
                    _why_not(rows, refused, case, kind),
                    va="center",
                    fontsize=7.5,
                    color="#888888",
                )
                continue
            ax.barh(
                yy, t, height=h * 0.9, color=colour, label=label if yi == y[0] else None
            )
            ax.text(
                t * 1.08,
                yy,
                f"{t * 1e3:.0f} ms" if t < 1 else f"{t:.1f} s",
                va="center",
                fontsize=8,
                color="#333333",
            )
    ax.set_yticks(y, [title for _, title in FIGURE_CASES])
    ax.set_xscale("log")
    ax.set_xlim(0.01, 200)
    ax.set_xlabel(
        "time to bring every sensor group within 1 % of the FEM reference (s), M4 Pro"
    )
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.savefig(out, dpi=150)
    print("saved", out)


def main():
    warnings.simplefilter("ignore")
    if sys.argv[1] == "measure":
        measure(RECORDS, sys.argv[2:])
    else:
        figure(RECORDS, sys.argv[2])


if __name__ == "__main__":
    main()
