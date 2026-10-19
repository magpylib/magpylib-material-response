"""Print the issue's tables from the raw result files, against the reviewed references.

usage: python tables.py [../results]
"""

import json
import re
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")

# references after the adversarial review (review/REPORT-fem.md) and the domain-size runs (results/fem-ucore-3-mu1000-Rair*.json)
UCORE = 2.152  # mT; 2.154 +- 0.004 from the h-series, minus the +0.08 % Dirichlet bias measured at h = 3 mm
UCORE_BAND = 0.2  # %
CUBE = 0.02888  # A m^2; the single-N formula 0.028894 is an upper bound (+0.06 %); FEM best 0.02889 +- 0.00003
CUBE_BAND = 0.1
PLATE = 0.5633  # A m^2; range 0.562..0.5639, 0.5639 a rigorous upper bound
PLATE_LO, PLATE_HI = -0.23, +0.11  # % relative to PLATE


def rows(name):
    out = []
    for line in (R / name).read_text().splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        d = json.loads(m.group())
        rss = re.search(r"peak RSS\s+(\d+) MB", line)
        d["gb"] = int(rss.group(1)) / 1024 if rss else None
        out.append(d)
    return out


def text_rows(name):
    """The adaptive runs were saved as formatted text: '  88 cells  peak 1.3779 mT (...)  solve 0.01s  indicator 0.06s ...'."""
    out = []
    for line in (R / name).read_text().splitlines():
        m = re.match(r"\s*(\d+) cells\s+peak ([\d.]+) mT.*solve\s+([\d.]+)s\s+indicator\s+([\d.]+)s", line)
        if m:
            out.append({"cells": int(m[1]), "peak_mT": float(m[2]), "solve_s": float(m[3]), "indicator_s": float(m[4])})
    return out


def pct(v, ref):
    return 100 * (v / ref - 1)


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def table(header, lines):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(" --- " for _ in header) + "|")
    for ln in lines:
        print("| " + " | ".join(str(c) for c in ln) + " |")
    print()


print(f"References: U-core peak {UCORE} mT ±{UCORE_BAND} %; cube moment {CUBE} A m² ±{CUBE_BAND} %; plate moment {PLATE} A m² ({PLATE_LO:+.2f}..{PLATE_HI:+.2f} %)\n")

print("## U-core, uniform cells (apply_demag on #52, direct)\n")
uni = {d["cells"]: d for d in rows("uniform.txt")}
app = {(d["solver"], d["cells"]): d for d in rows("apply_cuboids.txt")}
lines = []
for c, d in sorted(uni.items()):
    a = app.get(("direct", c), d)
    lines.append(["cuboids, uniform", f"{c:,}", f"{pct(d['peak_mT'], UCORE):+.1f} %", fmt(a["solve_s"]), fmt(a.get("gb"))])
for d in rows("tet_uniform.txt"):
    lines.append(["tetrahedra, U-core as one surface", f"{d['cells']:,}", f"{pct(d['peak_mT'], UCORE):+.1f} %", fmt(d["solve_s"]), fmt(d.get("gb"))])
for d in rows("boxes_tet.txt"):
    lines.append(["tetrahedra, each cuboid on its own", f"{d['cells']:,}", f"{pct(d['peak_mT'], UCORE):+.1f} %", fmt(d["solve_s"]), fmt(d.get("gb"))])
table(["mesh", "cells", "peak error", "solve (s)", "GB"], lines)

print("## U-core, #52 direct vs iterative on the same uniform cuboid grids\n")
lines = []
for c in sorted({k[1] for k in app}):
    d, i = app.get(("direct", c)), app.get(("iterative", c))
    lines.append([f"{c:,}", fmt(d and d["solve_s"], 2), fmt(d and d.get("gb")), fmt(i and i["solve_s"], 2), fmt(i and i.get("gb"))])
table(["cells", "direct (s)", "GB", "iterative (s)", "GB"], lines)

print("## U-core, adaptive refinement (residual indicator)\n")
lines = []
for d in text_rows("residual.txt"):
    lines.append(["cuboids, adaptive", f"{d['cells']:,}", f"{pct(d['peak_mT'], UCORE):+.1f} %", fmt(d["solve_s"]), fmt(d["indicator_s"])])
for d in rows("tet_adaptive.txt"):
    lines.append(["tetrahedra, adaptive", f"{d['cells']:,}", f"{pct(d['peak_mT'], UCORE):+.1f} %", fmt(d["solve_s"]), fmt(d["indicator_s"])])
table(["mesh", "cells", "peak error", "solve (s)", "indicator (s)"], lines)

print("## U-core, linear cells (12 unknowns per cell)\n")
lines = []
for d in rows("p1real.txt"):
    if d["case"] == "ucore" and d["basis"] == "p1":
        lines.append(["linear cuboids, LSQ at 8 points (±0.4)", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(d['peak_mT'], UCORE):+.2f} %", fmt(d["s"], 2), fmt(d.get("gb"))])
for d in rows("p1_galerkin.txt"):
    if d["case"] == "ucore":
        lines.append(["linear cuboids, Galerkin, 8-point quadrature", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(d['peak_mT'], UCORE):+.2f} %", fmt(d["s"], 2), "–"])
for cells, peak, s in ((88, 2.1349, 3.8), (297, 2.1431, 20.5)):
    lines.append(["linear cuboids, Galerkin, converged quadrature (reviewer)", f"{cells:,}", f"{12 * cells:,}", f"{pct(peak, UCORE):+.2f} %", fmt(s), "–"])
for d in rows("p1tet.txt"):
    if d["case"] == "ucore" and d["basis"] == "p1":
        lines.append(["linear tetrahedra, LSQ at 8 child centroids", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(d['peak_mT'], UCORE):+.2f} %", fmt(d["s"], 2), fmt(d.get("gb"))])
for d in rows("p1tet_corner4.txt"):
    if d["case"] == "ucore":
        lab = {"corner4": f"linear tetrahedra, 4 points at f = {d['frac']:.3g}", "corner4c": "linear tetrahedra, 4 points at f = 0.5 + centroid, LSQ"}[d["colloc"]]
        lines.append([lab, f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(d['peak_mT'], UCORE):+.2f} %", fmt(d["s"], 2), "–"])
table(["scheme", "cells", "unknowns", "peak error", "solve (s)", "GB"], lines)

print("## Iron cube moment\n")
lines = []
for d in rows("p1_cube.txt"):
    if d["basis"] == "p0":
        lines.append(["cuboids, uniform", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(d['moment'], CUBE):+.2f} %", fmt(d["s"], 2)])
for d in rows("p1real.txt"):
    if d["case"] == "cube" and d["basis"] == "p1":
        lines.append(["linear cuboids, LSQ at 8 points", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(0.02889 * (1 + d['err_pct'] / 100), CUBE):+.2f} %", fmt(d["s"], 2)])
for d in rows("p1_galerkin.txt"):
    if d["case"] == "cube":
        lines.append(["linear cuboids, Galerkin, 8-point quadrature", f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(0.02889 * (1 + d['err_pct_vs_0.02889'] / 100), CUBE):+.2f} %", fmt(d["s"], 2)])
for cells, e in ((8, -3.57), (27, -1.71), (64, -1.05)):
    lines.append(["linear cuboids, Galerkin, converged quadrature (reviewer)", f"{cells:,}", f"{12 * cells:,}", f"{pct(0.02889 * (1 + e / 100), CUBE):+.2f} %", "–"])
for d in rows("p1tet.txt"):
    if d["case"] == "cube":
        lab = "tetrahedra, uniform" if d["basis"] == "p0" else "linear tetrahedra, LSQ at 8 child centroids"
        lines.append([lab, f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(0.02889 * (1 + d['err_pct'] / 100), CUBE):+.2f} %", fmt(d["s"], 2)])
for d in rows("p1tet_corner4.txt"):
    if d["case"] == "cube":
        lab = {"corner4": f"linear tetrahedra, 4 points at f = {d['frac']:.3g}", "corner4c": "linear tetrahedra, 4 points at f = 0.5 + centroid, LSQ"}[d["colloc"]]
        lines.append([lab, f"{d['cells']:,}", f"{d['unknowns']:,}", f"{pct(0.02889 * (1 + d['err_pct_vs_0.02889'] / 100), CUBE):+.2f} %", fmt(d["s"], 2)])
table(["scheme", "cells", "unknowns", "moment error", "solve (s)"], lines)

print("## Thin plate moment\n")
lines = []
for line in (R / "plate.txt").read_text().splitlines():
    m = re.match(r"(main|#52 direct|#52 iterative)\s+(cub|tet)\s+(.*?)\s+cells=\s*(\d+)\s+solve\s+([\d.]+)s\s+m_x=([\d.]+).*peak RSS\s+(\d+) MB", line)
    if m:
        code, kind, args, cells, s, mx, rss = m.groups()
        lines.append([f"{'cuboids ' + args.replace(' ', '×') if kind == 'cub' else 'tetrahedra' + (' (relaxed quality)' if 'relaxed' in args else '')}, {code}", f"{int(cells):,}", f"{pct(float(mx), PLATE):+.2f} %", s, f"{int(rss) / 1024:.2f}"])
table(["mesh, code", "cells", "moment error", "solve (s)", "GB"], lines)
