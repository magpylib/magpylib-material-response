"""Step 3a: a soft-iron cylinder exported with N flat facets, against the exact
Cylinder. Writes results/step3_cylinder.json (one entry per run, resumable).

R = 5 mm, H = 10 mm, chi = 1000, uniform mu0 H = 10 mT along z (axis) or x.
Observers: 1 mm above the top face near the rim, 1 mm beside the mantle, and
50 mm away along the field (dipole-dominated: a proxy for the moment).
Face charge: N = 32, 64, 128 sides, m = 2, 3 (and 4 for 64, 128). MoM: material-response's
mesh_study on the exact Cylinder (cylinder-segment cells), 4, 8, 16 across.
"""
import json, os, sys, time
import numpy as np
import magpylib as magpy
HERE = os.path.dirname(os.path.abspath(__file__))
from meshes import cylinder  # noqa: E402
from refine import refine_rings  # noqa: E402
from tri_solver import MU0, TriBody  # noqa: E402

R, H, CHI, B0 = 0.005, 0.01, 1000.0, 0.01
OBS = np.array([[0.8 * R, 0, H / 2 + 1e-3], [R + 1e-3, 0, 0.3 * H], [0, 0, 0.05], [0.05, 0, 0]])
OUT = os.path.join(HERE, "results", "step3_cylinder.json")
done = json.load(open(OUT)) if os.path.exists(OUT) else []
have = {(r["method"], r["field"], r.get("N"), r.get("m")) for r in done}


def save(rec):
    done.append(rec)
    json.dump(done, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print({k: v for k, v in rec.items() if k != "B"}, flush=True)


for axis, name in ((2, "z"), (0, "x")):
    H0 = np.zeros(3); H0[axis] = B0 / MU0
    for N in (32, 64, 128):
        v, f = cylinder(N, R, H)
        for m in (2, 3) if N == 32 else (2, 3, 4):
            if ("face charge", name, N, m) in have:
                continue
            t = time.perf_counter()
            T = refine_rings(v, f, R, m)
            b = TriBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), chi=CHI)
            c = b.solve(load=b.load(lambda p: np.tile(H0, (len(p), 1))))
            save({"method": "face charge", "field": name, "N": N, "m": m, "unknowns": b.Nd,
                  "seconds": time.perf_counter() - t, "moment": b.moment(c).tolist(),
                  "B": b.B(OBS, c).tolist()})
    if ("MoM", name, None, None) not in have:
        from magpylib_material_response.convergence import mesh_study
        cyl = magpy.magnet.Cylinder(dimension=(2 * R, H), polarization=(0, 0, 0))
        cyl.susceptibility = CHI
        cyl.H_ext = tuple(B0 * np.eye(3)[axis])
        t = time.perf_counter()
        study = mesh_study(magpy.Collection(cyl), OBS, (4, 8, 16))
        save({"method": "MoM", "field": name, "N": None, "m": None, "cells": [int(x) for x in study.cells],
              "seconds": time.perf_counter() - t, "levels": [np.asarray(x).tolist() for x in study.values],
              "B": None if study.estimate is None else np.asarray(study.estimate).tolist(),
              "errors": None if study.errors is None else np.atleast_1d(study.errors).tolist()})

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE, AQUA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=170)
fig.patch.set_facecolor(SURF)
for ax in (a1, a2, a3):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fc = {(r["field"], r["N"], r["m"]): r for r in done if r["method"] == "face charge"}
mom = {r["field"]: r for r in done if r["method"] == "MoM"}

Ns = np.array([32, 64, 128])
for name, color in (("z", BLUE), ("x", AQUA)):
    axis = 2 if name == "z" else 0
    mz = np.array([fc[(name, N, 3)]["moment"][axis] for N in Ns])
    d1, d2 = mz[1] - mz[0], mz[2] - mz[1]
    p = np.log2(d1 / d2)  # observed order in 1/N
    limit = mz[2] + d2 / (2**p - 1)
    a1.loglog(Ns, np.abs(mz / limit - 1), "o-", color=color, lw=1.4,
              label=f"moment, field along {name} (order {p:.2f})")
a1.loglog(Ns, 1 - Ns / (2 * np.pi) * np.sin(2 * np.pi / Ns), "s:", color=INK2, lw=1.1, label="volume the facets cut off")
a1.set_xticks(Ns, [str(N) for N in Ns]); a1.minorticks_off()
a1.set_yticks([3e-4, 1e-3, 3e-3, 1e-2], ["3e-4", "1e-3", "3e-3", "1e-2"])
a1.set_xlabel("sides of the exported cylinder"); a1.set_ylabel("relative error against infinitely many sides")
a1.set_title("Faceting (face charge, m = 3)", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=8)

for ax, k, where in ((a2, 0, "1 mm above the top face, 1 mm inside the rim"), (a3, 1, "1 mm beside the mantle, 3 mm above the middle")):
    for name, color in (("z", BLUE), ("x", AQUA)):
        pts = [fc[(name, 128, m)] for m in (2, 3, 4) if (name, 128, m) in fc]
        ax.plot([r["unknowns"] for r in pts], [np.linalg.norm(r["B"][k]) * 1e3 for r in pts], "o-", color=color, lw=1.4,
                     label=f"face charge, 128 sides, m = 2–{pts[-1]['m']}, field along {name}")
        r = mom[name]
        ax.plot(3 * np.array(r["cells"]), [np.linalg.norm(lv[k]) * 1e3 for lv in r["levels"]], "s", color=color, ls=(0, (4, 2)), lw=1.1,
                mfc=SURF, label=f"MoM on the exact cylinder, 4, 8, 16 across, along {name}")
        if r["errors"] is not None and np.isfinite(r["errors"][k]):
            est = np.linalg.norm(r["B"][k]) * 1e3
            ax.axhspan(est * (1 - r["errors"][k]), est * (1 + r["errors"][k]), color=color, alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("unknowns (MoM: 3 per cell)"); ax.set_ylabel("|B| (mT)")
    ax.set_title(where, loc="left", fontsize=10)
fig.legend(*a2.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.66, 0))
a3.text(0.98, 0.04, "band: MoM's extrapolated value ± its error (z only;\nalong x the study declines: the rim points turn)",
        transform=a3.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(HERE, "results", "step3_cylinder.png"), facecolor=SURF)
