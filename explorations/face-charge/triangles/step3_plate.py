"""Step 3b: a soft-iron plate with filleted corners and an off-centre through-hole,
exported as a CAD tool would (meshes.plate), against material-response's MoM on
TetGen tetrahedra of the same export. Writes results/step3_plate.json (one entry
per run, resumable) and results/step3_plate.png.

Plate 20 x 12 x 4 mm, fillets r = 3 mm, hole r = 3 mm centred 3 mm off; chi =
1000; uniform mu0 H = 10 mT along (0.8, 0.36, 0.48), so no symmetry helps.
Exports at 15, 7.5 and 3.75 deg per facet. Face charge: 15 deg at m = 2, 3, 4;
7.5 deg at m = 2, 3; 3.75 deg at m = 2 (finer exports cost more: each facet strip
is cut into its own layers). MoM: mesh_study on the 15 deg export, TetGen targets
300, 1800, 4300 (about 1,600, 3,700 and 8,200 tets; the finest needs ~10 GB).
"""
import json, os, sys, time
import numpy as np
import magpylib as magpy
HERE = os.path.dirname(os.path.abspath(__file__))
from meshes import plate  # noqa: E402
from refine import refine_rings  # noqa: E402
from tri_solver import MU0, TriBody  # noqa: E402

CHI, B0 = 1000.0, 0.01
DIR = np.array([0.8, 0.36, 0.48])
ELL = 0.002  # half the thickness
EXPORTS = {15.0: (6, 24), 7.5: (12, 48), 3.75: (24, 96)}  # deg per facet: (per fillet, round the hole)
LEVELS = {15.0: (2, 3, 4), 7.5: (2, 3), 3.75: (2,)}
OBS = 1e-3 * np.array([[3, 0, 0], [8, 0, 3], [7 + 4 / np.sqrt(2), 3 + 4 / np.sqrt(2), 1], [0, 0, 3], [60, 0, 0], [0, 0, 60]])
NAMES = ["in the hole,\nmid-plane", "1 mm above\nthe bridge", "1 mm out\nfrom a fillet", "1 mm above\nthe hole's rim", "60 mm\nalong x", "60 mm\nalong z"]
TARGETS = (300, 1800, 4300)
OUT = os.path.join(HERE, "results", "step3_plate.json")
done = json.load(open(OUT)) if os.path.exists(OUT) else []
have = {(r["method"], r.get("deg"), r.get("m")) for r in done}


def save(rec):
    done.append(rec)
    json.dump(done, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print({k: v for k, v in rec.items() if k not in ("B", "levels")}, flush=True)


H0 = B0 / MU0 * DIR
for deg, (nf, nh) in EXPORTS.items():
    v, f = plate(nf, nh)
    for m in LEVELS[deg]:
        if ("face charge", deg, m) in have:
            continue
        t = time.perf_counter()
        T = refine_rings(v, f, ELL, m)
        b = TriBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), chi=CHI)
        c = b.solve(load=b.load(lambda p: np.tile(H0, (len(p), 1))))
        save({"method": "face charge", "deg": deg, "m": m, "unknowns": b.Nd,
              "seconds": time.perf_counter() - t, "moment": b.moment(c).tolist(),
              "B": b.B(OBS, c).tolist()})

if ("MoM", 15.0, None) not in have:
    from magpylib_material_response.convergence import mesh_study
    from magpylib_material_response.meshing import mesh_all
    v, f = plate(*EXPORTS[15.0])
    tm = magpy.magnet.TriangularMesh(vertices=v, faces=f, polarization=(0, 0, 0))
    tm.susceptibility = CHI
    tm.H_ext = tuple(B0 * DIR)
    t = time.perf_counter()
    study = mesh_study(magpy.Collection(tm), OBS, TARGETS, mesher=mesh_all)
    save({"method": "MoM", "deg": 15.0, "m": None, "targets": list(TARGETS),
          "cells": [int(x) for x in study.cells], "level_seconds": [float(s) for s in study.seconds],
          "seconds": time.perf_counter() - t, "order": study.order,
          "levels": [np.asarray(x).tolist() for x in study.values],
          "B": np.asarray(study.estimate).tolist(),
          "errors": np.atleast_1d(study.errors).tolist()})

# The tetrahedral MoM where the answer is known: a 10 mm cube from 12 triangles,
# same chi and field, about the same tet counts; its moment against face charge,
# whose grading of this cube matches Helsing & Perfekt to 4e-4 at m = 4 (step 2).
from scipy.spatial import ConvexHull  # noqa: E402
from meshes import outward  # noqa: E402
vc = 0.005 * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
vc, fcube = outward(vc, ConvexHull(vc).simplices)
for m in (4, 5, 6):
    if ("face charge cube", None, m) in have:
        continue
    t = time.perf_counter()
    T = refine_rings(vc, fcube, 0.005, m)
    b = TriBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), chi=CHI)
    c = b.solve(load=b.load(lambda p: np.tile(H0, (len(p), 1))))
    save({"method": "face charge cube", "deg": None, "m": m, "unknowns": b.Nd,
          "seconds": time.perf_counter() - t, "moment": b.moment(c).tolist()})
if ("MoM cube", None, None) not in have:
    from magpylib_material_response.convergence import mesh_study
    from magpylib_material_response.meshing import mesh_all

    def moment(solved):
        return sum(np.asarray(s.polarization) * s.volume for s in solved.sources_all
                   if isinstance(s, magpy.magnet.Tetrahedron)) / MU0

    tm = magpy.magnet.TriangularMesh(vertices=vc, faces=fcube, polarization=(0, 0, 0))
    tm.susceptibility = CHI
    tm.H_ext = tuple(B0 * DIR)
    t = time.perf_counter()
    study = mesh_study(magpy.Collection(tm), moment, (800, 1850, 4100), mesher=mesh_all)
    save({"method": "MoM cube", "deg": None, "m": None, "cells": [int(x) for x in study.cells],
          "level_seconds": [float(s) for s in study.seconds], "seconds": time.perf_counter() - t,
          "order": study.order, "levels": [np.asarray(x).tolist() for x in study.values],
          "B": np.asarray(study.estimate).tolist(), "errors": np.atleast_1d(study.errors).tolist()})

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE, AQUA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(17, 8.6), dpi=170)
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15])
a1, a2, a5 = (fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(3))
a3, a4 = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])
fig.patch.set_facecolor(SURF)
fc = {(r["deg"], r["m"]): r for r in done if r["method"] == "face charge"}
mom = next(r for r in done if r["method"] == "MoM")

v, f = plate(*EXPORTS[15.0])
top = v[f][np.all(np.isclose(v[f][:, :, 2], v[:, 2].max()), axis=1)]
T = refine_rings(v, f, ELL, 3)
ttop = T[np.all(np.isclose(T[:, :, 2], v[:, 2].max()), axis=1)]
for ax, tris, title in ((a1, top, f"The export: {len(f)} triangles, 15° per facet (top face)"),
                        (a2, ttop, f"Face charge, m = 3: {3 * len(T):,} unknowns (top face)")):
    ax.set_facecolor(SURF)
    for t_ in tris * 1e3:
        ax.fill(t_[:, 0], t_[:, 1], facecolor="#eef3fb", edgecolor=BLUE, lw=0.35)
    ax.plot(*(OBS[:4, :2] * 1e3).T, "o", color=ORANGE, ms=4, zorder=3)
    for k in range(4):
        ax.annotate(str(k + 1), OBS[k, :2] * 1e3, xytext=(4, 3), textcoords="offset points", color=ORANGE, fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-10.8, 12); ax.set_ylim(-6.6, 6.8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10)
a1.text(-10.5, -6.5, "field along (0.8, 0.36, 0.48); observers 5, 6 at 60 mm", fontsize=7.5, color=INK2)

V, H = 1e-6, B0 / MU0
fcc = [r for r in done if r["method"] == "face charge cube"]
mc = next(r for r in done if r["method"] == "MoM cube")
a5.set_facecolor(SURF); a5.grid(True, color=GRID, lw=0.7); a5.set_axisbelow(True)
for sp in ("top", "right"):
    a5.spines[sp].set_visible(False)
a5.plot([r["unknowns"] for r in fcc], [np.array(r["moment"]) @ DIR / (V * H) for r in fcc], "o-", color=BLUE, lw=1.4,
        label=f"face charge, m = {fcc[0]['m']}–{fcc[-1]['m']}")
a5.plot(3 * np.array(mc["cells"]), [np.ravel(lv) @ DIR / (V * H) for lv in mc["levels"]], "s", ls=(0, (4, 2)), color=ORANGE,
        mfc=SURF, lw=1.1, label=", ".join(f"{n:,}" for n in mc["cells"]) + " tets")
if mc["order"] is not None:
    est = np.ravel(mc["B"]) @ DIR / (V * H)
    err = max(mc["errors"])
    a5.errorbar(3 * mc["cells"][-1] * 1.5, est, yerr=est * err, fmt="D", color=ORANGE, mfc=SURF, mew=1.4, ms=5.5, capsize=3,
                label=f"MoM extrapolated, ± its error ({100 * err:.2g} %, largest component)")
a5.set_xscale("log"); a5.set_xlabel("unknowns (MoM: 3 per tet)"); a5.set_ylabel("moment / (volume × H)")
a5.set_title("Check: a 10 mm cube from 12 triangles, χ = 1000", loc="left", fontsize=10)
a5.legend(frameon=False, fontsize=7.5, loc="upper left")
a5.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

for ax in (a3, a4):
    ax.set_facecolor(SURF); ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_yscale("log")
    ax.set_xticks(range(len(OBS)), [f"{k + 1}  {n}" for k, n in enumerate(NAMES)], fontsize=7.5)


def rel(B, ref):
    B, ref = np.asarray(B), np.asarray(ref)
    return np.linalg.norm(B - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


ref = np.array(fc[(15.0, 4)]["B"])
x = np.arange(len(OBS))
shades = ["#f6c4ad", "#f19468", ORANGE]
for k, (lv, n) in enumerate(zip(mom["levels"], mom["cells"])):
    a3.plot(x - 0.24 + 0.08 * k, rel(lv, ref), "s", color=shades[k], ms=5, label=f"MoM, {n:,} tets")
if mom["order"] is not None:
    a3.plot(x + 0.06, rel(mom["B"], ref), "D", color=ORANGE, ms=5.5, mfc=SURF, mew=1.4, label="MoM extrapolated")
    a3.plot(x + 0.06, mom["errors"], "_", color=ORANGE, ms=14, mew=2, label="MoM's own error estimate")
else:
    a3.text(0.99, 0.98, "MoM: no extrapolation, its levels do not change steadily", transform=a3.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK2)
a3.plot(x + 0.22, rel(fc[(15.0, 3)]["B"], ref), "o", color=BLUE, ms=5, label=f"face charge, m = 3 ({fc[(15.0, 3)]['unknowns']:,})")
a3.plot(x + 0.30, rel(fc[(15.0, 2)]["B"], ref), "o", color=BLUE, ms=4, mfc=SURF, label=f"face charge, m = 2 ({fc[(15.0, 2)]['unknowns']:,})")
a3.set_ylabel(f"|B − face charge at m = 4| / |B|")
a3.set_title(f"Same export: against face charge at m = 4 ({fc[(15.0, 4)]['unknowns']:,} unknowns)", loc="left", fontsize=10)
a3.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.2))

base = np.array(fc[(15.0, 2)]["B"])
for deg, color, mk in ((7.5, AQUA, "o"), (3.75, BLUE, "D")):
    a4.plot(x, rel(fc[(deg, 2)]["B"], base), mk, color=color, ms=5.5,
            label=f"{deg:g}° per facet ({fc[(deg, 2)]['unknowns']:,} unknowns)")
a4.set_ylabel("|B − B on the 15° export| / |B|  (face charge, m = 2)")
a4.set_title("Finer exports of the same part", loc="left", fontsize=10)
a4.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.2))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "step3_plate.png"), facecolor=SURF)
