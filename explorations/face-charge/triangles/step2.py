"""Step 2: grading toward sharp edges, from crude meshes. Writes results/step2.json and .png.

- a cube given as 12 triangles (two per face), graded by refine_cells, against the
  cube's polarizability 3.644305 (Helsing & Perfekt 2013);
- the U-core of the ucore_electromagnet page as one closed mesh of 44 triangles,
  graded, in its coil's field, against the FEM profile of explore/cells-and-fem.
"""
import json, os, sys, time
import numpy as np
import magpylib as magpy
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "validation"))
from meshes import outward, ucore  # noqa: E402
from refine import refine_cells, sharp_edges  # noqa: E402
from tri_solver import TriBody  # noqa: E402
from examples import build_F, F_LINE, fem_ucore_reference  # noqa: E402

HP = 3.644305
out = {"cube": [], "ucore": []}
v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
v, f = outward(v, ConvexHull(v).simplices)
for m in [2, 3, 4]:
    t = time.perf_counter()
    T = refine_cells(v, f, ell=1.0, m=m)
    b = TriBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), chi=1e12)
    a = b.moment(b.solve(lambda p: np.tile([0, 0, 1.0], (len(p), 1))))[2] / 8
    out["cube"].append({"m": m, "triangles": len(T), "unknowns": b.Nd, "error": a / HP - 1, "seconds": time.perf_counter() - t})
    print(out["cube"][-1], flush=True)

ref, p, hs, peaks = fem_ucore_reference()
coil = [s for s in build_F().sources_all if isinstance(s, magpy.current.Polyline)]
v, f = ucore()
for m in [2]:  # m = 3 has 18,630 unknowns: about 45 GB dense
    t = time.perf_counter()
    T = refine_cells(v, f, ell=0.01, m=m)
    b = TriBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), chi=999.0)
    c = b.solve(lambda x: magpy.getH(coil, x, sumup=True))
    Bz = (b.B(F_LINE, c) + magpy.getB(coil, F_LINE, sumup=True))[:, 2] * 1e3
    e = np.abs(Bz - ref) / np.abs(ref).max()
    out["ucore"].append({"m": m, "triangles_in": len(f), "sharp_edges": len(sharp_edges(v, f)), "triangles": len(T),
                         "unknowns": b.Nd, "seconds": time.perf_counter() - t, "peak_mT": float(np.abs(Bz).max()),
                         "fem_peak_mT": float(np.abs(ref).max()), "max_error": float(e.max()), "rms_error": float(np.sqrt(np.mean(e**2))),
                         "Bz_mT": Bz.tolist()})
    print({k: v_ for k, v_ in out["ucore"][-1].items() if k != "Bz_mT"}, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "step2.json"), "w"), indent=1)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE, AQUA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=170)
fig.patch.set_facecolor(SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
s1 = json.load(open(os.path.join(HERE, "results", "step1.json")))["cube"]
a1.loglog([r["rect_unknowns"] for r in s1], [abs(r["rect_error"]) for r in s1], "o--", color=ORANGE, lw=1.2, label="cuboid panels (hand-built grading)")
a1.loglog([r["tri_unknowns"] for r in s1], [abs(r["tri_error"]) for r in s1], "s:", color=INK2, lw=1.2, label="those panels as triangles")
a1.loglog([r["unknowns"] for r in out["cube"]], [abs(r["error"]) for r in out["cube"]], "o-", color=BLUE, lw=1.4, label="12 triangles, graded automatically")
a1.set_xlabel("unknowns"); a1.set_ylabel("|α / 3.644305 − 1|")
a1.set_title("Cube, χ → ∞, against Helsing & Perfekt", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=8)
x = F_LINE[:, 0] * 1e3
vF = np.load(os.path.join(HERE, "..", "validation", "results", "F.npz"))
r = out["ucore"][-1]
scale = np.abs(ref).max()
a2.semilogy(x, 100 * np.abs(np.array(r["Bz_mT"]) - ref) / scale, color=BLUE, lw=1.4,
            label=f"one closed mesh, graded ({r['unknowns']:,} unknowns)")
a2.semilogy(x, 100 * np.abs(vF["fc_m3"] - ref) / scale, color=AQUA, lw=1.2, ls=(0, (4, 2)), label="three cuboids, face charge (3,240 unknowns)")
a2.axhspan(1e-4, 0.2, color=INK2, alpha=0.12, lw=0, label="within the FEM's own 0.2 %")
a2.set_ylim(1e-3, 1); a2.set_xlabel("x (mm), 30 mm above the poles"); a2.set_ylabel("|Bz − FEM| (% of the peak)")
a2.set_title("U-core with its coil, against FEM", loc="left", fontsize=10)
a2.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "step2.png"), facecolor=SURF)
