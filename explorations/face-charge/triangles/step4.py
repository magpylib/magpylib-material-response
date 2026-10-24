"""Step 4: assembly by a hierarchical matrix (HBody), against the dense
assembly (MacroBody), on the plate of step 3b with narrow facets sharing panels.
Writes results/step4.json (one entry per run, resumable) and results/step4.png.

Plate exports at 15 deg (m = 2, 4), 3.75 deg (m = 2, 4) and 1.875 deg (m = 4):
1,714 to 18,216 triangles. Dense where it takes under a minute; hierarchical on
all. Each entry records the machine's load average, as other jobs share it.
Version 1 of the hierarchical runs computed coplanar close pairs; version 2 drops
them (they exchange no flux), which the figure shows.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from meshes import plate  # noqa: E402
from refine import refine_rings  # noqa: E402
from tri_solver import MU0, HBody, MacroBody  # noqa: E402

H0 = 0.01 / MU0 * np.array([0.8, 0.36, 0.48])
OBS = 1e-3 * np.array([[3, 0, 0], [8, 0, 3], [7 + 4 / np.sqrt(2), 3 + 4 / np.sqrt(2), 1], [0, 0, 3], [60, 0, 0], [0, 0, 60]])
CASES = [(15.0, 2), (15.0, 4), (3.75, 2), (3.75, 4), (1.875, 4)]
DENSE = {(15.0, 2), (15.0, 4), (3.75, 2)}
OUT = os.path.join(HERE, "results", "step4.json")
done = json.load(open(OUT)) if os.path.exists(OUT) else []
VERSION = 2  # 1: coplanar pairs computed; 2: dropped (hierarchical)
have = {(r["method"], r["deg"], r["m"], r.get("version", 1)) for r in done}


def save(rec):
    done.append(rec)
    json.dump(done, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print({k: v for k, v in rec.items() if k not in ("B", "stats")}, flush=True)


for deg, m in CASES:
    todo = [k for k in ("hierarchical", "dense") if (k == "hierarchical" and (k, deg, m, VERSION) not in have)
            or (k == "dense" and (deg, m) in DENSE and not any(h[:3] == (k, deg, m) for h in have))]
    if not todo:
        continue
    t = time.perf_counter()
    T, G, Bd = refine_rings(*plate(int(90 / deg), int(360 / deg)), 0.002, m, groups=True)
    tmesh = time.perf_counter() - t
    for method in todo:
        cls = HBody if method == "hierarchical" else MacroBody
        b = cls(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), G, Bd, chi=1000.0)
        load0 = os.getloadavg()[0]
        t = time.perf_counter(); Gm = b.assemble(); tass = time.perf_counter() - t
        t = time.perf_counter(); c = b.solve(G=Gm, load=b.load(lambda q: np.tile(H0, (len(q), 1)))); tsol = time.perf_counter() - t
        save({"method": method, "deg": deg, "m": m, "version": VERSION if method == "hierarchical" else 1, "triangles": b.F, "unknowns": b.Nm, "mesh s": tmesh,
              "assemble s": tass, "solve s": tsol, "load": [load0, os.getloadavg()[0]],
              "stats": getattr(b, "stats", {}), "B": b.B(OBS, c).tolist()})

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE, AQUA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=170, gridspec_kw={"width_ratios": [1, 1.2]})
fig.patch.set_facecolor(SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
rec = {(r["method"], r["deg"], r["m"]): r for r in sorted(done, key=lambda r: r.get("version", 1))}  # latest wins
hier = [rec[("hierarchical", d, m)] for d, m in CASES]
dense = [rec[("dense", d, m)] for d, m in CASES if (d, m) in DENSE]
Fh = np.array([r["triangles"] for r in hier]); Fd = np.array([r["triangles"] for r in dense])
th = np.array([r["assemble s"] for r in hier]); td = np.array([r["assemble s"] for r in dense])
c2 = np.exp(np.mean(np.log(td / Fd**2)))
Ff = np.geomspace(Fd.min(), Fh.max() * 1.1, 50)
a1.loglog(Ff, c2 * Ff**2, "--", color=INK2, lw=1, label="dense, ∝ F² through its points")
a1.loglog(Fd, td, "s", color=INK2, mfc=SURF, ms=6, label="dense (MacroBody)")
a1.loglog(Fh, th, "o-", color=BLUE, lw=1.4, label="hierarchical (HBody)")
for r, F_, t_ in zip(hier, Fh, th):
    if (("dense", r["deg"], r["m"]) not in rec):
        a1.annotate(f"≈{c2 * F_**2 / t_:.0f}× the fit", (F_, t_), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.5, color=BLUE)
a1.set_xlabel("triangles"); a1.set_ylabel("assembly (s)")
a1.set_title("Assembly time, plate exports at m = 2 and 4", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=7.5, loc="upper left")

parts = [("setup s", "pairs and tree", GRID), ("near s", "close leaves (point rule + closed form)", AQUA),
         ("operators s", "cluster operators", ORANGE), ("far s", "far blocks", "#f6c4ad"), ("refined s", "refined pairs", BLUE)]
x = np.arange(len(hier))
bottom = np.zeros(len(hier))
for key, label, color in parts:
    v = np.array([r["stats"][key] for r in hier])
    a2.bar(x, v, bottom=bottom, color=color, width=0.6, label=label)
    bottom += v
a2.bar(x, [r["solve s"] for r in hier], bottom=bottom, color=INK2, width=0.6, label="solve")
a2.set_xticks(x, [f"{r['deg']:g}°, m = {r['m']}\n{r['triangles']:,} triangles\n{r['unknowns']:,} unknowns" for r in hier], fontsize=7.5)
a2.set_ylabel("s")
a2.set_title("Hierarchical: where the time goes", loc="left", fontsize=10)
a2.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "step4.png"), facecolor=SURF)
