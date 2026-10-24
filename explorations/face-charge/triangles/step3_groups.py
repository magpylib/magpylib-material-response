"""Step 3c: narrow facets share panels. refine_rings(groups=True) joins the ring
cells of one layer along one rim into runs as long as the last layer's size, and
MacroBody solves one linear charge per run, projected from the exact triangles.
Against the ungrouped runs of steps 3a and 3b, on the same exports. Writes
results/step3_groups.json (one entry per run, resumable) and step3_groups.png.

Plate: every export (15, 7.5, 3.75 deg) at m = 2, 3, 4. Cylinder: 128 sides, m =
2, 3, 4, field along z and x (one operator, two solves). The first runs, kept as
basis "tangent", took xi and eta along each group's mean tangent plane; "band"
takes them along the rim and away from it (see MacroBody).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from meshes import cylinder, plate  # noqa: E402
from refine import refine_rings  # noqa: E402
from tri_solver import MU0, MacroBody  # noqa: E402

# the cases of step3_plate.py and step3_cylinder.py
CHI = 1000.0
PLATE_H0 = 0.01 / MU0 * np.array([0.8, 0.36, 0.48])
PLATE_ELL = 0.002
EXPORTS = {15.0: (6, 24), 7.5: (12, 48), 3.75: (24, 96)}
PLATE_OBS = 1e-3 * np.array([[3, 0, 0], [8, 0, 3], [7 + 4 / np.sqrt(2), 3 + 4 / np.sqrt(2), 1], [0, 0, 3], [60, 0, 0], [0, 0, 60]])
R, H, B0 = 0.005, 0.01, 0.01
CYL_OBS = np.array([[0.8 * R, 0, H / 2 + 1e-3], [R + 1e-3, 0, 0.3 * H], [0, 0, 0.05], [0.05, 0, 0]])

OUT = os.path.join(HERE, "results", "step3_groups.json")
done = json.load(open(OUT)) if os.path.exists(OUT) else []
have = {(r["part"], r["deg"], r["m"], r.get("basis", "tangent")) for r in done}


def save(rec):
    done.append(rec)
    json.dump(done, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print({k: v for k, v in rec.items() if k not in ("B", "Bx")}, flush=True)


for deg, (nf, nh) in EXPORTS.items():
    v, f = plate(nf, nh)
    for m in (2, 3, 4):
        if ("plate", deg, m, "band") in have:
            continue
        t = time.perf_counter()
        T, G, Bd = refine_rings(v, f, PLATE_ELL, m, groups=True)
        b = MacroBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), G, Bd, chi=CHI)
        c = b.solve(load=b.load(lambda p: np.tile(PLATE_H0, (len(p), 1))))
        save({"part": "plate", "deg": deg, "m": m, "basis": "band", "triangles": b.F, "unknowns": b.Nm,
              "seconds": time.perf_counter() - t, "moment": b.moment(c).tolist(), "B": b.B(PLATE_OBS, c).tolist()})

v, f = cylinder(128, R, H)
for m in (2, 3, 4):
    if ("cylinder", 128, m, "band") in have:
        continue
    t = time.perf_counter()
    T, G, Bd = refine_rings(v, f, R, m, groups=True)
    b = MacroBody(T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), G, Bd, chi=CHI)
    Gm = b.assemble()
    rec = {"part": "cylinder", "deg": 128, "m": m, "basis": "band", "triangles": b.F, "unknowns": b.Nm}
    for axis, name in ((2, "z"), (0, "x")):
        H0 = np.zeros(3); H0[axis] = B0 / MU0
        c = b.solve(G=Gm, load=b.load(lambda p: np.tile(H0, (len(p), 1))))
        rec["moment" + name], rec["B" + name] = b.moment(c).tolist(), b.B(CYL_OBS, c).tolist()
    rec["seconds"] = time.perf_counter() - t
    save(rec)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE, AQUA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(16, 4.9), dpi=170)
gs = fig.add_gridspec(1, 7)
a1, a2, a3 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:])
for ax in (a1, a2, a3):
    ax.set_facecolor(SURF); ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_yscale("log")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
grp = {(r["part"], r["deg"], r["m"], r.get("basis", "tangent")): r for r in done}
und = {(r["deg"], r["m"]): r for r in json.load(open(os.path.join(HERE, "results", "step3_plate.json"))) if r["method"] == "face charge"}
cyl = {(r["field"], r["m"]): r for r in json.load(open(os.path.join(HERE, "results", "step3_cylinder.json")))
       if r["method"] == "face charge" and r["N"] == 128}

degs = [15.0, 7.5, 3.75]
x = np.arange(3)
for m, mk in ((2, "o"), (4, "s")):
    a1.plot(x, [3 * grp[("plate", d, m, "band")]["triangles"] for d in degs], mk + "--", color=INK2, mfc=SURF, lw=1.1,
            label=f"one charge per triangle, m = {m}")
    a1.plot(x, [grp[("plate", d, m, "band")]["unknowns"] for d in degs], mk + "-", color=BLUE, lw=1.4,
            label=f"narrow facets share panels, m = {m}")
a1.set_xticks(x, ["15°", "7.5°", "3.75°"]); a1.set_xlabel("plate export, degrees per facet"); a1.set_ylabel("unknowns")
a1.set_title("Unknowns (assembly time is the same)", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=7.5, loc="upper left")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def dots(ax, cols, band, tangent, step, labels):
    xx = np.arange(cols)
    ax.bar(xx, step, width=0.6, color=GRID, label="one charge per triangle: its own change from m = 3 to 4")
    ax.plot(xx - 0.12, tangent, "D", color=INK2, mfc=SURF, ms=5, label="shared, along the mean tangent plane")
    ax.plot(xx + 0.12, band, "o", color=BLUE, ms=6, label="shared, along the rim and away from it")
    ax.set_xticks(xx, labels, fontsize=7.5)
    ax.set_ylabel("|B shared − B one per triangle| / |B|, at m = 4")


dots(a2, 6, rel(grp[("plate", 15.0, 4, "band")]["B"], und[(15.0, 4)]["B"]),
     rel(grp[("plate", 15.0, 4, "tangent")]["B"], und[(15.0, 4)]["B"]), rel(und[(15.0, 3)]["B"], und[(15.0, 4)]["B"]),
     ["in the\nhole", "above the\nbridge", "out from\na fillet", "above the\nhole's rim", "60 mm\nalong x", "60 mm\nalong z"])
a2.set_title("Plate, 15° export", loc="left", fontsize=10)
band = np.concatenate([rel(grp[("cylinder", 128, 4, "band")]["B" + f], cyl[(f, 4)]["B"]) for f in "zx"])
tang = np.concatenate([rel(grp[("cylinder", 128, 4, "tangent")]["B" + f], cyl[(f, 4)]["B"]) for f in "zx"])
step = np.concatenate([rel(cyl[(f, 3)]["B"], cyl[(f, 4)]["B"]) for f in "zx"])
names = ["above\nthe rim", "beside the\nmantle", "50 mm\non axis", "50 mm\nacross"]
dots(a3, 8, band, tang, step, [f"{n}\n({f})" for f in ("axial", "cross") for n in names])
a3.axvline(3.5, color=GRID, lw=1)
a3.set_title("Cylinder, 128 sides: 11,544 → 1,266 unknowns", loc="left", fontsize=10)
a3.legend(frameon=False, fontsize=7.5, loc="upper right")
for ax in (a2, a3):
    ax.set_ylim(1e-5, 1e-2)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "step3_groups.png"), facecolor=SURF)
