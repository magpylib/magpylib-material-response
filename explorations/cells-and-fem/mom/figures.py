"""Figures for the material-response meshing issues. Data from the runs of 2026-10-19 (#52 at 4901d1d)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SIZES = {10.0: "#86b6ef", 5.0: "#2a78d6", 2.5: "#104281"}
mpl.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.handlelength": 3.2, "legend.numpoints": 1, "lines.linewidth": 1.6, "lines.markersize": 6,
})

UREF, PREF = 2.152, 0.5633  # reviewed references: ±0.2 % band; plate range 0.562..0.5639 (upper bound)
cub_uni = dict(cells=[88, 297, 704, 1375, 2376, 3773], t=[0.01, 0.08, 0.38, 1.19, 3.53, 9.27],
               v=[1.37788, 1.67203, 1.83279, 1.92662, 1.98505, 2.02356])
tet_uni = dict(cells=[89, 417, 989, 1928, 4016], t=[0.03, 0.46, 2.34, 9.03, 43.5],
               v=[2.11606, 2.15545, 2.14164, 2.14425, 2.14797])
tet_box = dict(cells=[72, 359, 1033, 1968], t=[0.02, 0.37, 2.64, 9.54], v=[2.21022, 2.19159, 2.13637, 2.19947])
cub_ada = dict(cells=[88, 242, 620, 1572, 3994], v=[1.37788, 1.79770, 1.84687, 2.02507, 2.05980])
tet_ada = dict(cells=[89, 292, 593, 1377, 1496], v=[2.11606, 2.12600, 2.12760, 2.14154, 2.14195])
plate_cub = dict(cells=["20×20×3", "40×40×3", "40×40×6", "40×40×12", "80×80×6", "80×80×12"],
                 n=[1200, 4800, 9600, 19200, 38400, 76800], t=[0.2, 1.1, 1.6, 2.6, 6.7, 11.5],
                 v=[0.558561, 0.557012, 0.559354, 0.559987, 0.560490, 0.561368])
plate_tet = dict(n=[1728, 3328], t=[7.1, 28.6], v=[0.567344, 0.564645])  # #52 direct


def err(v, ref):
    return 100 * (np.array(v) / ref - 1)


def series(ax, x, y, color, ls="-", hollow=False, label=None):
    ax.plot(x, y, ls=ls, color=color, marker="o", mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE,
            mew=1.4 if hollow else 1.2, solid_capstyle="round", label=label, zorder=3)


def ref_line(ax, text, lo=-0.2, hi=0.2):
    ax.axhspan(lo, hi, color=GRID, alpha=0.8, zorder=0, lw=0)
    ax.axhline(0, color=NEUTRAL, lw=1.2, zorder=1, label=text)


def note(ax, x, y, s, dx=6, dy=0, ha="left"):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha=ha, va="center")


# ---- figure 1: cell type ---------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
series(a, cub_uni["t"], err(cub_uni["v"], UREF), BLUE, label="cuboids, uniform grid")
series(a, tet_uni["t"], err(tet_uni["v"], UREF), ORANGE, label="tetrahedra, U-core as one surface")
series(a, tet_box["t"], err(tet_box["v"], UREF), ORANGE, ls="--", hollow=True, label="tetrahedra, each cuboid on its own")
for d, idx in ((cub_uni, (0, 5)), (tet_uni, (0, 4))):
    for i in idx:
        note(a, d["t"][i], err(d["v"], UREF)[i], f"{d['cells'][i]:,} cells", dy=-9 if d is tet_uni else 0)
ref_line(a, "FEM reference 2.152 mT, ±0.2 % band")
a.set_xscale("log")
a.set_xlabel("solve time (s), apply_demag direct")
a.set_ylabel("peak |Bz| error vs FEM (%)")
a.set_title("U-core, χ = 999: peak |Bz| 30 mm above the poles")
a.legend(loc="lower right", bbox_to_anchor=(1, 0.08), fontsize=8)

series(b, plate_cub["t"], err(plate_cub["v"], PREF), BLUE, label="cuboids, flat cells (#52 iterative, FFT)")
series(b, plate_tet["t"], err(plate_tet["v"], PREF), ORANGE, label="tetrahedra, TetGen (#52 direct)")
for i in (0, 5):
    note(b, plate_cub["t"][i], err(plate_cub["v"], PREF)[i], f"{plate_cub['cells'][i]} = {plate_cub['n'][i]:,} cells", dy=10 if i == 0 else -9)
for i in (0, 1):
    note(b, plate_tet["t"][i], err(plate_tet["v"], PREF)[i], f"{plate_tet['n'][i]:,} cells", dx=8 if i == 0 else -8, ha="left" if i == 0 else "right")
ref_line(b, "FEM reference 0.5633 A m², its band", -0.23, 0.11)
b.set_xscale("log")
b.set_xlabel("solve time (s)")
b.set_ylabel("moment error vs FEM (%)")
b.set_title("Thin plate 40 × 40 × 1 mm, χ = 1000: moment")
b.set_ylim(-1.6, 1.2)
b.legend(loc="upper left", fontsize=8)
fig.savefig("../results/fig_celltype.png", dpi=200)

# ---- figure 2: adaptive refinement -------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
series(a, cub_uni["cells"], err(cub_uni["v"], UREF), BLUE, label="cuboids, uniform")
series(a, cub_ada["cells"], err(cub_ada["v"], UREF), BLUE, ls="--", hollow=True, label="cuboids, adaptive")
series(a, tet_uni["cells"], err(tet_uni["v"], UREF), ORANGE, label="tetrahedra, uniform")
series(a, tet_ada["cells"], err(tet_ada["v"], UREF), ORANGE, ls="--", hollow=True, label="tetrahedra, adaptive")
note(a, 1572, err([2.02507], UREF)[0], "1,572", dx=-6, dy=7, ha="right")
note(a, 3773, err([2.02356], UREF)[0], "3,773", dy=-8)
ref_line(a, "FEM reference 2.152 mT, ±0.2 % band")
a.set_xscale("log")
a.set_xlabel("cells")
a.set_ylabel("peak |Bz| error vs FEM (%)")
a.set_title("U-core: adaptive vs uniform refinement")
a.legend(loc="lower right", bbox_to_anchor=(1, 0.08), fontsize=8)

m = np.load("../results/mesh-residual.npz")
pos, dim = m["pos"] * 1e3, m["dim"] * 1e3
cut = (pos[:, 1] - dim[:, 1] / 2 < 0.1) & (pos[:, 1] + dim[:, 1] / 2 > 0.1)
for p, d in zip(pos[cut], dim[cut]):
    b.add_patch(Rectangle((p[0] - d[0] / 2, p[2] - d[2] / 2), d[0], d[2], fc=SIZES[round(float(d.max()), 1)],
                          ec=SURFACE, lw=0.7, zorder=2))
xt = (np.arange(8) - 3.5) * 2.2
for zc in (14, 16.2, -14, -16.2):
    b.plot(xt, np.full(8, zc), "o", ms=2.6, color=INK2, zorder=3)
b.plot([-80, 80], [100, 100], color=NEUTRAL, lw=1.2, zorder=1)
b.text(-80, 103, "probe line (z = 100 mm)", fontsize=7.5, color=INK2)
b.text(0, -21, "coil, 16 turns × 25 A", fontsize=7.5, color=INK2, ha="center", va="top")
b.set_xlim(-85, 85)
b.set_ylim(-30, 110)
b.set_aspect("equal")
b.grid(False)
b.set_xlabel("x (mm)")
b.set_ylabel("z (mm)")
b.set_title(f"Where the adaptive cuboid cells went: {len(pos):,} cells, cut at y = 0")
b.legend(handles=[Patch(fc=c, ec=SURFACE, label=f"{s:g} mm cells") for s, c in SIZES.items() if s in set(np.round(dim[cut].max(1), 1))],
         loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=8)
fig.savefig("../results/fig_adaptive.png", dpi=200)
print("ok", cut.sum(), "cells in the cut")
