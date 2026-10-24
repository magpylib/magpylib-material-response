"""Step 1: the triangle solver on two bodies with known answers. Writes results/step1.json and .png.

- an icosphere in a uniform field (the exact sphere magnetizes uniformly, M = 3 chi/(3 + chi) H0);
- a cube given as triangles (each graded cuboid panel cut in two), against the cube's
  polarizability 3.644305 (Helsing & Perfekt 2013) and the cuboid solver on the same panels.
"""
import json, os, sys, time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bem import layer_panels  # noqa: E402
from p1 import P1  # noqa: E402
from meshes import icosphere, panels_to_triangles  # noqa: E402
from tri_solver import MU0, TriBody  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HP = 3.644305
out = {"sphere": [], "cube": []}
for level in [1, 2, 3]:
    v, f = icosphere(level)
    tri = v[f]
    vol = np.abs(np.einsum("fd,fd->f", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))).sum() / 6
    for chi in [1.0, 1000.0]:
        b = TriBody(v, f, chi=chi)
        t = time.perf_counter()
        m = b.moment(b.solve(lambda p: np.tile([0, 0, 1 / MU0], (len(p), 1))))[2]
        M_exact = 3 * chi / (3 + chi) / MU0
        out["sphere"].append({"level": level, "triangles": len(f), "unknowns": b.Nd, "chi": chi, "seconds": time.perf_counter() - t,
                              "moment_vs_sphere": m / (M_exact * 4 / 3 * np.pi) - 1, "volume_vs_sphere": vol / (4 / 3 * np.pi) - 1,
                              "M_vs_exact": m / vol / M_exact - 1})
        print(out["sphere"][-1], flush=True)
for m_ in [2, 3, 4]:
    pan = layer_panels(np.ones(3), m_, 2.5)
    s = P1(pan, nq=4)
    t = time.perf_counter()
    a_rect = s.moment(s.solve(1e12, s.assemble(), s.rhs(lambda x: np.tile([0, 0, 1.0], (len(x), 1)))))[2] / 8
    t_rect = time.perf_counter() - t
    v, f = panels_to_triangles(pan)
    b = TriBody(v, f, chi=1e12)
    t = time.perf_counter()
    a_tri = b.moment(b.solve(lambda p: np.tile([0, 0, 1.0], (len(p), 1))))[2] / 8
    out["cube"].append({"m": m_, "rect_unknowns": s.Nd, "rect_error": a_rect / HP - 1, "rect_seconds": t_rect,
                        "tri_unknowns": b.Nd, "tri_error": a_tri / HP - 1, "tri_seconds": time.perf_counter() - t})
    print(out["cube"][-1], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "step1.json"), "w"), indent=1)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=170)
fig.patch.set_facecolor(SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
sph = [r for r in out["sphere"] if r["chi"] == 1000.0]
tr = [r["triangles"] for r in sph]
a1.loglog(tr, [abs(r["volume_vs_sphere"]) for r in sph], "o-", color=INK2, lw=2.4, alpha=0.45, label="faceted volume vs sphere")
a1.loglog(tr, [abs(r["moment_vs_sphere"]) for r in sph], "o--", color=BLUE, lw=1.2, label="moment vs exact sphere")
a1.loglog(tr, [abs(r["M_vs_exact"]) for r in sph], "s-", color=ORANGE, lw=1.2, label="moment per faceted volume vs exact M")
a1.set_xlabel("triangles"); a1.set_ylabel("relative error")
a1.set_title("Icosphere, χ = 1000: the error is the faceting", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=8)
cb = out["cube"]
a2.loglog([r["rect_unknowns"] for r in cb], [abs(r["rect_error"]) for r in cb], "o--", color=ORANGE, lw=1.2, label="cuboid panels (p1.py)")
a2.loglog([r["tri_unknowns"] for r in cb], [abs(r["tri_error"]) for r in cb], "o-", color=BLUE, lw=1.2, label="the same panels as triangles")
a2.set_xlabel("unknowns"); a2.set_ylabel("|α / 3.644305 − 1|")
a2.set_title("Cube, χ → ∞, against Helsing & Perfekt (2013)", loc="left", fontsize=10)
a2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "step1.png"), facecolor=SURF)
