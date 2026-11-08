"""How far each faceting sits from the true curve, magnified: the radial gap
(micrometres) along the plate's hole and one fillet, 15 degree facets.
-> results/widening_view.png"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from geometry import HOLE_C, HOLE_R, RF, T, export_mesh_widened, faceted

INK2, GRID = "#52514e", "#e4e3df"
ARMS = [("a CAD export, vertices on the curve", "#9a988f", lambda: faceted("plate", 24, False)),
        ("the export widened from its mesh", "#1baf7a", lambda: export_mesh_widened("plate", 24)),
        ("solver facets the B-rep, widened", "#2a78d6", lambda: faceted("plate", 24, True))]
CURVES = [("the hole, r = 1.5 mm (material outside)", HOLE_C, HOLE_R, 0, 360),
          ("a fillet, r = 3 mm (material inside)", (7.0, 3.0), RF, 0, 90)]

def gap(V, centre, r, a0, a1):
    """Radial distance of the facets from the circle (µm) along its angle."""
    P = V[np.abs(V[:, 2] + T / 2) < 1e-6][:, :2] - centre  # bottom face's outline
    d = np.hypot(*P.T)
    a = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
    on = (np.abs(d - r) < 0.05 * r) & (a >= a0 - 1e-6) & (a <= a1 + 1e-6)
    P, a = P[on], a[on]
    P = P[np.argsort(a)]
    if a1 - a0 == 360:
        P = np.vstack([P, P[:1]])
    t = np.linspace(0, 1, 50)[:, None]
    pts = np.concatenate([p + t * (q - p) for p, q in zip(P[:-1], P[1:])])
    ang = np.degrees(np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))) % 360
    if a1 - a0 == 360:
        ang = np.degrees(np.unwrap(np.arctan2(pts[:, 1], pts[:, 0])))
        ang -= ang[0]
    return ang, 1e3 * (np.hypot(*pts.T) - r)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, (title, centre, r, a0, a1) in zip(axes, CURVES):
    for label, color, make in ARMS:
        V, F = make()
        ang, g = gap(np.asarray(V), np.asarray(centre), r, a0, a1)
        ax.plot(ang, g, color=color, lw=2, label=label)
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.text(a0 + 0.01 * (a1 - a0), 1.5, "the true curve", fontsize=8.5, color="#0b0b0b")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xlabel("position along the curve, degrees", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("facet's radial distance from the curve, µm\n(+ farther from the centre)", color=INK2)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Where the facets sit, 15° per facet: an export cuts inside the curve; widening moves the vertices out so the facets straddle it", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig("results/widening_view.png", dpi=130)
print("saved")
