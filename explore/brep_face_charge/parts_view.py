"""The study's parts as triangles: the 15 degree export's facets, and the
panels the solver makes on it and on the B-rep's own facets (level 2).
-> results/parts_view.png"""
import warnings
import magpylib as magpy
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geometry import FIELD, MM, as_magnet, faceted
from magpylib_material_response import UniformField, solve_face_charge
warnings.simplefilter("ignore")

def panels(V, F, level):
    solved = solve_face_charge(magpy.Collection(as_magnet(V, F, label="part"), UniformField(B=FIELD)), level=level)
    sheet = [c for c in solved.children if not isinstance(c, UniformField)][0]
    return np.asarray(sheet.triangles) / MM, solved.face_charge_info["unknowns"]

def draw(ax, tris, title, face="#dcdad3"):
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    shade = 0.55 + 0.45 * np.abs(n @ light)
    base = np.array(mpl.colors.to_rgb(face))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=base * shade[:, None],
                                         edgecolors="#2b2a27", linewidths=0.25))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=28, azim=-60)
    ax.set_title(title, fontsize=10, loc="left")

fig = plt.figure(figsize=(15, 9.5))
for row, part in enumerate(("plate", "pole")):
    V, F = faceted(part, 24, False)
    draw(fig.add_subplot(2, 3, 3 * row + 1, projection="3d"), np.asarray(V)[np.asarray(F)],
         f"{part}: a CAD export, 15° facets ({len(F)} triangles in)")
    T, u = panels(V, F, 2)
    draw(fig.add_subplot(2, 3, 3 * row + 2, projection="3d"), T,
         f"{part}: the solver's panels on it, level 2\n({len(T)} triangles, {u:,} unknowns)", face="#e9c2b0")
    Vb, Fb = faceted(part, 36, True)
    T, u = panels(Vb, Fb, 2)
    draw(fig.add_subplot(2, 3, 3 * row + 3, projection="3d"), T,
         f"{part}: its panels on the B-rep's facets, level 2 (10°, widened)\n({len(T)} triangles, {u:,} unknowns)", face="#b9cfea")
fig.suptitle("What the solver gets and what it makes of it: flat faces re-meshed from their outlines and graded toward sharp edges; "
             "each curved facet kept as a face of its own", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("results/parts_view.png", dpi=120)
print("saved")
