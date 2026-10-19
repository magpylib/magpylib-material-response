"""Data-only checks for C2, C3, C6, B1 from the txt files and mesh-residual.npz."""
import numpy as np
from scipy.optimize import least_squares

S = "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt/"
REF = 2.153
N = np.array([88, 297, 704, 1375, 2376, 3773.0])
v = np.array([1.3778767786303483, 1.6720344628059218, 1.832786595727041, 1.9266159525722906, 1.9850534287059043, 2.023563796538498])
h = np.array([10, 100 / 15, 5, 4, 100 / 30, 100 / 35])
err = 100 * (v / REF - 1)
print("B1 uniform cuboid errors (%):", np.round(err, 2))
print("B1 local order (error ~ h^p / N^(-p/3)):")
for i in range(len(N) - 1):
    p = np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])
    pN = 3 * np.log(err[i] / err[i + 1]) / np.log(N[i + 1] / N[i])
    print(f"   h {h[i]:.2f}->{h[i+1]:.2f}: p_h = {p:.2f}   p_N = {pN:.2f}")

print("\nC6 Richardson on uniform cuboids, v = v_inf - c N^(-p/3):")
def fit(idx):
    f = lambda x: x[0] - x[1] * N[idx] ** (-x[2] / 3) - v[idx]
    r = least_squares(f, [2.15, 10, 1.5])
    vinf, c, p = r.x
    return vinf, p, 100 * (v[idx][-1] / vinf - 1), r.fun
for idx in [slice(0, 6), slice(1, 6), slice(2, 6), slice(3, 6), slice(0, 5), slice(1, 5), slice(2, 5)]:
    vinf, p, e, res = fit(idx)
    print(f"   cells {N[idx].astype(int).tolist()}: v_inf {vinf:.4f} mT ({100*(vinf/REF-1):+.2f} % vs 2.153), p {p:.2f}, "
          f"predicted err of finest {e:+.2f} % (true {100*(v[idx][-1]/REF-1):+.2f} %), max resid {np.abs(res).max():.1e}")
# fixed-p variants
for p in (1.5, 2.0):
    for idx in [slice(3, 6), slice(4, 6)]:
        A = np.vstack([np.ones(len(N[idx])), -N[idx] ** (-p / 3)]).T
        vinf, c = np.linalg.lstsq(A, v[idx], rcond=None)[0]
        print(f"   fixed p={p}, cells {N[idx].astype(int).tolist()}: v_inf {vinf:.4f} ({100*(vinf/REF-1):+.2f} %), pred err finest {100*(v[idx][-1]/vinf-1):+.2f} %")

print("\nC2 uniform cell count at a given error (log-log interpolation between uniform points):")
ada = {242: 1.7977, 620: 1.8469, 1572: 2.0251, 3994: 2.0598}
E = -err  # positive
for n_a, v_a in ada.items():
    e_a = 100 * (1 - v_a / REF)
    # find bracket
    i = np.searchsorted(-E, -e_a)  # E decreasing
    i = min(max(i, 1), len(E) - 1)
    s = np.log(E[i] / E[i - 1]) / np.log(N[i] / N[i - 1])
    n_u = N[i - 1] * (e_a / E[i - 1]) ** (1 / s)
    tag = "extrapolated" if e_a < E[-1] else "interpolated"
    print(f"   adaptive {n_a} cells, {-e_a:+.2f} %: uniform needs ~{n_u:,.0f} cells ({tag}, slope {s:.2f}) -> ratio {n_u/n_a:.2f}x")

print("\nC6 change between rounds vs true error (true = 2.153 - value):")
for name, seq in (("cuboid residual", [(88, 1.3779), (242, 1.7977), (620, 1.8469), (1572, 2.0251), (3994, 2.0598)]),
                  ("cuboid targeted", [(88, 1.3779), (256, 1.7427), (536, 1.7395), (1628, 2.0090), (3770, 2.0118), (3994, 2.0224)]),
                  ("tet adaptive", [(89, 2.1160634), (292, 2.1260012), (593, 2.1276033), (1377, 2.1415444), (1496, 2.1419492)]),
                  ("cuboid uniform", list(zip(N.astype(int), v))),
                  ("tet uniform", [(89, 2.1160634), (417, 2.1554506), (989, 2.1416419), (1928, 2.1442488), (4016, 2.1479743)])):
    out = []
    for (n0, v0), (n1, v1) in zip(seq[:-1], seq[1:]):
        ch = abs(v1 - v0); tr = abs(REF - v1)
        out.append(f"{n1}: |change| {ch:.4f} vs true {tr:.4f} -> true/change {tr/ch:.1f}x")
    print(f"   {name}:\n      " + "\n      ".join(out))

print("\nC6 adjoint estimates vs true error (mT), residual.txt / targeted.txt:")
for name, rows in (("residual", [(88, 1.3779, 0.0305, 2.546), (242, 1.7977, -0.0119, 3.146), (620, 1.8469, 0.0011, 2.451), (1572, 2.0251, -0.0016, 2.332), (3994, 2.0598, -0.0055, 1.722)]),
                   ("targeted", [(88, 1.3779, 0.0305, 2.546), (256, 1.7427, -0.0130, 2.849), (536, 1.7395, 0.0182, 2.251), (1628, 2.0090, -0.0049, 2.089), (3770, 2.0118, 0.0040, 1.588), (3994, 2.0224, 0.0042, 1.552)])):
    for n, val, sg, ab in rows:
        tr = REF - val
        print(f"   {name} {n:>5}: true {tr:.4f}  signed {sg:+.4f} ({sg/tr:+.2f}x)  abs {ab:.3f} ({ab/tr:.1f}x)")

print("\nC3 mesh-residual.npz region count:")
m = np.load(S + "mesh-residual.npz")
pos, dim = m["pos"] * 1e3, m["dim"] * 1e3
size = np.round(dim.max(1), 2)
print("   cells:", len(pos), " by size:", {float(s): int((size == s).sum()) for s in np.unique(size)})
vol = np.prod(dim, 1)
print(f"   total volume {vol.sum():,.0f} mm^3 (expected 88,000)")
junction = (np.abs(pos[:, 0]) > 30) & (np.abs(pos[:, 0]) < 50) & (np.abs(pos[:, 2]) < 10)
print(f"   junction blocks |x| in (30,50), |z|<10: {junction.sum()} cells = {100*junction.mean():.0f} % of cells, "
      f"{100*vol[junction].sum()/vol.sum():.0f} % of volume; 2.5 mm cells there: {(junction & (size==2.5)).sum()} of {(size==2.5).sum()}")
for s in np.unique(size):
    sel = size == s
    print(f"   {s} mm cells: {sel.sum()} total; in junction {int((sel & junction).sum())}, in yoke outside junction {int((sel & ~junction & (np.abs(pos[:,2])<10)).sum())}, in legs z>10 {int((sel & (pos[:,2]>10)).sum())}")
# finer breakdown of the 2.5 mm cells by z and |x|
sel = size == 2.5
print("   2.5 mm cells by z band:", {f"{lo}..{lo+10}": int((sel & (pos[:,2]>=lo) & (pos[:,2]<lo+10)).sum()) for lo in range(-10, 70, 10)})
print("   2.5 mm cells by |x| band:", {f"{lo}..{lo+10}": int((sel & (np.abs(pos[:,0])>=lo) & (np.abs(pos[:,0])<lo+10)).sum()) for lo in range(0, 50, 10)})
sel5 = size == 5.0
print("   5 mm cells by z band:", {f"{lo}..{lo+10}": int((sel5 & (pos[:,2]>=lo) & (pos[:,2]<lo+10)).sum()) for lo in range(-10, 70, 10)})
