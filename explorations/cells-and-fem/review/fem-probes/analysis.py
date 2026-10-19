"""Richardson/extrapolation checks on the FEM series, plus analytic boundary estimates."""
import numpy as np
import magpylib as magpy

MU0 = magpy.mu_0


def fit_p(h, f):
    """f = L + C h^p from three points (h decreasing); returns p, L, C."""
    (a, b, c), (fa, fb, fc) = h, f
    ratio = (fa - fb) / (fb - fc)
    lo, hi = 0.05, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        val = (a**mid - b**mid) / (b**mid - c**mid)
        lo, hi = (mid, hi) if val < ratio else (lo, mid)
    p = mid
    C = (fb - fc) / (b**p - c**p)
    return p, fc - C * c**p, C


def lim_p(h, f, p):
    """L from the two finest points assuming order p."""
    (b, c), (fb, fc) = h[-2:], f[-2:]
    C = (fb - fc) / (b**p - c**p)
    return fc - C * c**p


print("=== U-core skfem P2 series (peak |Bz| mT) ===")
h = np.array([4, 3, 2, 1.5, 1.2, 1.0])
f = np.array([2.1897539, 2.1787490, 2.1683867, 2.1638563, 2.1617456, 2.1602822])
mx = np.array([6.1403914, 6.1240912, 6.1088656, 6.1024236, 6.0993446, 6.0972017])
for i in range(len(h) - 2):
    p, L, C = fit_p(h[i:i + 3], f[i:i + 3])
    pm, Lm, _ = fit_p(h[i:i + 3], mx[i:i + 3])
    print(f"  triple h={h[i:i + 3]}: peak p={p:.2f} L={L:.4f} | m_x p={pm:.2f} L={Lm:.4f}")
for p in (0.8, 1.0, 4 / 3, 1.5, 2.0):
    print(f"  assumed p={p:.2f}: peak limit {lim_p(h, f, p):.4f} mT ({100 * (f[-1] / lim_p(h, f, p) - 1):+.2f} % for h=1); m_x limit {lim_p(h, mx, p):.4f}")
# least-squares fit of all six points to L + C h^p
best = None
for p in np.linspace(0.3, 3, 541):
    Amat = np.stack([np.ones_like(h), h**p], 1)
    coef, res, *_ = np.linalg.lstsq(Amat, f, rcond=None)
    r = ((Amat @ coef - f) ** 2).sum()
    if best is None or r < best[0]:
        best = (r, p, coef)
print(f"  LSQ all six: p={best[1]:.2f} L={best[2][0]:.4f} rms={np.sqrt(best[0] / 6) * 1e3:.3f} uT")
best = None
for p in np.linspace(0.3, 3, 541):
    Amat = np.stack([np.ones_like(h[2:]), h[2:] ** p], 1)
    coef, *_ = np.linalg.lstsq(Amat, f[2:], rcond=None)
    r = ((Amat @ coef - f[2:]) ** 2).sum()
    if best is None or r < best[0]:
        best = (r, p, coef)
print(f"  LSQ finest four: p={best[1]:.2f} L={best[2][0]:.4f} rms={np.sqrt(best[0] / 4) * 1e3:.3f} uT")
print("  peak/m_x ratio:", " ".join(f"{v:.5f}" for v in f / mx))
for e in (-0.5, -0.26, -0.2, 0.1):
    print(f"  MoM at {e:+.2f} % of 2.153 = {2.153 * (1 + e / 100):.4f} mT")
# discrete vs continuous peak at h=1
b = np.array([-2.15739, -2.16022, -2.15923])
print(f"  parabola through the three points around the h=1 peak: continuous max {abs(b[1]) + (b[0] - b[2]) ** 2 / (8 * (b[0] - 2 * b[1] + b[2])) * -1:.5f}")

print("=== cube: NGSolve order series at h=2 and h=1 (moment A m^2) ===")
ref = 0.028894220
for hh, series in ((2, [0.02983, 0.02923, 0.02905]), (1, [0.02916, 0.02898, 0.02893])):
    d = np.diff(series)
    q = d[1] / d[0]
    rest = d[1] * q / (1 - q)
    print(f"  h={hh}: diffs {d}, ratio {q:.2f}, geometric remainder {rest:+.6f} -> limit {series[-1] + rest:.5f} ({100 * ((series[-1] + rest) / ref - 1):+.2f} % vs 0.02889)")
o4 = np.array([0.02905, 0.02893])
for p in (1, 4 / 3, 2, 4):
    L = o4[1] + (o4[1] - o4[0]) / (2**p - 1)
    print(f"  order-4 h=2->1, assumed p={p:.2f}: limit {L:.5f} ({100 * (L / ref - 1):+.2f} %)")
print("=== cube: skfem P2 h-series (mu_r=1000) ===")
hs = np.array([1, 0.7, 0.5, 0.35])
ms = np.array([0.029386397, 0.029186118, 0.029078024, 0.029007553])
for i in range(2):
    p, L, C = fit_p(hs[i:i + 3], ms[i:i + 3])
    print(f"  triple h={hs[i:i + 3]}: p={p:.2f} L={L:.6f} ({100 * (L / ref - 1):+.3f} %)")
for p in (1.0, 4 / 3, 1.5, 2.0):
    L = lim_p(hs, ms, p)
    print(f"  assumed p={p:.2f}: limit {L:.6f} ({100 * (L / ref - 1):+.3f} %)")
print("  skfem ratio m(mu1000)/m(mu1e6) at equal h:", end=" ")
for hh, m1, m6 in ((1, 0.029386397, 0.029477733), (0.7, 0.029186118, 0.029312235), (0.5, 0.029078024, 0.029207962)):
    print(f"h={hh}: {m1 / m6:.5f}", end="  ")
print(f"formula: {(1 / 3.6442 + 1e-6) / (1 / 3.6442 + 1 / 999):.5f}")

print("=== plate: NGSolve order series at h=2 ===")
s = np.array([0.5700, 0.5653, 0.5639])
d = np.diff(s)
q = d[1] / d[0]
print(f"  diffs {d}, ratio {q:.2f}, geometric remainder {d[1] * q / (1 - q):+.5f} -> limit {s[-1] + d[1] * q / (1 - q):.4f}; "
      f"with ratio 2.5: {s[-1] + d[1] * 0.4 / 0.6:.4f}; with ratio 5: {s[-1] + d[1] * 0.2 / 0.8:.4f}")
VH0 = 40 * 40 * 1 * 1e-9 * 0.01 / MU0
print(f"  V H0 = {VH0:.6f}; m/(V H0) = {0.5639 / VH0:.2f} -> N_eff + 1/chi = {VH0 / 0.5639:.5f}; 1/chi = 0.001 is {100 * 0.001 / (VH0 / 0.5639):.1f} % of it")

print("=== magnet cube, tier 0: octupole content at the probes and Robin reflection ===")
a = 0.01
cube = magpy.magnet.Cuboid(dimension=(a,) * 3, polarization=(0, 0, 1.0))
m = a**3 * 1.0 / MU0
d = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
for r in (7.5, 15, 30, 40, 50):
    X = r * 1e-3 * d
    B = cube.getB(X)
    rr = np.linalg.norm(X)
    Bd = MU0 / (4 * np.pi) * (3 * X * (m * X[2]) / rr**5 - np.array([0, 0, m]) / rr**3)
    frac = np.linalg.norm(B - Bd) / np.linalg.norm(B)
    R = 0.1
    refl = 0.3 * (r * 1e-3 / R) ** 7  # (3/4)*(2/5)*(r/R)^7, l=3 reflection by the 2/r Robin BC
    print(f"  r={r:5.1f} mm: |B-B_dipole|/|B| = {100 * frac:.3f} %  -> Robin(R=100 mm) reflected l=3 part ~ {100 * frac * refl:.2e} % of |B|")
print("=== fem.py cube: Dirichlet phi=0 at ~0.1 m ===")
H0 = 0.01 / MU0
print(f"  image field m/(4 pi R_d^3)/H0 = {100 * 0.02889 / (4 * np.pi * 0.1**3) / H0:.3f} % (R_d=0.1), {100 * 0.02889 / (4 * np.pi * 0.08**3) / H0:.3f} % (R_d=0.08)")
