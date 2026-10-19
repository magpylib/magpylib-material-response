"""Attack 6: convergence order from the cube series, p1real.txt (real kernel) and p1_cube.txt (S=5 emulation)."""
import numpy as np
def fit(cells, err, label, ref_shift=0.0):
    cells, err = np.array(cells, float), np.array(err, float) - ref_shift  # err in %, negative; ref_shift re-bases the reference
    h = 10 / np.cbrt(cells)
    p, c = np.polyfit(np.log(h), np.log(np.abs(err)), 1)
    local = -np.diff(np.log(np.abs(err))) / np.diff(np.log(cells)) * 3
    print(f"{label:38s} ref shift {ref_shift:+.2f} %: global p = {p:.2f}; pairwise p = {np.round(local, 2)}")
for shift in (0.0, +0.12, -0.12):
    fit([27, 64, 216, 512], [-8.83, -6.12, -3.46, -2.27], "uniform p0 (p1real/p1_cube, real)", shift)
    fit([8, 27, 64, 216], [-3.81, -1.91, -1.18, -0.64], "linear lsq8 (p1real, real kernel)", shift)
    fit([8, 27, 64, 125, 216], [-4.29, -2.18, -1.35, -0.96, -0.73], "linear lsq8 (p1emul S=5)", shift)
    fit([27, 64], [-2.1, -1.3], "linear galerkin (p1real)", shift)
    print()
# U-core
fit_u = lambda cells, err, label: print(f"{label:38s} global p = {np.polyfit(np.log(np.cbrt(np.array(cells, float))), np.log(np.abs(np.array(err))), 1)[0] * -1:.2f}")
fit_u([88, 297, 704, 1375, 2376, 3773], [100 * (x / 2.153 - 1) for x in (1.3779, 1.6720, 1.8328, 1.9266, 1.9851, 2.0236)], "U-core uniform cuboids (2.153 ref)")
fit_u([88, 297, 704], [-2.18, -1.05, -0.65], "U-core linear lsq8")
