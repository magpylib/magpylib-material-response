"""Spectrum of the cuboid's shape operator and the chi-dependence of its response."""
import numpy as np, time
from bem import Panels, kprime_galerkin, solve, moment
np.set_printoptions(precision=4, suppress=True, linewidth=150)
for half in [np.array([1., 1., 1.]), np.array([2., 1., 0.25])]:
    V = 8*np.prod(half)
    pan = Panels(half, 16, "power", 2.0)
    t = time.perf_counter(); K = kprime_galerkin(pan, 4); print("asm", time.perf_counter()-t)
    t = time.perf_counter(); lam, Vr = np.linalg.eig(K); Vl = np.linalg.inv(Vr); print("eig", time.perf_counter()-t)
    print("shape", half, "max|Im lam|", np.abs(lam.imag).max(), " Re lam range", lam.real.min(), np.sort(lam.real)[-3:])
    lam = lam.real; Vr = Vr.real; Vl = Vl.real
    eq = np.argmax(lam)
    for ax in range(3):
        f = pan.normal[:, ax]
        mvec = (pan.area * pan.center[:, ax]) @ Vr  # moment of each right eigvec
        w = mvec * (Vl @ f)
        w[eq] = 0.0
        n = 0.5 - lam
        w = w / V
        print(f" axis {ax}: sum w={w.sum():.6f} (expect 1)  sum w n={np.sum(w*n):.6f}  min w={w.min():.2e}  neg mass={-w[w<0].sum():.2e}  n-support of 99.9% mass: ", end="")
        o = np.argsort(-np.abs(w)); c = np.cumsum(np.abs(w[o]))/np.abs(w).sum()
        k = np.searchsorted(c, 0.999)+1
        print(f"{k} modes, n in [{n[o[:k]].min():.3f},{n[o[:k]].max():.3f}]")
        # Gauss quadrature of the spectral measure (Golub-Welsch via Lanczos on diag(n) with weights)
        chis = np.concatenate([[1e-3, 0.05, 0.3, 1, 3, 10, 30, 100, 1e3, 1e4, 1e6]])
        exact = np.array([np.sum(w*c/(1+c*n)) for c in chis])
        keep = np.abs(w) > 0
        wp, npos = w[keep], n[keep]
        # Lanczos with inner product sum wp * p q (wp may have tiny negatives; use |w| for robustness check)
        for J in [1, 2, 3, 4, 6]:
            # Stieltjes procedure
            q_prev = np.zeros_like(npos); q = np.ones_like(npos)/np.sqrt(wp.sum())
            a, b = [], [0.0]
            for j in range(J):
                aj = np.sum(wp*npos*q*q); a.append(aj)
                r = (npos-aj)*q - b[-1]*q_prev
                bj = np.sqrt(max(np.sum(wp*r*r), 1e-300)); b.append(bj)
                q_prev, q = q, r/bj
            T = np.diag(a) + np.diag(b[1:J], 1) + np.diag(b[1:J], -1)
            nodes, U = np.linalg.eigh(T); weights = wp.sum()*U[0]**2
            approx = np.array([np.sum(weights*c/(1+c*nodes)) for c in chis])
            print(f"   J={J} poles n_j={np.round(nodes,4)}  max rel err over chi in [1e-3,1e6]: {np.max(np.abs(approx/exact-1)):.1e}")
