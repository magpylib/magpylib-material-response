"""Attack 1: is p1real's 'galerkin' a Galerkin method, and how good is its 2x2x2 Gauss testing?

(a) uniform self term: 8-point Gauss mean of a cube's own field vs the exact volume mean -1/3
(b) full 12x12 self block (test 4 fn x 3 comps, trial 12 cols): 8-pt vs fine graded quadrature
(c) same for the face neighbour, edge neighbour, corner neighbour blocks
(d) asymmetry of the assembled 27-cell 'galerkin' interaction matrix (exact Galerkin is symmetric)
(e) re-solve the 8/27-cell cube with the test integrals done by fine quadrature ('true' Galerkin)
"""
import sys, time
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/e2f29b63-0bf2-4985-9167-bfac6cef8364/scratchpad/adapt")
import magpylib as magpy
from lincell import tensor
from p1emul import P, grid
MU0 = magpy.mu_0

def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w / 2  # weights sum to 1 on [-1, 1]

def graded(nint, npt, ratio=0.5):
    """Composite Gauss on [-1,1], intervals graded geometrically toward both ends."""
    # breakpoints from 0 to 1: 1 - ratio^k
    b = [0.0] + [1 - ratio**k for k in range(1, nint)] + [1.0]
    b = np.array(b)
    xg, wg = gl(npt)
    xs, ws = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        xs.append((lo + hi) / 2 + (hi - lo) / 2 * xg); ws.append(wg * (hi - lo))
    x = np.concatenate(xs); w = np.concatenate(ws)
    x = np.concatenate([-x[::-1], x]); w = np.concatenate([w[::-1], w]) / 2
    return x, w

def pts3(x, w):
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    W = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    return X, W

def block(xi, w, h, offset):
    """G[(a,m), b] = sum_g w_g psi_a(xi_g) mu0 H_b(offset + xi_g h), per unit coefficient; (12, 12)."""
    psi = np.concatenate([np.ones((len(xi), 1)), xi], axis=1)
    G = 0.0
    for s in range(0, len(xi), 100_000):
        T = tensor(offset[None] + xi[s : s + 100_000] * h, np.broadcast_to(h, xi[s : s + 100_000].shape), linear=True)  # (g, 3, 12)
        G = G + np.einsum("g,ga,gmb->amb", w[s : s + 100_000], psi[s : s + 100_000], T)
    return G.reshape(12, 12)

h = np.array([0.005, 0.005, 0.005])
g8 = 1 / np.sqrt(3)
xi8 = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * g8
w8 = np.full(8, 1 / 8)

MODE = sys.argv[1] if len(sys.argv) > 1 else "quad"
if MODE == "quad":
  print("(a) uniform self term, cube: 8-pt Gauss mean of N (per unit J, mu0 H = -N J):")
  T8 = tensor(xi8 * h, np.broadcast_to(h, (8, 3)), linear=False)
  N8 = -T8.mean(0)
  print("    mean N over 8 Gauss points =\n", np.round(N8, 6), "\n    exact volume mean = diag(1/3): max |err| =", np.abs(N8 - np.eye(3) / 3).max())
  # same for a 10 x 20 x 20 cell (not used in the runs; shows the cube is a special case)
  h2 = np.array([0.005, 0.01, 0.01])
  T8b = tensor(xi8 * h2, np.broadcast_to(h2, (8, 3)), linear=False)
  xf, wf = graded(10, 6); Xf, Wf = pts3(xf, wf)
  Tf = tensor(Xf * h2, np.broadcast_to(h2, (len(Xf), 3)), linear=False)
  print("    10x20x20 cell: 8-pt diag", np.round(-T8b.mean(0).diagonal(), 5), " fine diag", np.round(-np.einsum("g,gmb->mb", Wf, Tf).diagonal(), 5))

  print("\n(b,c) 12x12 blocks, 8-pt Gauss vs graded composite quadrature (10 intervals x 6 GL per axis, ratio 0.5 -> 120^3 pts):")
  for n_int, n_pt in ((6, 4), (10, 6)):
      xf, wf = graded(n_int, n_pt); Xf, Wf = pts3(xf, wf)
      print(f"    quadrature {n_int}x{n_pt} = {len(xf)} per axis, {len(Xf)} points")
      for name, off in (("self", (0, 0, 0)), ("face nb", (2, 0, 0)), ("edge nb", (2, 2, 0)), ("corner nb", (2, 2, 2))):
          off = np.array(off, float) * h
          Gf = block(Xf, Wf, h, off)
          G8 = block(xi8, w8, h, off)
          scale = np.abs(Gf).max()
          print(f"      {name:10s} max|G8 - Gfine| / max|Gfine| = {np.abs(G8 - Gf).max() / scale:.2e};"
                f"  uniform-uniform sub-block (a=1, trial a): {np.abs(G8[:3, :3] - Gf[:3, :3]).max() / np.abs(Gf[:3, :3]).max():.1e};"
                f"  linear test/trial part: {np.abs(G8[3:, 3:] - Gf[3:, 3:]).max() / max(np.abs(Gf[3:, 3:]).max(), 1e-300):.1e}")
      sys.stdout.flush()

# (d) and (e): assemble 'galerkin' matrices for the n^3 cube, 8-pt and fine, and solve
def assemble(pos, dim, chi, b0, xi, w, chunk_pairs=60_000):
    n, half = len(pos), dim / 2
    Q = len(xi)
    psi = np.concatenate([np.ones((Q, 1)), xi], axis=1)
    K = np.zeros((n, 12, n, 12))
    step = max(1, chunk_pairs // n)
    for i in range(n):
        Xi = pos[i] + xi * half[i]
        for s in range(0, Q, step):
            rel = Xi[s : s + step, None, :] - pos[None]
            T = tensor(rel.reshape(-1, 3), np.broadcast_to(half, rel.shape).reshape(-1, 3), linear=True).reshape(len(rel), n, 3, 12)
            K[i] += np.einsum("g,ga,gjmb->amjb", w[s : s + step], psi[s : s + step], T).reshape(12, n, 12)
    K = K.reshape(12 * n, 12 * n)
    M = np.einsum("g,ga,gmb->amb", w, psi, P(xi, 12)).reshape(12, 12)
    A = np.kron(np.eye(n), M) - chi * K
    b = np.zeros((n, 12)); b[:, 2] = chi * b0  # <psi_0 e_z b0>
    return A, b.reshape(-1), K

if MODE == "solve":
  # column b -> row index of the same basis function: col m (a=0) -> row m; col 3+3m+c -> row 3(c+1)+m
  perm = np.array([m for m in range(3)] + [3 * (c + 1) + m for m in range(3) for c in range(3)])
  print("\n(d) asymmetry of the interaction part K (columns permuted to row order; exact Galerkin K is symmetric):")
  for ncell, quads in ((2, ("8-pt Gauss", "GL 6^3", "GL 10^3")), (3, ("8-pt Gauss", "GL 6^3", "GL 10^3")), (4, ("8-pt Gauss", "GL 6^3"))):
    pos, dim = grid((0.01,) * 3, (0, 0, 0), (ncell,) * 3)
    for name in quads:
        xi, w = {"8-pt Gauss": (xi8, w8), "GL 6^3": pts3(*gl(6)), "GL 10^3": pts3(*gl(10))}[name]
        t0 = time.perf_counter()
        A, b, K = assemble(pos, dim, 1000.0, 0.01, xi, w)
        u = np.linalg.solve(A, b).reshape(-1, 12)
        m = (u[:, 2] * np.prod(dim, 1)).sum() / MU0
        n = len(pos)
        Kp = K.reshape(n, 12, n, 12)[:, :, :, perm].reshape(12 * n, 12 * n)
        asym = np.abs(Kp - Kp.T).max() / np.abs(Kp).max()
        print(f"    {ncell**3:3d} cells, {name:11s} ({len(xi):6d} pts/cell): max|K-K^T|/max|K| = {asym:.1e}; "
              f"moment err vs 0.02889 = {100 * (m / 0.02889 - 1):+.2f} %   ({time.perf_counter() - t0:.1f} s)")
        sys.stdout.flush()

if MODE == "ucore":
  from adapt import BLOCKS, CHI, COIL, LINE
  for h_mm, name in ((10, "8-pt Gauss"), (10, "GL 6^3"), (6.67, "8-pt Gauss"), (6.67, "GL 5^3")):
    xi, w = {"8-pt Gauss": (xi8, w8), "GL 5^3": pts3(*gl(5)), "GL 6^3": pts3(*gl(6))}[name]
    parts = [grid(d, c, np.maximum(np.round(np.array(d) / (h_mm * 1e-3)).astype(int), 1)) for d, c in BLOCKS]
    pos, dim = np.vstack([p for p, _ in parts]), np.vstack([d for _, d in parts])
    n, half, Q = len(pos), dim / 2, len(xi)
    t0 = time.perf_counter()
    A, b0, K = assemble(pos, dim, CHI, 0.0, xi, w)
    psi = np.concatenate([np.ones((Q, 1)), xi], axis=1)
    X = (pos[:, None, :] + xi[None] * half[:, None, :]).reshape(-1, 3)
    Bq = np.vstack([COIL.getB(X[s0 : s0 + 1000]) for s0 in range(0, len(X), 1000)]).reshape(n, Q, 3)
    b = CHI * np.einsum("g,ga,ngm->nam", w, psi, Bq).reshape(-1)
    u = np.linalg.solve(A, b).reshape(n, 12)
    t = time.perf_counter() - t0
    line = LINE + (0, 1e-6, 0)
    rel = line[:, None, :] - pos[None]
    T = tensor(rel.reshape(-1, 3), np.broadcast_to(half, rel.shape).reshape(-1, 3), linear=True)
    B = np.einsum("lncb,nb->lc", T.reshape(len(line), n, 3, 12), u) + COIL.getB(line)
    peak = float(np.abs(B[:, 2]).max() * 1e3)
    print(f"    U-core h={h_mm} mm, {n} cells, {name:11s} ({Q:4d} pts/cell): peak {peak:.4f} mT, err vs 2.153 = {100 * (peak / 2.153 - 1):+.2f} %  ({t:.1f} s)")
    sys.stdout.flush()
