# Review: solver (D4, D5, D6, D7, D9; E1–E3)

Scripts and logs in `S/review/solver/`: `galerkin_quad.py` (modes
quad/solve/ucore, logs `galerkin_quad.log`, `galerkin_solve.log`,
`galerkin_ucore.log`), `frac_sweep.sh` (`frac_sweep.log`), `sphere_f.py`
(`sphere_f.log`), `cond.py`, `order_fit.py`, `profile_assembly.py`,
`profile_1375.py`, `bench_e1.py`, `threads.py`, `f32.py`. Everything ran under
`guard.sh 2`; largest case 408 tets (1.96 GB peak), longest 20 s.

## 1. D4 "Galerkin" — OVERCLAIMED (mislabelled)

Claim: p1real.py `COLLOC=galerkin` is a Galerkin method.

What it is: Bubnov–Galerkin (test space = trial space) with every testing
integral replaced by the 2x2x2 Gauss rule — equivalently a weighted collocation
`Wc r = 0`, `Wc = psi^T (x) I_3`, at the 8 Gauss points (p1real.py lines 40–44).
The mass term and the applied-field term are integrated exactly by that rule;
the interaction term is not. Evidence:

- Uniform self term of a cube cell: 8-point mean of N = diag(1/3) to 1e-18 —
  exact, but only by symmetry (the points lie on body diagonals, where N_xx =
  N_yy = N_zz = 1/3). For a 10x20x20 cell the same 8-point mean gives N_xx =
  0.518 vs 0.496 exact (4.5 % off). All U-core and cube cells in the runs are
  cubes, so this term is exact there.
- 12x12 self block (test 4 fn x 3 comps vs 12 trial columns), 8-pt vs converged
  quadrature (48 and 120 points per axis agree): max error 4.7 % of the block's
  largest entry; linear test/trial entries 7.3 %. Face neighbour: 3.3 %
  (uniform–uniform part 0.1 %, linear part 13 %); edge neighbour 2.5 % (1.6 % /
  13 %); corner neighbour 1.9 % (1.9 % / 9.4 %).
- The exact Galerkin interaction matrix is symmetric (reciprocity). With columns
  mapped to row order, max|K−K^T|/max|K| = 1.7e-2 with 8 points, 4.1e-4 with
  6^3, 6.5e-5 with 10^3 Gauss points per cell. 6^3 = 216 points per cell is
  quadrature-converged (216, 1000 and 13,824 points give the same moment to 0.01
  %).

Corrected statement: "Galerkin testing with 2x2x2 Gauss under-integration of the
near-field integrals; self and neighbour blocks carry 2–13 % quadrature error."

## 2. D7 "needs a true Galerkin formulation" — OVERCLAIMED; now tested

True (quadrature-converged) Galerkin, same linear cuboid basis:

- Cube: −3.57 % (8 cells), −1.71 % (27), −1.05 % (64) vs 8-pt Galerkin −4.17 /
  −2.12 / −1.33 and LSQ8 −3.81 / −1.91 / −1.18. Fitted order 1.77 vs 1.65 /
  1.69.
- U-core: −0.84 % (88 cells, 216 pts/cell, 3.8 s) and −0.46 % (297 cells, 125
  pts/cell, 20.5 s) vs 8-pt −1.14 / −0.62 and LSQ8 −2.18 / −1.05.

So the testing scheme moves the error by 0.2–0.4 % absolute (20–30 % of it) at
16–27x the kernel work; the order and the picture do not change. "Collocation
decides everything" is wrong for cuboids: at a given mesh the three cuboid
schemes lie within ±0.3 % of each other. The sentence is only true for the tet
`corner4` scheme, where the parameter f changes the answer by percent (finding
3).

## 3. D5/D6/D7 corner4 f = 0.5 — WRONG as a scheme; "error cancellation" CONFIRMED

Sweep, `p1tet.py` with `COLLOC=corner4 FRAC=f`:

| f      | cube 108 tets | U-core 89 tets | icosphere 528 tets (exact ref) |
| ------ | ------------- | -------------- | ------------------------------ |
| 0.3    | −1.43 %       | −2.33 %        | −0.10 %                        |
| 0.4    | −0.88         | −1.41          | +0.02                          |
| 0.4472 | −0.50         | −0.82          | +0.10                          |
| 0.5    | +0.04         | −0.004         | +0.21                          |
| 0.55   | +0.70         | +1.01          | +0.34                          |
| 0.6    | +1.57         | +2.35          | +0.52                          |
| 0.7    | +4.40         | +6.86          | +1.07                          |
| 0.9    | +139          | +273           | —                              |

- The crossing is smooth and monotone; slope 10 %/0.05 (cube), 18 %/0.05
  (U-core). Zero crossing f*: cube 0.496 (108 tets) → 0.507 (408 tets: −0.44 /
  −0.06 / +0.39 % at 0.4472 / 0.5 / 0.55); U-core 0.4998 (89) → 0.483 (417, from
  the brief's −0.26 / +0.12); icosphere 0.39.
- Icosphere test (r = 5 mm, 320 faces; reference m = 3 V_poly H0 chi/(chi+3),
  geometric error O(dr^2) ~ 3e-5 because the l=2 perturbation vanishes for
  icosahedral symmetry): p0 centroid +0.02 %, child8 LSQ +0.02 %, corner4 f =
  0.5 **+0.21 %**. On a body where the uniform scheme is already exact, f = 0.5
  adds a +0.2 % bias. On the cube and U-core that positive bias cancels the
  negative discretization error from under-resolved edge/corner singularities.
  That is error cancellation, demonstrated.
- Barycentric check: point = (1−f) c + f v_i has coordinates ((1+3f)/4, (1−f)/4,
  (1−f)/4, (1−f)/4). f = 1/sqrt(5) gives (0.5854, 0.1382, …) = the degree-2
  four-point Gauss rule (b = (5−sqrt 5)/20; f = 1 − 4b = 1/sqrt 5 — the brief's
  arithmetic is right). f = 0.5 gives (0.625, 0.125, 0.125, 0.125): a degree-1
  rule only (any symmetric equal-weight 4-point rule is), and exactly the
  centroids of the four corner children of the midpoint 1→8 split — a geometric
  coincidence, not a quadrature principle.
- Usability: 1 % change of the answer per 0.05 in f, with the slope barely
  decreasing on refinement (10.0 → 9.1 %/0.05 from 108 to 408 tets): the
  f-dependence is an O(1) self-term sampling error at these resolutions, not a
  vanishing discretization error. A scheme with this free parameter is not
  usable; the parameter has to be removed (Galerkin testing or analytic
  self-integrals), not tuned.
- Also from the data: the Gauss choice f = 1/sqrt 5 stalls on the cube (−0.50 %
  at 108 → −0.44 % at 408 tets), and the parameter-free child8 LSQ is worse than
  uniform tets at equal cell count on the U-core (−3.2 % vs −1.7 % at 89; −1.46
  % vs +0.1 % at 417). Only the tuned f gives the headline numbers.

Corrected statement: "f = 0.5 is a fitted parameter whose zero crossing moves
with geometry (0.39–0.51) and mesh; the linear-tet collocation scheme is not
validated."

## 4. D7 "~16x faster" — OVERCLAIMED

- −0.26 % (417 linear tets), −0.2 % (4,016 uniform), −0.4 % (1,928 uniform, 9.0
  s) and −0.5 % (989 uniform, 2.3 s) all sit in or next to the reference's ±0.2
  % band; no accuracy ranking among them is supported.
- At matched unknown count — 417 linear cells = 5,004 unknowns vs 1,928 uniform
  = 5,784 — errors are −0.26 vs −0.40 % (indistinguishable) and times 2.7 vs 9.0
  s = 3.3x. The uniform time is 93 % magpylib's getH tetrahedron kernel (2,270
  ns/pair measured; 3.7 M pairs = 8.4 s of 9.0 s); the linear kernel costs 3,127
  ns/pair for 4x the columns. With D3's own uniform kernel (1,317 ns) the
  4,016-cell run would be ~21 s assembly + ~5 s LU, so the 16x is ~10x
  kernel-for-kernel and compares times to reach an accuracy the reference cannot
  resolve.
- Cube reference: 0.02889 = V H0/(1/3.6442 + 1/chi) = 0.028894. The other
  session's scikit-fem P1 series (+1.70 / +1.01 / +0.64 / +0.39 % at h = 1 / 0.7
  / 0.5 / 0.35 mm, order 1.40) extrapolates to 0.028894; NGSolve order 4 at h =
  1 is +0.12 % and not converged. So the reference is good to about ±0.05 %:
  −0.26 % is distinguishable from zero; −0.06 % and +0.04 % are not
  distinguishable from each other or from zero (and both lie inside the
  f-sensitivity: 0.005 in f = 0.1 %).

Supportable: "Linear tets reach the FEM reference's ±0.2 % band at 417 cells /
5k unknowns / 2.7 s; uniform tets need ≥1,928 cells / 5.8k unknowns / 9 s for
0.4 % and 4,016 / 12k / 43.5 s for 0.2 %. At equal unknown count the accuracy
gain is inside the reference uncertainty; the time ratio at that point is ~3x,
partly kernel implementation."

## 5. D4 normal equations — OK

27-cell cube lsq8: A is 648x324, cond(A) = 1.7e3, cond(A^T A) = 2.7e6 (8 cells
1.2e3 / 1.5e6; 64 cells 1.9e3 / 3.7e6). Normal-equation, `lstsq` and QR
solutions agree to 3e-10 relative in u; moments identical to 10 digits. Harmless
in float64. Note the fit residual is 3–6 % of |b| rms: the 24 equations per cell
are not satisfied, which is inherent to the over-determined collocation.

## 6. D5/D6 timing scope — OK with caveats

All `s` values are assembly + solve, excluding meshing and the output field:
p1real/p1tet time `assemble_solve` (Python chunk loop over the NumPy kernel +
normal equations or LU); adapt.py `run()` times demag_tensor + LU;
apply_cuboids.txt and tet_uniform.txt time `apply_demag` end to end (adds
collection copy and cluster analysis, small). Same scope. Caveats: (i) LSQ8
forms A^T A from a 24n x 12n matrix, ~3x the flops of an LU of the 12n square
system; (ii) uniform tets run magpylib's getH kernel, ~1.7x slower than D3's
uniform kernel; (iii) uniform cuboid assembly is np.unique-bound, not
kernel-bound (finding 9). Time ratios between the linear code and #52 therefore
mix cell type with implementation.

## 7. D9 "convergence order unchanged" — UNPROVEN (roughly right)

Fits of |err| ∝ h^p: uniform p0 cube (27–512 cells) p = 1.39 (pairwise 1.27 /
1.41 / 1.47, rising); linear LSQ8 (8–216) p = 1.63 (1.70 / 1.67 / 1.51,
falling); 8-pt Galerkin 1.65; true Galerkin 1.77 (8–64); U-core uniform 1.44,
linear LSQ8 1.75. Shifting the cube reference by ±0.12 % moves the linear p
between 1.50 and 1.79 and the uniform between 1.35 and 1.43. The numbers say p ≈
1.4 (uniform) vs 1.5–1.8 (linear), not "unchanged", but the difference is inside
what four points and the reference resolve; neither is order 2, as expected when
the edge/corner singularity of M, not the basis degree, limits the rate on
quasi-uniform meshes. Error ratio linear/uniform at equal cells: 0.22, 0.19,
0.18 (27 / 64 / 216).

Corrected: "both converge at roughly h^1.4–1.7; the linear basis cuts the error
~5x at equal cell count and may raise the apparent order slightly; neither
reaches order 2."

## 8. E1 jax.jit 8–11x — OVERCLAIMED (it is multi-core)

bench.py is right on the two points asked: compile time is excluded (warm-up
call, same shapes) and inputs/outputs are float64 (`JAX_ENABLE_X64` set before
`import jax`; verified `x64: True`, float64 in and out). Re-run: main 422 / 437
ns, #854 NumPy 439 / 433 (bit-identical), jax eager 555 / 156, jit 64 / 43 ns
(6.6x / 10.2x) at n = 10k / 100k. But the jit runs on ~9 of the 14 cores:
cpu/wall = 8.8 (NumPy 1.0). With `NPROC=1` (sizes the PjRt CPU thread pool;
`XLA_FLAGS --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1` do
nothing) jit = 386 ns/pair at cpu/wall 1.00 → 1.2x per core vs NumPy's 452;
`NPROC=2` → 197 ns. numpy→jax transfer adds ~5 ns/pair.

Corrected: "jax.jit: 1.2x per core, ~8–10x on 9 cores; the fair single-core gain
is ~20 %."

## 9. E2 "solves 4–8x faster with JAX" — WRONG for cuboid grids, plausible for tets

split.py labels the whole demag_tensor time "kernels". Profiles: 1,375 cuboids
(1.9 M pairs), demag_tensor 0.97 s: `point_block` kernel 0.145 s (15 %),
`_dedup_blocks` → `np.unique(axis=0)` argsort 0.90 s (92 %); 704 cells: 14 % /
89 %; 88 cells: 56 % / 66 % (0.005 s total). On grids the dedup leaves O(N)
unique displacements, so the kernel is already cheap; a 10x kernel makes the
dense cuboid solve ≤ 1.15x faster. The lever there is the dedup (integer grid
keys or a hash), not JAX. Tets: 95 % of assembly is magpylib's
`_BHJM_magnet_tetrahedron` → `triangle_Bfield` at 2,270 ns/pair (417 cells; raw
2,310 ns for three polarizations), so a faster kernel does transfer; a jitted
kernel on 9 cores (finding 8) could cut the 989-tet solve from 2.36 s to
~0.3–0.4 s.

Corrected: "cuboid grids: assembly is np.unique-bound (~90 %), JAX gives ≤ 15 %;
tets: assembly is kernel-bound (95 %), a jitted multi-core kernel could give
~5x."

## 10. E3 — OK (float32 claim now quantified; the rest untestable here)

float32 cuboid kernel vs float64, unit cube, 2000 random directions: max
relative error 2.8e-6 at 1.5 sides, 2.1e-5 at 3, 1.4e-3 at 10, 5.1e-2 at 30, 2.5
at 100, 2.7e3 at 1000. MoM pairs routinely span 10–100 cell sizes, so float32
needs a far-field switch at ~5–10 sides (the float64 switch is at 50). "Apple
GPU has no float64" not tested (no GPU JAX in the env); "gradients are the
bigger prize" is an opinion, not a claim.
