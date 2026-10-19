# Claims under adversarial review

All work from 2026-10-19/20 on magpylib-material-response (MoM demag solver) vs
FEM, and on higher-order cells. Your job: find what is wrong, overclaimed,
unfair, or unproven. Be concrete: a finding is "claim X is wrong/unsupported
because Y; here is the check I ran / the line of code". Do not soften. Do not
pad. If a claim survives, say so in one line.

## Paths

The reviewers ran against the session's scratch copies on 2026-10-20; in this
branch the same files live under `explorations/cells-and-fem/`:

- `S/adapt/*.py`, `S/plate.py` → `mom/`; `S/ngs_*.py` → `fem/`; result files →
  `results/`.
- material-response #52 code (branch feat/demag-solver @ 4901d1d): a worktree of
  that branch on PYTHONPATH.
- The sibling session's scikit-fem FEM: `fem/fem.py`, `fem/ucore_setup.py` (was
  `setup.py`), results `results/fem-ucore-*.json`, `results/fem-cube-*.json`.
- Memory guard: `guard.sh <GB> cmd...` kills cmd above the RSS limit and prints
  peak RSS.
- Environments: see `../README.md`.
- GitHub issues already written from these claims:
  magpylib/magpylib-material-response #70, #72, #73, #74;
  magpylib/magpylib-studio #56.

## HARD RULES

- The machine has 24 GB and crashed on 2026-10-19 from parallel FEM jobs. Four
  reviewers run at once. Run ANY script only through `S/guard.sh 2 ...` (2 GB
  cap). Nothing longer than ~60 s. Never run: ngs_*.py at order 4, adapt.py
  with >1,600 cells, p1real.py ucore 5, p1tet.py ucore 200, tet targets > 200,
  plate.py tet, split.py. Small cases (<= 100 cells) are enough to test a claim.
- Do NOT modify any file under S except inside your own folder
  `S/review/<your-name>/`.
- Do NOT post to GitHub, do not run `gh` with write verbs, do not commit
  anything.
- Report: a Markdown file `S/review/<your-name>/REPORT.md` AND the same text as
  your final message. Format: numbered findings, each with severity (WRONG /
  OVERCLAIMED / UNPROVEN / OK), the claim, the evidence, and what the corrected
  statement should be.

## Physical setup used everywhere

- U-core: docs example `docs/examples/ucore_electromagnet.md` in
  material-response. Three cuboids (yoke 100x20x20 mm, two legs 20x20x60 mm at x
  = +-40 mm, z = 40 mm), chi = 999, coil of 16 Polylines x 25 A. Quantity: peak
  |Bz| along the line y = 0, z = 100 mm (161 points, x in [-80, 80] mm).
  Geometry in `S/adapt/adapt.py` (BLOCKS, COIL, LINE).
- FEM reference for it: the other session's scikit-fem P1 scalar-potential run:
  peak 2.190 / 2.179 / 2.168 / 2.164 / 2.162 / 2.160 mT at h = 4 / 3 / 2 / 1.5 /
  1.2 / 1 mm, "approaching from above", extrapolated to about 2.153 mT. All MoM
  errors below are vs 2.153.
- Iron cube: 10 mm, chi = 1000, 10 mT applied; moment reference 0.02889 A m^2 =
  V H0 / (1/3.6442 + 1/chi) (3.6442 a^3 = polarizability of a perfectly
  permeable cube), stated as material-response's own converged value from a mesh
  study (#66) and checked by NGSolve order 4 (0.02893, from above).
- Thin plate 40x40x1 mm, chi = 1000, 10 mT in-plane; moment reference 0.5639 A
  m^2 from NGSolve order 4 at 2 mm (orders 2,3,4: 0.5700, 0.5653, 0.5639).

## Claims

### A. FEM (S/ngs_cube.py, S/ngs_plate.py, S/diyfem.py is gone; other session's fem.py)

A1. Scalar potential: H = H0 - grad psi, weak form int mur grad psi . grad v + ∮
(2/r) psi v = int (J/mu0 + (mur-1) H0) . grad v. "The Robin term 2/r is exact
for a dipole far field." A2. Tier 0 (10 mm cube, 1 T, mur=1, 30 probes 7.5-50 mm
along (1,2,3), air sphere R = 100 mm, grading 0.3): NGSolve order 2/3/4 at h = 1
mm: worst pointwise error 14 % / 2.8 % / 0.31 %; 0.9 / 1.2 / 1.5 s; 0.9 / 1.2 /
1.4 GB. Claimed "passes the plan's 0.5 % gate on the first mesh tried". A3. Iron
cube chi=1000 moment: NGSolve order 2/3/4 at h=1 mm: +0.93 / +0.30 / +0.12 % vs
0.02889, "approaching from above". A4. Plate moment: 0.5639 A m^2 (order 4, 2
mm) used as reference with "about +-0.2 %" implied. A5. U-core FEM reference
2.153 mT: extrapolated from the P1 series above. Error bar claimed "about +-0.2
%". A6. "FEM vs MoM cost": at ~0.3 % on the cube, MoM graded cuboids (−0.3 %
from 512 cells, <1 s) ≈ FEM order 3 (1.2 s); at 0.1 % FEM wins (1.5 s vs ~190k
MoM cells extrapolated). On the U-core: 89 tets 0.03 s −1.7 % vs FEM 89 s for
0.3 %.

### B. Cell type (S/adapt/adapt.py uniform, S/adapt/adapt_tet.py, S/adapt/boxes_tet.py, S/plate.py)

B1. U-core uniform cuboids (near-cubic, per-body grids of 10..2.86 mm): −36.0 %
(88 cells) ... −6.0 % (3,773 cells, 9.3 s, 5.1 GB). Via apply_demag direct on
#52. B2. U-core tetrahedra, U-core as ONE closed TriangularMesh via
mesh_TriangularMesh(target): 89 cells −1.7 % (0.03 s), 417 +0.1 %, 989 −0.5 %
(2.3 s), 1,928 −0.4 %, 4,016 −0.2 % (43.5 s, 7.3 GB). B3. Tetrahedra of each
cuboid separately (non-conforming at joints): 72..1,968 cells, −0.8..+2.7 %, not
monotone. Claim: "the joints matter". B4. Thin plate: cuboids 20x20x3 (#52
iterative/FFT) −0.95 % 0.2 s 0.12 GB; 80x80x12 −0.45 % 11.5 s; tets 1,728
(relaxed quality) +0.6 % 7.1 s 1.7 GB; 3,328 (default) +0.13 % 28.6 s 4.1 GB.
Claim "cuboids ~35x cheaper at ~1 %". B5. TetGen ignores target_elems on thin
plate: 3,328 cells for targets 50/200/1000 (1,728 with minratio=2,
mindihedral=5). (#72) B6. "#52's iterative solver is slower than direct on the
U-core: 12.6 s vs 3.5 s at 2,376 cells; killed at 8 GB at 3,773 where direct
needs 5.1 GB." Hypothesis given: FFT covers each body's own grid only. B7. Cube
moment convergence of uniform cuboids: −2.3 % at 512, −1.2 % at 1,728, "order
~1.5"; graded (#66) −0.3 % from 512.

### C. Adaptive refinement (S/adapt/adapt.py adaptive, S/adapt/adapt_tet.py adaptive; results residual.txt, targeted.txt, tet_adaptive.txt, uniform.txt, tet_uniform.txt; mesh-residual.npz)

C1. Residual indicator: eta_i = vol_i * rms over the 8 would-be child centres of
|chi mu0 H(x) − J_i|; Dörfler theta = 0.5; split 1→8 (cuboid octants / tet edge
midpoints); non-conforming allowed. C2. Adaptive cuboids: −5.9 % at 1,572 cells
vs uniform −6.0 % at 3,773 → "2.4x fewer cells"; −4.3 % at 3,994. C3. Where
cells went: round 1 splits all 10→5 mm; later rounds go to the two leg–yoke
junction blocks (|x| 30..50, z −10..10): 62 % of cells in 18 % of volume. (An
earlier count said "not concentrated"; retracted.) C4. Adaptive tets: −0.5 % at
1,377 vs uniform −0.5 % at 989: no gain. C5. Goal-oriented (adjoint) indicator
with child adjoint = parent/8: no better than uniform. C6. Error estimates:
change between rounds ~3x too small; adjoint-weighted signed sum ≈ 0; abs sum
3–18x too large; Richardson on uniform cuboids 7–9 % predicted vs 6 % true.
"None usable as an accuracy target." C7. Indicator cost > solve cost (getH at 8
points per cell).

### D. Linear cells (S/adapt/lincell.py, lintet.py, *_check.py, p1emul.py, p1real.py, p1tet.py, bench_lincell.py)

D1. Linear cuboid kernel: closed form = uniform volume charge (−div J) +
linearly varying surface charge per face; corner sums of sqrt/ln/atan shared by
12 columns. Validated: uniform columns vs magpylib cuboid_field 4e-13; linear
columns vs slab quadrature of magpylib's uniform kernel 1e-9 inside / 5e-11
outside (300 random points, 10 inside). D2. Linear tet kernel: Wilton face
potential/field + edge integrals via surface divergence theorem; volume charge =
sum n_f P_f. Validated: uniform part vs magpylib tetrahedron_field 5e-14; 6 Kuhn
tetrahedra of a box reproduce the linear cuboid kernel to 3e-14 inside and
outside. D3. Cost per pair (float64, CPU): cuboid: magpylib uniform 3x3 via
three calls 1,430 ns; "this uniform" 800 ns; linear 3x12 1,640 ns NumPy; jit
64–70 / 86–109 ns. Tet: magpylib uniform 2,134; this uniform 1,317 / jit 61;
linear 3,127 / jit 101 ns. D4. Solve: 12 unknowns per cell, collocation at 8
points (cuboid: xi = ±0.4 least squares via normal equations, or "Galerkin" =
project 24 residuals at the 2x2x2 Gauss points onto the 4 shape functions ⊗ 3
comps; tet: 8 child centroids LSQ, or 4 points at fraction f from centroid to
vertices → square system). D5. Results, cube moment: linear cuboids LSQ −3.8 %
(8 cells) −1.9 % (27) −1.2 % (64, 0.06 s) −0.64 % (216, 0.8 s); Galerkin −2.1 %
(27) −1.3 % (64). Linear tets, 4 Gauss points f=1/sqrt(5): −0.50 % (108 cells,
0.16 s), −0.44 % (408); f=0.5: +0.04 % (108), −0.06 % (408); 8 child centroids:
−1.07 % (108), −0.84 % (408). D6. Results, U-core peak: linear cuboids LSQ −2.2
% (88, 0.16 s) −1.05 % (297, 1.6 s) −0.65 % (704, 10.7 s, 3.7 GB); Galerkin
−1.14 % (88) −0.62 % (297). Linear tets f=1/sqrt(5): −0.82 % (89, 0.13 s), −0.26
% (417, 2.7 s); f=0.5: −0.004 % (89), +0.12 % (417); child centroids −3.2 %
(89), −1.46 % (417). D7. Claims: "linear tets reach −0.26 % at 417 cells in 2.7
s vs uniform tets −0.2 % at 4,016 in 43.5 s: ~16x faster"; "f = 0.5 results are
probably error cancellation"; "collocation decides everything; needs a true
Galerkin formulation". D8. Staircase emulation (p1emul.py, S^3 sub-cuboids):
only S = 5, 15 (points at sub-cell centres) valid; S=3/7 swing ±30 %. Real
kernel agrees with S=15 emulation (−1.9 % both at 27 cube cells; U-core −2.5 %
(S=15) vs −2.2 % real at 88). D9. "The convergence order is unchanged; only the
constant improves." D10. Singularities: observers exactly in a face plane / on
an edge line hit 0/0; probe line offset by 1 µm.

### E. JAX / Array API (S/pr854/bench.py)

E1. #854's cuboid core function: main 425 ns/pair, #854 NumPy 425
(bit-identical), jax eager 120–510, jax.jit 38–52 ns (8–11x), float64 CPU, 14
cores. E2. Kernel assembly is 83–96 % of a material-response dense solve
(split.py: cub 1,375 cells assembly 0.96 s LU 0.14 s; tet 989 assembly 2.27 s LU
0.09 s) → "solves 4–8x faster with JAX". E3. "Apple GPU no float64; float32
unsafe (far-field cancellation)"; "gradients are the bigger prize".
