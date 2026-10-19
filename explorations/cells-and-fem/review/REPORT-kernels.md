# Review `kernels`: claims D1, D2, D3, D8, D10

(The reviewer could not write this file itself; the text below is its final
report, saved verbatim by the coordinating session. Scripts and raw outputs are
in `kernels-probes/`: `indep_cuboid.py/.out`, `indep_tet.py/.out`,
`indep_tet4.py/.out` (doubly graded surface rule), `flat_near.out`,
`bench_100k.py/.out`, `jit_threads.py/.out`, `emul_s.py/.out`, `emul_sweep.out`,
`farfield_ctx.out`, `author_*.out` (the author's own check/bench scripts re-run
as-is). All runs under `guard.sh 2`; peak RSS 1.2 GB once, typically < 400 MB.)

Independent references used (none derived from the author's formulas): (a)
brute-force 3D Gauss-Legendre quadrature of the dipole kernel over the cell
(outside points); (b) layer-cake slab identity evaluated with magpylib's uniform
cuboid kernel, composite GL graded geometrically toward the kink at t = X_k
(cuboid, inside/near-face points); (c) surface-charge formulation sigma = J.n
per face, volume charge -div J via the gradient theorem (sum_f n_f int 1/R dS),
2D quadrature graded radially toward the closest face point and angularly toward
its foot on each edge (tet, inside and near-surface points; checked against
magpylib's uniform tet kernel at every point to 1e-10..1e-9, which is the rule's
own precision); (d) point-dipole / analytic quadrupole far field.

## 1. D1 linear cuboid kernel — OK (validation now independent; author's "1e-9 inside" was quadrature-limited)

Claim: linear columns validated to 1e-9 inside / 5e-11 outside on 300 points, 10
inside.

Evidence:

- Outside, 50 points at >= 0.6 h from the cell, all 12 columns vs (a) GL(48):
  max |diff| 1.4e-14 (scale 0.10); GL(48) vs GL(32) 3e-15.
- Inside, 240 points (40 at 1e-3 h from a corner, 60 at 1e-3 h from a face, 20
  near the centre, 120 random) vs (b): 1.5e-10 at the near-corner points, <=
  9e-14 everywhere else (scale 0.95). magpylib's own uniform kernel deviates
  from the same reference by the same 1.5e-10 at those corner points, so that
  figure is the reference's, not the kernel's. Surface rule (c) at 40 interior
  points: 1.7e-7 (quadrature-limited).
- Author's `lincell_check.py` re-run as-is: worst 1.2e-9 inside — that is GL-64
  error across the kink at t = X_k, not kernel error.
- Signs: cube centre field exactly 0.0 for every linear column; uniform diagonal
  -1/3; J_x = xi_x gives H_x = +0.147 just outside the +x face (slab reference
  +0.147, away from the positive sheet) and -0.147 outside the -x face; jump of
  mu0 H_n across the +x face = +0.999999 = sigma for both the uniform and the
  linear column. Volume-charge sign (-div J) and surface-charge sign are right.
- Far field with a = 0: decay exponents 4.00/4.00/4.00 (quadrupole); matches the
  analytic quadrupole (second moment V h_k/3) to 2.4e-3 at r = 40 h, consistent
  with the O((h/r)^2) next term.

Corrected statement: "linear cuboid columns agree with an independent dipole
quadrature to 1e-14 outside and with a slab identity built on magpylib's uniform
kernel to 1e-13 inside (1e-10 within 1e-3 h of corners, where magpylib's kernel
itself is at 1e-10)."

New caveat (precision, not correctness): the linear columns lose relative
precision with distance much faster than the uniform ones: 1e-8 at r = 40 h,
1.4e-5 at 160 h, 19 % at 1000 h, pure noise at 1e4 h (uniform columns: 1.4e-6 at
1000 h, 2e-4 at 1e4 h — identical to magpylib's cuboid kernel). In absolute
terms the linear-column error at 1000 h is 2.4e-4 of the uniform column at the
same distance, so harmless for cell-to-cell assembly (max r/h ~ 50 in the
U-core) but worth a docstring line.

## 2. D2 linear tet kernel — OK (6-Kuhn-tet test was a special case; random tets now checked independently)

Claim: uniform part 5e-14 vs magpylib; 6 Kuhn tets reproduce the linear cuboid
kernel to 3e-14.

Evidence (random tets, vertices uniform in [-1,1]^3, i.e. many slivers):

- Outside, 57 points at >= 1.6 scale from the centroid, all 12 columns vs (a)
  collapsed GL(40): 3.6e-16 (scale 2e-3, ~2e-13 relative); GL(40) vs GL(28)
  2e-17.
- Inside/near-surface, 24 points per group vs (c): random interior 2.7e-10, 1e-3
  of a face 3.2e-10, 1e-3 of an edge 4.6e-10, 1e-3 of a vertex 8.6e-10, outside
  0.05 height beyond a face 1.6e-10 (scale ~1). magpylib's uniform tet kernel
  differs from the rule by the same amounts (rule self-consistency 1e-8..4e-8).
  Uniform and linear columns alike.
- Kuhn box extended to 120 points at 1e-3 h from box corners/faces/edges, from
  the internal diagonal faces and from the main diagonal, vs lincell (validated
  above): 2.4e-11.
- Flat tet (height 1e-3 of a unit base): at 0.5 above/below vs (a) 1.3e-12; at
  2x height above and below, inside at half height, and just above the apex, vs
  (c): <= 3.4e-10. Far field: uniform columns vs point dipole 1.6e-2 / 1.4e-3 /
  1.6e-4 / 1.4e-5 at r = 3 / 10 / 30 / 100 (the (size/r)^2 truncation, as it
  should be); linear columns decay exponent 4.01/4.02.
- Reversed vertex order (chirality): 1.8e-15.
- Code: outward normal (`sign(dot(n, a - o))`), the in-plane edge normal m, the
  `sign(wf) * beta` solid-angle term for observers below a face (w < 0) and
  `_safe_sum` are all exercised by random points on both sides of every face; no
  sign error found.

Corrected statement: as D2, plus "random (including sliver and flat) tets
checked inside and within 1e-3 of faces/edges/vertices against an independent
surface-charge quadrature to 1e-9."

Caveat: same far-field precision loss as the cuboid — linear columns 6e-5 at r =
30 scale, 3 % at r = 100, noise at 1000 (absolute: 4e-5 of the uniform column at
r = 100); uniform columns match magpylib's tet kernel (4e-6 at 1000; 6e-3 vs
3e-3 at 1e4).

## 3. D3 cost per pair — OVERCLAIMED (jit numbers are wall-clock on ~9 cores; uniform comparison omits magpylib's special-case handling)

Claim: cuboid: magpylib uniform 3x3 via three calls 1,430 ns; this uniform 800;
linear 3x12 1,640 NumPy; jit 64–70 / 86–109 ns. Tet: 2,134 / 1,317 / 61 / 3,127
/ 101 ns.

Evidence (n = 100,000, float64, same arrays as `bench_lincell.py`, best of 5):

- magpylib `cuboid_field("H")` three calls 1,552 ns; one call with 3n stacked
  pairs 1,529 (the three-call structure costs nothing, and #52's `demag.py` also
  loops over the three unit polarizations); `magpy.core.magnet_cuboid_Bfield`
  (no input checks, no edge/corner masks) three calls 1,339; this uniform 880;
  linear 1,700; jit uniform 67, linear 85. The author's tet bench re-run as-is:
  2,161 / 1,347 / 58 / 3,146 / 104 ns. All within 10 % of the claim.
- Fairness 1: ~15 % of magpylib's time is input validation plus the masks that
  keep it finite on edge lines, edges and corners (see D10); the new kernels do
  none of that. Core-to-core the uniform cuboid kernel is 1.5x faster (880 vs
  1,339 ns), not 1.8x.
- Fairness 2 (`jit_threads.py`): NumPy linear: wall 1,657 ns/pair, process CPU
  1,656 (CPU/wall 1.0). jit linear: wall 83 ns, process CPU 752 ns (CPU/wall
  9.1). XLA's CPU backend spreads the fused kernel over ~9 cores; NumPy is
  single-threaded. Per core the jit gain is 2.2x, not 20–25x.
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`
  did not disable the pool in jax 0.12 (67 -> 76 ns), so a true single-thread
  jit figure could not be measured; the CPU-time figure stands in for it.
- Method: compilation excluded (warm-up call before timing), `block_until_ready`
  inside the timed region: correct. jit output differs from NumPy by 1e-10
  absolute (cuboid; 1e-12 tet) at scale ~1 — XLA's transcendental
  approximations; fine for solving, not bit-exact.
- Output size: the linear kernel writes an (n,3,12) array, 4x the uniform one;
  an elementwise op on that array costs 5 ns/pair, so memory traffic is not what
  the 1,700 ns measures.

Corrected statement: "cuboid: linear 3x12 costs 1.9x the uniform 3x3 of the same
kernel (1,700 vs 880 ns) and about 1.1x magpylib's three uniform calls (1,550
ns, of which ~200 ns is validation and edge/corner handling the new kernel
lacks). jax.jit: 67 / 85 ns wall on a 14-core CPU with ~9 cores busy = ~750 ns
CPU per pair, i.e. ~2x per core; most of the headline speedup is multithreading,
which a chunked NumPy assembly could also get."

## 4. D8 staircase emulation — mechanism OK; "S = 5 valid" OVERCLAIMED; recorded emulation numbers are S = 5

Claim: only S = 5, 15 valid because the collocation points +-0.4 fall on
sub-cell centres for S = 5, 15, 25; S = 3/7 swing +-30 %; real kernel agrees
with S = 15 (-1.9 % both at 27 cube cells; U-core -2.5 % vs -2.2 %).

Evidence:

- Sub-cell centres (i + 0.5)/S \* 2 - 1: S = 5 -> {-0.8, -0.4, 0, 0.4, 0.8}, S =
  15 -> i = 10, S = 25 -> i = 17 contain 0.4; S = 3 -> {-2/3, 0, 2/3}; for S = 7
  and S = 3 the point 0.4 sits 0.8 sub-cell half-widths off the nearest centre
  (0.2 half-widths from a sub-cell face); S = 9: 0.4 half-widths. Arithmetic
  claim true.
- Direct test (`emul_s.py`): field of the S^3 staircase of one unit cell at its
  own 8 collocation points vs the exact kernel. Uniform columns exact for every
  S (<= 2e-14, as a uniform staircase must be). Linear columns, relative to max
  |exact|: S = 3: 110 %, S = 5: 1.0 %, S = 7: 44 %, S = 9: 18 %, S = 15: 0.11 %,
  S = 25: 0.04 %.
- Moments (`p1emul.py cube 2|3 p1 lsq8`, reference 0.02889): 8 cells: S = 3
  -8.56 %, S = 5 -4.29, S = 7 +4.93, S = 15 -3.86, S = 25 -3.83, real kernel
  -3.81. 27 cells: S = 3 -4.83, S = 5 -2.18, S = 7 +1.56, real -1.91. (S = 15 at
  27 cells needs ~90 s and was not run; S = 7 at 8 cells swings by +13 points
  relative to S = 3, not +-30 %.)
- `p1_cube.txt` and `p1_ucore.txt` on disk are S = 5 runs (default `S=5` in
  `p1emul.py`): cube -4.29 / -2.18 / -1.35 / -0.96 / -0.73 % and U-core -4.42 %
  (88) / -2.19 % (297). The real kernel gives -3.81 / -1.91 / -1.18 / -0.64 %
  and U-core -2.18 % (88) / -1.05 % (297). The S = 5 U-core number at 88 cells
  is 2.2 points off.
- The U-core S = 15 vs real gap (-2.5 vs -2.2, from the brief; not reproducible
  within the time budget) is not explained by the self-term (0.1 % at S = 15):
  the staircase fields of neighbouring cells at a collocation point 0.6 h from a
  shared face are also approximate and were never checked.

Corrected statement: "S must put the collocation points on sub-cell centres (S =
5, 15, 25); even then the self-cell linear terms are 1 % off at S = 5 (moment
0.5 points off at 8 cells, 0.27 at 27), 0.1 % at S = 15. The emulation results
in `p1_cube.txt`/`p1_ucore.txt` are S = 5 and are superseded by `p1real.txt`;
the S = 15 U-core figure still differs from the real kernel by 0.3 points."

## 5. D10 singular observers — WRONG in detail (face planes are fine; edge-line extensions are not); fix is right

Claim: observers exactly in a face plane / on an edge line hit 0/0; probe line
offset by 1 µm.

Evidence (`indep_cuboid.py [3]`, `indep_tet.py [5]`, `np.errstate` off):

- Cuboid, observer in the plane x = +h but outside the face rectangle: all 36
  entries finite, 2.7e-15 vs dipole quadrature, 6e-17 vs magpylib. Same for a
  y-face plane. A point on a face (inside the rectangle): finite. Tet, in the
  plane of a face but outside it: finite, 4e-17 vs quadrature.
- Observer on an edge-line extension (two face planes at once): cuboid 16–20 of
  36 entries NaN/inf, tet 36 of 36. On an edge: NaN. Corner/vertex: NaN. Cause:
  `arctan(py*pz/(px*R))` with px = py = 0 (0/0) and `_ln_plus`/`_safe_sum` with
  rest2 = R0^2 = 0 (log 0 - log 0). magpylib's kernels are finite at all of
  these points.
- The U-core probe line y = 0, z = 100 mm lies in the y = 0 face plane of every
  cell (each block is split at y = 0), which by itself is fine; the NaNs come
  from the 1 mm probe spacing hitting the x-face planes of the cells, i.e.
  edge-line extensions. The 1 µm offset in y leaves both. (`p1real.py`'s comment
  "off the y = 0 face planes, where the corner formulas divide by zero" has the
  same imprecision.)

Corrected statement: "the corner formulas are singular on edge lines (and edges
and corners), not in face planes; the probe line was shifted by 1 µm because it
crosses edge-line extensions. Unlike magpylib's kernels, `lincell`/`lintet` do
not mask these cases, which is part of their speed advantage (D3)."

## 6. Not run / breaches

- `p1emul.py` at S = 15 for 27 cells or any U-core case, and S = 7 for the
  U-core: 90 s to many minutes, over the brief's 60 s rule. The brief's U-core
  emulation figures (-2.5 % at S = 15, "+-30 % swing") are UNPROVEN here.
- A true single-thread jit timing could not be forced in jax 0.12; the CPU-time
  measurement is the substitute.
- Reviewer's own breach: a first emulation sweep used `cube 8` (8 cells per side
  = 512 cells, not 8 cells) — S = 3 ran 67 s / 1.4 GB (-1.08 %) and S = 5 ran
  254 s / 1.8 GB (-0.49 %) before it was killed; both stayed under the 2 GB
  guard. No file outside the reviewer's folder was written; nothing posted to
  GitHub.
