# Review: meshing (B1–B7, C1–C7, issues #70 #72 #73 #74)

Reviewer `meshing`. All runs under `guard.sh 2`, ≤ 350 cells or mesh-only; raw
outputs in
`S/review/meshing/{data_checks,plate_tetgen,ucore_probe,tet_small,apply_small,plate_small}.txt`.

## 1. B6 "#52 iterative slower on the U-core: the FFT covers each body's own grid" — WRONG (explanation), OK (numbers)

Claim: 12.6 s vs 3.5 s at 2,376 cells, > 8 GB at 3,773, because "the FFT covers
each body's own grid".

Evidence: `demag_fft.analyze_structure` tries `_detect_grid` on the whole
geometry cluster, which requires the cells to fill a complete Nx×Ny×Nz box; a U
does not. It then splits into connected components with radius 1.5× the median
nearest-neighbour distance; the legs sit on the yoke, so the three bodies are
ONE component, which again is not a box. Result at every h (my probe,
`ucore_probe.txt`): a single `loose(n)` cluster, no `grid`, hence no FFT at any
size. In `_build_operator` a loose self-block is a dense ndarray while 9n² ≤
`DENSE_BLOCK_MAX_ENTRIES` = 2e7 (n ≤ 1,490): GMRES with a dense matvec, same
cost as direct (88 cells 0.01/0.02 s, 297: 0.08/0.08 s today; 1,375: 1.19/1.14 s
in apply_cuboids.txt). At 2,376 cells (5.1e7 entries) `_pair_block` builds the
block in 29 chunks of 84 observers and `_row_sparsify` runs two `argsort`s over
each 252×7,128 chunk, then `sp.csr_matrix` from ~5e7 triplets; the row budget
0.1·1e-6/(999·K) ≈ 1e-10 drops essentially nothing. That is the 12.6 s and 3.3
GB; at 3,773 (1.3e8 entries, values + indices + sort temporaries) it passes 8
GB.

Corrected: "On the U-core #52 detects no grid (the three touching bodies form
one non-box component), so the iterative solver has no FFT to use: it is a dense
GMRES up to ~1,500 cells and, above that, a chunked sparsification into CSR that
is slower and larger than the dense matrix." The FFT sentence in #73 and "split
cells leave #52's FFT grids" in #74 should go (there is no grid to leave on this
geometry).

## 2. B5 / #72 TetGen ignores target_elems — OK (facts), OVERCLAIMED (diagnosis and fix)

Counts reproduce mesh-only (`plate_tetgen.txt`): targets 50/200/1,000 → 3,328
(default) / 1,728 ("relaxed"); target 5,000 → 8,412 / 8,678, so the target does
bite once (V/target)^(1/3) = 0.68 mm < thickness. `quality=False` gives 6 tets,
so `fixedvolume=True, maxvolume=V/target` is not what forces the count;
`minratio` is. At maxvolume = V/50: minratio 1.5 / 2 / 3 / 5 / 10 → 3,328 /
1,728 / 729 / 294 / 90 cells (mindihedral ≤ 1; at minratio ≥ 5 `mindihedral=10`
adds cells back, 552 at ratio 5). `maxvolume=V` or no `-a` at all also gives
1,728: the volume bound is inert. Note tetgen 0.8.4's own defaults are
`minratio=2.0, mindihedral=0.0`; #52 tightened the default to 1.5/10, and the
issue's "relaxed" column (2, 5) is just tetgen's default ratio.

Corrected: `target_elems` is a lower bound on the count (it only sets
`maxvolume`); on a thin body `minratio` sets the count. A warning at 2× is not a
lever: the docstring should say which knob decides, and `mesh_TriangularMesh`
should relax or derive `minratio` for thin bodies (3–5 gives 729–294 cells on
this plate) or default to tetgen's 2.0. The "(volume/target_elems)^(1/3) much
larger than a dimension" trigger in #72 is right.

## 3. B2 vs B3 "the joints matter" — UNPROVEN

Evidence (`tet_small.txt`, `ucore_probe.txt`, tet_uniform.txt, boxes_tet.txt),
error vs 2.153 mT:

- one surface: 53 cells −1.4 %, 59 −1.2, 89 −1.7, 123 −1.7, 229 −1.0, 317 +0.2,
  417 +0.1, 989 −0.5, 1,928 −0.4, 4,016 −0.2: not monotone either, range 1.9
  points.
- per cuboid: 48 +4.3 %, 72 +2.7, 100 −0.05, 152 +0.65, 347 −0.1, 359 +1.8,
  1,033 −0.8, 1,968 +2.2: range 5.1 points, and near-identical meshes (347 vs
  359 cells; per-body [165,91,91] vs [175,92,92]) differ by 1.9 points.
- Shape is not the difference: radius-edge median 0.8–1.1 and max ≤ 1.5 for
  both; volume max/min similar at matched counts.
- Hanging-node non-conformity alone does not do it: the midpoint-split
  one-surface mesh (tet_adaptive.txt: 292 −1.25 %, 593 −1.2 %, 1,377 −0.5 %)
  stays inside the one-surface band.

So the per-cuboid series is more mesh-sensitive, consistent with the joint, but
nothing separates "faces that cross at the joint plane" from "TetGen's per-body
cell distribution": there is no conforming per-body mesh to compare, and the
one-surface series scatters 1.9 points by itself. Corrected: "Meshing the
cuboids separately gives results that jump by 2 % between near-identical meshes
and range −0.8..+4.3 %; one closed surface stays within −1.7..+0.2 % from 53
cells. Whether the joint or the per-body meshing causes it is not separated."
Keep "not monotone"; drop "the joints matter" as a conclusion.

## 4. C1 / C3 indicator and where the cells went — OK

`adapt.residual()`: child centres `pos + CHILD·dim`, CHILD = ±1/4 (octant
centres); H from all cells plus the coil; r = χμ0H − J_i; η = vol·rms over the 8
children. Marking: sort η descending, smallest k with cumulative sum ≥ θ·Σ
(searchsorted + 1), capped by the budget. Adjoint per child = parent/8 as
described. Count from mesh-residual.npz (1,600 cells = the 1,572 mesh plus 4
more splits; caption says 1,600, table 1,572, same mesh): 996/1,600 = 62 % of
cells in |x| 30–50, |z| < 10, which is 16,000/88,000 = 18 % of the volume. 992
of the 1,024 2.5 mm cells are there, 32 at the pole faces (z 60–70), 4 five-mm
cells remain in the junction: the junction blocks are refined to a near-uniform
2.5 mm. Worth saying in #74: the "adaptive" mesh is uniform 5 mm plus uniform
2.5 mm in two 20 mm blocks.

## 5. C6 error estimates — OVERCLAIMED on two of four rows

(a) "change between rounds about 3× too small, for both cell types":
true/|change| from the txt files: cuboid residual 0.8, 6.2, 0.7, 2.7; targeted
1.1, 129, 0.5, 50, 12; tets 2.7, 16, 0.8, 27. Erratic from 0.5× to 130×; "3×" is
the last cuboid round and the first tet round. On uniform cuboids it is a steady
1.6 → 3.4×. Corrected: "on the adaptive series the change between rounds bears
no fixed relation to the error (0.5–130×)".

(b) "Richardson: 7–9 % predicted against 6 % true": fit v = v∞ − cN^(−p/3) on
the uniform cuboids: last 3 points → v∞ = 2.174 mT (+1.0 % vs FEM), p = 1.48,
finest-mesh error predicted −6.9 % (true −6.0 %); last 4 → 2.188 (+1.6 %), −7.5
%; last 5 → 2.218, −8.75 %; all 6 → −12 %. Fixed p = 1.5: 2.171; p = 2:
2.123–2.130 (−1.1..−1.4 %). The local order is still rising (1.18 → 1.69), so a
constant-p fit errs high: conservative. "7–9 %" is the 4–5-point fit; the
3-point fit is 6.9 %. Corrected: "Richardson over the three finest uniform
meshes gives the limit to +1 % and the error of the finest mesh to 6.9 vs 6.0 %:
usable, conservatively, for uniform cuboid series; not for tets (non-monotone)
or adaptive meshes." "None of these is an error estimate" is wrong for this row.

(c) signed adjoint sum ≈ 0: |signed| ≤ 0.06× true — OK. (d) abs sum 3–18× too
large: 3.3–18.5× (residual), 3.3–14.5× (targeted) — OK.

## 6. C2 "2.4× fewer cells" — OK at one point, OVERCLAIMED as a factor

Adaptive 1,572 → −5.94 %; uniform 3,773 → −6.01 %; the uniform count for −5.94 %
is ~3,850 (log-log slope −0.56, extrapolated 2 % past the last point) → 2.45×.
At the other rounds: 242 cells 2.3×, 620 cells 1.24×, 3,994 cells (−4.33 %,
uniform ≈ 6,760 extrapolated) 1.7×. Corrected: "1.2–2.5× fewer cells depending
on the round; 2.4× at 1,572".

## 7. B4 plate "cuboids ~35× cheaper" — OVERCLAIMED (solver mixed into cell type)

My runs (`plate_small.txt`): 20×20×3 on #52 direct 0.9 s / 0.59 GB; iterative
(FFT) 0.2 s / 0.12 GB; main dense 2.1 s / 1.7 GB (brief). Tets 1,728 on #52
direct 7.1 s / 1.7 GB. Like-for-like on the same solver: 7.1/0.9 ≈ 8× (1.44×
cells → 3× by n³, the rest the tetrahedron kernel per pair); vs main 3.4×; the
FFT adds 4.5× and only a full cuboid grid gets it. Corrected for #73: "8× on the
same direct solver; 35× when the cuboid grid also uses the FFT, which tetrahedra
cannot". Likewise "tetrahedra win by about 40× on the U-core" has no
like-for-like basis: cuboids never reach the tets' accuracy; 3,773/89 = 42× in
cells compares −6.0 % with −1.7 %. Say "cuboids are still −6 % at 3,773 cells /
9.3 s where 89 tets / 0.03 s are at −1.7 %".

## 8. Issue texts vs raw files

- #73: every U-core table number matches uniform.txt / apply_cuboids.txt /
  tet_uniform.txt / boxes_tet.txt (704: −14.87 %, 0.38 s, 0.45 GB; 3,773: −6.01
  %, 9.27 s, 5.14 GB; 89: −1.72 %; 417: +0.11 %, 0.46 s, 0.95 GB; 989: −0.53 %,
  2.34 s, 1.21 GB; per-cuboid +2.66/+1.79/−0.77/+2.16 %). Plate rows match only
  figures.py literals; no raw file for the plate runs exists in S. The 20×20×1
  −3.0 % and ×2 −1.6 % reproduce (−2.97, −1.59 % today). "FFT covers each body's
  own grid": wrong (1). "35×" and "40×": (7). "On main, 1,728 tets need > 10
  GB": no record in S.
- #74: table matches residual.txt / tet_adaptive.txt / uniform.txt. "62 % in 18
  %": matches. "about 3× too small, both cell types": wrong (5a). "7 to 9 %":
  only for 4–5-point fits (5b). "2.4×": one point (6). "Split cells leave #52's
  FFT grids": moot on the U-core (1).
- #72: table matches; "up to 66×" = 3,328/50. Diagnosis incomplete (2). 7.1 s /
  28.6 s rows come from figures.py literals only.
- #70: "−2.3 % at 512" matches p1_cube.txt (−2.27 %); "−1.2 % at 1,728" is in no
  result file in S (p1_cube.txt stops at 512; it is what order 1.47 from 216→512
  predicts). Plate numbers match. "16× the entries per cell pair" is right.

## 9. Survivors, one line each

- B1 uniform cuboids −36.0..−6.0 %, times and memory: OK. "About order 1.5":
  local order 1.18 → 1.69, still rising; say "1.2 rising to 1.7".
- B2 one-surface tets −1.7 / +0.1 / −0.5 / −0.4 / −0.2 %: OK, but note the
  series is not monotone.
- B7 cube −2.3 % at 512: OK; −1.2 % at 1,728: not in the files.
- C4 adaptive tets no gain (−0.53 % at 1,377 vs −0.53 % at 989): OK.
- C5 goal-oriented no better than uniform (3,994 → −6.1 % vs uniform 3,773 →
  −6.0 %): OK.
- C7 indicator 3.3–3.7× the solve time (6.6/2.0 s, 41.7/11.2 s): OK.
