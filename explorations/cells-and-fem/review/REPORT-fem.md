# Review `fem`: section A (FEM formulation and the three reference values)

Probe scripts, logs: `S/review/fem/` (`ngs_cube_R.py`, `ngs_plate_diag.py`,
`postproc.py`, `chi_ratio.py`, `dirichlet_est.py`, `analysis.py`, `*.log`). All
runs under `guard.sh 2`, peak RSS <= 1.3 GB, none longer than 10 s.

## 1. A1 weak form and Robin term — OK for `ngs_*.py`; WRONG as a description of `fem.py`

Claim: H = H0 − ∇ψ, ∫μr∇ψ·∇v + ∮(2/r)ψv = ∫(J/μ0 + (μr−1)H0)·∇v; "2/r exact for
a dipole far field".

Evidence: from div(μ0μr H + J) = 0 with H = H0 − ∇ψ and div H0 = 0: −div(μr∇ψ) =
−div M, M = J/μ0 + (μr−1)H0 in the body. Multiply by v: ∫μr∇ψ·∇v − ∮(∂ψ/∂n)v =
∫_body M·∇v (the M·n surface charge is inside the `dx("body")` integral, M = 0
on the outer sphere). For ψ ∝ cosθ/r², ∂ψ/∂r = −2ψ/r, so −∮(∂ψ/∂n)v = +∮(2/r)ψv.
Signs and both RHS terms match `ngs_cube.py:35-37` and `ngs_plate.py:34-36`. The
sphere is centred on the body, so the l=1 part is absorbed exactly; the l=3 part
is reflected with amplitude (3/4)(2/5)(r/R)^7. With magpylib I measured the
non-dipole content of the 10 mm cube field: 0.15 % at r = 30 mm, 0.019 % at 50
mm; times the reflection factor the boundary error at the probes is 1e-5 %
(analysis.log). Domain-size sweep: none existed; I ran R = 100/200/400 mm at h =
2 mm, orders 2 and 3 (magnet_R.log): no trend with R, see finding 7.

`fem.py` (the U-core and the skfem cube) does NOT use this formulation: it is
the two-potential Simkin–Trowbridge form (total ψ in the iron, reduced φ in the
air, coupled by ψ = φ + χ_s on the iron surface, χ_s itself an FE Neumann solve
of the coil potential), `ElementTetP2`, and **Dirichlet φ = 0 on every dof
farther than R/2 from the centre** (`fem.py:137`), i.e. at ≈0.4–0.5 m for the
U-core (elements there are 125 mm) and ≈0.1 m for the cube. No Robin term
anywhere in it. Its weak form (`fem.py:7`, `f_i = −∮(H_s·n)v`) is correct; the
coil field is evaluated by magpylib at the facet quadrature points
(`fem.py:107-109`), μr = 1000 ↔ setup.py's χ = 999 and adapt.py's CHI = 999,
consistent. In this formulation φ is sourced by the iron only (the coil enters
through H_s·n on the iron surface), so the air far field is the iron's dipole:
confirmed.

Corrected: "Robin 2/r, exact for the dipole, reflected octupole 1e-5 % at the
probes" holds for `ngs_cube.py`/`ngs_plate.py` only. `fem.py` is P2,
two-potential, Dirichlet at ~R/2; #73 and studio #56 ("scalar-potential solve
with a dipole boundary condition on an air sphere, scikit-fem, linear elements")
are wrong on all three points.

## 2. A5 U-core reference 2.153 mT — OVERCLAIMED precision; +0.1 % Dirichlet bias; "P1" wrong

Series (fem-ucore-*.json): 2.18975, 2.17875, 2.16839, 2.16386, 2.16175, 2.16028
mT at h = 4, 3, 2, 1.5, 1.2, 1 mm; dofs 57k → 743k; the iron mesh scales with h
but the air mesh does not (34.9k → 206k tets, SizeMax R/8 = 125 mm fixed).

Richardson (analysis.log): three-point fits of f = L + C h^p give p = 1.18,
1.38, 1.97, 0.81 for successive triples — not one power law. Limits: p = 0.8 →
2.1510; p = 1 → 2.1530 (this is where 2.153 comes from); p = 4/3 → 2.1550; p = 2
→ 2.1570; least squares on all six: p = 1.36, L = 2.1549; on the finest four: p
= 1.65, L = 2.1565. Honest statement: **2.154 ± 0.004 mT (±0.2 %)**, the band
being assumption-driven, not a data error bar. m_x (6.140 → 6.097) behaves the
same way.

Dirichlet truncation bias: φ = 0 at R_d acts as a μ = ∞ shell, adding a uniform
field m_x/(4πR_d³) = 3.9–7.6 A/m along x (m_x = 6.1 A m², R_d = 0.5–0.4 m). MoM
sensitivity on 88 and 704 cuboids (dirichlet.log, ratio stable to 1 %):
Δpeak/peak = +0.08 % (R_d = 0.5), +0.11 % (0.45), +0.15 % (0.4). The reference
is therefore about **0.1 % too high**; every MoM "−x %" is really "−(x−0.1) %".
Cheap check nobody ran: `fem.py ucore 3` with R = 2 m (5 s).

"Approaching from above": there is no one-sided bound for a pointwise field 30
mm away (the Galerkin energy argument bounds l(ψ), not B at a point; see finding
4 for where it does hold). The data are monotone, nothing more.

Post-processing (coordinator's question): `peak_mT` = max|Bz| over the same 161
points (x ∈ [−80, 80] mm, y = 0, z = 100 mm) as adapt.py's LINE, from
element-mean J per tet summed with `tetrahedron_field` plus the magpylib coil
field (`fem.py:144-161`). (a) The cell mean keeps each tet's dipole exact, so
the omitted term is the tet's own quadrupole: relative error ≈
(I₂/h²)·h²/(d·L_J), i.e. O(h²), not the P0 rate O(h). (b) Measured with a smooth
prescribed J = μ0·1000·H_coil on cuboid grids (postproc.log): +0.135 % at 10 mm
cells, +0.036 % at 5 mm, +0.006 % at 2 mm, observed order 1.85–2.0; sign
positive (peak overestimated) for this J. A tet of edge h has the second moment
of a cube of side ≈0.55 h, so the gmsh meshes sit at <= +0.01 % (h = 4) down to
~1e-4 % (h = 1) for smooth J; the real J has sharper gradients at the pole
edges, so allow up to ~0.1 % at h = 4 and <= 0.01 % at h <= 2. (c) Hence the 1.4
% drop of the series is FEM discretisation error, not post-processing, and the
extrapolation above stands; the P0 representation does not shift the limit (it
vanishes as h²) and is exactly the representation the MoM uses, so no bias
enters the MoM/FEM comparison from it.

Discrete vs continuous peak: parabola through the three points around the h = 1
peak gives 2.16011 vs 2.16028 (0.01 %); MoM uses the same 161 points, so it
cancels.

Consequence for section B/D: MoM −0.5 % = 2.142 mT lies outside the band
(2.150–2.158; after the Dirichlet correction ≈ 2.149–2.156); −0.26 % = 2.147 is
at its edge; −0.2 % = 2.149 and +0.1 % = 2.155 are inside. **Any ranking of MoM
variants by differences below ~0.3 % on the U-core is not supported by the
reference.**

## 3. A3 cube reference 0.02889 — OK to 0.1 %; provenance misstated; formula is an upper bound (+0.06 %)

Provenance: #66 is a dependabot bump. #67 says: "The converged moment, 0.02889
A·m², comes from the polarizability of a perfectly permeable cube" — it is the
formula V·H0/(1/3.6442 + 1/χ), not a MoM mesh study. (fem.py uses χ = 999,
ngs_cube.py χ = 1000: 4e-6 relative, irrelevant.)

Formula: write m(χ)/H0 = Σ_k w_k/(1/χ + n_k) over the eigenmodes n_k of the
demagnetising operator (Σw_k = V, Σw_k n_k/V = 1/3, Σw_k/n_k/V = 3.6442). With
g(x) = χx/(χ+x), x = 1/n, concave: Σp_k g(x_k) <= g(x̄) — the single-N formula is
an **upper bound**, exact only at χ → 0 and χ → ∞, error ≈ Var(1/n)/χ. Measured:
MoM ratio m(1000)/m(1e6) on identical cell grids (chi_ratio.log; discretisation
cancels): 0.99623, 0.99603, 0.99593, 0.99587 for 4³…10³ cells → 0.9958 ± 0.0001,
vs formula 0.99637; skfem at equal h: 0.9969, 0.9957, 0.9956. So the formula is
**+0.05–0.06 % high** at χ = 1000: true m ≈ 0.02887–0.02888 if α = 3.6442 is
exact. 3.6442 is Sihvola et al. 2004 (PEC cube, α/ε0V); I could not re-verify
the fourth digit online; the skfem μr = 1e6 series extrapolates to α =
3.649–3.66 (from above), consistent at the 0.1–0.4 % level only.

FEM's own estimate: NGSolve order series at h = 2 (0.02983, 0.02923, 0.02905 —
first two reproduced: +3.22 %, +1.15 %) and h = 1 (0.02916, 0.02898, 0.02893),
ratio ≈ 3.4 per order; geometric remainder → 0.02897 (h = 2), 0.02891 (h = 1);
algebraic p-convergence (corner singularity, see below) → 0.02886. Order 4, h
2→1 with p = 4/3…2 → 0.02885–0.02889. skfem P2 h-series (1, 0.7, 0.5, 0.35 mm):
triple fits p = 1.6, 1.4 → 0.02893, 0.02890; p = 1…2 → 0.02884–0.02894. **Best
FEM estimate 0.02889 ± 0.00003 (±0.1 %).** The reference is consistent with it;
"+0.12 % at order 4, from above" is inside the reference's own uncertainty.

Rate: the order-2 h-series I ran (0.03105, 0.03038, 0.02983, 0.02916 at 4, 3, 2,
1 mm) converges as h^0.86–0.9 — the μ → ∞ exterior corner (Fichera) exponent λ ≈
0.454 gives h^{2λ} = h^0.91 for the moment at any order on uniform body meshes.
So "FEM gets accuracy by raising the order" (#70) is true at these h only
pre-asymptotically; orders 2/3/4 at the same h sit on the same h^0.9 tail with
smaller constants.

## 4. "Approaching from above" — OK and stronger than stated: it is a theorem for the moments

F(v) = ½a(v,v) − l(v) with l(v) = ∫_body(μr−1)H0·∇v; Galerkin gives F(ψ_h) >=
F(ψ) ⇒ l(ψ_h) <= l(ψ); m_z = (μr−1)VH0 − l(ψ)/H0 ⇒ **m_h >= m for every
conforming mesh and order** (a symmetric, coercive; curved-boundary and l=3
reflection effects ~1e-5). Hence rigorous upper bounds: cube m <= 0.02893, plate
m <= 0.5639. Any MoM value above these is wrong by at least the excess (used in
finding 5). It does not transfer to the U-core peak (pointwise) nor to the
U-core m_x (coil source: l(v) ≠ moment functional).

## 5. A4 plate reference 0.5639 — OVERCLAIMED: it is an upper bound, best estimate 0.5633 ± 0.001; tets "+0.13 %" is a lower bound of their error

Evidence (plate.log): the 1 mm plate at body.maxh = 2 mm is **one element
thick** (vertices strictly inside the thickness: 8, 1, 11 against 195–844 on the
faces, at h = 4, 3, 2); order 2 and 3 reproduce (0.569986, 0.565264). Order-2
h-series 0.57542, 0.57280, 0.56999 at 4, 3, 2 mm: equal steps, ~h^0.9 again (p =
1 would extrapolate to 0.5644, above the order-4 upper bound, so p < 1; p =
0.8–0.9 → 0.5627–0.5636). R = 200 vs 400 mm at order 2: 0.56858 vs 0.56999
(−0.25 %) — mesh-to-mesh noise at order 2, not the boundary (reflection ~1e-9).
Order series at h = 2: diffs −0.0047, −0.0014, ratio 3.4 → geometric limit
0.5633 (0.5630–0.5635 for ratios 2.5–5); algebraic p-convergence → as low as
0.562. **Reference: 0.5633, range 0.562–0.5639, upper bound 0.5639 rigorous.**
Order 4 across one element is enough for the interior; the error is the
edge/corner singularity, not the thickness.

MoM: cuboids 76,800 "−0.45 %" = 0.5614 → true error between −0.45 % and −0.1 %.
Tets 3,328 "+0.13 %" = 0.5646 > 0.5639 upper bound → **the tet error is at least
+0.13 % and probably +0.2–0.5 %**; "cuboids −0.45 % vs tets +0.13 %" cannot rank
the two.

## 6. A3 "FEM agrees to 0.12 %", A6 "at 0.1 % FEM wins in 1.5 s" — UNPROVEN

Both references are known to ±0.1 % (findings 3, 5) and FEM's own extrapolation
uncertainty is ±0.1 %, so no number at the 0.1 % level is resolved. Corrected:
"FEM order 4 at 1 mm agrees with the reference to within the ~0.1–0.2 % that
either is known".

## 7. A2 tier-0 "worst pointwise 0.31 %" — OK as a number, OVERCLAIMED as a statement about the method; boundary is not the limit

Error = |B_FEM − B_magpylib|/|B_magpylib| at 30 points r = 7.5…50 mm along
(1,2,3)/√14 (`ngs_cube.py:48-53`); the nearest probe (2, 4, 6) mm is 1 mm above
the top face and 1.4 mm from an edge, where ∇B is log-singular. At h = 2 mm
(magnet_R.log) per-probe errors scatter 10× between neighbours (order 2: 0.7–20
%, order 3: 0.1–5 %), do not grow with r, and reshuffle with R = 100/200/400 mm
or outer maxh 25/50/100 mm — no R trend. The far-probe error is the local
air-cell size (grading 0.3 → ~10–15 mm cells at r = 30–50 mm) through the
discontinuous ∇ψ_h, not the boundary (1e-5 %). Corrected: "0.31 % is the worst
of 30 samples of a noisy pointwise-gradient metric on one mesh; the gate is
passed, but a different mesh could give 0.2 or 0.6 %; G3's refinement and domain
checks (which issue-fem.md admits were not done) are what would make it a
statement".

## 8. A6 cost statements — UNFAIR as written

"FEM 89 s for 0.3 % on the U-core" is scikit-fem **P2** (Python assembly, pyamg
CG) at h = 1 mm, 743k dofs, and "0.3 %" is the distance to the extrapolated
limit of the same series (±0.2 %). Fair version: "scikit-fem P2: 5 s → +1.2 %,
14 s → +0.7 %, 89 s → +0.3 % relative to its own extrapolation (uncertain to
±0.2 %, biased +0.1 % by the Dirichlet truncation). No NGSolve or higher-order
U-core run exists, so this FEM cost is an upper bound. MoM tets: 0.03 s −1.7 %
(89 cells), 2.3 s −0.5 % (989). At the 0.5 % level MoM tets are >= 10× cheaper
than this FEM; at 0.3 % neither method has demonstrated the accuracy." Cube: "at
~0.3 %, MoM graded 512 cells (<1 s) ≈ NGSolve order 3 (1.2 s)" is fine; "at 0.1
% FEM wins" falls under finding 6.

## Survives

- A1 weak form, signs, RHS and the Robin term as implemented in
  `ngs_cube.py`/`ngs_plate.py`.
- A3/A4 "from above" (now a proven bound, not an observation).
- The three references as 3-digit numbers: 2.15 mT, 0.0289 A m², 0.563 A m².
  Nothing at the fourth digit, and MoM differences below ~0.3 % (U-core) / ~0.2
  % (cube, plate) are inside the references' uncertainty.
