# Method of Moments for Demagnetization

This page describes the physics and numerical method behind `apply_demag`,
starting from the material model and ending with all the algorithmic
optimizations that make the iterative solver fast.

---

## 1. Physical model

A _soft magnetic_ material responds to an applied magnetic field by acquiring a
magnetization proportional to the local total field:

$$\mathbf{M} = \chi \, \mathbf{H}_\text{total}$$

where $\chi$ is the magnetic susceptibility — an isotropic scalar, or a 3-vector
$(\chi_x, \chi_y, \chi_z)$ along the **cell's own axes**, which turn with its
orientation like its polarization: a cell rotated by $R$ responds with the
tensor $R\,\operatorname{diag}(\chi_x, \chi_y, \chi_z)\,R^\mathsf{T}$ in the global
frame.
In magpylib's convention, the primary field quantity is the **intrinsic
polarization** $\mathbf{J} = \mu_0 \mathbf{M}$ (units of Tesla), so the
constitutive relation becomes

$$\mathbf{J} = \mu_0 \chi \, \mathbf{H}_\text{total} = \chi \, \mathbf{B}_\text{total}$$

The total field seen by a cell has three contributions:

| Contribution          | Symbol                    | Origin                                                  |
| --------------------- | ------------------------- | ------------------------------------------------------- |
| Remanent polarization | $\mathbf{J}^{(0)}$        | Hard-magnet-like "frozen" moment                        |
| External applied flux | $\mathbf{B}_\text{ext}$   | User-specified `H_ext` on the object                    |
| Demagnetizing flux    | $\mathbf{B}_\text{demag}$ | The cell's own magnetization (and neighbors') acts back |

Current sources (`magpylib.current.*`) in the collection contribute an
additional applied-field term: their field at the cell barycentres is added to
the right-hand side just like $\mathbf{B}_\text{ext}$.

Self-consistency requires that the polarization **satisfies this equation
simultaneously for every cell**:

$$\mathbf{J}_i = \mathbf{J}_i^{(0)} + \chi_i \bigl( \mathbf{B}_{\text{ext},i} + \mathbf{B}_{\text{demag},i} \bigr)$$

---

## 2. Method of Moments — general formulation

### Discretization

The magnetic body is divided into $N$ cuboid cells, each assumed to carry a
**uniform polarization** $\mathbf{J}_i$ (Method of Moments / volume-element
approach). Under this assumption the demagnetizing flux at cell $i$ due to cell
$j$ is linear in $\mathbf{J}_j$:

$$\mathbf{B}_{\text{demag},i} = \sum_{j=1}^{N} \mathsf{T}_{ij} \, \mathbf{J}_j$$

where $\mathsf{T}_{ij}$ is a **3×3 demagnetization tensor block**
(dimensionless): the field that a unit polarization in cell $j$ produces at
cell $i$, taken at its centre or averaged over it depending on the
interaction rule (section 3).

### Linear system

Substituting into the self-consistency equation and collecting terms:

$$
\mathbf{J}_i = \mathbf{J}_i^{(0)} + \chi_i \mathbf{B}_{\text{ext},i}
               + \chi_i \sum_{j} \mathsf{T}_{ij} \mathbf{J}_j
$$

In matrix–vector form (stacking all 3-component cell vectors into a single
$3N$-vector):

$$
\boxed{(\mathbf{I} - \mathbf{S}\,\mathbf{T})\,\mathbf{J} =
         \mathbf{J}^{(0)} + \mathbf{S}\,\mathbf{B}_\text{ext}}
$$

where:

- $\mathbf{J} \in \mathbb{R}^{3N}$ — unknown polarization vector (Fortran-flat:
  $[J_{x,1},\ldots,J_{x,N},\, J_{y,1},\ldots,J_{z,N}]$)
- $\mathbf{S}$ — susceptibility matrix, block-diagonal: cell $i$ contributes
  its 3×3 tensor $R_i \operatorname{diag}(\boldsymbol{\chi}_i) R_i^\mathsf{T}$
  (diagonal when the susceptibility is isotropic or the cell unrotated)
- $\mathbf{T} \in \mathbb{R}^{3N \times 3N}$ — full demag tensor matrix
  ($\mathsf{T}_{ij}$ blocks on rows $i$, columns $j$)

Solving this linear system gives the self-consistent polarization of every cell,
accounting for all mutual demagnetization interactions.

---

## 3. The demagnetization tensor

### Two interaction rules

A cell's field is known exactly; the method of moments only has to decide
_where_ in the observer cell to take it. `apply_demag` offers two answers,
chosen per solve with `interaction` and applied to **every pair**, never mixed:

| `interaction`         | The field of cell $j$ at cell $i$       | Works for                                    |
| --------------------- | --------------------------------------- | -------------------------------------------- |
| `"point"` (default)   | at the centre of cell $i$               | any cell type and orientation                |
| `"newell"`            | averaged over the whole of cell $i$     | cuboid cells, all with one orientation       |

**Point matching** (Chadebec 2006) is the rule this package has always used.
**Volume averaging** is the Galerkin form of the problem: with it, the induced
moment of a soft body in a uniform field is a **lower bound** that rises with
every refinement of the mesh and never overshoots, so a convergence study knows
which side the converged value is on. Point matching gives no such guarantee:
on a thin soft plate its moment overshoots and then falls back as the mesh is
refined.

At equal cell count neither rule is more accurate in general. On cubic cells
they agree to fourth order in cell size over distance; on flat or long cells
each is closer to the converged answer on different quantities (on a
20 × 20 × 1 mm soft plate with 4 × 4 × 1 mm cells, point matching is closer on
the moment and volume averaging on the field beside the plate). Both converge
to the same answer.

The rules are never mixed within a solve. A mix would use volume averaging only
between parallel cuboids, so turning one of two touching bodies by any angle,
however small, would switch their interaction to point matching: the result
would jump (0.28 % for a magnet and an iron block turned by a millionth of a
degree), and the lower bound would be lost. `interaction="newell"` therefore
raises when the cells are not all cuboids with one orientation.

### Point matching

The field of a uniformly polarized cell at the observer's centre is evaluated
with magpylib's closed-form field of the cell type (`magpy.getH`, Chadebec
2006). For cuboid source cells this is
{func}`~magpylib_material_response.newell.point_block`: the tensor depends only
on the displacement and the source shape, so cells sharing a shape and
orientation get the same fast paths as volume averaging (deduplicated blocks,
reverse blocks by reciprocity, FFT kernels). It is symmetric and even in the
displacement outside the source. Beyond 50 source sides it switches to the
far-field expansion below, where magpylib's closed form would slowly lose
precision; the error stays below 3 × 10⁻⁸ on both sides of the switch.

### Newell 1993 analytical formula

For **identical, axis-aligned** rectangular prisms of dimensions $(a,b,c)$, the
volume-averaged interaction tensor $\mathsf{N}_{ij}$ (a.k.a. the _Newell
kernel_) has a closed analytical form given by
> A. J. Newell, W. Williams, D. J. Dunlop (1993) _A generalization of the
> demagnetizing tensor for nonuniform magnetization._ J. Geophys. Res.
> **98**(B6), 9551–9555.

The six independent components ($\mathsf{N}_{xx}$, $\mathsf{N}_{yy}$,
$\mathsf{N}_{zz}$, $\mathsf{N}_{xy}$, $\mathsf{N}_{xz}$, $\mathsf{N}_{yz}$) are
computed from two auxiliary functions $f(x,y,z)$ and $g(x,y,z)$ via a 27-point
second-difference operator:

$$
N_{xx}(\mathbf{r};\,a,b,c) =
  -\frac{1}{4\pi abc} \sum_{i,j,k \in \{-1,0,1\}} w_i w_j w_k \;
  f\!\bigl(\mathbf{r} + (ia,\, jb,\, kc)\bigr), \quad w_{\pm1}=1,\; w_0=-2
$$

Implemented in
[`newell.py`](https://github.com/magpylib/magpylib-material-response/blob/main/src/magpylib_material_response/newell.py)
(`newell_f`, `newell_g`, `demag_block`).

### Generalization to different-size prisms

For two **parallel prisms of different sizes** $(a_1,b_1,c_1)$ (source) and
$(a_2,b_2,c_2)$ (observer), the double volume integral collapses per axis to a
**4-point difference** at offsets $\pm(a_1+a_2)/2$, $\pm(a_1-a_2)/2$ with
weights $(1,-1,-1,1)$ (which degenerates to the $(1,-2,1)$ second difference
when $a_1 = a_2$), normalised by the _observer_ volume:

$$
N_{xx}(\mathbf{r}) = -\frac{1}{4\pi\, a_2 b_2 c_2}
  \sum_{\alpha,\beta,\gamma} w_\alpha w_\beta w_\gamma\,
  f\!\bigl(\mathbf{r} + (o_\alpha,\, o_\beta,\, o_\gamma)\bigr)
$$

This satisfies the exact volume-weighted reciprocity
$V_\text{obs} \mathsf{N}(\mathbf{d}; s, o) = V_\text{src}\,
\mathsf{N}(\mathbf{d}; o, s)^\mathsf{T}$,
used to derive reverse cross-blocks by transposition. Implemented in
`demag_block_general` and validated against Gauss–Legendre volume averages of
magpylib's exact cuboid field.

### Far field

Far apart, the exact formulas lose precision: Newell's double differences
cancel nearly equal terms, about $\varepsilon\, r^6 / (V_\text{src} V_\text{obs})$
relative. Beyond a switch distance both rules use the far-field expansion of
the tensor, the point dipole plus its leading cell-size correction,

$$
\mathsf{N}(\mathbf{r}) \approx -\frac{V_\text{src}}{4\pi}
  \Bigl[\nabla\nabla \tfrac{1}{r}
  + \sum_k \tfrac{a_k^2 + b_k^2}{24}\, \partial_k^2 \nabla\nabla \tfrac{1}{r}\Bigr],
$$

accurate to $(h_\text{max}/r)^4$, with $a$ and $b$ the source and observer
sides ($b = 0$ for point matching). Volume averaging switches where the two
errors meet, at $r^{10} = 10^{15}\, h_\text{max}^4 V_\text{src} V_\text{obs}$,
calibrated against 60-digit evaluations of the exact formulas: the worst error
from 5 to 10 000 cell sizes is $3 \times 10^{-7}$ for cubes and
$10^{-5}$ for 10:1 rods. Without the switch, an iron cube 3 m from a magnet
meshed with 1 mm cells received 300 times its induced polarization.

### Sign convention in the code

The default assembly builds the dimensionless $(3N \times 3N)$ operator
directly, block by block, in the Fortran (component-major) layout:

```text
T[(m, j), (k, i)]  =  -N_mk( pos[j] - pos[i] )
```

so `T` maps a polarization $\mathbf{J}$ (Tesla) to the demagnetizing flux
$\mathbf{B}_\text{demag}$ (Tesla). The public
{func}`~magpylib_material_response.demag.demag_tensor` and the legacy
point-matching path use the historical _pre_-`mu_0` 4-index layout
`T[k, i, j, m] = -N_mk(pos[j] - pos[i]) / mu_0`, promoted by a `T *= mu_0` step
inside `apply_demag`; both conventions describe the same operator.

### Self-demagnetization

For the **diagonal block** $i = j$ (a cell acting on itself), volume
averaging gives the classical magnetometric demagnetization factors
$(N_{xx}^{(\text{self})},\, N_{yy}^{(\text{self})},\, N_{zz}^{(\text{self})})$
satisfying Brown's identity:

$$N_{xx}^{(\text{self})} + N_{yy}^{(\text{self})} + N_{zz}^{(\text{self})} = 1$$

Point matching gives the factors at the cell's centre instead, which also sum
to 1. For a cube both are $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$.

The chosen rule's self-factors build the GMRES preconditioner (see below).

---

## 4. Solvers

### 4.1 Direct solver (`solver="direct"`)

1. Assemble the full $3N \times 3N$ matrix
   $\mathbf{Q} = \mathbf{I} - \mathbf{S}\,\mathbf{T}$ explicitly.
2. Call `numpy.linalg.solve(Q, rhs)`.

Cost: $O(N^2)$ memory, $O(N^3)$ solve. Exact to floating-point precision.
Practical for $N \lesssim 3000$.

The tensor $\mathbf{T}$ is assembled block-wise under the chosen rule
(`_assemble_T_dense`). The iterative solver applies the _same_ entries
matrix-free, so **both solvers solve the same system**. GMRES stops when the
residual falls below `solver_tol` relative to the right-hand side; the
polarizations then agree to about `solver_tol` for hard magnets, and for soft
ones the gap can grow to about $\chi \cdot$ `solver_tol` / 3 (35 times
`solver_tol` measured at $\chi = 1000$), so tighten `solver_tol` for high
susceptibilities.

The legacy options (`pairs_matching`, `max_dist`, `split`) take the historical
point-matching path; they do not combine with `interaction="newell"`.

### 4.2 Iterative solver (`solver="iterative"`) — GMRES

Instead of assembling $\mathbf{Q}$ explicitly, GMRES only requires a **matvec**
$\mathbf{v} \mapsto \mathbf{Q}\,\mathbf{v}$ at each iteration. This opens the
door to specialized fast matvec implementations.

The preconditioner is block-Jacobi: one 3×3 block per cell, built from the
cell's susceptibility tensor and its self-demag factors under the chosen rule,

$$\mathbf{M}_i = \mathbf{I} + \boldsymbol{\chi}_i \,
  \operatorname{diag}\bigl(N^{(\text{self})}_i\bigr)$$

which is the cell's own block of $\mathbf{Q}$ (exact for a rotated anisotropic
cell) and dramatically reduces the number of GMRES iterations needed for
high-susceptibility materials.

Two implementation details worth knowing: the solve is **warm-started** at the
right-hand side (`x0 = rhs`), and `max_iter` counts scipy's _restart cycles_
(default restart length 20), so the worst-case matvec budget is about
`20 * max_iter`. Non-convergence raises a `RuntimeError` rather than returning a
partially converged result.

---

## 5. Optimizations

### 5.1 FFT convolution for a uniform single-group grid

When all $N$ cells share the same dimensions and orientation, and their centres
lie on a regular Cartesian grid of shape $(N_x, N_y, N_z)$ with spacing
$(s_x, s_y,
s_z)$, the demag tensor is **translation-invariant**:

$$\mathsf{T}_{ij} = \mathsf{T}(\mathbf{r}_j - \mathbf{r}_i)$$

The matvec $\mathbf{H} = \mathbf{T}\,\mathbf{J}$ is then a **3-D discrete
convolution**, computable in $O(N \log N)$ with an FFT on a doubled
(zero-padded) grid for open boundary conditions.

```text
Physical grid (Nx × Ny × Nz)   →   Doubled grid (2Nx × 2Ny × 2Nz)
   _________                          _________________
  |         |                        |         |0 0 0 0|
  |  cells  |                        |  cells  |0 0 0 0|
  |_________|                        |_________|0 0 0 0|
                                     |0 0 0 0 0|0 0 0 0|
                                     |_________________|
```

The kernel (the 6 independent components of the chosen rule's tensor,
pre-FFTed) is built **once** by `build_fft_kernel` and reused for every GMRES
iteration.

**Flow for a single-group FFT matvec:**

```text
J (3N flat, original-cell order)
    │
    ▼ reorder by grid permutation
J_grid (Nx, Ny, Nz, 3)
    │
    ▼ zero-pad to doubled grid
J_pad (2Nx, 2Ny, 2Nz, 3)
    │
    ▼ rfftn
J_fft (2Nx, 2Ny, Nz+1, 3)  [half-spectrum]
    │
    ▼ multiply by kernel FFT (Nxx,Nyy,Nzz,Nxy,Nxz,Nyz)
H_fft (2Nx, 2Ny, Nz+1, 3)
    │
    ▼ irfftn  →  crop to (Nx, Ny, Nz, 3)
    │
    ▼ reorder back to original-cell order
H (3N flat)
    │
    ▼  return  J - S H       [the (I - ST) matvec result; S per cell]
```

Implemented in `demag_fft_matvec`; the kernel supports grid spacings larger than
the cell size (non-touching cells).

---

### 5.2 Structure analysis — clusters with guaranteed coverage

`analyze_structure` partitions the cells into **clusters**; every cell lands in
exactly one cluster (asserted), so no interaction can be silently dropped:

> - **grid**: identical cuboids with a common orientation whose centres form a
>   complete uniform grid (per spatially-connected component, so two separate
>   meshed bodies with identical cells become two grid clusters — and merge into
>   one when their grids align);
> - **loose**: identical parallel cuboids without grid structure;
> - **generic**: everything else — non-cuboid magnets, rotated singletons, and
>   (under point matching) groups too small to be worth dedicated bookkeeping.

Clusters only decide _how fast_ a block is computed, never which rule it
follows.

Orientation keying uses sign-canonicalized quaternions (largest component made
positive — stable for 180° rotations where $w \approx 0$) with a two-stage
bucket-then-merge scheme that is immune to rounding-boundary splits.

The demag operator becomes a **block structure** over cluster pairs:

```text
 Clusters:   C₁        C₂        C₃
          ┌─────────┬─────────┬─────────┐
     C₁   │ T₁₁ FFT │ T₁₂     │ T₁₃     │   self blocks of grid clusters → FFT
          ├─────────┼─────────┼─────────┤   all other blocks, chosen rule:
     C₂   │ T₂₁     │ T₂₂ FFT │ T₂₃     │     cuboid sources → deduplicated
          ├─────────┼─────────┼─────────┤     other sources  → getH per pair
     C₃   │ T₃₁     │ T₃₂     │ T₃₃     │   (dense if small, sparse if huge)
          └─────────┴─────────┴─────────┘
```

**Rotation handling:** each cluster may have an arbitrary orientation $R_C$. The
FFT operates in the cluster's local frame — the polarization is rotated in and
the field rotated back **inside the block** (`_fft_apply`), so the GMRES solve
itself always runs in the global frame. There each cell's susceptibility is
its 3×3 tensor $R_i \operatorname{diag}(\boldsymbol{\chi}_i) R_i^\mathsf{T}$:
an anisotropic susceptibility follows the cell's own axes, whatever its
rotation.

Two size gates keep the bookkeeping proportionate: a detected grid smaller than
`MIN_GRID_CELLS` (16) is handled as a dense _loose_ block instead of an FFT
kernel, and under point matching a geometry cluster smaller than
`MIN_CLUSTER_CELLS` (8) is folded into the _generic_ cluster. Volume averaging
keeps every cuboid in a geometry cluster, since each pair needs both shapes.

For dense blocks the reverse block comes for free by reciprocity wherever it
is exact: volume averaging obeys
$\mathbf{T}_{BA} = (V_A / V_B)\,\mathbf{T}_{AB}^\mathsf{T}$ for any two parallel
cuboid clusters, point matching $\mathbf{T}_{BA} = \mathbf{T}_{AB}^\mathsf{T}$
when both clusters share one cell shape and orientation (sparsified blocks are
rebuilt per direction to preserve their row-sum error bound). Displacements
are deduplicated before evaluation when fewer than 25 % are distinct — for
same-spacing grids only $O(N)$ of the $N^2$ pair displacements are.

---

### 5.3 Bounded sparsification of large blocks

A block with more than `DENSE_BLOCK_MAX_ENTRIES` dense entries is built in
observer chunks and stored as CSR. Entries are dropped **per observer row,
smallest first, only while the sum of dropped magnitudes stays below a budget**
(`_row_sparsify`):

$$
\sum_{\text{dropped } j} |T_{ij}| ;\le; \varepsilon_\text{row}
  = \frac{0.1\, \varepsilon_\text{tol}}{\max(1, \|\mathbf{S}\|_\infty)\, K}
$$

where $\varepsilon_\text{tol}$ is `solver_tol` and $\|\mathbf{S}\|_\infty$ the
largest row sum of the cells' $|\boldsymbol{\chi}_i|$. Summing over the $\le K$ blocks
that touch a row, the operator perturbation is rigorously bounded:

$$
\|\mathbf{S}(\mathbf{T}-\tilde{\mathbf{T}})\|_\infty
  \;\le\; 0.1\, \varepsilon_\text{tol}
$$

— the truncation error can never exceed the requested solver accuracy. This
replaces the earlier point-estimate triage ($\chi V / r^3$ thresholding), which
bounded individual entries but not their accumulated sum.

---

## 6. Complete algorithm flow

```text
apply_demag(collection, solver=..., interaction=...)
│
├─ 0. Legacy flags (pairs_matching / max_dist / split)?
│      YES → historical point-matching dense tensor (refused with
│            interaction="newell"); iterative then runs a dense matvec
│
├─ 1. Collect cells: positions, dimensions, orientations; one 3×3
│      susceptibility tensor per cell, R diag(χ) Rᵀ; check the rule
│      ("newell": all cells cuboid, one orientation, else raise)
│
├─ 2. analyze_structure → clusters (grid / loose / generic), full coverage
│
├─ 3. Build demag operator under the chosen rule
│      ├─ [direct]     _assemble_T_dense: every cluster-pair block scattered
│      │               into the dense (3N × 3N) T
│      └─ [iterative]  _build_operator:
│            · grid self-blocks   → build_fft_kernel (FFT convolution)
│            · other blocks       → dense (small) or row-budget CSR (large)
│            · reverse blocks by reciprocity where it is exact
│
├─ 4. Solve linear system  (I - S T) J = J⁽⁰⁾ + S B_ext   [global frame]
│      ├─ [direct]     numpy.linalg.solve(Q, rhs)                  O(N³)
│      └─ [iterative]  GMRES + block-Jacobi preconditioner (each cell's
│                      3×3 self block); RuntimeError on non-convergence
│
└─ 5. Assign self-consistent polarizations back to the magnet objects
```

---

## 7. Complexity summary

| Configuration                   | Operator build                         | Matvec per GMRES iteration  |
| ------------------------------- | -------------------------------------- | --------------------------- |
| Single uniform grid             | $O(N \log N)$                          | $O(N \log N)$               |
| $K$ grid clusters               | $O(N \log N + \sum_{A\neq B} n_A n_B)$ | $O(N \log N + \text{nnz})$  |
| Loose / generic cells           | $O(n^2)$ for those cells               | $O(n^2)$ or $O(\text{nnz})$ |
| Legacy (`split`, `max_dist`, …) | $O(N^2)$ dense                         | $O(N^2)$                    |

Cross-block build cost is halved by reciprocity and reduced further by
displacement deduplication when cluster grids share their spacing. Indicative
timings for one meshed cube (Apple Silicon, $\chi = 3$, `solver_tol=1e-8`,
point matching): about 40× faster than the direct solver at $N=3\,375$;
$N=27\,000$ solves in about 1.5 s where the dense solver would need ~50 GB
($(3N)^2 \times 8$ bytes).

---

## 8. References

1. Newell, A. J., Williams, W., & Dunlop, D. J. (1993). _A generalization of the
   demagnetizing tensor for nonuniform magnetization._ Journal of Geophysical
   Research: Solid Earth, 98(B6), 9551–9555.

2. Chadebec, O., Coulomb, J.-L., & Janet, F. (2006). _A review of magnetostatic
   moment method._ IEEE Transactions on Magnetics, 42(4), 515–520.
