# Method of Moments for Demagnetization

This page describes the physics and numerical method behind `apply_demag`,
starting from the material model and ending with all the algorithmic
optimizations that make the iterative solver fast.

---

## 1. Physical model

A _soft magnetic_ material responds to an applied magnetic field by acquiring a
magnetization proportional to the local total field:

$$\mathbf{M} = \chi \, \mathbf{H}_\text{total}$$

where $\chi$ is the (isotropic scalar) magnetic susceptibility. In magpylib's
convention, the primary field quantity is the **intrinsic polarization**
$\mathbf{J} = \mu_0 \mathbf{M}$ (units of Tesla), so the constitutive relation
becomes

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
(dimensionless) that encodes the volume-averaged field produced at cell $i$'s
centre by a unit polarization in cell $j$.

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
- $\mathbf{S} = \operatorname{diag}(\chi_1,\ldots,\chi_N,\chi_1,\ldots,\chi_N,\chi_1,\ldots,\chi_N)$
  — susceptibility matrix (3N×3N diagonal)
- $\mathbf{T} \in \mathbb{R}^{3N \times 3N}$ — full demag tensor matrix
  ($\mathsf{T}_{ij}$ blocks on rows $i$, columns $j$)

Solving this linear system gives the self-consistent polarization of every cell,
accounting for all mutual demagnetization interactions.

---

## 3. The demagnetization tensor

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

### The unified pair rule

Every entry of $\mathbf{T}$ is defined by one rule, shared by both solvers:

- both cells are Cuboids with a **common orientation** (any relative position,
  any sizes) → analytical volume-averaged Newell tensor;
- anything else (non-cuboid magnets, differently-rotated cells) → **point
  matching**: the field of a unit source polarization evaluated at the observer
  barycentre via `magpy.getH` (Chadebec 2006).

### Sign convention in the code

The default assembly builds the dimensionless $(3N \times 3N)$ operator
directly, block by block, in the Fortran (component-major) layout:

```text
T[(m, j), (k, i)]  =  -N_mk( pos[j] - pos[i] )
```

so `T` maps a polarization $\mathbf{J}$ (Tesla) to the demagnetizing flux
$\mathbf{B}_\text{demag}$ (Tesla). The public :func:`demag_tensor` and the
legacy point-matching path use the historical _pre_-`mu_0` 4-index layout
`T[k, i, j, m] = -N_mk(pos[j] - pos[i]) / mu_0`, promoted by a `T *= mu_0` step
inside `apply_demag`; both conventions describe the same operator.

### Self-demagnetization

For the **diagonal block** $i = j$ (a cell acting on itself), the Newell kernel
reduces to the classical shape demagnetization factors
$(N_{xx}^{(\text{self})},\, N_{yy}^{(\text{self})},\, N_{zz}^{(\text{self})})$
satisfying Brown's identity:

$$N_{xx}^{(\text{self})} + N_{yy}^{(\text{self})} + N_{zz}^{(\text{self})} = 1$$

For a cube: $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$.

These self-factors are used to build the Jacobi preconditioner for GMRES (see
below).

---

## 4. Solvers

### 4.1 Direct solver (`solver="direct"`)

1. Assemble the full $3N \times 3N$ matrix
   $\mathbf{Q} = \mathbf{I} - \mathbf{S}\,\mathbf{T}$ explicitly.
2. Call `numpy.linalg.solve(Q, rhs)`.

Cost: $O(N^2)$ memory, $O(N^3)$ solve. Exact to floating-point precision.
Practical for $N \lesssim 3000$.

The tensor $\mathbf{T}$ is assembled block-wise from the unified pair rule
(`_assemble_T_dense`): analytical Newell blocks for parallel-cuboid cluster
pairs, `magpy.getH` point matching otherwise. Because the iterative solver
applies the _same_ entries matrix-free, **both solvers agree to solver tolerance
for any input**.

The legacy options (`pairs_matching`, `max_dist`, `split`) force the historical
all-point-matching evaluation.

### 4.2 Iterative solver (`solver="iterative"`) — GMRES

Instead of assembling $\mathbf{Q}$ explicitly, GMRES only requires a **matvec**
$\mathbf{v} \mapsto \mathbf{Q}\,\mathbf{v}$ at each iteration. This opens the
door to specialized fast matvec implementations.

The preconditioner is a diagonal (Jacobi) operator using the analytical
self-demag factors:

$$[\mathbf{M}_\text{prec}]_{ii} = 1 + \chi_i \, N_{ii}^{(\text{self})}$$

which approximates the diagonal of $\mathbf{Q}$ and dramatically reduces the
number of GMRES iterations needed for high-susceptibility materials.

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

The kernel (6 independent Newell components pre-FFTed) is built **once** by
`build_fft_kernel` and reused for every GMRES iteration.

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
    ▼  return  J - S * H     [the (I - ST) matvec result]
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
>   clusters too small to be worth dedicated bookkeeping.

Orientation keying uses sign-canonicalized quaternions (largest component made
positive — stable for 180° rotations where $w \approx 0$) with a two-stage
bucket-then-merge scheme that is immune to rounding-boundary splits.

The demag operator becomes a **block structure** over cluster pairs:

```text
 Clusters:   C₁        C₂        C₃
          ┌─────────┬─────────┬─────────┐
     C₁   │ T₁₁ FFT │ T₁₂     │ T₁₃     │   self blocks of grid clusters → FFT
          ├─────────┼─────────┼─────────┤   all other blocks → pair rule:
     C₂   │ T₂₁     │ T₂₂ FFT │ T₂₃     │     parallel cuboids → Newell
          ├─────────┼─────────┼─────────┤     otherwise        → getH
     C₃   │ T₃₁     │ T₃₂     │ T₃₃     │   (dense if small, sparse if huge)
          └─────────┴─────────┴─────────┘
```

**Rotation handling:** each cluster may have an arbitrary orientation $R_C$. The
FFT operates in the cluster's local frame — the polarization is rotated in and
the field rotated back **inside the block** (`_fft_apply`), so the GMRES solve
itself always runs in the global frame where the susceptibility matrix
$\mathbf{S}$ is diagonal. Anisotropic susceptibility therefore works for
arbitrary rotations.

Two size gates keep the bookkeeping proportionate: a detected grid smaller than
`MIN_GRID_CELLS` (16) is handled as a dense _loose_ block instead of an FFT
kernel, and a geometry cluster smaller than `MIN_CLUSTER_CELLS` (8) is folded
into the point-matched _generic_ cluster.

Cross-blocks between parallel cuboid clusters use the generalized Newell
formula; for dense blocks the reverse block is obtained for free from
volume-weighted reciprocity,
$\mathbf{T}_{BA} = (V_A / V_B)\,
\mathbf{T}_{AB}^\mathsf{T}$ (sparsified blocks
are rebuilt per direction to preserve their row-sum error bound). Displacements
are deduplicated before evaluation when fewer than 25 % are distinct — for
same-spacing grids only $O(N)$ of the $N^2$ pair displacements are.

---

### 5.3 Bounded sparsification of large blocks

A block with more than `DENSE_BLOCK_MAX_ENTRIES` dense entries is built in
observer chunks and stored as CSR. Entries are dropped **per observer row,
smallest first, only while the sum of dropped magnitudes stays below a budget**
(`_row_sparsify`):

$$ \sum_{\text{dropped } j} |T_{ij}| ;\le; \varepsilon_\text{row}
  = \frac{0.1\, \varepsilon_\text{tol}}{\max(1, \chi_{\max})\, K}$$

where $\varepsilon_\text{tol}$ is `solver_tol`. Summing over the $\le K$
blocks that touch a row, the operator perturbation is rigorously bounded:

$$\|\mathbf{S}(\mathbf{T}-\tilde{\mathbf{T}})\|_\infty
  \;\le\; 0.1\, \varepsilon_\text{tol}$$

— the truncation error can never exceed the requested solver accuracy. This
replaces the earlier point-estimate triage ($\chi V / r^3$ thresholding),
which bounded individual entries but not their accumulated sum.

---

## 6. Complete algorithm flow

```text
apply_demag(collection, solver=...)
│
├─ 0. Legacy flags (pairs_matching / max_dist / split)?
│      YES → historical all-point-matching dense tensor; iterative then
│            runs a dense matvec without the Jacobi preconditioner
│
├─ 1. Collect cells: positions, dimensions, orientations, χ
│
├─ 2. analyze_structure → clusters (grid / loose / generic), full coverage
│
├─ 3. Build demag operator from the unified pair rule
│      ├─ [direct]     _assemble_T_dense: every cluster-pair block scattered
│      │               into the dense (3N × 3N) T
│      └─ [iterative]  _build_operator:
│            · grid self-blocks   → build_fft_kernel (FFT convolution)
│            · other blocks       → dense (small) or row-budget CSR (large)
│            · reverse Newell blocks by reciprocity transpose
│
├─ 4. Solve linear system  (I - S T) J = J⁽⁰⁾ + S B_ext   [global frame]
│      ├─ [direct]     numpy.linalg.solve(Q, rhs)                  O(N³)
│      └─ [iterative]  GMRES + Jacobi preconditioner (rotated analytical
│                      self-demag factors); RuntimeError on non-convergence
│
└─ 5. Assign self-consistent polarizations back to the magnet objects
```

---

## 7. Complexity summary

| Configuration                   | Operator build                  | Matvec per GMRES iteration    |
| ------------------------------- | ------------------------------- | ----------------------------- |
| Single uniform grid             | $O(N \log N)$                   | $O(N \log N)$                 |
| $K$ grid clusters               | $O(N \log N + \sum_{A\neq B} n_A n_B)$ | $O(N \log N + \text{nnz})$ |
| Loose / generic cells           | $O(n^2)$ for those cells        | $O(n^2)$ or $O(\text{nnz})$   |
| Legacy (`split`, `max_dist`, …) | $O(N^2)$ dense                  | $O(N^2)$                      |

Cross-block build cost is halved by reciprocity and reduced further by
displacement deduplication when cluster grids share their spacing. Indicative
timings (Apple Silicon, `solver_tol=1e-8`, vs. direct): 16× at $N=3\,375$;
$N=27\,000$ solves in ~2 s where the dense solver would need ~50 GB
($(3N)^2 \times 8$ bytes).

---

## 8. References

1. Newell, A. J., Williams, W., & Dunlop, D. J. (1993). _A generalization of the
   demagnetizing tensor for nonuniform magnetization._ Journal of Geophysical
   Research: Solid Earth, 98(B6), 9551–9555.

2. Chadebec, O., Coulomb, J.-L., & Janet, F. (2006). _A review of
   magnetostatic moment method._ IEEE Transactions on Magnetics, 42(4),
   515–520.
$$
