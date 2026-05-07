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

Implemented in [`newell.py`](../src/magpylib_material_response/newell.py)
(`newell_f`, `newell_g`, `demag_block`).

### Sign convention in the code

```text
T_code[k, i, j, m]  =  -(1/mu_0) * N_mk( pos[j] - pos[i] )
```

After the `T *= mu_0` step in `apply_demag` (which promotes from H-field units
to B-field units), `T_code` becomes dimensionless: it maps a polarization
$\mathbf{J}$ (Tesla) to the demagnetizing flux $\mathbf{B}_\text{demag}$
(Tesla).

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

The tensor $\mathbf{T}$ is built via `magpy.getH` with three orthogonal unit
polarizations applied to all cells simultaneously (the "3 pol unit" loop in
`demag_tensor`).

### 4.2 Iterative solver (`solver="iterative"`) — GMRES

Instead of assembling $\mathbf{Q}$ explicitly, GMRES only requires a **matvec**
$\mathbf{v} \mapsto \mathbf{Q}\,\mathbf{v}$ at each iteration. This opens the
door to specialized fast matvec implementations.

The preconditioner is a diagonal (Jacobi) operator using the analytical
self-demag factors:

$$[\mathbf{M}_\text{prec}]_{ii} = 1 + \chi_i \, N_{ii}^{(\text{self})}$$

which approximates the diagonal of $\mathbf{Q}$ and dramatically reduces the
number of GMRES iterations needed for high-susceptibility materials.

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

Implemented in `demag_fft_matvec` and `_build_fft_matvec`.

---

### 5.2 Block-FFT for multi-group assemblies

When the input is a collection of **several meshed cuboids** (each with a
different orientation or size), no single uniform grid spans all cells. Instead,
`detect_grid_groups` partitions the cells into groups:

> **Group**: a maximal subset of cells sharing the same cell dimensions _and_
> the same orientation (quaternion), whose centres form a valid regular
> Cartesian grid.

In practice, each meshed cuboid produces exactly one FFT group.

The full $3N \times 3N$ demag matrix is then naturally split into a **block
structure**:

```text
 Groups:    G₁        G₂        G₃
         ┌─────────┬─────────┬─────────┐
    G₁   │ T₁₁ FFT │ T₁₂ CSR │ T₁₃ CSR │
         ├─────────┼─────────┼─────────┤
    G₂   │ T₂₁ CSR │ T₂₂ FFT │ T₂₃ CSR │
         ├─────────┼─────────┼─────────┤
    G₃   │ T₃₁ CSR │ T₃₂ CSR │ T₃₃ FFT │
         └─────────┴─────────┴─────────┘

 Diagonal blocks  T_GG  →  handled by group-local FFT kernel
 Off-diagonal blocks T_AB  →  handled by sparse CSR matrices (see §5.3)
```

The block matvec for one GMRES step is:

```text
For each group G:
    H_G += FFT_self_block(J_G)          ← group-local FFT kernel
    for each other group A ≠ G:
        H_G += T_AG (CSR) @ J_A         ← sparse cross-block matvec
```

**Rotation handling:** each group may have an arbitrary orientation $R_G$. The
FFT operates in the group's local frame, so the polarization is rotated into
that frame before the FFT and the result is rotated back:

```text
J_global  →  R_G⁻¹  →  J_local  →  FFT  →  H_local  →  R_G  →  H_global
```

Implemented in `_fft_block_h` (per-group FFT self-block) and
`_build_block_fft_matvec` (full block matvec).

---

### 5.3 Sparse triage for cross-block interactions

For two groups $A$ and $B$ separated by a large distance, most cell pairs
$(i \in A,\, j \in B)$ have negligible mutual influence. The
**$\chi \cdot V / r^3$ triage criterion** exploits this:

$$\text{keep}(i,j) \iff \max(\chi_i, \chi_j)\,\frac{V_j}{|\mathbf{r}_{ij}|^3} > \varepsilon$$

where
$\varepsilon = $ `solver_tol * 0.1` and $\mathbf{r}_{ij} = \mathbf{p}_i - \mathbf{p}_j$
is the cell-centre displacement.

**Physical interpretation:** $V_j / r^3$ is proportional to the solid angle
subtended by cell $j$ as seen from cell $i$. The $\chi$ weighting accounts for
the fact that a low-susceptibility cell scarcely responds even to a strong
incident field.

The **critical distance** below which no pairs are dropped:

$$r_\text{triage} = \left(\frac{\chi_{\max} \cdot V_\text{cell}}{\varepsilon}\right)^{1/3}$$

For 128-element 1 mm³ cubes with $\chi = 1$ and `solver_tol = 1e-6`:
$r_\text{triage} \approx 200\,\text{mm}$.

**Algorithm in `_compute_cross_block`:**

```text
Input: n_a cells in group A, n_b cells in group B

1. Compute influence(i,j) = max(χᵢ,χⱼ) * Vⱼ / |rᵢⱼ|³  for all (i,j)  [O(n_a·n_b)]

2. Build boolean mask = influence > ε

3. Prune active sources:
     active_b = {j : mask[:,j].any()}   ← columns with ≥1 significant pair
     active_a = {i : mask[i,:].any()}   ← rows   with ≥1 significant pair

4. Call magpy.getH only for (active_a × active_b) sub-problem          [reduced cost]

5. Zero out remaining below-threshold entries within the sub-block

6. Scatter into full CSR matrix of shape (3·n_a, 3·n_b)
```

The resulting `T_AB` is stored as a `scipy.sparse.csr_matrix`. Each GMRES matvec
then costs $O(\text{nnz})$ instead of $O(n_a \cdot n_b)$.

---

## 6. Complete algorithm flow

```text
apply_demag(collection, solver="iterative")
│
├─ 1. Collect cells: positions, dimensions, orientations, χ
│
├─ 2. Detect grid structure
│      ├─ detect_uniform_grid  → single FFT group?
│      │     YES → fft_info path
│      ├─ detect_grid_groups   → multiple FFT groups?
│      │     YES → block-FFT + sparse cross-blocks path
│      └─ otherwise            → dense T matrix (magpy.getH)
│
├─ 3. Build demag operator
│      ├─ [single group]   build_fft_kernel  →  kernel_fft stored in fft_info
│      ├─ [multi-group]    build_fft_kernel per group
│      │                   _compute_cross_block for every (A,B) pair  →  CSR T_AB
│      └─ [dense]          demag_tensor via magpy.getH  →  T (3N×3N dense)
│
├─ 4. Solve linear system  (I - S T) J = J⁽⁰⁾ + S B_ext
│      ├─ [direct]     numpy.linalg.solve(Q, rhs)                  O(N³)
│      └─ [iterative]  GMRES with Jacobi preconditioner
│             matvec options:
│               · _build_fft_matvec           (single group)  O(N log N)
│               · _build_block_fft_matvec     (multi-group)   O(N log N + nnz)
│               · _build_dense_matvec         (fallback)      O(N²)
│
└─ 5. Assign self-consistent polarizations back to source objects
```

---

## 7. Complexity summary

| Configuration              | Tensor build                  | Matvec per GMRES iteration    |
| -------------------------- | ----------------------------- | ----------------------------- |
| Single uniform grid        | $O(N \log N)$                 | $O(N \log N)$                 |
| $K$ groups, well-separated | $O(N \log N) + O(\text{nnz})$ | $O(N \log N) + O(\text{nnz})$ |
| $K$ groups, touching       | $O(K^2 N^2 / K^2) = O(N^2)$   | $O(N^2)$                      |
| Dense fallback             | $O(N^2)$                      | $O(N^2)$                      |

where nnz $\ll K^2 (N/K)^2$ when groups are separated beyond $r_\text{triage}$.

---

## 8. References

1. Newell, A. J., Williams, W., & Dunlop, D. J. (1993). _A generalization of the
   demagnetizing tensor for nonuniform magnetization._ Journal of Geophysical
   Research: Solid Earth, 98(B6), 9551–9555.

2. Chadbec, O. et al. (2006). _Micromagnetic simulation using the
   demagnetization tensor approach._ (Method of Moments references in
   `demag_tensor` docstring.)
