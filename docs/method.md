# Method of Moments for Demagnetization

This page describes the physics and numerical method behind `apply_demag`, from
the material model to the linear system and the two solvers. For the ideas
first, with 3D views and charts to play with, see
[How it works](how_it_works.md).

---

## 1. Physical model

Each cell's material is linear: its polarization $\mathbf{J} = \mu_0 \mathbf{M}$
(in tesla, the quantity magpylib uses for magnets) responds to the magnetic
field $\mathbf{H}$ at the cell as

$$\mathbf{J} = \mathbf{J}^{(0)} + \chi \, \mu_0 \mathbf{H}$$

where $\mathbf{J}^{(0)}$ is the remanent polarization (zero for a soft material)
and $\chi$ the susceptibility: a scalar, or a 3-vector $(\chi_x, \chi_y,
\chi_z)$ along the **cell's own axes**, which turn with its orientation like its
polarization. A cell rotated by $R$ responds with the tensor
$R\,\operatorname{diag}(\chi_x, \chi_y, \chi_z)\,R^\mathsf{T}$ in the global
frame.

The field at a cell has three parts, all handled as $\mu_0 \mathbf{H}$ in tesla:

| Part          | Symbol                        | Origin                                                       |
| ------------- | ----------------------------- | ------------------------------------------------------------ |
| Applied       | $\mu_0 \mathbf{H}_\text{ext}$ | The uniform `H_ext` of the cell or of a parent `Collection`  |
| Currents      | $\mu_0 \mathbf{H}_\text{cur}$ | The current sources (`magpylib.current.*`) in the collection |
| Demagnetizing | $\mu_0 \mathbf{H}_\text{d}$   | The polarization of all cells, the cell itself included      |

Inside a polarized cell this is not the flux density: $\mathbf{B} = \mu_0
\mathbf{H} + \mathbf{J}$.

Self-consistency requires, for every cell $i$ at once,

$$\mathbf{J}_i = \mathbf{J}_i^{(0)} + \chi_i \, \mu_0 \bigl( \mathbf{H}_{\text{ext},i} + \mathbf{H}_{\text{cur},i} + \mathbf{H}_{\text{d},i} \bigr)$$

---

## 2. Method of Moments

### Discretization

The body is divided into $N$ cells (cuboids, cylinder segments, tetrahedra),
each carrying a **uniform polarization** $\mathbf{J}_j$. The demagnetizing field
is then linear in the polarizations:

$$\mu_0 \mathbf{H}_{\text{d},i} = \sum_{j=1}^{N} \mathsf{T}_{ij} \, \mathbf{J}_j$$

where the dimensionless 3×3 block $\mathsf{T}_{ij}$ gives $\mu_0 \mathbf{H}$ at
cell $i$ per unit polarization of cell $j$, taken at the centroid of cell $i$ or
averaged over it (section 3).

### Linear system

Stacking the polarizations of all cells into one $3N$-vector gives

$$
\boxed{(\mathbf{I} - \mathbf{S}\,\mathbf{T})\,\mathbf{J} =
         \mathbf{J}^{(0)} + \mathbf{S}\,\mu_0 (\mathbf{H}_\text{ext} + \mathbf{H}_\text{cur})}
$$

where:

- $\mathbf{J} \in \mathbb{R}^{3N}$ — the unknown polarizations, stacked
  component by component: $[J_{x,1},\ldots,J_{x,N},\, J_{y,1},\ldots,J_{z,N}]$
- $\mathbf{S}$ — the susceptibilities, block-diagonal: cell $i$ contributes its
  3×3 tensor $R_i \operatorname{diag}(\boldsymbol{\chi}_i) R_i^\mathsf{T}$
- $\mathbf{T} \in \mathbb{R}^{3N \times 3N}$ — the interaction matrix, with the
  blocks $\mathsf{T}_{ij}$

---

## 3. The interaction tensor

### Two interaction rules

A cell's field is known exactly; the method only has to decide _where_ in the
observer cell to take it. `apply_demag` offers two answers, chosen per solve
with `interaction` and applied to **every pair**, never mixed:

| `interaction`       | The field of cell $j$ at cell $i$   | Works for                              |
| ------------------- | ----------------------------------- | -------------------------------------- |
| `"point"` (default) | at the centroid of cell $i$         | any cell type and orientation          |
| `"newell"`          | averaged over the whole of cell $i$ | cuboid cells, all with one orientation |

**Volume averaging** is the Galerkin form of the problem. For a soft body with a
positive susceptibility in a uniform applied field, the induced moment along the
field is then a **lower bound**: it never exceeds the converged value and rises
each time every cell is split into smaller cells, so a convergence study knows
which side the answer lies on. Point matching gives no such guarantee: on a thin
soft plate its moment overshoots and then falls back as the mesh is refined.

At equal cell count neither rule is more accurate in general. On cubic cells
they agree to fourth order in cell size over distance. On a soft cube of 1,728
cells, point matching is closer on the field above it (0.43 % against 0.52 %)
and volume averaging on its moment (1.1 % against 1.2 %). Both converge to the
same answer.

The rules are never mixed within a solve. A mix would use volume averaging only
between parallel cuboids, so turning one of two touching bodies by any angle,
however small, would switch their interaction to point matching: the result
would jump (0.28 % for a magnet and an iron block turned by a millionth of a
degree), and the lower bound would be lost. `interaction="newell"` therefore
raises when the cells are not all cuboids with one orientation.

### Point matching

Each block $\mathsf{T}_{ij}$ is $\mu_0$ times the field $\mathbf{H}$ that cell
$j$, carrying a unit polarization along $x$, $y$ or $z$, produces at the
**centroid** of cell $i$ (Chadebec 2006), from magpylib's closed-form field of
the cell type. For cuboid source cells this is
{func}`~magpylib_material_response.newell.point_block`: the tensor depends only
on the displacement and the source shape, so cells sharing a shape and
orientation get the same fast paths as volume averaging (section 5). Beyond 50
source sides it switches to the far-field expansion below, where magpylib's
closed form slowly loses precision.

### Volume averaging (Newell)

For **identical, axis-aligned** rectangular prisms of sides $(a, b, c)$, the
field averaged over the observer cell has a closed form (Newell, Williams and
Dunlop 1993). The six independent components follow from two auxiliary functions
$f(x,y,z)$ and $g(x,y,z)$ through a 27-point second difference:

$$
N_{xx}(\mathbf{r};\,a,b,c) =
  -\frac{1}{4\pi abc} \sum_{i,j,k \in \{-1,0,1\}} w_i w_j w_k \;
  f\!\bigl(\mathbf{r} + (ia,\, jb,\, kc)\bigr), \quad w_{\pm1}=1,\; w_0=-2
$$

with $\mathsf{T} = -\mathsf{N}$. For two **parallel prisms of different sizes**,
source $(a_1, b_1, c_1)$ and observer $(a_2, b_2, c_2)$, the double volume
integral becomes per axis a 4-point difference at offsets $\pm(a_1+a_2)/2$ and
$\pm(a_1-a_2)/2$ with weights $(1, -1, -1, 1)$, normalized by the observer
volume:

$$
N_{xx}(\mathbf{r}) = -\frac{1}{4\pi\, a_2 b_2 c_2}
  \sum_{\alpha,\beta,\gamma} w_\alpha w_\beta w_\gamma\,
  f\!\bigl(\mathbf{r} + (o_\alpha,\, o_\beta,\, o_\gamma)\bigr)
$$

It satisfies the volume-weighted reciprocity $V_\text{obs}
\mathsf{N}(\mathbf{d}; s, o) = V_\text{src}\, \mathsf{N}(\mathbf{d}; o,
s)^\mathsf{T}$, which gives reverse blocks by transposition. Implemented in
`demag_block` and `demag_block_general`, and checked against Gauss–Legendre
volume averages of magpylib's cuboid field.

### Far field

Far apart, the exact formulas lose precision: Newell's differences cancel nearly
equal terms. Beyond a switch distance both rules use the far-field expansion of
the tensor, the point dipole plus its leading cell-size correction,

$$
\mathsf{N}(\mathbf{r}) \approx -\frac{V_\text{src}}{4\pi}
  \Bigl[\nabla\nabla \tfrac{1}{r}
  + \sum_k \tfrac{a_k^2 + b_k^2}{24}\, \partial_k^2 \nabla\nabla \tfrac{1}{r}\Bigr],
$$

accurate to $(h_\text{max}/r)^4$, with $a$ and $b$ the source and observer sides
($b = 0$ for point matching). Volume averaging switches where the two errors
meet, at $r^{10} = 10^{15}\, h_\text{max}^4 V_\text{src} V_\text{obs}$,
calibrated against 60-digit evaluations of the exact formulas: the worst error
from 5 to 10,000 cell sizes is $3 \times 10^{-7}$ for cubes and $1.2 \times
10^{-5}$ for 10:1 rods. Without the switch, an iron cube 3 m from a magnet
meshed with 1 mm cells received 300 times its induced polarization.

### Layout

`demag_tensor` returns the blocks with shape $(3, N, N, 3)$: polarization
component, source cell, observer cell, field component, as $\mathbf{H}$ in A/m
per tesla. `apply_demag` multiplies them by $\mu_0$ and arranges them into
$\mathbf{T}$ in the component-major layout of $\mathbf{J}$.

### Self-demagnetization

The diagonal block is $\mathsf{T}_{ii} = -\mathsf{N}_i$, the demagnetizing
tensor of cell $i$: at its centroid under point matching, averaged over it (the
magnetometric factors) under volume averaging. Either way its trace is 1. The
tensor at a point is $\mathsf{N} = -\nabla\nabla\psi$ with $\psi(\mathbf{r}) =
\frac{1}{4\pi}\int_V \frac{\mathrm{d}V'}{|\mathbf{r} - \mathbf{r}'|}$, and
Poisson's equation gives $\nabla^2\psi = -1$ inside the cell; the average over
the cell keeps the trace (Moskowitz and Della Torre 1966):

$$N_{xx}^{(i)} + N_{yy}^{(i)} + N_{zz}^{(i)} = 1$$

For a cube both rules give $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$. Under
point matching the centroid must lie inside the cell: `mesh_Cylinder` cuts thin
rings into enough sectors that it does.

---

## 4. Solvers

### 4.1 Direct solver (`solver="direct"`)

1. Assemble the full $3N \times 3N$ matrix $\mathbf{Q} = \mathbf{I} -
   \mathbf{S}\,\mathbf{T}$, block by block under the chosen rule (section 5
   gives the shortcuts).
2. Build the right-hand side from the remanent polarizations, `H_ext`, and the
   field of the current sources at the cell centroids.
3. Factor $\mathbf{Q}$ in place with LAPACK (`scipy.linalg.solve`) and write
   each cell's polarization back, expressed in the cell's own frame.

The solve is direct, exact up to floating-point rounding. Memory grows with
$N^2$ — the matrix alone takes $(3N)^2 \times 8$ bytes, 650 MB for 3,000 cells —
and time with up to $N^3$. The matrix is assembled in Fortran order, every chunk
of a block written straight into it, so the solve stays within about 1.1 times
the matrix (a C-ordered matrix would be copied before the factorization, twice
the matrix in memory).

### 4.2 Iterative solver (`solver="iterative"`)

GMRES (Saad and Schultz 1986) never assembles $\mathbf{Q}$: it only needs the
product $\mathbf{v} \mapsto \mathbf{Q}\,\mathbf{v}$, which the FFT kernels and
the stored blocks of section 5 provide. Both solvers apply the same entries, so
they solve the same system.

GMRES stops when the residual falls below `solver_tol` relative to the
right-hand side, which bounds the residual, not the error. Measured against the
direct solve, the polarizations agree to about 2 × `solver_tol` for hard
magnets. For soft iron at $\chi = 1000$ the gap grows with the cell count, to
160 × `solver_tol` at 4,096 cells and 580 × at 32,768: tighten `solver_tol` for
high susceptibilities. `max_iter` caps the iterations, rounded up to restart
cycles of 100; GMRES that does not converge raises `RuntimeError` rather than
return a partial result. The solve starts from the right-hand side.

The preconditioner is block-Jacobi, one 3×3 block per cell:

$$
\mathbf{M}_i = \mathbf{I} + \boldsymbol{\chi}_i \, \mathsf{N}_i
$$

with $\boldsymbol{\chi}_i$ the cell's susceptibility tensor and $\mathsf{N}_i$
its self-demagnetizing tensor under the solve's rule, in the global frame. For
cuboids that is the cell's own block of $\mathbf{Q}$, rotated with the cell.
Other cells take $\mathsf{N}_i = \mathbf{I}/3$, the isotropic part of their own
block, whose trace is 1 for any cell with its centroid inside. Their full
point-matched blocks vary with the cell's shape, and scaling tetrahedra by them
triples the iterations; leaving such cells unscaled puts them about $\chi$ away
from the scaled cuboids, where GMRES stalls. A grid with 650 tetrahedra at χ =
1000 converges in 80 iterations this way, and not in 1,000 without.

(auto-solver)=

### 4.3 Automatic choice (`solver="auto"`, the default)

The two solvers give the same polarization, so the choice is one of time and
memory, and it follows from the cell structure that section 5 detects. `"auto"`
counts the entries of the interaction matrix that FFT products cover: the cells
of each uniform grid among themselves, and large grids of one spacing between
each other. From a tenth on, it takes the iterative solver. Below, it takes the
direct one: GMRES applies the rest as blocks built pair by pair, as the direct
solver builds them, and many small bodies or cells off any grid make it slower.
Measured on soft iron at $\chi = 1000$, out of 39 models:

| Model                         | Cells | FFT   | Direct         | Iterative      | `"auto"`  |
| ----------------------------- | ----- | ----- | -------------- | -------------- | --------- |
| one cube, identical cells     | 4,096 | 100 % | 3.4 s, 2.1 GB  | 0.41 s, 0.2 GB | iterative |
| three cubes, three cell sizes | 3,240 | 40 %  | 9.2 s, 1.9 GB  | 8.3 s, 1.0 GB  | iterative |
| a grid and 3,900 tetrahedra   | 8,031 | 26 %  | 112 s, 5.7 GB  | 99 s, 4.4 GB   | iterative |
| 100 cubes of 27 cells         | 2,700 | 1 %   | 5.1 s, 0.6 GB  | 6.2 s, 0.7 GB  | direct    |
| voxelized sphere              | 1,304 | 3 %   | 0.80 s, 0.3 GB | 1.35 s, 0.3 GB | direct    |
| tetrahedra                    | 1,931 | 0     | 9.3 s, 1.1 GB  | 9.1 s, 1.1 GB  | direct    |
| cuboids of 216 sizes          | 1,728 | 0     | 34 s, 0.3 GB   | 61 s, 0.4 GB   | direct    |

Under `"auto"` GMRES runs to `solver_tol / 100`, so the polarization stays
within about `solver_tol` of the direct solve on a few thousand cells and within
10 × `solver_tol` on 32,768. If GMRES does not converge, `"auto"` falls back to
the direct solver when its matrix takes at most 2 GB (about 5,000 cells), and
raises otherwise. With the legacy options below, `"auto"` takes the direct
solver.

### Legacy options

Three options of `apply_demag` trade generality for speed or memory. They take
the point-matching path and do not combine with `interaction="newell"`:

| Option                | Effect                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `pairs_matching=True` | Cell pairs with the same offset, orientations and source dimensions are evaluated once; pays off on regular grids |
| `max_dist=d`          | Pairs whose centroids lie $d$ or more times the larger cell side apart are dropped, an approximation              |
| `split=k`             | The field evaluations run in $k$ chunks of source cells, bounding the memory of the field evaluation              |

`pairs_matching` and `max_dist` take cuboid cells only; `split` takes any cell.
`max_dist` does not combine with the other two, and `pairs_matching` does not
combine with `split`.

---

## 5. Fast paths

### 5.1 FFT convolution on a uniform grid

When cells share one shape and orientation and their centres fill a regular grid
of shape $(N_x, N_y, N_z)$, the interaction depends only on the offset between
cells, $\mathsf{T}_{ij} = \mathsf{T}(\mathbf{r}_j - \mathbf{r}_i)$, and the
product $\mathbf{T}\,\mathbf{J}$ is a 3-D discrete convolution: an FFT on the
grid zero-padded to twice its size computes it in $O(N \log N)$. The kernel, the
six independent components of the chosen rule's tensor, is built once by
`build_fft_kernel` and reused at every GMRES iteration (`demag_fft_matvec`). The
grid spacing may exceed the cell size.

### 5.2 Clusters

`analyze_structure` sorts the cells into **clusters**, and every cell lands in
exactly one:

- **grid**: identical cuboids with one orientation whose centres fill a complete
  uniform grid, one cluster per meshed body. Bodies are told apart by the
  Collection their mesher put the cells in, or, when the cells were flattened
  into one Collection, by the full boxes they fill on their lattice, so a yoke
  with its legs is three grids and not one connected, non-box component;
- **loose**: identical parallel cuboids without that grid;
- **generic**: everything else — other cell types, rotated singletons and, under
  point matching, groups smaller than `MIN_CLUSTER_CELLS` (8).

Clusters only decide _how fast_ a block is computed, never which rule it
follows. Grid self-blocks are FFT convolutions in the iterative solver (a grid
smaller than `MIN_GRID_CELLS`, 16, is treated as loose), and so is the block
between two grids of one spacing and orientation when it is large
(`FFT_CROSS_MIN_ENTRIES`): the displacement is $\mathbf{o}_A - \mathbf{o}*B +
\mathbf{p} \odot (\mathbf{i} - \mathbf{j})$ for grid indices $\mathbf{i}$,
$\mathbf{j}$, a convolution on a box of $N_A + N_B - 1$ per axis. The same index
differences give the dense solver that block from one table, without sorting.
Every other block is computed once per distinct displacement when fewer than 25
% of the displacements are distinct, and reverse blocks follow by reciprocity
where it is exact: $\mathsf{T}*{BA} = (V_A / V_B)\,\mathsf{T}_{AB}^\mathsf{T}$
for volume averaging between parallel cuboid clusters, $\mathsf{T}_{BA} =
\mathsf{T}_{AB}^\mathsf{T}$ for point matching when both clusters share one cell
shape and orientation. Each FFT block works in its cluster's own frame, rotating
the polarization in and the field back, so the solve runs in the global frame.

### 5.3 Bounded sparsification of large blocks

In the iterative solver, a block with more than `DENSE_BLOCK_MAX_ENTRIES`
entries is stored sparse when that pays: one chunk of observers spread over the
whole body must show that at least 90 % of the entries can go
(`SPARSIFY_MIN_DROP`), as a sparse product runs on one thread and is only then
about as fast as the dense one. Above `DENSE_BLOCK_MAX_BYTES` (2 GB) memory
comes first and 60 % suffices (`SPARSIFY_MIN_DROP_LARGE`): a sparse matrix takes
12 bytes per kept entry against 8 per dense entry, and about 20 while it is
built. At high susceptibility the budget below rarely allows either, and the
block stays dense at any size. When the explicit blocks pass
`DENSE_BLOCK_MAX_BYTES` in all, the solve logs a warning. Entries are dropped
per observer row, smallest first, only while the sum of the dropped magnitudes
stays within a budget:

$$
\sum_{\text{dropped } j} |T_{ij}| \;\le\; \varepsilon_\text{row} = \frac{0.1\, \varepsilon_\text{tol}}{\max(1, \|\mathbf{S}\|_\infty)\, K}
$$

where $\varepsilon_\text{tol}$ is `solver_tol`, $\|\mathbf{S}\|_\infty$ the
largest row sum of the cells' $|\boldsymbol{\chi}_i|$, and $K$ the number of
clusters. Summed over the at most $K$ blocks in a row, the operator changes by
at most

$$
\|\mathbf{S}(\mathbf{T}-\tilde{\mathbf{T}})\|_\infty \;\le\; 0.1\, \varepsilon_\text{tol}
$$

---

## 6. Algorithm flow

```text
apply_demag(collection, solver=..., interaction=...)
│
├─ 0. Legacy options (pairs_matching / max_dist / split)?
│      YES → point-matched dense tensor (refused with interaction="newell")
│
├─ 1. Collect the magnet cells; one 3×3 susceptibility tensor per cell,
│      R diag(χ) Rᵀ; check the rule ("newell": all cells cuboid, one
│      orientation, else raise); refuse objects with paths
│
├─ 2. analyze_structure → clusters (grid / loose / generic)
│
├─ 3. Build the interaction operator under the chosen rule
│      ├─ [direct]     every cluster-pair block into the dense (3N × 3N) T
│      └─ [iterative]  grid self-blocks as FFT kernels, other blocks dense,
│                      or sparse where the error budget drops most of them,
│                      reverse blocks by reciprocity where it is exact
│
├─ 4. Solve (I − S T) J = J⁽⁰⁾ + S μ0 (H_ext + H_cur)       [global frame]
│      ├─ [direct]     numpy.linalg.solve(Q, rhs)                  O(N³)
│      └─ [iterative]  GMRES with the block-Jacobi preconditioner
│
└─ 5. Write the self-consistent polarizations back to the magnet cells
```

---

## 7. Complexity

| Configuration                   | Operator build                         | Product per GMRES iteration |
| ------------------------------- | -------------------------------------- | --------------------------- |
| Single uniform grid             | $O(N \log N)$                          | $O(N \log N)$               |
| $K$ grid clusters               | $O(N \log N + \sum_{A\neq B} n_A n_B)$ | $O(N \log N + \text{nnz})$  |
| Loose / generic cells           | $O(n^2)$ for those cells               | $O(n^2)$ or $O(\text{nnz})$ |
| Legacy (`split`, `max_dist`, …) | $O(N^2)$ dense                         | $O(N^2)$                    |

Measured on one meshed cube (Apple Silicon, $\chi = 3$, `solver_tol=1e-8`, point
matching): the iterative solver is about 40 times faster than the direct one at
$N = 3{,}375$, and solves $N = 27{,}000$ in about 1.5 s, where the dense matrix
alone would take 52 GB. Several separate bodies are slower: their blocks against
each other are not convolutions.

---

## 8. References

1. Newell, A. J., Williams, W., & Dunlop, D. J. (1993). _A generalization of the
   demagnetizing tensor for nonuniform magnetization._ Journal of Geophysical
   Research: Solid Earth, 98(B6), 9551–9555.
   [doi:10.1029/93JB00694](https://doi.org/10.1029/93JB00694)

2. Chadebec, O., Coulomb, J.-L., & Janet, F. (2006). _A review of magnetostatic
   moment method._ IEEE Transactions on Magnetics, 42(4), 515–520.
   [doi:10.1109/TMAG.2006.870929](https://doi.org/10.1109/TMAG.2006.870929)

3. Moskowitz, R., & Della Torre, E. (1966). _Theoretical aspects of
   demagnetization tensors._ IEEE Transactions on Magnetics, 2(4), 739–744.
   [doi:10.1109/TMAG.1966.1065973](https://doi.org/10.1109/TMAG.1966.1065973)

4. Saad, Y., & Schultz, M. H. (1986). _GMRES: a generalized minimal residual
   algorithm for solving nonsymmetric linear systems._ SIAM Journal on
   Scientific and Statistical Computing, 7(3), 856–869.
   [doi:10.1137/0907058](https://doi.org/10.1137/0907058)
