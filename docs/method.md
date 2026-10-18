# Method of Moments for Demagnetization

This page describes the physics and numerical method behind `apply_demag`, from
the material model to the linear system and its solution.

---

## 1. Physical model

Each cell's material is linear: its polarization $\mathbf{J} = \mu_0 \mathbf{M}$
(in tesla, the quantity magpylib uses for magnets) responds to the magnetic
field $\mathbf{H}$ at the cell as

$$\mathbf{J} = \mathbf{J}^{(0)} + \chi \, \mu_0 \mathbf{H}$$

where $\mathbf{J}^{(0)}$ is the remanent polarization (zero for a soft material)
and $\chi$ the susceptibility: a scalar, or a 3-vector
$(\chi_x, \chi_y, \chi_z)$ along the **cell's own axes**, which turn with its
orientation like its polarization. A cell rotated by $R$ responds with the
tensor $R\,\operatorname{diag}(\chi_x, \chi_y, \chi_z)\,R^\mathsf{T}$ in the
global frame.

The field at a cell has three parts, all handled as $\mu_0 \mathbf{H}$ in tesla:

| Part          | Symbol                        | Origin                                                       |
| ------------- | ----------------------------- | ------------------------------------------------------------ |
| Applied       | $\mu_0 \mathbf{H}_\text{ext}$ | The uniform `H_ext` of the cell or of a parent `Collection`  |
| Currents      | $\mu_0 \mathbf{H}_\text{cur}$ | The current sources (`magpylib.current.*`) in the collection |
| Demagnetizing | $\mu_0 \mathbf{H}_\text{d}$   | The polarization of all cells, the cell itself included      |

Inside a polarized cell this is not the flux density:
$\mathbf{B} = \mu_0 \mathbf{H} + \mathbf{J}$.

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
the centroid of cell $i$ per unit polarization of cell $j$ (section 3).

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

### Point matching

Each block $\mathsf{T}_{ij}$ is $\mu_0$ times the field $\mathbf{H}$ that cell
$j$, carrying a unit polarization along $x$, $y$ or $z$, produces at the
**centroid** of cell $i$ (point matching, Chadebec 2006). `apply_demag`
evaluates it with magpylib's closed-form field of each cell type (`magpy.getH`),
for all cells at all centroids, once per unit polarization.

`demag_tensor` returns these fields with shape $(3, N, N, 3)$: polarization
component, source cell, observer cell, field component, in A/m per tesla.
`apply_demag` multiplies them by $\mu_0$ and arranges them into the matrix
$\mathbf{T}$.

### Self-demagnetization

The diagonal block is $\mathsf{T}_{ii} = -\mathsf{N}_i$, the demagnetizing
tensor of cell $i$ at its centroid. Its trace is 1 at every point inside a
uniformly polarized body, whatever its shape:

$$N_{xx}^{(i)} + N_{yy}^{(i)} + N_{zz}^{(i)} = 1$$

For a cube, $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$. The centroid must lie
inside the cell: `mesh_Cylinder` cuts thin rings into enough sectors that it
does.

---

## 4. Solving the system

1. Assemble the full $3N \times 3N$ matrix
   $\mathbf{Q} = \mathbf{I} - \mathbf{S}\,\mathbf{T}$.
2. Build the right-hand side from the remanent polarizations, `H_ext`, and the
   field of the current sources at the cell centroids.
3. Call `numpy.linalg.solve(Q, rhs)` and write each cell's polarization back,
   expressed in the cell's own frame.

The solve is direct, exact up to floating-point rounding. Memory grows with
$N^2$ — the matrix alone takes $(3N)^2 \times 8$ bytes, 650 MB for 3,000 cells —
and time with up to $N^3$. On an Apple Silicon Mac, 1,744 tetrahedra take about
16 s and 7 GB.

### Legacy options

Three options of `apply_demag` trade generality for speed or memory:

| Option                | Effect                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `pairs_matching=True` | Cell pairs with the same offset, orientations and source dimensions are evaluated once; pays off on regular grids |
| `max_dist=d`          | Pairs whose centroids lie $d$ or more times the larger cell side apart are dropped, an approximation              |
| `split=k`             | The field evaluations run in $k$ chunks of source cells, bounding the memory of the field evaluation              |

`pairs_matching` and `max_dist` take cuboid cells only; `split` takes any cell.
`max_dist` does not combine with the other two, and `pairs_matching` does not
combine with `split`.

---

## 5. Algorithm flow

```text
apply_demag(collection, susceptibility=..., pairs_matching=..., max_dist=..., split=...)
│
├─ 1. Collect the magnet cells; one 3×3 susceptibility tensor per cell,
│      R diag(χ) Rᵀ; refuse objects with paths
│
├─ 2. demag_tensor: field of every cell at every centroid, per unit
│      polarization (all at once, in chunks with split, or for unique
│      pairs only with pairs_matching / near pairs only with max_dist)
│
├─ 3. T = μ0 · tensor, flattened to (3N × 3N); Q = I − S T
│
├─ 4. rhs = J⁽⁰⁾ + S μ0 (H_ext + H_cur), both taken at the centroids
│
├─ 5. numpy.linalg.solve(Q, rhs)                              O(N³)
│
└─ 6. Write the self-consistent polarizations back to the magnet cells
```

---

## 6. References

1. Chadebec, O., Coulomb, J.-L., & Janet, F. (2006). _A review of magnetostatic
   moment method._ IEEE Transactions on Magnetics, 42(4), 515–520.
