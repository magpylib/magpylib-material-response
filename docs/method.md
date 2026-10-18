# Method of Moments for Demagnetization

This page describes the physics and numerical method behind `apply_demag`, from
the material model to the linear system and its solution.

---

## 1. Physical model

A _soft magnetic_ material responds to an applied magnetic field by acquiring a
magnetization proportional to the local total field:

$$\mathbf{M} = \chi \, \mathbf{H}_\text{total}$$

where $\chi$ is the magnetic susceptibility — an isotropic scalar, or a 3-vector
$(\chi_x, \chi_y, \chi_z)$ along the **cell's own axes**, which turn with its
orientation like its polarization: a cell rotated by $R$ responds with the
tensor $R\,\operatorname{diag}(\chi_x, \chi_y, \chi_z)\,R^\mathsf{T}$ in the
global frame. In magpylib's convention, the primary field quantity is the
**intrinsic polarization** $\mathbf{J} = \mu_0 \mathbf{M}$ (units of Tesla), so
the constitutive relation becomes

$$\mathbf{J} = \mu_0 \chi \, \mathbf{H}_\text{total} = \chi \, \mathbf{B}_\text{total}$$

The total field seen by a cell has three contributions:

| Contribution          | Symbol                    | Origin                                                  |
| --------------------- | ------------------------- | ------------------------------------------------------- |
| Remanent polarization | $\mathbf{J}^{(0)}$        | Hard-magnet-like "frozen" moment                        |
| External applied flux | $\mathbf{B}_\text{ext}$   | User-specified `H_ext` on the object                    |
| Demagnetizing flux    | $\mathbf{B}_\text{demag}$ | The cell's own magnetization (and neighbors') acts back |

Current sources (`magpylib.current.*`) in the collection contribute an
additional applied-field term: their field at the cell centroids is added to the
right-hand side just like $\mathbf{B}_\text{ext}$.

Self-consistency requires that the polarization **satisfies this equation
simultaneously for every cell**:

$$\mathbf{J}_i = \mathbf{J}_i^{(0)} + \chi_i \bigl( \mathbf{B}_{\text{ext},i} + \mathbf{B}_{\text{demag},i} \bigr)$$

---

## 2. Method of Moments — general formulation

### Discretization

The magnetic body is divided into $N$ cells (cuboids, cylinder segments,
tetrahedra), each assumed to carry a **uniform polarization** $\mathbf{J}_i$
(Method of Moments / volume-element approach). Under this assumption the
demagnetizing flux at cell $i$ due to cell $j$ is linear in $\mathbf{J}_j$:

$$\mathbf{B}_{\text{demag},i} = \sum_{j=1}^{N} \mathsf{T}_{ij} \, \mathbf{J}_j$$

where $\mathsf{T}_{ij}$ is a **3×3 demagnetization tensor block**
(dimensionless): the field that a unit polarization in cell $j$ produces at cell
$i$, taken at the centroid of cell $i$ (section 3).

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
- $\mathbf{S}$ — susceptibility matrix, block-diagonal: cell $i$ contributes its
  3×3 tensor $R_i \operatorname{diag}(\boldsymbol{\chi}_i) R_i^\mathsf{T}$
  (diagonal when the susceptibility is isotropic or the cell unrotated)
- $\mathbf{T} \in \mathbb{R}^{3N \times 3N}$ — full demag tensor matrix
  ($\mathsf{T}_{ij}$ blocks on rows $i$, columns $j$)

Solving this linear system gives the self-consistent polarization of every cell,
accounting for all mutual demagnetization interactions.

---

## 3. The interaction tensor

### Point matching

Each block $\mathsf{T}_{ij}$ is the flux density that cell $j$, carrying a unit
polarization along $x$, $y$ or $z$, produces at the **centroid** of cell $i$
(point matching, Chadebec 2006). `apply_demag` evaluates it with magpylib's
closed-form field of the cell type (`magpy.getH`), for all cells at all
centroids at once, once per unit polarization. Any magnet class magpylib can
compute a field for can be a cell.

`demag_tensor` returns this field with shape $(3, N, N, 3)$: polarization
component, source cell, observer cell, field component, in A/m per tesla.
`apply_demag` multiplies it by $\mu_0$ and flattens it into the $(3N \times 3N)$
matrix $\mathbf{T}$ in the component-major layout of $\mathbf{J}$, so that
$\mathbf{T}$ maps polarizations to demagnetizing flux density, both in tesla.

### Self-demagnetization

For the **diagonal block** $i = j$, the field of a cell at its own centroid
gives its demagnetization factors at that point. They sum to 1 for any cell
shape, because the trace of the demagnetizing tensor is 1 at every point inside
a uniformly polarized body:

$$N_{xx}^{(\text{self})} + N_{yy}^{(\text{self})} + N_{zz}^{(\text{self})} = 1$$

For a cube, $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$. The centroid must lie
inside the cell: `mesh_Cylinder` cuts thin rings into enough sectors that it
does.

---

## 4. Solving the system

1. Assemble the full $3N \times 3N$ matrix
   $\mathbf{Q} = \mathbf{I} - \mathbf{S}\,\mathbf{T}$.
2. Build the right-hand side
   $\mathbf{J}^{(0)} + \mathbf{S}\,\mathbf{B}_\text{ext}$, where
   $\mathbf{B}_\text{ext}$ holds `H_ext` and the field of the current sources at
   the cell centroids.
3. Call `numpy.linalg.solve(Q, rhs)` and write each cell's polarization back,
   expressed in the cell's own frame.

The solve is exact to floating-point precision. Memory grows with $N^2$ — the
matrix alone takes $(3N)^2 \times 8$ bytes, 650 MB for 3,000 cells — and time
with up to $N^3$. On an Apple Silicon Mac, 1,744 tetrahedra take about 16 s and
7 GB.

### Legacy options

Three options of `apply_demag` trade generality for speed or memory on meshes of
cuboid cells:

| Option                | Effect                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `pairs_matching=True` | Cell pairs with the same offset, orientations and source dimensions are evaluated once; pays off on regular grids    |
| `max_dist=d`          | Pairs whose centroids lie $d$ or more times the larger cell side apart are dropped, an approximation                 |
| `split=k`             | The field evaluations run in $k$ chunks of source cells, bounding the memory of the field evaluation (any cell type) |

`pairs_matching` and `max_dist` take cuboid cells only. `max_dist` does not
combine with the other two, and `pairs_matching` does not combine with `split`.

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
├─ 4. rhs = J⁽⁰⁾ + S (H_ext + current sources' field at the centroids)
│
├─ 5. numpy.linalg.solve(Q, rhs)                              O(N³)
│
└─ 6. Write the self-consistent polarizations back to the magnet cells
```

---

## 6. References

1. Chadebec, O., Coulomb, J.-L., & Janet, F. (2006). _A review of magnetostatic
   moment method._ IEEE Transactions on Magnetics, 42(4), 515–520.
