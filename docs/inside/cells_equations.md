# The cell equations

`solve_cells` replaces each part by cells of uniform polarization and finds all
the polarizations at once. In equations, that raises four questions: what holds
in each cell, how the field of one cell at another is taken, how the system is
solved, and how close the two solvers come to each other. This page answers them
in that order. [Inside cells](cells.md) shows the same ideas with 3D views and
charts.

## What holds in each cell

Each cell's material is linear: its polarization $\mathbf{J} = \mu_0 \mathbf{M}$
(in tesla, as Magpylib takes it) responds to the field $\mathbf{H}$ at the cell
as

$$
\mathbf{J} = \mathbf{J}_r + \chi \, \mu_0 \mathbf{H}
$$

where $\mathbf{J}_r$ is the remanence (zero for a soft material) and $\chi$ the
susceptibility: a scalar, or a 3-vector $(\chi_x, \chi_y, \chi_z)$ along the
**cell's own axes**, which turn with its orientation like its polarization. A
cell rotated by $R$ responds with the tensor $R\,\operatorname{diag}(\chi_x,
\chi_y, \chi_z)\,R^\mathsf{T}$ in the global frame.

The field $\mu_0 \mathbf{H}$ at a cell has three parts, all in tesla:

| Part           | Symbol                      | Origin                                                       |
| -------------- | --------------------------- | ------------------------------------------------------------ |
| Applied        | $\mathbf{B}_\text{ext}$     | a `UniformField` in the collection                           |
| Currents       | $\mathbf{B}_\text{c}$       | the current sources (`magpylib.current.*`) in the collection |
| The cells' own | $\mu_0 \mathbf{H}_\text{d}$ | the polarization of every magnet, the cell itself included   |

Every magnet in the collection is a cell; one without a susceptibility keeps its
polarization and acts only as a source. Inside a polarized cell
$\mu_0\mathbf{H}$ is not the flux density: $\mathbf{B} = \mu_0 \mathbf{H} +
\mathbf{J}$.

Self-consistency requires, for every cell $i$ at once,

$$
\mathbf{J}_i = \mathbf{J}_{r,i} + \chi_i \bigl( \mathbf{B}_{\text{ext}} + \mathbf{B}_{\text{c},i} + \mu_0\mathbf{H}_{\text{d},i} \bigr)
$$

## One linear system

With a uniform polarization $\mathbf{J}_j$ in each of the $N$ cells (cuboids,
cylinder segments, tetrahedra), the cells' own field is linear in the
polarizations:

$$
\mu_0 \mathbf{H}_{\text{d},i} = \sum_{j=1}^{N} \mathsf{T}_{ij} \, \mathbf{J}_j
$$

The dimensionless 3×3 block $\mathsf{T}_{ij}$ gives $\mu_0 \mathbf{H}$ at cell
$i$ per unit polarization of cell $j$. Stacking the polarizations of all cells
into one $3N$-vector gives

$$
\boxed{(\mathbf{I} - \mathbf{S}\,\mathbf{T})\,\mathbf{J} =
         \mathbf{J}_r + \mathbf{S}\,(\mathbf{B}_\text{ext} + \mathbf{B}_\text{c})}
$$

with $\mathbf{S}$ the susceptibilities, block-diagonal, one 3×3 tensor per cell,
and $\mathbf{T}$ the interaction matrix of the blocks $\mathsf{T}_{ij}$. This is
the magnetostatic method of moments {cite}`chadebec2006`.

## The field of one cell at another

A cell's field is known exactly; the method only has to decide _where_ in the
observer cell to take it. `solve_cells` offers two answers, chosen per solve
with `interaction` and applied to **every pair**:

| `interaction`       | The field of cell $j$ at cell $i$   | Works for                              |
| ------------------- | ----------------------------------- | -------------------------------------- |
| `"point"` (default) | at the centroid of cell $i$ ¹       | any cell type and orientation          |
| `"newell"`          | averaged over the whole of cell $i$ | cuboid cells, all with one orientation |

¹ Close cylinder segments with χ ≥ 1 are averaged, as set out below.

**Volume averaging** is the Galerkin form of the problem. For a soft body with a
positive susceptibility in a uniform applied field, the induced moment along the
field is then a **lower bound**: it never exceeds the converged value and rises
each time every cell is split into smaller cells, so a convergence study knows
which side the answer lies on. Point matching gives no such guarantee.

At equal cell count neither rule is more accurate in general. On cubic cells
they agree to fourth order in cell size over distance. On a soft cube of 1,728
cells, point matching is closer on the field above it (0.28 % against 0.33 %)
and volume averaging on its moment (1.1 % against 1.2 %), as the
[moment study in Inside cells](cells.md) shows. Both converge to the same
answer.

One rule holds for every pair of a solve. A mix would switch a pair's rule when
one of two touching bodies turns by any angle, however small: the result would
jump (0.26 % for a magnet on an iron block turned by a millionth of a degree),
and the lower bound would be lost. `interaction="newell"` therefore raises when
the cells are not all cuboids with one orientation.

### At the centroid (point matching)

Each block $\mathsf{T}_{ij}$ is $\mu_0$ times the field $\mathbf{H}$ that cell
$j$, carrying a unit polarization along $x$, $y$ or $z$, produces at the
**centroid** of cell $i$ {cite}`chadebec2006`, from Magpylib's closed-form field
of the cell type.

Between close cylinder segments the centroid value can make the equations
unsolvable at a high χ: curved cells give $\mathsf{T}$ positive eigenvalues,
which no demagnetizing field has. A tube with a 1 mm wall at χ = 1000 came out
−68 % off its moment with two cells across the wall. Those pairs are therefore
averaged over cell $i$ by Gauss quadrature, which holds the tube at −11 %, an
error that falls as more cells go around the ring
([#92](https://github.com/magpylib/magpylib-material-response/issues/92)). Pairs
with both susceptibilities below 1, or farther apart than twice the sum of their
half-diagonals, keep the centroid.

### Averaged over the cell (Newell)

For **identical, axis-aligned** rectangular prisms of sides $(a, b, c)$, the
field averaged over the observer cell has a closed form {cite}`newell1993`. The
six independent components follow from two auxiliary functions $f(x,y,z)$ and
$g(x,y,z)$ through a 27-point second difference:

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
s)^\mathsf{T}$, and is checked against Gauss–Legendre volume averages of
Magpylib's cuboid field.

### Far apart

Far apart, the exact formulas lose precision: Newell's differences cancel nearly
equal terms, and Magpylib's closed form does too beyond 50 source sides. There
both rules use the far-field expansion of the tensor, the point dipole plus its
leading cell-size correction,

$$
\mathsf{N}(\mathbf{r}) \approx -\frac{V_\text{src}}{4\pi}
  \Bigl[\nabla\nabla \tfrac{1}{r}
  + \sum_k \tfrac{a_k^2 + b_k^2}{24}\, \partial_k^2 \nabla\nabla \tfrac{1}{r}\Bigr],
$$

accurate to $(h_\text{max}/r)^4$, with $a$ and $b$ the source and observer sides
($b = 0$ for point matching). Volume averaging switches where the two errors
meet, where the error is $3 \times 10^{-7}$ for cubes and $2 \times 10^{-5}$ for
10:1 rods.

### A cell on itself

The diagonal block is $\mathsf{T}_{ii} = -\mathsf{N}_i$, the demagnetizing
tensor of cell $i$: at its centroid under point matching, averaged over it (the
magnetometric factors) under volume averaging. Either way its trace is 1. The
tensor at a point is $\mathsf{N} = -\nabla\nabla\psi$ with $\psi(\mathbf{r}) =
\frac{1}{4\pi}\int_V \frac{\mathrm{d}V'}{|\mathbf{r} - \mathbf{r}'|}$, and
Poisson's equation gives $\nabla^2\psi = -1$ inside the cell; the average over
the cell keeps the trace {cite}`moskowitz1966`:

$$
N_{xx}^{(i)} + N_{yy}^{(i)} + N_{zz}^{(i)} = 1
$$

For a cube both rules give $N_{xx} = N_{yy} = N_{zz} = \tfrac{1}{3}$. Under
point matching the centroid must lie inside the cell: `mesh_Cylinder` cuts thin
rings into enough sectors that it does.

## Solving the system

### Direct

`solver="direct"` assembles the full $3N \times 3N$ matrix $\mathbf{Q} =
\mathbf{I} - \mathbf{S}\,\mathbf{T}$, builds the right-hand side from the
remanence and the applied field at the cell centroids, and factors $\mathbf{Q}$
with LAPACK {cite}`lapack1999`. The answer is exact up to floating-point
rounding. Memory grows with $N^2$, the matrix alone taking $(3N)^2 \times 8$
bytes, 650 MB for 3,000 cells, and time with up to $N^3$.

### Iterative

`solver="iterative"` runs GMRES {cite}`saad1986`, as SciPy implements it
{cite}`scipy2020`. It never assembles $\mathbf{Q}$: it only needs the product
$\mathbf{v} \mapsto \mathbf{Q}\,\mathbf{v}$, with the same entries as the direct
solver, so both solve the same system.

GMRES stops when the residual falls below `solver_tol` relative to the
right-hand side, which bounds the residual, not the error. Against the direct
solve, the polarizations agree within a few times `solver_tol` for hard magnets:
0.03 to 3.4 times on four models at χ 0.05 to 0.2. For soft iron at χ = 1000 the
gap grows with the cell count, to 160 × `solver_tol` at 4,096 cells and 580 × at
32,768: tighten `solver_tol` for high susceptibilities. GMRES that does not
converge within `max_iter` raises `RuntimeError` rather than return a partial
result.

The preconditioner is block-Jacobi, one 3×3 block per cell:

$$
\mathbf{M}_i = \mathbf{I} + \boldsymbol{\chi}_i \, \mathsf{N}_i
$$

with $\boldsymbol{\chi}_i$ the cell's susceptibility tensor and $\mathsf{N}_i$
its self-demagnetizing tensor, in the global frame. Cells other than cuboids
take $\mathsf{N}_i = \mathbf{I}/3$, the isotropic part of their own block: a
grid with 902 tetrahedra beside it at χ = 1000 converges in 69 iterations this
way, and not in 1,000 without a preconditioner on them.

### FFT products on a grid

When cells share one shape and orientation and their centres fill a regular grid
of shape $(N_x, N_y, N_z)$, the interaction depends only on the offset between
cells, $\mathsf{T}_{ij} = \mathsf{T}(\mathbf{r}_j - \mathbf{r}_i)$, and the
product $\mathbf{T}\,\mathbf{J}$ is a 3-D discrete convolution: an FFT on the
grid zero-padded to twice its size computes it in $O(N \log N)$ time
{cite}`hayashi1996`. A body on one lattice that does not fill its box, such as a
voxelized sphere, runs the same way when its cells fill at least an eighth of
the box and it is at least four cells deep along every axis: the empty places
hold zero polarization. Large blocks between two grids of one spacing are
convolutions too.

The iterative solver applies every other block as a stored matrix. A large one
may be stored sparse: its smallest entries are dropped only while the operator
changes by at most a tenth of the tolerance,
$\|\mathbf{S}(\mathbf{T}-\tilde{\mathbf{T}})\|_\infty \le 0.1\,
\varepsilon_\text{tol}$, with $\varepsilon_\text{tol}$ the `solver_tol`. Many
separate bodies, cells of many sizes or cells off any grid leave little to the
FFT, and then the iterative solver is no faster than the direct one.

### Which one runs

`solver="auto"` takes the iterative solver when FFT products cover enough of the
interaction matrix, and runs it to `solver_tol / 100`, so its answer stays
within about `solver_tol` of the direct solve on a few thousand cells and within
10 × `solver_tol` on 32,768. [Cells](../guide/cells.md) gives the rule, and
[Cells: direct and iterative solvers, measured](cells_performance.md) the
measurements behind it.

## In short

<!-- prettier-ignore -->
Each cell obeys $\mathbf{J} = \mathbf{J}_r + \chi\,\mu_0\mathbf{H}$, and the
cells' own field is linear in their polarizations, so one linear system
$(\mathbf{I} - \mathbf{S}\mathbf{T})\mathbf{J} = \mathbf{J}_r +
\mathbf{S}(\mathbf{B}_\text{ext} + \mathbf{B}_\text{c})$ holds the whole model.
The field of one cell at another is taken at the centroid or averaged over the
cell; the average gives a lower bound on a soft part's moment. The direct solver
factors the matrix exactly; the iterative one applies it, by FFT on grids, to
within its tolerance.

Code: `src/magpylib_material_response/cells.py`, `newell.py` and `demag_fft.py`.
