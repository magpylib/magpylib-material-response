# Inside saturation

A `BHCurve` makes the problem non-linear: a cell's susceptibility depends on the
field inside it, which depends on every cell's polarization. `solve_cells` and
`solve_fem` both solve it as a sequence of linear solves. Code references are to
`src/magpylib_material_response/_saturation.py`.

## The curve

`BHCurve(H, B)` takes the measured curve, H in A/m and B in T, both from 0. The
polarization is $J = B - \mu_0 H$. Between the points B follows a monotone cubic
(PCHIP), so the curve never wiggles between measurements; past the last point J
keeps its last value, and B rises with the slope of vacuum. A curve whose J
passes 5 T warns: its columns are swapped or its units wrong.

## The fixed point

Each cell on a curve takes the **secant** susceptibility at its own field,

$$
\chi = \frac{J(H)}{\mu_0 H},
$$

the line from the origin to its point on the curve. That is the way Radia treats
non-linear iron {cite}`chubar1998`. Then:

1. Solve the linear problem with every cell's current $\chi$.
2. Read the field $\mu_0 H$ in each cell, from the interaction operator.
3. Move each $\chi$ to the curve's secant at that field.
4. Stop when no cell's $\chi$ changes by more than 1e-4, relative.

The iteration runs on $\log \chi$, which spans many decades across a part that
saturates in places. Plain substitution converges slowly near the knee, so each
step is accelerated by Anderson's method {cite}`anderson1965,walker2011`, with a
memory of the last 6 steps and a damping of 0.7. That cuts the solves 3 to 4
times: 3 to 40 solves, more on a sharp knee, at most 200. Each step is clipped
to the susceptibilities the curve can give, as Anderson's steps can overshoot.

## What it costs

In `solve_cells` the interaction matrix does not depend on $\chi$, so it is
built once; each step is one more linear solve. With the direct solver the
system matrix $\mathbf{I} - \mathbf{S}\mathbf{T}$ changes with $\chi$ and is
factored again each time.

In `solve_fem` the susceptibility is a field on each element, its logarithm
linear across it, and each step assembles the matrix again. The iteration stops
when $\chi$ on the worst element moves by less than 1e-3: 5 to 40 solves, at
most 70.

`solve_face_charge` refuses a curve: a part that saturates is no longer uniform
inside, and has charge in its volume. A sheet's `mean_polarization` tells
whether that matters: while it stays below the knee of the curve, the linear
solve holds.

## How well it does

A sphere in a uniform field stays uniformly polarized while it saturates, so its
answer is exact: $\mu_0 H$ inside is $B_0 - J/3$, with $J$ on the curve at that
field. `solve_cells` is within 1 % of it along the whole curve
([Saturation](../examples/saturation.md)).
