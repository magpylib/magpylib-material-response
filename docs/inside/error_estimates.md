# Inside the error estimates

A solve gives an answer, not its error. Face charge reads an indicator of its
error off the solve, the jump of its charge between panels. All three solvers
estimate their error from several solves: `study_cells` over meshes,
`study_face_charge` over panel levels, `study_fem` over orders. How does each
estimate work, when does it hold, and when do two estimates agree?
[How accurate is a result?](../guide/accuracy.md) shows how to call them.

## `study_cells`: the grid convergence index

Cells converge as a power of the cell size $h$ once the mesh resolves the parts'
features:

$$
f(h) \approx f_0 + C h^p .
$$

Three meshes, $h_1 < h_2 < h_3$, give three values and two changes. Their ratio
gives the observed order $p$, and Richardson extrapolation
{cite}`richardson1911` the value $f_0$ at $h = 0$. The grid convergence index
{cite}`roache1994`, as {cite:t}`celik2008` set it out, takes the distance of the
finest value to $f_0$, relative to the finest value, times a factor of safety of
1.25, as its error.

What `study_cells` does differently from Celik et al.:

- **One order for the whole quantity**, from the sizes of the last two changes
  of all its entries together, instead of an average of each entry's order.
  Single sensors don't converge monotonically; the whole field does.
- **No estimate unless the changes are steady**: the order must lie between 0.3
  and 3, and entries moving against the trend may carry at most 10 % of the last
  change. Otherwise `order` is None and the errors NaN, which says: refine once
  more.
- **A warning when the meshes are too close**: when the cell count grows by less
  than 2.2 times from one mesh to the next (cells less than 1.3 times smaller),
  the least Celik et al. recommend. Meshes 4 to 8 times apart work best.

Each entry's error is its own last change, extrapolated with that one order,
relative to its own finest value, times 1.25. The order is fitted in the count
of cells, which is what the meshes control; only the cells that respond are
counted, so a permanent magnet that is not meshed doesn't inflate it. The
[mesh convergence example](../examples/mesh_convergence.md) runs it on three
parts.

## `study_fem`: rising orders on one mesh

On a mesh refined geometrically toward the edges, the finite-element error falls
geometrically with the polynomial order {cite}`babuska1996`, until the finest
layer limits it. The error of an order is then about its difference to the next.
`study_fem` takes the last change, times the same factor of 1.25, as the error
of the highest order, per entry and in all. With two orders, or when the last
change is not smaller than the one before, the error is NaN, with a warning.

It extrapolates nothing: the rate slows as the layers start to limit, so an
extrapolation would claim too much. On a cube at orders 3, 4 and 5, the last
ratio of changes, 0.09, would put the remaining error of the moment at 5e-6,
where it is 2e-5. The last change itself, 1.25 times over, stays above the
remaining error while the error still falls. The
[finite-element check](../examples/fem_check.md) runs it.

## `study_face_charge`: rising panel levels

Each level adds a layer of panels graded toward the faces' edges, so the error
falls faster with the level than a power of the panel size that the grid
convergence index could fit: it declines every face-charge series.
`study_face_charge` therefore takes the rule of `study_fem`, the last change
times 1.25 as the error of the highest level. On the 18 validation cases face
charge solves, at levels 1–3, 2–4 and 3–5, that bar stayed above the error over
all sensors wherever the finite-element reference was accurate enough to tell.

A sphere is the exception. Its facets are cut anew at each level, not split, so
its error does not fall steadily: against the exact field of a soft sphere the
bar fell short by up to 2.6 times at levels 3–5. The study warns when the model
holds a sphere.

## `compare`: two estimates against each other

`compare(a, b)` takes the last value of each study, the one its error is for,
and both error bars per entry. Two values agree when they differ by no more than
the sum of the two bars. A disagreement means at least one estimate is too
small, or one method has a fault the other has not: the two methods share only
the model.

## Face charge's `jump_indicator`

On a flat face the true charge is continuous, while the panels' linear charges
jump from one panel to the next. The largest jump, weighted by panel size, comes
with the solve. It is an indicator, not an error bar: for a plate under a magnet
the worst error of B 0.25 mm above the plate is 1 to 3 times it.
[How accurate is a result?](../guide/accuracy.md) shows how to use it with
`level` and `refine`.

## In short

Face charge gives an indicator with every solve; all three solvers give an error
bar from three solves or more. The grid convergence index extrapolates over
meshes and needs steady changes; the order and level studies extrapolate nothing
and overestimate while the error falls. Two error bars that do not overlap mean
one of them is too small.

Code: `study_cells` in `src/magpylib_material_response/convergence.py`,
`study_face_charge` in `face_charge/_study.py`, `study_fem` in `fem/_study.py`.
