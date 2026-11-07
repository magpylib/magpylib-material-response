# How accurate is a result?

Every solver discretizes: panels, cells or elements. Its answer is as good as
that discretization. How large is the error, and how do you find out? One solve
does not tell you: face charge reports an indicator with each solve, and an
error estimate takes several solves, at finer settings or by another solver.
This page shows each solver's way, then how to set two against each other.

| Solver              | Check                                                                      | Cost                         |
| ------------------- | -------------------------------------------------------------------------- | ---------------------------- |
| `solve_face_charge` | `jump_indicator` on each sheet, `refine` where it is large; a second level | free; one more solve         |
| `solve_cells`       | `study_cells`: three or more meshes, Richardson extrapolation              | the solves of all the meshes |
| `solve_fem`         | `study_fem`: three or more orders on one mesh                              | the solves of all the orders |
| any two studies     | `compare(a, b)`: both error bars side by side, per sensor                  | the two studies              |

## Face charge: `level`, `jump_indicator`, `refine`

`level`, 3 by default, sets the panels in each graded layer toward a face's
edges. Fields far from the parts converge at low levels; fields within a panel's
size of a surface need higher ones.

Each solved sheet reports its `jump_indicator`: the largest jump of the charge
between neighbouring panels, weighted by panel size. The true charge is
continuous across a flat face, so a jump is discretization error, read off the
solve for free. It indicates the error; it does not bound it. For the
validation's magnet on a soft plate, the worst error of B 0.25 mm off the plate
is 0.3 to 1.2 times it from level 1 to level 6.

`refine=k` uses the same jumps: each of k rounds splits the panels where they
are largest and solves again. It helps where the error sits at those jumps, as
on a thin wall that shields: one round takes the field outside
[the magnet in a steel tube](../examples/shield_tube.md) from 5.6 % to 1.0 %. It
does not help where the charge is smooth: on the plate, B 0.25 mm above it 7 mm
from the magnet stays 2.0 % off after two rounds from level 3, while uniform
panels at level 6 (10,044 unknowns, 2 s) bring it within the finite-element
reference's own 0.4 %.

To check a result, solve it again at a higher level: where the field you read
changes less than you can accept, it has converged. The change is a guide, not a
bound. On the validation case of a magnet on a soft plate, 1.25 times the change
from level 3 to level 4 fell short of the level-4 error at 1 of the 16 sensors,
by a factor of 1.3. `study_face_charge` does that for you:

```python
from magpylib_material_response import study_face_charge

study = study_face_charge(model, sensors)  # levels 2, 3 and 4
print(f"{study.error:.2%}", study.errors)  # whole and per sensor
```

It takes 1.25 times the change from the second-highest level to the highest as
the error of the highest, as `study_fem` does with the order. On the validation
cases that face charge solves, that bar covered the error over all sensors
wherever the finite-element reference was accurate enough to tell; per sensor it
is indicative. A sphere's facets are cut anew at each level: there it fell short
by up to 2.6 times, and the study warns.

## Cells: `study_cells`

```python
from magpylib_material_response import study_cells

study = study_cells(model, sensors, levels=(4, 8, 16))
print(study.values[-1], study.error, study.errors)
```

`study_cells` solves the model on three or more meshes, fits how the result
changes from mesh to mesh, extrapolates it to an infinitely fine mesh, and
reports the distance from the finest mesh to that limit, times 1.25 for safety.
That is the grid convergence index {cite}`roache1994`, as {cite:t}`celik2008`
set it out, with Richardson extrapolation {cite}`richardson1911` over the three
finest meshes. It gives one error for the whole quantity, `error`, and one per
sensor, `errors`.

- **`levels`** are the cells across the thinnest part that responds, one mesh
  each, made by `mesh_uniform`. Doubling them gives 8 times the cells: choose
  meshes 4 to 8 times apart in cells. Closer ones change little without being
  converged; it warns below 2.2 times.
- **The quantity** is B at positions or sensors, or a function of the solved
  model, such as a force or a moment. A force converges more slowly than a field
  ([A holding magnet on a steel plate](../examples/holding_magnet.md)).
- **`mesher=`** replaces `mesh_uniform`: `mesher(model, level)` returns the
  meshed model, such as tetrahedra for a CAD part.
- **No estimate**, a NaN error with a warning, means the three finest results
  don't change steadily yet: add a finer mesh.
- **It measures the mesh's error**, not the model's: a linear susceptibility
  does not saturate, and a faceted surface stays faceted.

[Error bars for cells](../examples/mesh_convergence.md) runs it on a flux
concentrator, a Halbach array and a steel shield.

## Finite elements: `study_fem`

`study_fem(model, quantity)` solves at orders 2, 3 and 4 by default, on one
mesh, and takes 1.25 times the last change as the error of the highest order. If
the changes don't shrink, the error is NaN, with a warning. `quantity` is a set
of positions or sensors, or a function of the solution, such as
`lambda s: s.moment`. The error falls geometrically with the order on a mesh
refined toward the edges {cite}`babuska1996`. While it does, the last change
overstates the remaining error, so the estimate errs on the safe side.

`solve_fem` on its own uses order 3. With a `BHCurve`, order 2 can stop with
"did not settle": start the study at order 3, with `orders=(3, 4, 5)`.

## Two solvers against each other: `compare`

```python
from magpylib_material_response import compare, study_cells, study_fem

comparison = compare(study_cells(model, sensors, (4, 8, 16)), study_fem(model, sensors))
print(comparison.agree)  # per sensor: do the error bars overlap?
```

`compare` takes any two studies of the same quantity: from `study_cells`,
`study_face_charge`, `study_fem`, or `MeshStudy.from_values(cells, values)` for
results of another volume mesh. Two values agree when they are no farther apart
than their two error bars together. `agree` says it per sensor, `agreement`
gives the share of sensors that agree, and `relative_difference` the difference
over the whole quantity. Where they disagree, at least one bar is too short:
refine there, or take the finite-element value.
[Checking a result with finite elements](../examples/fem_check.md) shows a case
where the cells' bar is the short one.

## How the solvers are checked

The repository's
[`validation/`](https://github.com/magpylib/magpylib-material-response/tree/main/validation)
folder holds 23 cases: the models of these pages and others that are hard for
one solver or another, nearly all with a stored finite-element reference at
their sensors. Every pull request holds cells and face charge to their stored
accuracy, and every night all solvers climb their settings against references
made again.

## In short

One solve gives an indicator at best. For a design decision, check one
configuration with `study_face_charge`, `study_cells` or `study_fem`, or two
solvers with `compare`, and trust the cheaper solver for the sweep.
[Inside the error estimates](../inside/error_estimates.md) shows how each
estimate works and where it holds.
