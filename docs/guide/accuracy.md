# How accurate is a result?

Every solver discretizes: panels, cells or elements. Its answer is as good as
that discretization, and none of them knows its own error from a single solve.
Each one has a way to estimate it, and they can check each other.

| Solver              | Estimate                                                      | Cost                         |
| ------------------- | ------------------------------------------------------------- | ---------------------------- |
| `solve_face_charge` | `jump_indicator` on each sheet; `refine` where it is large    | free with the solve          |
| `solve_cells`       | `study_cells`: three or more meshes, Richardson extrapolation | the solves of all the meshes |
| `solve_fem`         | `study_fem`: three or more orders on one mesh                 | the solves of all the orders |
| any two             | `compare(a, b)`: both error bars side by side, per sensor     | the two studies              |

## Face charge: `level`, `jump_indicator`, `refine`

`level` sets the panels in each graded layer toward a face's edges. Fields far
from the parts converge at low levels; fields within a panel's size of a surface
need higher ones.

Each solved sheet reports its `jump_indicator`: the largest jump of the charge
between neighbouring panels, weighted by panel size. The true charge is
continuous across a flat face, so the jump is the discretization error, read off
the solve for free. For a plate under a magnet, the worst error of B 0.25 mm
above the plate is 1 to 3 times it.

`refine=k` uses the same jumps: each of k rounds splits the panels where they
are largest and solves again. For the plate, `refine=2` from level 3 (2,688
unknowns, under 2 s) puts B 0.25 mm above the plate within 0.04 %, where uniform
panels need 10,350 unknowns and 20 s for 0.13 %.

To check a face-charge result fully, solve it at two levels: where the field you
read changes less than you can accept, it has converged.

## Cells: `study_cells`

```python
from magpylib_material_response import study_cells

study = study_cells(model, sensors, levels=(4, 8, 16))
print(study.values[-1], study.error, study.errors)
```

`study_cells` meshes the model with `mesh_uniform` at each level, cells across
the thinnest part that responds, solves each mesh, and estimates the error of
the finest from how the results change between them. The estimate is the grid
convergence index {cite}`roache1994`, as {cite:t}`celik2008` set it out:
Richardson extrapolation {cite}`richardson1911` over the three finest meshes,
times a factor of safety of 1.25. It gives one error for the whole quantity and
one per sensor, and none while the results do not yet change steadily.

- **Choose meshes 4 to 8 times apart** in cells. Closer ones change little
  without being converged; it warns below 2.2 times.
- **No estimate** means the three finest results don't change steadily yet:
  refine once more.
- **It measures the mesh's error**, not the model's: a linear susceptibility
  does not saturate, and a faceted surface stays faceted.

[Mesh convergence](../examples/mesh_convergence.md) runs it on a flux
concentrator, a Halbach array and a steel shield.

## Finite elements: `study_fem`

`study_fem(model, sensors)` raises the order, 2, 3 and 4 by default, on one
mesh, and takes 1.25 times the last change as the error of the highest. The
error falls geometrically with the order on a mesh refined toward the edges
{cite}`babuska1996`, so the estimate is an overestimate while it still falls.

## Two solvers against each other: `compare`

```python
from magpylib_material_response import compare, study_cells, study_fem

comparison = compare(study_cells(model, sensors, (2, 4, 8)), study_fem(model, sensors))
print(comparison.agree)  # per sensor: do the error bars overlap?
```

Two values agree when they are no farther apart than their two error bars
together. Where they disagree, at least one bar is too short: refine there, or
take the finite-element value.
[Checking a result with finite elements](../examples/fem_check.md) shows a case
where the cells' bar is the short one.

## The validation suite

The repository's `validation/` folder holds 23 cases: the models of these pages
and others that are hard for one method or another, nearly all with a stored
finite-element reference at their sensors. On every pull request a test holds
cells and face charge to their stored accuracy, and every night all methods
climb their settings against references made again, so that a change in any
method shows. From a clone of the repository:

```bash
python -m validation.run --cases plate --methods magpylib MoM FEM  # error against time
python -m validation.references --check plate                      # make a reference again
```
