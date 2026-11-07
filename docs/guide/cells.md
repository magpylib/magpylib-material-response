---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Cells

`solve_cells` cuts each part into cells, small magnets of one uniform
polarization each, and finds the polarization of every cell at once. It is the
solver for what face charge does not do: saturation on a B–H curve, a
susceptibility that differs along a part's axes or varies inside it, and the
field inside a part. It takes three steps, mesh, solve and read, and this page
answers the question each raises: which mesher, which solver settings, how many
cells, and where to read.

```{code-cell} ipython3
import magpylib as magpy

from magpylib_material_response import UniformField, solve_cells
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3
magnet = magpy.magnet.Cuboid(dimension=(1 * MM,) * 3, polarization=(0, 0, 1))
magnet.susceptibility = 0.3  # μr = 1.3: a weak magnet demagnetizes itself

meshed = mesh_uniform(magnet, 5)  # 5 cells across, 125 cells
solved = solve_cells(meshed)

observer = (0, 0, 1.5 * MM)
print(f"Bz without response: {magpy.getB(magnet, observer)[2] * 1e3:.2f} mT")
print(f"Bz with response   : {magpy.getB(solved, observer)[2] * 1e3:.2f} mT")
```

The field drops by about 10 %: the magnet's own field opposes its polarization.

## Mesh the parts

| Mesher                      | Cells                                                            | Use it for                               |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| `mesh_uniform(model, n)`    | one size for the whole model, `n` across the thinnest part       | most models: the fastest solve           |
| `mesh_Cuboid(cuboid, n)`    | a grid of cuboids                                                | one cuboid at its own resolution         |
| `mesh_Cylinder(cyl, n)`     | a core and rings of cylinder segments                            | cylinders, rings and tubes               |
| `mesh_TriangularMesh(m, n)` | tetrahedra, through TetGen {cite}`si2015` (the `[tetgen]` extra) | any closed surface, such as a CAD export |
| `voxelize(magnet, n)`       | cubes on one lattice, holding the part's volume                  | any shape, on the fast grid path         |
| `mesh_all(model, n)`        | each part with its own mesher, `n` cells shared by volume        | mixed models                             |

`mesh_uniform` cuts every part that responds into cells of one size, and leaves
fixed magnets and currents as they are: a uniformly polarized magnet's field
does not change when it is meshed. Boxes whose sides are multiples of that size
become regular grids, the fastest case for the solver.

Prefer cuboid grids where the parts are boxes. Tetrahedra follow any shape and
reach a given error with fewer cells, but a grid of identical cuboids solves far
faster per cell ([Inside cells](../inside/cells.md) shows why).

## Solve

`solve_cells` takes no setting for most models. The ones there are:

| Argument         | Default        | What it does                                                                                                                       |
| ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `solver`         | `"auto"`       | `"direct"` solves the full matrix; `"iterative"` never stores it, and is faster on grids; `"auto"` picks                           |
| `solver_tol`     | `1e-6`         | the iterative solver's tolerance, which `"auto"` tightens a hundredfold; under `"iterative"`, tighten it for a high susceptibility |
| `interaction`    | `"point"`      | how a cell's field is taken at another: at its centre, or `"newell"`, averaged over it (cuboids of one orientation)                |
| `susceptibility` | from the parts | one value for all cells, or one per cell, overriding the parts' attributes                                                         |
| `inplace`        | `False`        | write the polarizations into the given cells instead of a copy                                                                     |

[The cell equations](../inside/cells_equations.md) set out the two interaction
rules.

`"auto"` takes the iterative solver when FFT products cover at least a tenth of
the interaction matrix, as on the grids `mesh_uniform` makes of boxes, and the
direct solver otherwise, which is then as fast or faster. Its log says which it
took ([Logging](logging.md)). On one grid of cubes, the iterative solver is as
fast as the direct one at 216 cells and 4.7 times faster at 1,728, where it
needs 17 MB against 433 MB; it solves 8,000 cells in 0.5 s. Off a grid, as with
tetrahedra, memory limits the direct solver: its matrix takes 72 n² bytes for n
cells, and the solve peaks near twice that, 433 MB at 1,728 cells. If GMRES does
not converge under `"auto"`, the direct solver takes over when its matrix fits
in 2 GB, about 5,000 cells; above that the solve raises. Leave `solver="auto"`:
[the direct and the iterative solver, measured](../inside/cells_performance.md)
has the data.

## Saturation

A `BHCurve` as `susceptibility` puts every cell on the curve at its own field.
The linear solve repeats until no cell's susceptibility changes, 3 to 40 solves
on one matrix. [Saturation](saturation.md) shows how to enter a curve and check
the result.

## How many cells

The answer converges as the cells shrink, slowly: halving the cell size takes
eight times the cells and cuts the error to about a third. Soft iron is the hard
case, because its polarization gathers at edges and corners. Start coarse,
multiply the cells by four to eight, and stop when the quantity you need, a
field at a sensor or a force, changes by less than you can accept. `study_cells`
does that for you and estimates the error of the finest mesh:
[How accurate is a result?](accuracy.md).

## Read the result

Touching cells each count a point on their shared surface as inside, so the
field there is wrong, with no warning: J counts 2, 4 or 8 times, and B can be
far off. The centre of a part with an even number of cells across is such a
point, as is any point on a symmetry plane through it. A 10 mm cube with χ = 10
in 0.1 T, 6 cells across, reads B = 0.107 T at its centre, which is wrong, and
0.29 T 1 µm off it along each axis. Use an odd number of cells across, or read a
little off the boundaries.

## In short

Mesh boxes with `mesh_uniform`, other shapes with the mesher that fits them.
Leave the solver settings at their defaults. Refine until the quantity you need
stops changing, or let `study_cells` estimate its error. Read off the cell
boundaries.
