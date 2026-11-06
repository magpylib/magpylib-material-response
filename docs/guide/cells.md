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
field inside a part. It takes three steps: mesh, solve, read.

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

The field drops by about 10 %: the magnet's own field opposes its polarization.

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
does not change when it is meshed. Cuboids whose sides are multiples of that
size become grids of one spacing, which the solver multiplies by FFT, within a
part and between parts. A mesh is a `Collection` with `sealed = True`, which
magpylib-studio shows as one object; its cells stay ordinary magnets.

Prefer cuboid grids where the parts are boxes. Tetrahedra follow any shape and
reach a given error with fewer cells, but a grid of identical cuboids solves far
faster per cell ([Inside cells](../inside/cells.md) shows why).

## Solve

`solve_cells` takes no setting for most models. The ones there are:

| Argument         | Default        | What it does                                                                                                        |
| ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------- |
| `solver`         | `"auto"`       | `"direct"` builds and factors the whole matrix, `"iterative"` runs GMRES with FFT products; `"auto"` picks          |
| `solver_tol`     | `1e-6`         | GMRES's tolerance on the residual; tighten it for a high susceptibility                                             |
| `interaction`    | `"point"`      | how a cell's field is taken at another: at its centre, or `"newell"`, averaged over it (cuboids of one orientation) |
| `susceptibility` | from the parts | one value for all cells, or one per cell, overriding the parts' attributes                                          |
| `inplace`        | `False`        | write the polarizations into the given cells instead of a copy                                                      |

The direct solver needs 72 n² bytes for n cells, 1 GB at about 3,700 cells.
`"auto"` takes the iterative solver when FFT products cover at least a tenth of
the matrix, as on `mesh_uniform` grids, which then solves 27,000 cells in
seconds.

## Saturation

A `BHCurve` as `susceptibility` puts every cell on the curve at its own field.
The linear solve repeats until no cell's susceptibility changes, 3 to 40 solves
with Anderson acceleration.

```{code-cell} ipython3
import numpy as np

from magpylib_material_response import BHCurve

H = np.r_[0.0, np.geomspace(1e-2, 1e6, 200)]  # A/m
mu0H = magpy.mu_0 * H
cube = magpy.magnet.Cuboid(dimension=(10 * MM,) * 3, polarization=(0, 0, 0))
cube.susceptibility = BHCurve(H=H, B=1000 * mu0H / (1 + 1000 * mu0H / 1.6) + mu0H)

for B0 in (0.01, 0.5, 2.0):
    solved = solve_cells(magpy.Collection(mesh_uniform(cube, 4), UniformField(B=(0, 0, B0))))
    J = np.mean([c.polarization[2] for c in solved.children[0].sources_all])
    print(f"applied {B0 * 1e3:6.0f} mT: mean polarization {J * 1e3:5.0f} mT, {J / B0:.2f} times")
```

At 10 mT the cube is linear, and its polarization is 3.4 times the applied field
on these 64 cells: a cube of infinite susceptibility reaches 3.64 times
{cite}`helsing2013`. At 0.5 T it is past the knee of its curve, and at 2 T it
saturates near the curve's 1.6 T. [Saturation](../examples/saturation.md) checks
a sphere against its exact answer.

## How many cells

The answer converges as the cells shrink, slowly: halving the cell size takes
eight times the cells and cuts the error to about a third. Soft iron is the hard
case, because its polarization gathers at edges and corners. Start coarse,
multiply the cells by four to eight, and stop when the quantity you need, a
field at a sensor or a force, changes by less than you can accept. `study_cells`
does that for you and estimates the error of the finest mesh:
[How accurate is a result?](accuracy.md).

## Don't read on a cell boundary

Touching cells each count a point on their shared surface as inside, so the
field there is wrong, with no warning: J counts 2, 4 or 8 times, and B can be
far off. The centre of a part with an even number of cells across is such a
point, as is any point on a symmetry plane through it. A 10 mm cube with χ = 10
in 0.1 T, 6 cells across, reads B = 0.107 T at its centre and 0.29 T 1 µm off it
along each axis. Use an odd number of cells across, or read a little off the
boundaries.
