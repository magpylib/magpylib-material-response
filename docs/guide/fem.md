# Finite elements

`solve_fem` solves the same Magpylib model by finite elements, with
[NGSolve](https://ngsolve.org/). It is independent of the other two solvers: the
parts keep their own shapes, the air around them is meshed too, and the field is
a polynomial on each element. Use it to check a result of `solve_face_charge` or
`solve_cells`, or where you need an error bar you can trust.

```{include} ../_install.md

```

```python
import magpylib as magpy

from magpylib_material_response import solve_fem

solved = solve_fem(model, observers=sensor_positions)  # order 3 by default
B = magpy.getB(solved, sensor_positions)
print(solved.ndof, solved.seconds, solved.moment)
```

## What it takes

- **Shapes:** cuboids, cylinders, cylinder segments, spheres, tetrahedra and
  closed triangle meshes, unmeshed.
- **Materials:** remanence and a susceptibility, isotropic or along the part's
  own axes, or a `BHCurve`.
- **Drives:** magnets without susceptibility, a `UniformField`, and currents.
  Currents are never meshed: their field is Magpylib's.
- **Refused:** a current through a part or through the hole of a ring magnet, an
  open current path, and parts that overlap.

## Settings

| Argument      | Default                     | What it does                                                              |
| ------------- | --------------------------- | ------------------------------------------------------------------------- |
| `order`       | 3                           | the polynomial order of the elements; the error falls fast with it        |
| `observers`   | none                        | positions or sensors where you will read: the mesh is made fine there     |
| `maxh`        | each part's thinnest extent | the element size in the parts, one value or one per part                  |
| `edge_layers` | 4                           | layers of refinement toward the parts' edges, where the field is singular |
| `solver_tol`  | 1e-10                       | the iterative solver's tolerance; a saturating part may need about 1e-7   |

Pass `observers` whenever you know where you will read: the field there
converges much faster.

## What it returns

A `FemSolution`, a Magpylib source: `magpy.getB`, `getH`, `getJ` and `getM` read
it, alone or with other sources. It holds the field of the whole model, the
magnets' own and the drives', and the solve's `order`, `ndof` (unknowns),
`elements`, `seconds` and `moment`.

- **Inside a part** with a high susceptibility, the values are pointwise and
  show the elements: H there is small next to its error.
- **Moving it** moves the field as it is, without solving again.
- **It cannot be saved.** A pickled solution crashes on its first read, so
  pickling is refused: save the values you read, not the solution.

## Its error bar: `study_fem`

`study_fem(model, quantity, orders=(2, 3, 4))` solves at rising orders on one
mesh, and takes the last change, times 1.25, as the error of the highest order,
per entry and in all: with three orders whose changes shrink, otherwise the
error is NaN, with a warning. `quantity` is a set of positions or sensors, or a
function of the solution such as `lambda s: s.moment`. On a cube of infinite
susceptibility it finds the known polarizability to 2e-5 at order 5; it reaches
0.1 % on every validation case in 24 s at most.
[How accurate is a result?](accuracy.md) shows how to set it beside the other
solvers.

## What it costs

Memory is the limit, not time. The
[magnet in a steel tube](../examples/shield_tube.md) takes 1.9 million unknowns
and 8.4 GB at order 4, in 50 s. On the validation cases it reaches 1 % in 0.5 to
11 s, 3 to 104 times slower than face charge ([Choosing a solver](choosing.md)).

[Inside finite elements](../inside/fem.md) shows the formulation: the
potentials, the exact open boundary and the treatment of currents.
