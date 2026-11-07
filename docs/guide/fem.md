# Finite elements

`solve_fem` solves the same Magpylib model by finite elements, with
[NGSolve](https://ngsolve.org/). It is independent of the other two solvers: the
parts keep their own shapes, the air around them is meshed too, and the field is
a polynomial on each element. Use it to check a result of `solve_face_charge` or
`solve_cells`, or where you need an error bar you can trust. This page answers
what it takes, how to set it, what it returns, how far to trust it and what it
costs.

It needs the `fem` extra, `uv add "magpylib-material-response[fem]"`, or with
pip as [Getting started](../getting_started.md) shows. NGSolve (LGPL-2.1) has no
wheels for Linux on ARM or for musl.

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

What it refuses, overlapping parts and some currents among them, is listed in
[Building a model](model.md#what-is-refused).

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

[Reading a solved model](reading.md) reads all three solvers' results side by
side.

## Its error bar: `study_fem`

`study_fem(model, quantity)` solves at orders 2, 3 and 4 by default, on one
mesh, and takes 1.25 times the last change as the error of the highest order. On
a cube of infinite susceptibility it finds the known magnetic moment to 2e-5 at
order 5. On 12 validation cases it reaches 0.1 % in 2 to 21 s; on the other
three their references are not that good ([Face charge](face_charge.md) has the
table). [How accurate is a result?](accuracy.md) says when the estimate holds,
which orders to take with a `BHCurve`, and how to set it beside the other
solvers; [Checking a result with finite elements](../examples/fem_check.md) runs
it.

## What it costs

Memory is the limit, not time. The
[magnet in a steel tube](../examples/shield_tube.md) takes 2.0 million unknowns
and 8.4 GB at order 4, in 50 s. On the validation cases finite elements reach
1 % in 0.5 to 11 s, 3 to over 100 times later than face charge
([Choosing a solver](choosing.md)).

## In short

Use `solve_fem` to check, not to sweep: pass `observers`, run `study_fem` for an
error bar, and keep the values, not the solution.
[Inside finite elements](../inside/fem.md) shows the formulation: the
potentials, the exact open boundary and the treatment of currents.
