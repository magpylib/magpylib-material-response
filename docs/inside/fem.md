# Inside finite elements

`solve_fem` hands the model to NGSolve {cite}`schoberl2014`. It is built to be
independent of the other two solvers: the parts keep their shapes, the air is
meshed, and the open space is closed exactly. Code references are to
`src/magpylib_material_response/fem/`.

## The potentials

Without currents the field is a gradient everywhere. The solver takes the
reduced scalar potential $u$,

$$
\mu_0 \mathbf{H} = \mathbf{B}_\text{ext} - \nabla u ,
\qquad
\nabla\cdot\bigl(\mu_r \nabla u\bigr) = \nabla\cdot\bigl(\chi \mathbf{B}_\text{ext} + \mathbf{J}_r\bigr)
$$

in all space, with $u \to 0$ far away, $\mathbf{B}_\text{ext}$ the field of the
fixed magnets and the `UniformField`, and $\mu_r = 1 + \chi$. A uniform
$\mathbf{B}_\text{ext}$ is linear in $u$, which every order holds exactly, so a
high susceptibility costs no accuracy.

**With currents**, the coil's field $\mathbf{B}_c$ comes from Magpylib, and the
coil is never meshed. In soft iron the field is small against a coil's, and the
reduced potential would have to cancel the coil's field there almost exactly. So
the parts take the total potential $\psi$, $\mu_0\mathbf{H} = -\nabla\psi$, and
the air keeps the reduced one, $\mu_0\mathbf{H} = \mathbf{B}_\text{ext} +
\mathbf{B}_c - \nabla u$ {cite}`simkin1979`. On a part's surface the two are
joined through a potential of the applied field there, projected on each solve's
elements, so the unknowns and the matrix stay those of the solve without
currents; only the right-hand side changes.

## The open boundary

Space is cut by a sphere of radius $R$, 1.5 times the radius of the box around
the parts. The outside is mapped onto a second ball by the Kelvin transform,

$$
w(\mathbf{y}) = \frac{R}{|\mathbf{y}|}\, u(\mathbf{x}), \qquad \mathbf{y} = \frac{R^2\,\mathbf{x}}{|\mathbf{x}|^2},
$$

which keeps $w$ harmonic, and the two balls are glued along the sphere: the
boundary is exact, not a distant box {cite}`freeman1988,chen1997`. In three
dimensions the energy outside gains a surface term,

$$
\int_\text{outside} |\nabla u|^2\, \mathrm{d}x = \int_\text{ball} |\nabla w|^2\, \mathrm{d}y + \frac{1}{R}\oint_\text{sphere} w^2\, \mathrm{d}s ,
$$

and with it a soft sphere's moment comes out within 1e-7 of the exact one.

## The mesh

Netgen meshes the parts and the air {cite}`schoberl1997` (`_geometry.py` builds
the parts as OpenCASCADE solids).

- **In the parts** the elements are `maxh` across, by default each part's
  thinnest extent.
- **Toward the edges and corners**, where a soft part's field is singular, the
  mesh is refined geometrically: `edge_layers` layers, 4 by default, each 0.3
  times as thick as the last. With high orders this gives an error that falls
  geometrically with the order {cite}`babuska1996`.
- **At the observers** passed as `observers`, the elements shrink to a share of
  their distance to the nearest part: 0.03 mm at 0.1 mm from a 1 mm cube's edge
  converged the field there to 0.006 mT, where the parts' element size left it 1
  mT off.
- **Near the sphere**, the elements are capped in size, as curved elements at
  the far size read the field wrong close to it.

## The solve

The elements are polynomials of `order` 3 by default. The system is solved by
conjugate gradients {cite}`hestenes1952` with a BDDC preconditioner
{cite}`dohrmann2003`, to `solver_tol`, 1e-10 by default.

A part on a `BHCurve` repeats the linear solve until its susceptibility stops
moving, as `solve_cells` does ([Saturation](saturation.md)). Here the
susceptibility is a field on each element, its logarithm linear across it, and
each solve assembles the matrix again: 5 to 40 solves, at most 70. The iteration
stops when the susceptibility on the worst element moves by less than 1e-3.

## The result

A `FemSolution` is a Magpylib source. Its `getB` reads the solution at any
point: inside the ball from the elements, outside from their image in the Kelvin
ball. It holds the field of the whole model, drives included, and its `moment`.

It cannot be pickled: a pickled NGSolve solution crashes on its first read, so
pickling is refused. `study_fem` raises the order on one mesh and estimates the
error of the highest ([Error estimates](error_estimates.md)).
