# Inside face charge

`solve_face_charge` puts the unknowns on the parts' surfaces only, with no mesh
inside them or in the air. That raises five questions: why the surface is
enough, what equation gives the charge on it, how the surface is cut into
panels, how parts that touch are handled, and how the system is solved and read.
This page answers them in that order, with the papers each step comes from.
[Face charge](../guide/face_charge.md) in the user guide gives the settings.

## Why the surface is enough

Face charge works in magnetization, $\mathbf{M} = \mathbf{J}/\mu_0$ in A/m,
because the charge it solves for, $\sigma = \mathbf{M}\cdot\mathbf{n}$, is one.
Inside a part of uniform, isotropic susceptibility $\chi$ and uniform remanence
$\mathbf{J}_r$, the magnetization is $\mathbf{M} = \mathbf{J}_r/\mu_0 + \chi
\mathbf{H}$, so $\nabla\cdot\mathbf{H} = \nabla\cdot\mathbf{M} / \chi$. With
$\nabla\cdot\mathbf{B} = \mu_0 \nabla\cdot(\mathbf{H} + \mathbf{M}) = 0$ that
gives

$$
\mu_0 \left(\frac{1}{\chi} + 1\right) \nabla\cdot\mathbf{M} = 0 ,
$$

so $\nabla\cdot\mathbf{M} = 0$: the part carries no magnetic charge in its
volume. Its whole field is that of the charge $\sigma$ on its surface,
$\mathbf{n}$ the outward normal. An anisotropic susceptibility, one that varies
inside the part, or a B–H curve breaks this, which is why those go to
`solve_cells`.

## The equation

Just inside the surface, the normal field is the applied one, plus the field of
all the charge, minus half the local charge, the jump across a charged sheet.
Writing $\mathbf{M}\cdot\mathbf{n} = \sigma$ there gives one equation for
$\sigma$:

$$
\left(\frac{1}{\chi} + \frac{1}{2}\right) \sigma - K'\sigma
= \frac{\mathbf{B}_\text{ext}\cdot\mathbf{n}}{\mu_0} + \frac{\mathbf{J}_r\cdot\mathbf{n}}{\mu_0\chi}
$$

$\mathbf{B}_\text{ext}$ is the applied field at the surface, from Magpylib: that
of every fixed source, magnets, coils and a `UniformField`. $K'$ is the adjoint
double-layer operator, the normal field at one point of the surface of a unit
charge at another, which depends on the shape only. Several parts share one
system: each part's equation holds the field of the others' charge.

The total charge of a closed part is zero. Each part's system is bordered with
one row and one column that fix it, or one per closed shell of a triangle mesh.

## Panels

Each face is cut into panels, and each panel carries a charge linear across it,
$c_0 + c_1 s + c_2 t$ in its local coordinates: three unknowns per panel,
discontinuous from one panel to the next.

- **Cuboids**: rectangles. A soft part's charge is singular at its edges, so the
  panels are graded toward each edge. With $m$ = `level`, $m$ panels fill a
  layer of depth $\ell$, the part's smallest side, their edges at $\ell
  (k/m)^{2.5}$ for $k = 1 \ldots m$; panels of the layer's largest size fill the
  rest of the face.
- **Triangle meshes**: the flat faces are merged back into polygons, so that a
  CAD export's long diagonals do not cross the layers, and each polygon is cut
  in rings inward from its sharp edges, where the triangles turn by more than
  20°. The layers are as deep as half the part's thickness. A flat face with
  vertices inside it, as a finite-element surface mesh has, is rebuilt from its
  outline, so it costs the same however it was cut.
- **Cylinders and spheres**: a curved face becomes `12 (level + 1)` flat facets
  per turn, 48 at level 3, widened so that the part keeps its true volume, and
  is then meshed as a triangle mesh. The widening matters more than the facets'
  number: on a soft cylinder, 24 widened facets are 0.02 % off 50 mm away, 96
  facets without it 0.16 %. A sphere is a geodesic polyhedron, finer with each
  level.

![The plate under a magnet at level 3 with its panels drawn, and at level 5: the panels shrink toward every edge and toward the magnet's footprint.](../_static/images/face_charge_plate.png)

## The Galerkin system

The equation is tested with the panels' own functions, a Galerkin method: each
entry of the matrix is the flux of one panel's charge through another panel,
weighted by a test function.

$$
\left(\frac{1}{\chi} + \frac{1}{2}\right) \mathsf{M} c - \mathsf{G} c = F + \frac{F_r}{\chi}
$$

$\mathsf{M}$ is the (diagonal) mass matrix, $F$ and $F_r$ the loads of the
applied field and of the remanence. $\mathsf{G}$ holds the fluxes. They are
integrals, over the source panel, of the test panel's weighted solid angle,
which stays bounded where the field itself is logarithmically singular, at a
shared edge. Panels on one face give each other no flux.

- **Closed forms**: the field of a flat rectangle or triangle with a linear
  charge is a sum over its edges of logarithms and arctangents
  {cite}`wilton1984,graglia1993`, and the solid angle of a triangle has a closed
  form too {cite}`oosterom1983`. They are checked against numerical integration
  to 1e-9.
- **Quadrature over the source panel**: the closed-form weighted solid angle of
  the test panel is integrated at the source panel's Gauss points; 7 points on a
  triangle, 4 × 7 for pairs about as close as they are long.
- **Far pairs**: in large systems, cuboid panel pairs at least 3 panel sides
  apart take 3 Gauss points per side instead of the closed form, and pairs 16
  sides apart take 2 points per side. This moves answers by under 1e-6 and makes
  assembly up to 1.55 times faster on 8 threads.

On a cube of infinite susceptibility the method reproduces the known
polarizability, 3.6443 times the volume {cite}`helsing2013`, from below, as a
Galerkin method should: 4.5e-4 off at level 4.

## Parts that touch or come close

- **Faces in one plane** couple exactly. The flux of a charge through a test
  panel in its own plane jumps, and the one-sided limit from inside the test
  part gives −1/2 of the integral of the two charges over their overlap. The
  overlap of the two panels is clipped as a polygon and integrated exactly, so
  the two meshes need not match.
- **Parts of one material that touch** are solved as one: the faces they share
  carry no charge. A short bar between two wide ones is 0.83 % off the same
  solid as one mesh at level 2, where leaving both sides' charges to cancel was
  16 % off.
- **A part close to another**, nearer than a fifth of the smaller part's largest
  half-side, is graded toward the other's outline, as for touching parts: a 10 ×
  10 × 2 mm plate beside a larger part is graded once the gap is below 1 mm. A
  magnet without susceptibility counts too, when it is a cuboid: the panels
  follow its footprint. Near any magnet, the load takes a finer rule.

## Solving

All the parts form one dense system: $8 n^2$ bytes for $n$ unknowns, 3.2 GB at
the default `max_unknowns` of 20,000.

- **Below 2,000 unknowns**, a double-precision LU from LAPACK
  {cite}`lapack1999`.
- **From 2,000 unknowns**, the LU is taken in single precision and refined
  against the double matrix to double's accuracy, as LAPACK's `dsgesv` does
  {cite}`langou2006,carson2018`: 2 to 4 times faster, fields within 2e-9. The
  single copy takes the 20,000-unknown system from 3.2 to 4.8 GB.
- **`reuse=True`** keeps the double LU on the result. Its fingerprint is taken
  from the meshed panels, not the input, so a part whose panels a nearby magnet
  has changed never reuses a wrong matrix. [Many solves](../guide/loops.md) says
  what keeps the factorization.
- **`refine=k`** splits, k times, the panels where the solved charge jumps most
  and solves again: a rectangle in two along both axes, a triangle into four.

## The result: `ChargeSheet`

Each solved part becomes a `ChargeSheet`, a Magpylib `CustomSource` that holds
triangles and the charge at their corners, in the part's place in the
collection.

- **Its field** is the closed form near the sheet. An observer farther from the
  whole sheet than 64 of its longest triangle sides takes 6 points per triangle
  instead, exact to degree 4 {cite}`dunavant1985`: there the closed form loses
  digits to cancellation.
- **Inside the part** it gives μ0 H and a warning: B there needs the part's
  magnetization, which a surface charge does not carry.
- **`mean_polarization`** is the part's mean J: μ0 times the first moment of the
  charge, the integral of $\sigma \mathbf{r}$ over the surface, divided by the
  volume. It tells whether saturation matters
  ([Inside saturation](saturation.md)).
- **`jump_indicator`** is the largest jump of the charge between neighbouring
  panels, weighted by panel size: the true charge is continuous on a flat face.
  [Inside the error estimates](error_estimates.md) says what it measures.
- **`field_matrix`** writes the sheets' field at fixed points as one matrix
  times their charges, for many solves read at the same points
  ([Many solves](../guide/loops.md)).

Assembly and fields run on `magpy.defaults.threads` threads, in chunks sized so
that all of them together take the memory one would alone.

## In short

The surface is enough because a linear, isotropic, uniform part has no charge in
its volume. Panels graded toward edges and contacts carry a linear charge,
tested in Galerkin form; one dense LU solves all the parts together, in mixed
precision above 2,000 unknowns, and can be kept for a loop. The
<a href="map.html#fc">solver map</a> shows this path box by box.

Code, in `src/magpylib_material_response/face_charge/`: panels in `_panels.py`
and `_bodies.py` (cuboids), `_mesh.py` and `_tribody.py` (triangle meshes) and
`_apply.py` (curved faces); closed forms in `_rectangle.py` and `_triangle.py`;
contacts in `_contact.py`; the kept factorization in `_kept.py`; the result in
`_sheet.py` and `_matrix.py`; threads in `_parallel.py`.
