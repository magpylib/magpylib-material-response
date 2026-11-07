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

# Saturation

A susceptibility χ is linear: a part with χ = 1000 carries any flux it is given,
while real steel saturates between about 1.5 and 2.1 T
([Material values](materials.md)). A part that carries more than its
steel can needs its B–H curve. That raises five questions: does a part saturate?
How does a datasheet's curve go in? Which solver follows it? How do you check
the result? What does it cost? This page answers them on a 4 × 4 × 8 mm steel
pin standing on a 20 × 20 × 10 mm NdFeB magnet, a pole piece that gathers the
magnet's flux into its tip.

## Does the part saturate?

Solve it linear first, with face charge, and read each sheet's
`mean_polarization`, the part's mean polarization in T:

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

from magpylib_material_response import BHCurve, solve_cells, solve_face_charge
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3  # lengths are in metres; this writes them in millimetres

magnet = magpy.magnet.Cuboid(
    dimension=(20 * MM, 20 * MM, 10 * MM), position=(0, 0, -5 * MM), polarization=(0, 0, 1.3)
)
pin = magpy.magnet.Cuboid(
    dimension=(4 * MM, 4 * MM, 8 * MM), position=(0, 0, 4 * MM), polarization=(0, 0, 0)
)
pin.susceptibility = 1000
model = magpy.Collection(magnet, pin)

linear = solve_face_charge(model)
print(f"mean polarization of the pin: {linear.children[1].mean_polarization[2]:.2f} T")
```

That is past this page's curve, which saturates at 1.58 T, and near the limit of
any steel: the linear answer overstates the pin, and the
field at its tip. Below the knee of the curve, the linear solve holds.

The mean is only a mean. Where the flux turns inside a part, as in a plate under
a magnet, flux in opposite directions cancels in it, and the mean stays low
while the steel saturates locally. There, read the flux density where the flux
is densest, as
[A holding magnet on a steel plate](../examples/holding_magnet.md) does under
the magnet's rim and
[A Hall current sensor](../examples/hall_current_sensor.md#at-what-current-does-the-ferrite-saturate)
does around its ferrite ring.

## Entering a datasheet's curve

`BHCurve(H=..., B=...)` takes the initial magnetization curve: H in A/m and B in
T, as two arrays of one length. The points below are the stand-in curve these
pages use (χ = 1000 at the origin, saturating at 1.6 T), listed as a datasheet
lists them; they are not a measured steel.

```{code-cell} ipython3
H = np.array([0, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 1e5])  # A/m
B = np.array([0, 0.061, 0.117, 0.217, 0.452, 0.705, 0.98, 1.282, 1.432, 1.706])  # T
steel = BHCurve(H=H, B=B)
print(steel)
print(f"J at H = 1 kA/m: {steel.polarization(1e3):.3f} T")
```

The curve is checked as it is built:

- **It starts at (0, 0),** with H increasing and B not decreasing, and at least
  two points.
- **B stays at or above μ0 H,** so that the polarization J = B − μ0 H is never
  negative.
- **A J above 5 T warns:** the columns are swapped, or the units wrong. Convert
  oersted to A/m (1 Oe = 1000/(4π) A/m) and gauss or mT to T first.
- **A table is refused** as a susceptibility: give the two columns to `BHCurve`.

Between the points B follows a monotone cubic, so it never wiggles. Past the
last point J keeps its last value, 1.58 T here, and B rises as in vacuum. End
the table well into saturation. A curve has no hysteresis: a field that falls
again goes back down the same curve.

A part's `polarization` stays its remanence, and the curve's J adds to it. For a
part that saturates, leave `polarization` at zero.

## Which solver follows it?

Set the curve as the part's `susceptibility`, on the part or on a collection
above it, and solve with `solve_cells` or `solve_fem`. Face charge refuses a
part on a curve: its charge sits on the surface, and a part that saturates in
places has charge inside too.

```{code-cell} ipython3
pin.susceptibility = steel
saturated = solve_cells(mesh_uniform(model, 8))  # 8 cells across the pin

tip = (0, 0, 9 * MM)  # 1 mm above the pin
print(f"Bz at the tip, magnet alone: {magpy.getB(magnet, tip)[2] * 1e3:.0f} mT")
print(f"Bz at the tip, linear pin  : {magpy.getB(linear, tip)[2] * 1e3:.0f} mT")
print(f"Bz at the tip, on the curve: {magpy.getB(saturated, tip)[2] * 1e3:.0f} mT")
```

Saturation cuts the field at the tip by a tenth. Each cell takes the
susceptibility its own field gives on the curve, and the solve repeats until
none of them changes. `solve_fem(model)` takes the same model, with the
susceptibility per element. A run for this page, at order 3 with the tip as
observer, gave 346 mT there in 90 s. Start finite elements at order 3: at order
2 the iteration can stop with "did not settle".

## How do you check the result?

Compare the cells with the curve. Their polarization says how much of the part
is saturated, and their field whether the table reaches far enough: a cell past
its last point sits on the flat extension, at the last J.

```{code-cell} ipython3
cells = saturated.children[1].sources_all
J = np.linalg.norm([c.polarization for c in cells], axis=1)
volumes = np.array([c.volume for c in cells])
centres = np.array([c.position for c in cells])
H_cells = np.linalg.norm(magpy.getH(saturated, centres), axis=1)
print(f"mean Jz: {np.array([c.polarization[2] for c in cells]) @ volumes / volumes.sum():.2f} T")
print(f"cells within 1 % of the curve's last J, {steel.J_last:.2f} T: {np.mean(J > 0.99 * steel.J_last):.0%}")
print(f"cells past its last point, {steel.H_last:,.0f} A/m: {np.mean(H_cells > steel.H_last):.0%}")
```

The pin's mean falls from the linear 1.88 T to 1.41 T, and 14 % of its cells lie
past the table. If the steel still rises there, extend the table: the extension
holds J at its last value.

The field on the curve converges with the cells as a linear one does. A finer
mesh moves it a little: 16 cells across gave 348 mT, 1 % below the 8 across, in
13 s. [How accurate is a result?](accuracy.md) shows how to estimate that error
with `study_cells`, and `study_fem` with `orders=(3, 4, 5)` for a part on a
curve.

## What it costs

A part on a curve takes 3 to 40 linear solves, more on a sharp knee. In
`solve_cells` the interaction matrix is built once, so each further solve is
cheap. Against the same mesh solved linear:

```{code-cell} ipython3
import time

seconds = {}
for name, chi in (("linear", 1000), ("on the curve", steel)):
    pin.susceptibility = chi
    start = time.perf_counter()
    solve_cells(mesh_uniform(model, 8))
    seconds[name] = time.perf_counter() - start
    print(f"{name:>12}: {seconds[name]:.2f} s")
print(f"the curve takes {seconds['on the curve'] / seconds['linear']:.0f} times as long")
```

`solve_fem` assembles its matrix again at each of its 5 to 40 solves; a
saturating part may need `solver_tol` of about 1e-7 ([Finite elements](fem.md)).
A sharp knee, rising within about 100 A/m, can stop either iteration with an
error: give it more points, or a gentler rise. [Logging](logging.md) reports how
many solves a curve took.

## In short

- Solve linear first: a `mean_polarization` past the knee of the curve, or a
  flux density past it where the flux is densest, means the part saturates.
- Enter the curve as `BHCurve(H=..., B=...)`, in A/m and T, from (0, 0), well
  into saturation, on a part with zero `polarization`.
- Solve with `solve_cells`, or `solve_fem` from order 3; face charge refuses a
  curve.
- Check the cells against the curve's last point, and refine as for a linear
  part.

[The saturation example](../examples/saturation.md) checks a sphere against its
exact answer and a U-core at high current;
[Inside saturation](../inside/saturation.md) shows how the solve follows the
curve.
