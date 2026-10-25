# Face charge: linear parts without a volume mesh

`apply_demag` fills every part with cells. For a part whose susceptibility is
uniform and isotropic, such as soft iron in its linear range or a magnet with a
small susceptibility, that is more than the physics needs: such a part carries
no magnetic charge inside, and its whole response is a charge on its surface.
`apply_face_charge` solves only that charge, on panels over the faces, without a
volume mesh.

```python
import magpylib as magpy

from magpylib_material_response.face_charge import apply_face_charge

plate = magpy.magnet.Cuboid(
    dimension=(0.01, 0.01, 0.004),
    polarization=(0, 0, 0),
)
plate.susceptibility = 1000
magnet = magpy.magnet.Cuboid(
    dimension=(0.004, 0.004, 0.003),
    position=(0, 0, 0.0045),
    polarization=(0, 0, 1.2),
)

solved = apply_face_charge(magpy.Collection(plate, magnet))
B = solved.getB((0, 0, 0.0035))
```

The result is a copy of the collection in which the plate is a `ChargeSheet`, an
ordinary magpylib source: `getB`, observer paths, `show()` and studio work as
for any other. A cuboid whose susceptibility is exactly 0 is not solved and
stays as it was. `show()` colours the solved charge in the magnet colours,
positive charge being a north pole; `style.mesh.grid.show` draws the panels.

![The plate under the magnet: the plate turns north (red) under the magnet's south face and south (green) where the flux returns, on its rim and sides; level 3 with its panels drawn, and level 5.](_static/images/face_charge_plate.png)

## When it fits

- **What it solves:** cuboids, alone or touching, placed and turned freely, each
  with a uniform isotropic susceptibility and a uniform remanence, driven by any
  other magpylib sources and by the applied field `H_ext`.
- **How well:** on the six documented examples it lands within each reference's
  own uncertainty, 0.64 % at worst, where `apply_demag` at the documented
  settings is off by up to 23 %. On soft iron it is 20 to 40 times more accurate
  than `apply_demag` in the same time.
- **Where it is slower:** a few weak magnets alone (`apply_demag` reaches the
  same accuracy faster), many touching bodies (one dense coupled system) and
  thin sheets (panels sized to the thickness across the whole sheet).
- **What it leaves to `apply_demag`:** other shapes, saturation, an anisotropic
  susceptibility and one varying inside a part, all of which bring back charge
  inside the part, and objects moving along paths. `apply_face_charge` raises
  for these.

## How accurate a solve is

`level` sets the panels in each graded layer toward a face's edges. Fields far
from the parts converge at low levels; fields within a panel's size of a surface
need higher ones.

Each solved sheet reports its `jump_indicator`: the largest jump of the charge
between neighbouring panels, weighted by panel size. The true charge is
continuous across a flat face, so the jump is discretization error, read off the
solve for free. For the plate under the magnet the worst error of B 0.25 mm
above the plate is 1 to 3 times it.

`refine` uses the same jumps: each round splits the panels where they are
largest and solves again. For the plate, `refine=2` from level 3 (2,688
unknowns, under 2 s) puts B 0.25 mm above the plate within 0.04 %, where uniform
panels need 10,350 unknowns and 20 s for 0.13 %.

## Whether saturation matters

A sheet's `mean_polarization` is the average polarization of the part it
encloses. While it stays below the knee of the material's B–H curve, the linear
solve holds. Above it, saturation takes over: mesh the part and use
`apply_demag` with a B–H iteration.

## How it works

With `M = M0 + chi H` and `div B = 0`, a uniform isotropic `chi` gives
`div M = 0` inside, so the part is fully described by the surface charge
`sigma = M.n`, which solves

$$
(1/\chi + 1/2)\,\sigma - K'\sigma = H_{ext}\cdot n + M_0\cdot n / \chi ,
$$

with `K'` the adjoint double-layer operator of the surface, which depends on the
shape only. The faces are cut into panels graded toward the edges, where a soft
part's charge is singular. Each panel carries a charge linear across it, with
fields in closed form, and the equation is solved in its Galerkin form. Touching
parts are coupled exactly, even when their panels do not match. On a cube the
method reproduces the polarizability of Helsing and Perfekt (2013) from below,
as a Galerkin method should: 4.5e-4 off at level 4.
