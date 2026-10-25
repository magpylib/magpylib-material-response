# Face charge: linear, uniform cuboids solved on their surface

A solver for cuboids with a uniform isotropic susceptibility χ (and remanence)
in the field of other magpylib sources, without a volume mesh. The solve takes
milliseconds; the result is a magpylib source per body. Not package code yet:
ruff and codespell skip this folder and CI does not run it.

The study behind it (validation against the MoM and the six documented examples,
triangle meshes and CAD exports, the figures) is on the branch
[`explore/face-charge`](https://github.com/magpylib/magpylib-material-response/tree/explore/face-charge/explorations/face-charge).

## Use

```python
import fc  # validation/fc.py

solved = fc.apply_face_charge(collection, m=3)  # a copy: each soft cuboid is now a ChargeSheet
B = solved.getB(sensors)  # like getB of a collection solved by material-response
```

Cuboids with a susceptibility (on themselves or a parent collection) are solved;
every other source stays as it is and acts on them. `H_ext` is a uniform applied
field, excluded from `getB` as in material-response. `m` is the mesh level.

## How

- **Faces only.** With uniform isotropic χ and uniform remanence, `div M = 0`
  inside the body, so each body is a surface charge σ = M·n on its faces. The
  faces are split into panels graded toward the edges (power 2.5), each carrying
  a linear charge (P1 discontinuous Galerkin), with closed-form panel fields.
- **Coupled bodies.** Touching faces get contact terms and footprint lines;
  bodies may be placed and turned freely.
- **`ChargeSheet`** (`source.py`) carries the solved charge: vertices (n, 3),
  faces (m, 3) and `charge_densities` (m, 3), per face corner as the charge
  jumps from one face to the next, like magpylib's `current.TriangleSheet`. It
  is a `magpylib.misc.CustomSource`, so `getB`, paths, collections and `show()`
  work as for any source. It gives H everywhere and B outside the bodies.
- **Drawing.** `ChargeSheet` takes a `TriangularMesh`'s style. Its `get_trace`
  colours the charge in `style.magnetization.color`, positive charge being a
  north pole (plotly and pyvista; matplotlib draws one colour), smoothed per
  flat face and on a square-root scale: the jumps between panels are
  discretization error, and the square root shows the weak charge where flux
  returns into a soft part. `style.mesh.grid.show` draws the panel edges. A body
  with χ exactly 0 is not solved and keeps magpylib's own drawing.
- **Accuracy hint.** `ChargeSheet.jump_indicator`, and `jump_indicator` in
  `fc.solve`'s info, is the largest jump of the charge between panels meeting at
  a corner, as a share of the peak charge, weighted by the panel's size (√(panel
  area / sheet area)). On a flat face the true charge is continuous, so the jump
  is discretization error. Unweighted it stalls at about 16 % in the first
  panels at a soft body's edges, where the charge is singular; weighted, it
  points where the panels are too coarse for the field around the body. For the
  plate under a magnet it sits under the magnet's corners and falls from 5.2 %
  at level 3 to 1.9 % at level 5 and 0.35 % at level 9; the worst error of B
  0.25 mm above the plate is 1.1 to 2.8 times it, against level 11. Far fields
  are 0.01 to 0.8 % off at level 3. The study is in `adaptive/` on
  `explore/face-charge`.

## Files

| File               | What                                                               |
| ------------------ | ------------------------------------------------------------------ |
| `bem.py`           | rectangle kernel, graded panel meshes                              |
| `p1.py`            | linear-density closed forms, P1 Galerkin assembly, load, field     |
| `multibody.py`     | bodies, contact coupling, footprint lines, coupled solve           |
| `triangles/tri.py` | closed-form field of a triangle with linear charge                 |
| `source.py`        | `ChargeSheet`                                                      |
| `validation/fc.py` | `split`, `solve` and `apply_face_charge` for a magpylib collection |
| `t_source.py`      | checks, a few seconds: `python t_source.py`                        |

## Checks (`t_source.py`)

- uniform charge against magpylib's own `misc.Triangle`: 5e-15;
- linear charge against uniform pieces: 1.9e-3, 4.7e-4, 1.2e-4 at 64, 256 and
  1,024 pieces, the second-order convergence of the pieces to it;
- a collection through `apply_face_charge` against `fc.solve`: 1e-11; the
  sheet's `jump_indicator` equals what `fc.solve` reports;
- a path of two positions turned 30°, against the same by hand: 5e-16; a copy
  with doubled charge, and two sources in one `getB`: exact;
- `show()` in plotly, matplotlib and pyvista, plotly's colours exactly the
  square root of the smoothed charge. Smoothing moves the charge by 0.2 % or
  less of its peak at half the corners, and by 27 % at most, at the corners of
  the four 2 mm panels under the magnet (14 % at level 5). The grid draws the
  532 panel edges, the 798 triangle edges less one diagonal per panel.

On magpylib 6.0.0 the field function, a bound method, is set past the setter,
and the source stays in memory after `getB`; magpylib main fixes both
([#1045](https://github.com/magpylib/magpylib/pull/1045),
[#1054](https://github.com/magpylib/magpylib/pull/1054)).

## Limits

- Anisotropic χ, saturation and a χ varying inside a body bring back volume
  charge; they stay with the MoM.
- Sensors closer than about 0.25 mm to a soft body's edge or corner see about
  1e-2 error.
- Moving a coupled part means a rebuild of about 3 s; a new polarization, χ or
  external field takes milliseconds.
