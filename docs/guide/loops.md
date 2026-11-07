# Many solves: sweeps, moving parts, Monte Carlo

A tolerance study, a magnet moving past a sensor, a sweep of a coil's current:
each step is a solve. Why can't one solve serve every step, and how do you make
many solves fast? This page answers in order: why each step needs its own solve,
how `reuse=` keeps the factorization, how `field_matrix` reads a field map in
one product, how to solve a solved model again, and what cells and finite
elements keep between calls.

## A solved model is a snapshot

Moving a magnet after the solve leaves the parts' response where it was. For a 5
mm cube magnet 1 mm above a 20 × 20 × 2 mm plate with χ = 1000, Bz 0.5 mm above
the plate beside the magnet changes by 17 mT per mm of sideways magnet shift in
face charge, and by 16 mT per mm in finite elements. Moving the magnet in the
solved model gives 1.9 mT per mm, the change of the magnet's own field, as if
the plate were not there. So a tolerance study needs a solve per sample.

A model whose objects move along a Magpylib path is refused
([Building a model](model.md#what-is-refused)): loop over the steps instead,
building the model at each.

## Keep the factorization: `reuse=`

The matrix of a face-charge solve depends only on the solved parts. A loop that
changes only what drives them, such as a magnet's pose, a coil's current or a
`UniformField`, or that moves all solved parts as one, can keep its
factorization. Start with `reuse=True`, so that the first solve keeps it:

```python
solved = True
for pose in poses:
    solved = solve_face_charge(model_at(pose), reuse=solved)
    B.append(magpy.getB(solved, sensor))
```

Each call compares its meshed parts with the previous result's. If they make the
same matrix, it takes only a new load and a back-substitution, and
`face_charge_info["reused"]` is `True`. If not, it solves afresh and keeps that
factorization instead. A poor match costs time, never a wrong field: the fields
match a fresh solve to rounding error.

[A gear-tooth sensor](../examples/gear_tooth_sensor.md),
[An angle sensor on a steel shaft](../examples/angle_sensor.md) and
[Assembly tolerances, by Monte Carlo](../examples/tolerance_monte_carlo.md)
solve in such loops.

### What keeps the matrix

The matrix depends on the solved parts' panels, their susceptibilities and their
poses relative to each other. Everything else enters only the load. A cuboid
magnet close to a part regrades the part's panels toward its footprint. Close
means within a tenth of the smaller one's largest side: 0.5 mm for a 5 mm
magnet. Measured on a 20 × 20 × 2 mm plate with χ = 1000 at level 2:

| What changes between solves                                                      | Matrix kept |
| -------------------------------------------------------------------------------- | ----------- |
| A magnet's polarization, a coil's current, a dipole's moment                     | yes         |
| A `UniformField`'s value or direction                                            | yes         |
| A coil or a polyline moving or changing shape, even 0.05 mm from the part        | yes         |
| A magnet without susceptibility moving, turning or resized, not close to a part  | yes         |
| A magnet that is not a cuboid moving anywhere: only cuboids grade the panels     | yes         |
| The solved parts moving or turning all together                                  | yes         |
| A solved part's remanence                                                        | yes         |
| A cuboid magnet moving, turning or resized close to a part                       | no          |
| Solved parts moving against each other, or a magnet with a susceptibility moving | no          |
| A solved part's size                                                             | no          |

### What it saves

For the plate under the magnet at levels 2 and 5, the time per call:

| Unknowns | Plain solve | First call with `reuse` | Later calls |
| -------: | ----------: | ----------------------: | ----------: |
|    1,848 |       95 ms |                   95 ms |       21 ms |
|    8,526 |      0.90 s |                   1.3 s |      0.11 s |

The first call keeps the factorization, which from about 2,000 unknowns takes up
to 1.5 times as long as a plain solve; the first match pays that back. A loop in
which every call changes the matrix, the rows marked "no" above, gains nothing
from `reuse` and pays that extra on every call.

The factorization takes 8 n² bytes for n unknowns and lives on the result: it
goes when the result goes. Keep the last result only, as the loop above does.

## A field map in a loop: `field_matrix`

Each solve of the loop above gives sheets with the same mesh and a new charge,
as long as the parts stay where they are. Their field at a fixed set of
observers, a field map or the points field lines start from, is then the same
matrix times a new charge each time. `field_matrix` builds that matrix once, and
each later field is one product:

```python
from magpylib_material_response.face_charge import ChargeSheet, field_matrix


def sheets_of(solved):
    return [s for s in solved.sources_all if isinstance(s, ChargeSheet)]


solved = solve_face_charge(model_at(poses[0]), reuse=True)
fm = field_matrix(sheets_of(solved), points)  # B at points = fm.matrix @ charges
for pose in poses:
    solved = solve_face_charge(model_at(pose), reuse=solved)
    sheets = sheets_of(solved)
    others = [s for s in solved.sources_all if not any(s is k for k in sheets)]
    B = fm(sheets) + magpy.getB(others, points, sumup=True)
```

Only the sheets are in the matrix: add the field of the magnets and other
sources with `getB`. For the plate under the magnet, 560 triangles solved at
level 1, B at 2,601 points takes 21 ms with `getB` and 0.3 ms from the matrix,
which took 46 ms to build: the build is paid back by the third field. The two
agree to 2e-15 of the largest field.

The matrix holds `3 M n` numbers for M observers and n unknowns, three per
panel: 52 MB for those 2,601 points and 840 unknowns. With `dtype=np.float32` it
takes 26 MB and stays within 2e-7 of the largest field around the sheets, but
far away, at 2,000 times the plate's size, it is up to 3e-4 off: keep float64
for observers far away. A sheet whose mesh or pose differs from the matrix's is
refused, so a magnet close enough to grade a part's panels, or a part that
moves, needs a new matrix.

## Solve a solved model again

A solved model can be solved again from what it holds. Each sheet's `part` is
the magnet it was solved from, at the sheet's pose now, with its susceptibility
on it. Everything else is a drive, and goes in as a copy, as an object has one
parent:

```python
from magpylib_material_response.face_charge import ChargeSheet

sources = solved.sources_all
sheets = [s for s in sources if isinstance(s, ChargeSheet) and s.part is not None]
drives = [s.copy() for s in sources if not any(s is k for k in sheets)]
parts = [s.part for s in sheets]
parts[0].move((0.001, 0, 0))  # a drag
level = solved.face_charge_info["level"]
again = solve_face_charge(magpy.Collection(*parts, *drives), level=level)
```

The parts are copies, so `solved` stays as it was. Pass `reuse=solved` as well,
if the first solve had `reuse=True`, to keep its factorization when only the
drives moved. Leave the sheets out: one kept beside its own part counts its
response twice, and warns.

`to_json` keeps each sheet's part, so a result loaded from a file
([Saving and loading](saving.md)) can be solved again too. A sheet without a
part, such as one built by hand, cannot, and a result solved with `refine` comes
back without its split panels: solve the original model instead.

## Cells and finite elements in a loop

`solve_cells` and `solve_fem` keep nothing between calls: each solves afresh.
For `solve_cells` on a grid of identical cuboids that is fast anyway, as the
iterative solver never builds the matrix ([Inside cells](../inside/cells.md)).

## In short

Solve once per configuration. When only the drives change, pass `reuse=` and
keep the last result. To read the same points after every solve, build a
`field_matrix` once.
