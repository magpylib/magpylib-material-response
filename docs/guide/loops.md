# Many solves

A tolerance study, a magnet moving past a sensor, a sweep of a coil's current:
each step is a solve. This page is about doing many of them fast.

## A solved model is a snapshot

A solve is not a source you can move things around in: a magnet moved after the
solve leaves the part's charge as it was. B at the sensor stays right, and its
sensitivity to the magnet is wrong. For a plate under a magnet, the field
changes by 218 mT per mm of magnet shift, as finite elements agree, and by 146
mT if the magnet moves after one solve, as if the plate were not there. So a
tolerance study needs a solve per sample.

## Keep the factorization: `reuse=`

The matrix of a face-charge solve depends only on the solved parts. A loop that
changes only what drives them, such as a magnet's pose, a coil's current or a
`UniformField`, or that moves all solved parts as one, can keep its
factorization:

```python
solved = True
for pose in poses:
    solved = solve_face_charge(model_at(pose), reuse=solved)
    B.append(magpy.getB(solved, sensor))
```

Each call compares its meshed parts with the previous result's. If they make the
same matrix it takes only a new load and a back-substitution, as
`face_charge_info["reused"]` tells; if not, it solves afresh and keeps that
factorization instead. So a poor match costs time, never a wrong field. Fields
agree with a fresh solve to 1e-13 of the largest, 2e-11 at a weak observer.

### What keeps the matrix

The matrix depends on the solved parts' panels, their susceptibilities and their
poses relative to each other. Everything else enters only the load. A cuboid
magnet close to a part regrades the part's panels toward its footprint: close
means within 20 % of the largest half-side of the smaller of the two, 0.5 mm for
a 5 mm magnet. Measured on a 20 × 20 × 2 mm plate with χ 1000 at level 2:

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

For the plate under a magnet, a call takes 72 ms at level 3 and then 18 ms, and
1.15 s at 9,558 unknowns and then 0.12 s. The call that factors takes up to
twice as long as a plain solve from 2,000 unknowns on, 1.9 s at 9,558, which the
first match pays back.

The factorization takes 8 n² bytes for n unknowns, on the result: it goes when
the result goes. Keep the last result only, as the loop above does. A loop in
which every call changes the matrix gains nothing from `reuse`: the rows marked
"no" above.

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
sources with `getB`. For a plate under a magnet, 664 triangles solved at level
1, B at 2,601 points takes 25 ms with `getB` and 0.4 ms from the matrix, which
took 43 ms to build: the build is paid back by the second field. The two agree
to 3e-15 of the largest field.

The matrix holds `3 M n` numbers for M observers and n unknowns, three per
panel: 62 MB for those 2,601 points and 996 unknowns, or 31 MB with
`dtype=np.float32`, which stays within 1e-6 of the largest field around the
sheets. A sheet whose mesh or pose differs from the matrix's is refused, so a
magnet close enough to grade a part's panels, or a part that moves, needs a new
matrix.

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

The parts are copies, so `solved` stays as it was. Moving a sheet of `solved`
instead moves its field too, without solving, and its part with it. Each `copy`
of a drive adds a suffix to its label, as magpylib's does.

Pass `reuse=solved` as well, if the first solve had `reuse=True`, to keep its
factorization when only the drives moved. Leave the sheets out: one kept beside
its own part counts its response twice, and warns.

`to_json` keeps it, so a triangle-mesh part holds its mesh twice in the file. A
result solved with `refine` cannot be reproduced exactly from its parts, as the
panels it split are not kept. A sheet built without `part=`, with its mesh or
charge changed, or kept as a fixed source of a later solve has no part, nor has
a result solved before version 0.9.1: solve its original model instead. Inside a
studio `@scene` function, build the parts with their class: a part read from
`part` is not recorded, and the scene refuses it.

## Paths

A model whose objects move along a Magpylib path is refused by every solver: one
solve is one configuration. Loop over the steps as above, with
`magpy.Collection` built at each step. What a solve along a whole path would
take, step by step and for each kind of change, is recorded in
[#184](https://github.com/magpylib/magpylib-material-response/issues/184).

## Cells and finite elements in a loop

`solve_cells` and `solve_fem` keep nothing between calls: each solves afresh.
For `solve_cells` on a grid of identical cuboids that is fast anyway, as the
iterative solver never builds the matrix ([Inside cells](../inside/cells.md)).
